package systems

import (
	"log"
	"math"
	"minimetro-go/components"
	"minimetro-go/config"
	"minimetro-go/state"
	"minimetro-go/systems/graph"
	"sort"
	"time"
)

// GhostPhase tracks the lifecycle of a temporary "ghost" line used to rapidly
// evacuate a critically overcrowded station.
type GhostPhase int

const (
	GhostIdle  GhostPhase = iota
	GhostArmed            // line + train exist; waiting for boarding or timeout
)

type ghostLineState struct {
	Phase   GhostPhase
	Line    *components.Line
	Train   *components.Train
	ArmedAt float64 // sim time when armed; abort after 8 000 ms
}

// Circle-cluster detection thresholds.
const (
	circleDensityRadius   = 200.0 // pixels: radius for cluster scan
	circleDensityMinCount = 3     // minimum circles in range to trigger Titan Line
)

// SA launch interval: how often (in sim-ms) we start a new background search.
const saLaunchIntervalMs = 2000.0

// Solver is the autonomous AI player. When Enabled, it calls tick() every
// runInterval ms of simulation time and makes exactly one topology-changing
// action per tick, prioritised by urgency.
//
// Advanced capabilities added on top of the base solver:
//   - Simulated Annealing topology search (background goroutine, forward sim).
//   - Betweenness Centrality tracking for preemptive interchange placement.
//   - Circle-cluster detection ("Paradise of Circles") → Titan Line response.
//   - TOC-style priority queue: stations sorted by overcrowd fraction.
type Solver struct {
	Enabled     bool
	lastRunMs   float64
	runInterval float64
	ghost       ghostLineState

	// SA background search
	saResultCh  <-chan SAResult
	lastSAMs    float64

	// Centrality cache (recomputed on graph change)
	centrality       map[*components.Station]float64
	centralityDirty  bool
}

func NewSolver() *Solver {
	return &Solver{
		Enabled:         false,
		runInterval:     300.0,
		centralityDirty: true,
	}
}

// Update is called every frame from main.go, after GameSystem.Update.
func (s *Solver) Update(gs *state.GameState, gm *graph.GraphManager, nowMs float64) {
	if !s.Enabled || gs.Paused || gs.GameOver {
		return
	}

	// Recompute centrality whenever graph topology has changed.
	// CachedCentrality deduplicates the O(V·E) Brandes pass with obs.go.
	if gs.GraphDirty || s.centralityDirty {
		s.centrality = gm.CachedCentrality(gs)
		s.centralityDirty = false
	}

	// Poll completed SA result (non-blocking).
	if s.saResultCh != nil {
		select {
		case result := <-s.saResultCh:
			s.saResultCh = nil
			s.applySAResult(gs, result, nowMs)
		default:
		}
	}

	// Respond faster when stations are near overcrowding.
	interval := s.runInterval
	if len(s.stationsAbove(gs, config.OvercrowdCriticalThreshold)) > 0 {
		interval = 80.0
	} else if len(s.stationsAbove(gs, 0.6)) > 0 {
		interval = 150.0
	}
	if nowMs-s.lastRunMs < interval {
		return
	}
	s.lastRunMs = nowMs
	s.tick(gs, gm, nowMs)

	// Launch a new SA search if none is running and enough time has elapsed.
	if s.saResultCh == nil && nowMs-s.lastSAMs >= saLaunchIntervalMs {
		s.lastSAMs = nowMs
		s.saResultCh = SAOptimize(gs, 500*time.Millisecond)
		log.Printf("[Solver] SA search launched")
	}
}

// applySAResult applies the best SA action to the real game state, provided it
// represents a genuine improvement over the current network cost.
func (s *Solver) applySAResult(gs *state.GameState, result SAResult, nowMs float64) {
	if result.Action == nil {
		return
	}
	baseline := NetworkCost(gs)
	if result.Score >= baseline {
		log.Printf("[Solver] SA result (%.1f) not better than baseline (%.1f) — discarded", result.Score, baseline)
		return
	}
	if ApplyPerturbation(gs, result.Action) {
		log.Printf("[Solver] SA action applied (type=%d line=%d) score %.1f → %.1f",
			result.Action.Type, result.Action.LineIdx, baseline, result.Score)
	}
}

// ChooseUpgrade picks the most beneficial upgrade for the current game state.
func (s *Solver) ChooseUpgrade(gs *state.GameState, choices []string) string {
	if len(choices) == 0 {
		return ""
	}
	best, bestScore := choices[0], math.Inf(-1)
	for _, c := range choices {
		sc := s.scoreUpgrade(gs, c)
		if sc > bestScore {
			bestScore = sc
			best = c
		}
	}
	log.Printf("[Solver] Upgrade chosen: %s (score=%.1f)", best, bestScore)
	return best
}

func (s *Solver) scoreUpgrade(gs *state.GameState, upgrade string) float64 {
	switch upgrade {
	case UpgradeNewLine:
		if gs.AvailableLines >= gs.MaxLines {
			return -1000
		}
		isolated := s.isolatedStations(gs)
		score := float64(len(isolated)) * 25
		for _, st := range gs.Stations {
			score += st.OvercrowdProgress * 0.04
		}
		// Bonus proportional to how geographically extended the existing lines
		// are: long lines signal that the network needs more routes, not longer ones.
		score += s.avgLineArcLength(gs) * 0.06
		// Base preference so that new lines are always competitive early on.
		score += 18
		return score

	case UpgradeCarriage:
		score := 0.0
		for _, t := range gs.Trains {
			if t.Line == nil || !t.Line.Active || t.Line.MarkedForDeletion {
				continue
			}
			cap := t.TotalCapacity()
			if cap == 0 {
				continue
			}
			ratio := float64(len(t.Passengers)) / float64(cap)
			if ratio > 0.5 {
				score += ratio * 20
			}
		}
		for _, st := range gs.Stations {
			score += st.OvercrowdProgress * 0.03
		}
		return score

	case UpgradeBridge:
		if gs.Bridges == 0 {
			return 60
		}
		if gs.Bridges == 1 {
			return 35
		}
		// Extra value if isolated stations can only be reached via river crossings.
		isolated := s.isolatedStations(gs)
		for _, st := range isolated {
			for _, other := range gs.Stations {
				if other != st && s.riverCost(gs, st, other) > 0 {
					return 40
				}
			}
		}
		return 15

	case UpgradeInterchange:
		score := float64(gs.Week) * 2
		for _, st := range gs.Stations {
			if st.IsInterchange {
				continue
			}
			lines := s.linesThroughStation(gs, st)
			if len(lines) >= 2 {
				score += float64(len(lines))*8 + st.OvercrowdProgress*0.1
			}
		}
		return score
	}
	return 0
}

// tick executes at most one action, in strict priority order.
func (s *Solver) tick(gs *state.GameState, gm *graph.GraphManager, nowMs float64) {
	// Always advance ghost state machine first (no topology change, no early return).
	s.updateGhost(gs, nowMs)

	// --- Priority 1: EMERGENCY (> 85 % overcrowd) — TOC priority queue ---
	// Stations are sorted by overcrowd fraction descending (Theory of Constraints:
	// attack the binding constraint first).
	critical := s.stationsAbove(gs, config.OvercrowdCriticalThreshold)
	if len(critical) > 0 {
		sort.Slice(critical, func(i, j int) bool {
			return critical[i].OvercrowdProgress > critical[j].OvercrowdProgress
		})
		for _, st := range critical {
			if s.tryGhostLine(gs, st, nowMs) {
				log.Printf("[Solver] Ghost line armed for %s station", st.Type)
				return
			}
			if s.tryAddCarriageToLineThroughStation(gs, st) {
				log.Printf("[Solver] Carriage added to line through %s station", st.Type)
				return
			}
			if s.tryAddTrainToLineThroughStation(gs, st) {
				log.Printf("[Solver] Train deployed to line through %s station", st.Type)
				return
			}
			if s.tryInterchange(gs, st) {
				log.Printf("[Solver] Interchange upgraded for %s station", st.Type)
				return
			}
		}
	}

	// --- Priority 2: connect isolated stations ---
	isolated := s.isolatedStations(gs)
	if len(isolated) > 0 {
		sort.Slice(isolated, func(i, j int) bool {
			return isolated[i].OvercrowdProgress > isolated[j].OvercrowdProgress
		})
		for _, st := range isolated {
			if s.tryConnectIsolated(gs, st) {
				log.Printf("[Solver] Connected isolated %s station", st.Type)
				return
			}
		}
	}

	// --- Priority 3: deploy spare trains to active lines ---
	if gs.AvailableTrains > 0 {
		if s.tryDeployTrain(gs) {
			log.Printf("[Solver] Deployed spare train")
			return
		}
	}

	// --- Priority 3.5: rebalance trains between over/under-provisioned lines ---
	if s.tryRebalanceTrains(gs) {
		log.Printf("[Solver] Rebalanced train between lines")
		return
	}

	// --- Priority 4: add carriages to loaded trains ---
	if gs.Carriages > 0 {
		if s.tryAddCarriage(gs) {
			log.Printf("[Solver] Added carriage to busy train")
			return
		}
	}

	// --- Priority 4.5: close loops proactively ---
	// Loops are the preferred line topology: they eliminate terminal starvation
	// and distribute trains evenly. Close any eligible line (≥ 3 stations) as
	// soon as the network is fully connected.
	if len(s.isolatedStations(gs)) == 0 {
		if s.tryCloseLoop(gs) {
			log.Printf("[Solver] Closed loop on a line")
			return
		}
	}

	// --- Priority 5: respond to circle-cluster ("Paradise of Circles") ---
	// A Titan Line is a short dedicated drain connecting a circle-heavy cluster
	// to the nearest non-circle station, absorbing overflow before it cascades.
	clusters := s.detectCircleClusters(gs, circleDensityRadius, circleDensityMinCount)
	for _, hub := range clusters {
		if s.tryTitanLine(gs, hub) {
			log.Printf("[Solver] Titan Line deployed for circle cluster at (%.0f,%.0f)", hub.X, hub.Y)
			return
		}
	}

	// --- Priority 5.5: split overly long loops into two interlinked rings ---
	// A loop with too many stations has a huge cycle time, causing local starvation
	// at the far end. Splitting it at the midpoint creates two smaller rings that
	// share a junction station, halving the worst-case wait time.
	if len(s.isolatedStations(gs)) == 0 {
		if s.trySplitLoop(gs) {
			log.Printf("[Solver] Split long loop into two rings")
			return
		}
	}

	// --- Priority 6: preemptive interchange at high-centrality junction ---
	// Upgrade the highest-centrality non-interchange station before it overflows.
	if gs.Interchanges > 0 {
		if s.tryUpgradeCentralNode(gs) {
			log.Printf("[Solver] Preemptive interchange at high-centrality station")
			return
		}
	}

	// --- Priority 7: upgrade interchange on moderately overcrowded station ---
	if gs.Interchanges > 0 {
		if s.tryUpgradeInterchange(gs) {
			log.Printf("[Solver] Upgraded interchange")
			return
		}
	}
}

// ---------------------------------------------------------------------------
// Ghost line tactic
// ---------------------------------------------------------------------------

func (s *Solver) updateGhost(gs *state.GameState, nowMs float64) {
	if s.ghost.Phase != GhostArmed {
		return
	}
	boarded := len(s.ghost.Train.Passengers) > 0
	timedOut := nowMs-s.ghost.ArmedAt > config.GhostLineTimeoutMs
	if boarded || timedOut {
		s.ghost.Line.MarkedForDeletion = true
		s.ghost = ghostLineState{Phase: GhostIdle}
		log.Printf("[Solver] Ghost line marked for deletion (boarded=%v, timedOut=%v)", boarded, timedOut)
	}
}

func (s *Solver) tryGhostLine(gs *state.GameState, target *components.Station, nowMs float64) bool {
	if s.ghost.Phase != GhostIdle {
		return false
	}
	if gs.AvailableTrains == 0 {
		return false
	}

	spareLine := s.firstSpareLine(gs)
	if spareLine == nil {
		return false
	}

	destType := s.mostNeededType(target)
	dest := s.nearestStationOfType(gs, target, destType)
	if dest == nil {
		return false
	}

	cost := s.riverCost(gs, target, dest)
	if gs.Bridges < cost {
		return false
	}

	markDirty := func() { gs.GraphDirty = true }
	spareLine.AddStation(target, -1, markDirty)
	spareLine.AddStation(dest, -1, markDirty)
	gs.Bridges -= cost

	cityCfg := config.Cities[gs.SelectedCity]
	train := components.NewTrain(gs.TrainIDCounter, spareLine, cityCfg.TrainCapacity, config.TrainMaxSpeed)
	gs.AddTrain(train)
	spareLine.Trains = append(spareLine.Trains, train)
	gs.AvailableTrains--

	s.ghost = ghostLineState{
		Phase:   GhostArmed,
		Line:    spareLine,
		Train:   train,
		ArmedAt: nowMs,
	}
	return true
}

func (s *Solver) mostNeededType(station *components.Station) config.StationType {
	counts := make(map[config.StationType]int)
	for _, p := range station.Passengers {
		counts[p.Destination]++
	}
	var best config.StationType
	bestN := 0
	for t, n := range counts {
		if n > bestN {
			bestN = n
			best = t
		}
	}
	return best
}

func (s *Solver) nearestStationOfType(gs *state.GameState, from *components.Station, stationType config.StationType) *components.Station {
	var best *components.Station
	bestDist := math.MaxFloat64
	for _, st := range gs.Stations {
		if st == from || st.Type != stationType {
			continue
		}
		d := math.Hypot(st.X-from.X, st.Y-from.Y)
		if d < bestDist {
			bestDist = d
			best = st
		}
	}
	return best
}

// ---------------------------------------------------------------------------
// Connect isolated stations
// ---------------------------------------------------------------------------

// avgActiveLineLength returns the mean number of (real) stations across all
// active lines. Used to decide when to favour starting a new line.
func (s *Solver) avgActiveLineLength(gs *state.GameState) float64 {
	total, count := 0, 0
	for i := 0; i < gs.AvailableLines && i < len(gs.Lines); i++ {
		l := gs.Lines[i]
		if !l.Active || l.MarkedForDeletion {
			continue
		}
		n := len(l.Stations)
		isLoop := n >= 3 && l.Stations[0] == l.Stations[n-1]
		if isLoop {
			n-- // don't count the duplicate closing station
		}
		total += n
		count++
	}
	if count == 0 {
		return 0
	}
	return float64(total) / float64(count)
}

func (s *Solver) tryConnectIsolated(gs *state.GameState, station *components.Station) bool {
	// When a spare line and train are available AND existing lines are already
	// long (avg ≥ 4 real stations), prefer starting a new dedicated line over
	// growing an already-large one. This keeps individual lines short, which
	// produces smaller cycle times and better geographic coverage.
	const newLineAvgLenThreshold = 4.0
	if gs.AvailableTrains > 0 {
		spare := s.firstSpareLine(gs)
		if spare != nil && s.avgActiveLineLength(gs) >= newLineAvgLenThreshold {
			if s.startNewLineFor(gs, spare, station) {
				return true
			}
		}
	}

	// Extend an existing active line (open or loop).
	line, idx, ok := s.bestLineForStation(gs, station)
	if ok {
		s.attachStation(gs, line, station, idx)
		return true
	}

	// Open a closed loop to make room, then extend to the isolated station.
	if s.tryOpenLoopAndExtend(gs, station) {
		return true
	}

	// Last resort: start a new line even if avg length is below threshold.
	if gs.AvailableTrains > 0 {
		spare := s.firstSpareLine(gs)
		if spare != nil {
			if s.startNewLineFor(gs, spare, station) {
				return true
			}
		}
	}

	return false
}

// startNewLineFor starts a brand-new line on spare for station, pairing it with
// the best available partner. Returns false when no viable partner exists.
func (s *Solver) startNewLineFor(gs *state.GameState, spare *components.Line, station *components.Station) bool {
	partner := s.bestPartnerForNewLine(gs, station)
	if partner == nil {
		return false
	}
	cost := s.riverCost(gs, station, partner)
	if gs.Bridges < cost {
		return false
	}
	gs.Bridges -= cost
	markDirty := func() { gs.GraphDirty = true }
	spare.AddStation(station, -1, markDirty)
	spare.AddStation(partner, -1, markDirty)
	gs.AvailableTrains--
	cityCfg := config.Cities[gs.SelectedCity]
	train := components.NewTrain(gs.TrainIDCounter, spare, cityCfg.TrainCapacity, config.TrainMaxSpeed)
	gs.AddTrain(train)
	spare.Trains = append(spare.Trains, train)
	return true
}

// tryOpenLoopAndExtend opens a closed loop on the line best positioned to reach
// station, then extends the newly-opened tail to include station.
func (s *Solver) tryOpenLoopAndExtend(gs *state.GameState, station *components.Station) bool {
	markDirty := func() { gs.GraphDirty = true }

	var bestLine *components.Line
	bestDist := math.MaxFloat64

	for i := 0; i < gs.AvailableLines && i < len(gs.Lines); i++ {
		line := gs.Lines[i]
		if !line.Active || line.MarkedForDeletion {
			continue
		}
		n := len(line.Stations)
		if n < 2 || line.Stations[0] != line.Stations[n-1] {
			continue // not a loop
		}
		// Distance from the tail (which equals head after opening) to station.
		tail := line.Stations[n-2] // the real last station before loop closure
		dist := math.Hypot(tail.X-station.X, tail.Y-station.Y)
		cost := s.riverCost(gs, tail, station)
		if gs.Bridges < cost {
			continue
		}
		if dist < bestDist {
			bestDist = dist
			bestLine = line
		}
	}

	if bestLine == nil {
		return false
	}

	n := len(bestLine.Stations)
	// Remove the loop-closure duplicate (the last element equals the first).
	bestLine.RemoveEndStation(bestLine.Stations[n-1], markDirty)
	// Now extend to the isolated station at the tail.
	cost := s.riverCost(gs, bestLine.Stations[len(bestLine.Stations)-1], station)
	gs.Bridges -= cost
	bestLine.AddStation(station, -1, markDirty)
	return true
}

// bestPartnerForNewLine finds the best second station when starting a brand-new
// line for station. Prefers a different type, prefers proximity, excludes stations
// that would require more bridges than available.
// Axis-aligned partners (roughly N-S or E-W from station) receive a bonus
// because they form straight segments that minimise angle penalties and act as
// the "spine" of the network.
func (s *Solver) bestPartnerForNewLine(gs *state.GameState, station *components.Station) *components.Station {
	isolatedSet := make(map[*components.Station]bool)
	for _, st := range s.isolatedStations(gs) {
		isolatedSet[st] = true
	}

	var best *components.Station
	bestScore := -math.MaxFloat64
	for _, candidate := range gs.Stations {
		if candidate == station {
			continue
		}
		cost := s.riverCost(gs, station, candidate)
		if gs.Bridges < cost {
			continue
		}
		dist := math.Hypot(candidate.X-station.X, candidate.Y-station.Y)
		score := -dist * 0.05
		if candidate.Type != station.Type {
			score += 30.0
		}
		// Extra value for connecting two isolated stations simultaneously.
		if isolatedSet[candidate] {
			score += 20.0
		}
		// Prefer candidates with more waiting passengers (higher demand).
		score += float64(len(candidate.Passengers)) * 2
		// Axis-alignment bonus: prefer N-S or E-W segments as they form the
		// backbone grid and create natural soft-turn junctions later.
		angle := math.Atan2(candidate.Y-station.Y, candidate.X-station.X)
		axisBonus := math.Abs(math.Cos(2*angle)) * 12 // peaks at 0°,90°,180°,270°
		score += axisBonus
		if score > bestScore {
			bestScore = score
			best = candidate
		}
	}
	return best
}

// insertionTurnPenalty returns a score penalty for inserting a station B between
// A and C such that the turn A→B→C is sharp. The penalty is proportional to
// cos(ABC): 0 for straight/obtuse (≥ 90°), up to 50 for a direct U-turn.
func (s *Solver) insertionTurnPenalty(a, b, c *components.Station) float64 {
	if a == nil || b == nil || c == nil || a == c {
		return 0
	}
	v1x, v1y := a.X-b.X, a.Y-b.Y
	v2x, v2y := c.X-b.X, c.Y-b.Y
	d1 := math.Hypot(v1x, v1y)
	d2 := math.Hypot(v2x, v2y)
	if d1 < 1e-6 || d2 < 1e-6 {
		return 0
	}
	cosTheta := (v1x*v2x + v1y*v2y) / (d1 * d2)
	if cosTheta <= 0 {
		return 0
	}
	return cosTheta * 50
}

// ---------------------------------------------------------------------------
// Train deployment
// ---------------------------------------------------------------------------

func (s *Solver) tryDeployTrain(gs *state.GameState) bool {
	var bestLine *components.Line
	bestScore := -math.MaxFloat64
	for i := 0; i < gs.AvailableLines && i < len(gs.Lines); i++ {
		line := gs.Lines[i]
		if !line.Active || line.MarkedForDeletion {
			continue
		}
		if len(line.Trains) >= config.MaxTrainsPerLine {
			continue
		}
		load := s.passengerLoadOnLine(gs, line)
		score := float64(load) - float64(len(line.Trains))*10
		if score > bestScore {
			bestScore = score
			bestLine = line
		}
	}
	if bestLine == nil {
		return false
	}
	gs.AvailableTrains--
	cityCfg := config.Cities[gs.SelectedCity]
	train := components.NewTrain(gs.TrainIDCounter, bestLine, cityCfg.TrainCapacity, config.TrainMaxSpeed)
	gs.AddTrain(train)
	bestLine.Trains = append(bestLine.Trains, train)
	return true
}

func (s *Solver) tryAddTrainToLineThroughStation(gs *state.GameState, station *components.Station) bool {
	if gs.AvailableTrains == 0 {
		return false
	}
	lines := s.linesThroughStation(gs, station)
	var bestLine *components.Line
	bestLoad := -1
	for _, l := range lines {
		if !l.Active || l.MarkedForDeletion {
			continue
		}
		if len(l.Trains) >= config.MaxTrainsPerLine {
			continue
		}
		load := s.passengerLoadOnLine(gs, l)
		if load > bestLoad {
			bestLoad = load
			bestLine = l
		}
	}
	if bestLine == nil {
		return false
	}
	gs.AvailableTrains--
	cityCfg := config.Cities[gs.SelectedCity]
	train := components.NewTrain(gs.TrainIDCounter, bestLine, cityCfg.TrainCapacity, config.TrainMaxSpeed)
	gs.AddTrain(train)
	bestLine.Trains = append(bestLine.Trains, train)
	return true
}

// ---------------------------------------------------------------------------
// Carriage management
// ---------------------------------------------------------------------------

func (s *Solver) tryAddCarriage(gs *state.GameState) bool {
	var bestTrain *components.Train
	bestRatio := 0.6 // only act if > 60 % full
	for _, t := range gs.Trains {
		if t.CarriageCount >= config.MaxCarriagesPerTrain {
			continue
		}
		if t.Line == nil || !t.Line.Active || t.Line.MarkedForDeletion {
			continue
		}
		cap := t.TotalCapacity()
		if cap == 0 {
			continue
		}
		ratio := float64(len(t.Passengers)) / float64(cap)
		if ratio > bestRatio {
			bestRatio = ratio
			bestTrain = t
		}
	}
	if bestTrain == nil {
		return false
	}
	bestTrain.CarriageCount++
	gs.Carriages--
	return true
}

func (s *Solver) tryAddCarriageToLineThroughStation(gs *state.GameState, station *components.Station) bool {
	if gs.Carriages == 0 {
		return false
	}
	lines := s.linesThroughStation(gs, station)
	for _, l := range lines {
		if !l.Active || l.MarkedForDeletion {
			continue
		}
		for _, t := range l.Trains {
			if t.CarriageCount < config.MaxCarriagesPerTrain {
				t.CarriageCount++
				gs.Carriages--
				return true
			}
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Train rebalancing
// ---------------------------------------------------------------------------

// tryRebalanceTrains steals an empty train from an over-provisioned line
// (few passengers per train) and returns it to the available pool, where the
// normal deployment logic will redeploy it to a needier line.
func (s *Solver) tryRebalanceTrains(gs *state.GameState) bool {
	const donorThreshold    = 4.0 // passengers-per-train; below this a line is over-provisioned
	const recipientMinRatio = 8.0 // passengers-per-train; above this a line genuinely needs more

	var bestDonor *components.Line
	lowestRatio := math.MaxFloat64

	for i := 0; i < gs.AvailableLines && i < len(gs.Lines); i++ {
		line := gs.Lines[i]
		if !line.Active || line.MarkedForDeletion || len(line.Trains) < 2 {
			continue
		}
		load := s.passengerLoadOnLine(gs, line)
		ratio := float64(load) / float64(len(line.Trains))
		if ratio < lowestRatio {
			lowestRatio = ratio
			bestDonor = line
		}
	}

	if bestDonor == nil || lowestRatio > donorThreshold {
		return false
	}

	// Check that at least one other line actually needs more trains.
	needsTrains := false
	for i := 0; i < gs.AvailableLines && i < len(gs.Lines); i++ {
		line := gs.Lines[i]
		if !line.Active || line.MarkedForDeletion || line == bestDonor {
			continue
		}
		if len(line.Trains) >= config.MaxTrainsPerLine || len(line.Trains) == 0 {
			continue
		}
		load := s.passengerLoadOnLine(gs, line)
		ratio := float64(load) / float64(len(line.Trains))
		if ratio >= recipientMinRatio {
			needsTrains = true
			break
		}
	}

	// Also steal if there are isolated stations but no available trains.
	if !needsTrains && (gs.AvailableTrains > 0 || len(s.isolatedStations(gs)) == 0) {
		return false
	}

	return s.stealTrainFromLine(gs, bestDonor)
}

// stealTrainFromLine removes an empty (no passengers) train from line and
// returns it to the global pool (gs.AvailableTrains++).
func (s *Solver) stealTrainFromLine(gs *state.GameState, line *components.Line) bool {
	for i, t := range line.Trains {
		if len(t.Passengers) > 0 {
			continue
		}
		line.Trains = append(line.Trains[:i], line.Trains[i+1:]...)
		for j, gt := range gs.Trains {
			if gt == t {
				gs.Trains = append(gs.Trains[:j], gs.Trains[j+1:]...)
				break
			}
		}
		gs.Carriages += t.CarriageCount
		gs.AvailableTrains++
		gs.GraphDirty = true
		return true
	}
	return false
}

// ---------------------------------------------------------------------------
// Loop closing
// ---------------------------------------------------------------------------

func (s *Solver) tryCloseLoop(gs *state.GameState) bool {
	// Prefer short lines first: a 3-station triangle loop has the smallest cycle
	// time and is the ideal building block before the map grows.
	var best *components.Line
	bestLen := math.MaxInt32

	for i := 0; i < gs.AvailableLines && i < len(gs.Lines); i++ {
		line := gs.Lines[i]
		if !line.Active || line.MarkedForDeletion {
			continue
		}
		n := len(line.Stations)
		// Minimum 3 stations to form a meaningful loop (triangle).
		if n < 3 {
			continue
		}
		// Already a loop?
		if line.Stations[0] == line.Stations[n-1] {
			continue
		}
		cost := s.riverCost(gs, line.Stations[n-1], line.Stations[0])
		if gs.Bridges < cost {
			continue
		}
		if n < bestLen {
			bestLen = n
			best = line
		}
	}

	if best == nil {
		return false
	}
	n := len(best.Stations)
	cost := s.riverCost(gs, best.Stations[n-1], best.Stations[0])
	gs.Bridges -= cost
	best.AddStation(best.Stations[0], -1, func() { gs.GraphDirty = true })
	return true
}

// trySplitLoop finds the longest active loop with at least splitMinStations
// unique stations and splits it at its midpoint into two smaller interlinked
// rings. The split station becomes a shared junction between both rings.
//
// Before: A→B→C→D→E→F→A  (6 unique stations, long cycle time)
// After:  A→B→C→A  +  C→D→E→F→C  (two 3-station rings, half the cycle time)
//
// The junction station (C) is served by both rings, acting as an implicit
// interchange that keeps the network connected.
func (s *Solver) trySplitLoop(gs *state.GameState) bool {
	const splitMinStations = 6 // only split loops that are already large

	spare := s.firstSpareLine(gs)
	if spare == nil || gs.AvailableTrains == 0 {
		return false
	}

	// Find the longest active loop to split.
	var bestLoop *components.Line
	bestLen := 0
	for i := 0; i < gs.AvailableLines && i < len(gs.Lines); i++ {
		line := gs.Lines[i]
		if !line.Active || line.MarkedForDeletion {
			continue
		}
		n := len(line.Stations)
		if n < 2 || line.Stations[0] != line.Stations[n-1] {
			continue
		}
		unique := n - 1
		if unique > bestLen {
			bestLen = unique
			bestLoop = line
		}
	}

	if bestLoop == nil || bestLen < splitMinStations {
		return false
	}

	markDirty := func() { gs.GraphDirty = true }
	n := len(bestLoop.Stations)
	uniqueN := n - 1
	mid := uniqueN / 2 // index of the junction/split station

	splitStation := bestLoop.Stations[mid]
	lastStation := bestLoop.Stations[uniqueN-1]

	// Compute total bridge cost before committing.
	costClose2 := s.riverCost(gs, lastStation, splitStation)   // close new ring
	costClose1 := s.riverCost(gs, splitStation, bestLoop.Stations[0]) // close original ring
	if gs.Bridges < costClose1+costClose2 {
		return false
	}

	// Step 1: Open the original loop (remove the closing duplicate).
	bestLoop.RemoveEndStation(bestLoop.Stations[n-1], markDirty)
	clampTrainIndices(bestLoop)

	// Step 2: Build the spare line with the second half (mid .. uniqueN-1).
	for i := mid; i < uniqueN; i++ {
		spare.AddStation(bestLoop.Stations[i], -1, markDirty)
	}
	gs.Bridges -= costClose2
	spare.AddStation(splitStation, -1, markDirty) // close second ring

	// Spawn a train for the new ring.
	gs.AvailableTrains--
	cityCfg := config.Cities[gs.SelectedCity]
	train := components.NewTrain(gs.TrainIDCounter, spare, cityCfg.TrainCapacity, config.TrainMaxSpeed)
	gs.AddTrain(train)
	spare.Trains = append(spare.Trains, train)

	// Step 3: Trim the original line back to stations[0..mid].
	currentLen := len(bestLoop.Stations)
	for currentLen > mid+1 {
		tail := bestLoop.Stations[len(bestLoop.Stations)-1]
		bestLoop.RemoveEndStation(tail, markDirty)
		clampTrainIndices(bestLoop)
		currentLen = len(bestLoop.Stations)
	}

	// Step 4: Close the first ring back to stations[0].
	gs.Bridges -= costClose1
	bestLoop.AddStation(bestLoop.Stations[0], -1, markDirty)

	return true
}

// ---------------------------------------------------------------------------
// Interchange upgrade
// ---------------------------------------------------------------------------

func (s *Solver) tryInterchange(gs *state.GameState, station *components.Station) bool {
	if gs.Interchanges == 0 {
		return false
	}
	if station.IsInterchange {
		return false
	}
	if station.OvercrowdProgress < float64(config.OvercrowdTime)*config.OvercrowdCriticalThreshold {
		return false
	}
	station.IsInterchange = true
	gs.Interchanges--
	return true
}

func (s *Solver) tryUpgradeInterchange(gs *state.GameState) bool {
	var best *components.Station
	for _, st := range gs.Stations {
		if st.IsInterchange {
			continue
		}
		if st.OvercrowdProgress < float64(config.OvercrowdTime)*0.5 {
			continue
		}
		if best == nil || st.OvercrowdProgress > best.OvercrowdProgress {
			best = st
		}
	}
	if best == nil {
		return false
	}
	best.IsInterchange = true
	gs.Interchanges--
	return true
}

// ---------------------------------------------------------------------------
// Line selection helpers
// ---------------------------------------------------------------------------

// bestLineForStation finds the best active line to extend to station. For open
// lines it evaluates prepend and append; for closed loops it evaluates all
// interior insertion positions (matching the original game mechanic where you
// drag a station onto any segment of a loop).
func (s *Solver) bestLineForStation(gs *state.GameState, station *components.Station) (*components.Line, int, bool) {
	type candidate struct {
		line  *components.Line
		idx   int
		score float64
	}
	var candidates []candidate

	for i := 0; i < gs.AvailableLines && i < len(gs.Lines); i++ {
		line := gs.Lines[i]
		if !line.Active || line.MarkedForDeletion {
			continue
		}
		n := len(line.Stations)
		if n == 0 {
			continue
		}
		// Check if station is already on the line.
		alreadyOn := false
		for _, ls := range line.Stations {
			if ls == station {
				alreadyOn = true
				break
			}
		}
		if alreadyOn {
			continue
		}

		isLoop := n >= 3 && line.Stations[0] == line.Stations[n-1]
		covered := make(map[config.StationType]bool)
		for _, ls := range line.Stations {
			covered[ls.Type] = true
		}
		typeCoverage := 0.0
		if !covered[station.Type] {
			typeCoverage = 40.0
		}
		demand := s.demandMatchScore(line, station)
		urgency := station.OvercrowdProgress * 0.02

		// Penalty that grows quadratically with line length, discouraging
		// piling more stations onto already-large lines.
		lengthPenalty := math.Max(0, float64(n-3)) * math.Max(0, float64(n-3)) * 1.5

		if isLoop {
			// For loops, try every interior segment as the insertion point.
			// Real stations are indices 0..n-2; stations[n-1] == stations[0].
			for pos := 1; pos < n; pos++ {
				prev := line.Stations[pos-1]
				next := line.Stations[pos] // pos == n-1 means next == stations[0]
				// Net bridge cost: new crossings minus the crossing we eliminate.
				netCost := s.riverCost(gs, prev, station) + s.riverCost(gs, station, next) - s.riverCost(gs, prev, next)
				if netCost < 0 {
					netCost = 0
				}
				if gs.Bridges < netCost {
					continue
				}
				detour := math.Hypot(prev.X-station.X, prev.Y-station.Y) +
					math.Hypot(station.X-next.X, station.Y-next.Y) -
					math.Hypot(prev.X-next.X, prev.Y-next.Y)
				hs := 0.0
				if !covered[station.Type] {
					hs = 30.0
				}
				// Penalise sharp turns at the insertion point: prefer soft curves.
				turnPenalty := s.insertionTurnPenalty(prev, station, next)
				score := hs + demand + typeCoverage - detour*0.008 + urgency - lengthPenalty - turnPenalty
				candidates = append(candidates, candidate{line, pos, score})
			}
		} else {
			// Open line: evaluate prepend (idx=0) and append (idx=-1).
			for _, insertIdx := range []int{0, -1} {
				var neighbor *components.Station
				if insertIdx == 0 {
					neighbor = line.Stations[0]
				} else {
					neighbor = line.Stations[n-1]
				}
				cost := s.riverCost(gs, neighbor, station)
				if gs.Bridges < cost {
					continue
				}
				hs := s.hygieneScore(line, station, insertIdx)
				dist := math.Hypot(station.X-neighbor.X, station.Y-neighbor.Y)
				// Penalise the turn angle created at the current endpoint when
				// extending the line. A straight or obtuse continuation incurs
				// no penalty; a sharp bend is discouraged.
				turnPenalty := 0.0
				if insertIdx == -1 && n >= 2 {
					turnPenalty = s.insertionTurnPenalty(line.Stations[n-2], neighbor, station)
				} else if insertIdx == 0 && n >= 2 {
					turnPenalty = s.insertionTurnPenalty(station, neighbor, line.Stations[1])
				}
				score := hs + demand + typeCoverage - dist*0.008 + urgency - lengthPenalty - turnPenalty
				candidates = append(candidates, candidate{line, insertIdx, score})
			}
		}
	}

	if len(candidates) == 0 {
		return nil, 0, false
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})
	best := candidates[0]
	return best.line, best.idx, true
}

// hygieneScore rates how well inserting newStation at insertIdx improves line
// type diversity. Higher is better.
func (s *Solver) hygieneScore(line *components.Line, newStation *components.Station, insertIdx int) float64 {
	// Build hypothetical slice.
	src := line.Stations
	n := len(src)
	hyp := make([]*components.Station, 0, n+1)
	if insertIdx == 0 {
		hyp = append(hyp, newStation)
		hyp = append(hyp, src...)
	} else {
		hyp = append(hyp, src...)
		hyp = append(hyp, newStation)
	}

	score := 0.0
	runLen := 1
	for i := 1; i < len(hyp); i++ {
		if hyp[i].Type == hyp[i-1].Type {
			runLen++
			score -= math.Pow(2, float64(runLen)) * 10
		} else {
			runLen = 1
		}
	}

	// Bonus for introducing a type not yet on the line.
	covered := make(map[config.StationType]bool)
	for _, st := range line.Stations {
		covered[st.Type] = true
	}
	if !covered[newStation.Type] {
		score += 30
	}

	// Prefer shorter lines (more room to grow).
	score += 20 / float64(max(n, 1))

	return score
}

// attachStation adds station to line at insertIdx, paying the correct bridge
// cost and auto-spawning a train if the line just became active.
//
// For open lines insertIdx must be 0 (prepend) or -1 (append).
// For closed loops insertIdx is an interior position (1 ≤ idx < n); the net
// bridge cost accounts for the two new segments and the removed segment.
func (s *Solver) attachStation(gs *state.GameState, line *components.Line, station *components.Station, insertIdx int) {
	n := len(line.Stations)
	isLoop := n >= 3 && line.Stations[0] == line.Stations[n-1]

	if isLoop && insertIdx > 0 && insertIdx < n {
		prev := line.Stations[insertIdx-1]
		next := line.Stations[insertIdx]
		netCost := s.riverCost(gs, prev, station) + s.riverCost(gs, station, next) - s.riverCost(gs, prev, next)
		if netCost > 0 {
			gs.Bridges -= netCost
		}
	} else {
		var neighbor *components.Station
		if insertIdx == 0 && n > 0 {
			neighbor = line.Stations[0]
		} else if n > 0 {
			neighbor = line.Stations[n-1]
		}
		if neighbor != nil {
			cost := s.riverCost(gs, neighbor, station)
			gs.Bridges -= cost
		}
	}

	wasActive := line.Active
	line.AddStation(station, insertIdx, func() { gs.GraphDirty = true })

	if !wasActive && line.Active && gs.AvailableTrains > 0 && len(line.Trains) == 0 {
		gs.AvailableTrains--
		cityCfg := config.Cities[gs.SelectedCity]
		train := components.NewTrain(gs.TrainIDCounter, line, cityCfg.TrainCapacity, config.TrainMaxSpeed)
		gs.AddTrain(train)
		line.Trains = append(line.Trains, train)
	}
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

func (s *Solver) stationsAbove(gs *state.GameState, threshold float64) []*components.Station {
	limit := float64(config.OvercrowdTime) * threshold
	var result []*components.Station
	for _, st := range gs.Stations {
		if st.OvercrowdProgress >= limit {
			result = append(result, st)
		}
	}
	return result
}

func (s *Solver) isolatedStations(gs *state.GameState) []*components.Station {
	onLine := make(map[*components.Station]bool)
	for i := 0; i < gs.AvailableLines && i < len(gs.Lines); i++ {
		for _, st := range gs.Lines[i].Stations {
			onLine[st] = true
		}
	}
	var result []*components.Station
	for _, st := range gs.Stations {
		if !onLine[st] {
			result = append(result, st)
		}
	}
	return result
}

func (s *Solver) linesThroughStation(gs *state.GameState, station *components.Station) []*components.Line {
	var result []*components.Line
	for i := 0; i < gs.AvailableLines && i < len(gs.Lines); i++ {
		line := gs.Lines[i]
		for _, st := range line.Stations {
			if st == station {
				result = append(result, line)
				break
			}
		}
	}
	return result
}

func (s *Solver) firstSpareLine(gs *state.GameState) *components.Line {
	for i := 0; i < gs.AvailableLines && i < len(gs.Lines); i++ {
		l := gs.Lines[i]
		if len(l.Stations) == 0 && !l.MarkedForDeletion {
			return l
		}
	}
	return nil
}

// avgLineArcLength returns the mean geographic arc length (sum of segment
// distances) across all active lines. Used to gauge how stretched the current
// network is when scoring the new-line upgrade.
func (s *Solver) avgLineArcLength(gs *state.GameState) float64 {
	total := 0.0
	count := 0
	for i := 0; i < gs.AvailableLines && i < len(gs.Lines); i++ {
		l := gs.Lines[i]
		if !l.Active || l.MarkedForDeletion {
			continue
		}
		arc := 0.0
		for j := 0; j < len(l.Stations)-1; j++ {
			a, b := l.Stations[j], l.Stations[j+1]
			if a != b {
				arc += math.Hypot(b.X-a.X, b.Y-a.Y)
			}
		}
		total += arc
		count++
	}
	if count == 0 {
		return 0
	}
	return total / float64(count)
}

func (s *Solver) passengerLoadOnLine(gs *state.GameState, line *components.Line) int {
	load := 0
	for _, st := range line.Stations {
		load += len(st.Passengers)
	}
	return load
}

func (s *Solver) riverCost(gs *state.GameState, s1, s2 *components.Station) int {
	if CheckRiverCrossing(gs, s1, s2) {
		return 1
	}
	return 0
}

// demandMatchScore rewards connecting a station to a line that already serves
// the destination types that passengers at the station are waiting for.
func (s *Solver) demandMatchScore(line *components.Line, station *components.Station) float64 {
	covered := make(map[config.StationType]bool)
	for _, st := range line.Stations {
		covered[st.Type] = true
	}
	score := 0.0
	for _, p := range station.Passengers {
		if covered[p.Destination] {
			score += 5
		}
	}
	return score
}

// ---------------------------------------------------------------------------
// Circle-cluster detection and Titan Line
// ---------------------------------------------------------------------------

// detectCircleClusters scans the map for areas dense with Circle-type stations.
// For each cluster of at least minCount circles within radius pixels, it returns
// the most overcrowded station in that cluster as the representative hub.
//
// Complexity: O(n²) on circle count, acceptable for n < 50.
func (s *Solver) detectCircleClusters(gs *state.GameState, radius float64, minCount int) []*components.Station {
	var circles []*components.Station
	for _, st := range gs.Stations {
		if st.Type == config.Circle {
			circles = append(circles, st)
		}
	}

	visited := make(map[*components.Station]bool)
	var hubs []*components.Station

	for _, c := range circles {
		if visited[c] {
			continue
		}
		var cluster []*components.Station
		for _, other := range circles {
			if !visited[other] && math.Hypot(other.X-c.X, other.Y-c.Y) <= radius {
				cluster = append(cluster, other)
			}
		}
		if len(cluster) < minCount {
			continue
		}
		for _, st := range cluster {
			visited[st] = true
		}
		// Representative = most overcrowded station in the cluster.
		best := cluster[0]
		for _, st := range cluster[1:] {
			if st.OvercrowdProgress > best.OvercrowdProgress {
				best = st
			}
		}
		hubs = append(hubs, best)
	}
	return hubs
}

// tryTitanLine deploys a short, high-capacity line connecting a circle-cluster
// hub to the nearest non-circle station. It acts as a dedicated drain that
// prevents circle stations from cascading into game-over.
func (s *Solver) tryTitanLine(gs *state.GameState, hub *components.Station) bool {
	spare := s.firstSpareLine(gs)
	if spare == nil || gs.AvailableTrains == 0 {
		return false
	}

	// Nearest non-circle station.
	var dest *components.Station
	bestDist := math.MaxFloat64
	for _, st := range gs.Stations {
		if st.Type == config.Circle || st == hub {
			continue
		}
		d := math.Hypot(st.X-hub.X, st.Y-hub.Y)
		if d < bestDist {
			bestDist = d
			dest = st
		}
	}
	if dest == nil {
		return false
	}

	cost := s.riverCost(gs, hub, dest)
	if gs.Bridges < cost {
		return false
	}
	gs.Bridges -= cost

	markDirty := func() { gs.GraphDirty = true }
	spare.AddStation(hub, -1, markDirty)
	spare.AddStation(dest, -1, markDirty)

	gs.AvailableTrains--
	cityCfg := config.Cities[gs.SelectedCity]
	train := components.NewTrain(gs.TrainIDCounter, spare, cityCfg.TrainCapacity, config.TrainMaxSpeed)
	gs.AddTrain(train)
	spare.Trains = append(spare.Trains, train)
	return true
}

// ---------------------------------------------------------------------------
// Centrality-based preemptive interchange
// ---------------------------------------------------------------------------

// tryUpgradeCentralNode upgrades the highest-centrality non-interchange station
// that is also a junction (served by ≥ 2 lines) and has at least moderate
// overcrowding. This places interchanges at network bottlenecks before they
// become critical.
func (s *Solver) tryUpgradeCentralNode(gs *state.GameState) bool {
	if gs.Interchanges == 0 || s.centrality == nil {
		return false
	}

	const minCentrality = 0.05 // only act on genuinely important junctions
	const minOvercrowdFraction = 0.3

	var best *components.Station
	bestScore := -math.MaxFloat64

	for _, st := range gs.Stations {
		if st.IsInterchange {
			continue
		}
		c, ok := s.centrality[st]
		if !ok || c < minCentrality {
			continue
		}
		lines := s.linesThroughStation(gs, st)
		if len(lines) < 2 {
			continue
		}
		overcrowdFraction := st.OvercrowdProgress / float64(config.OvercrowdTime)
		if overcrowdFraction < minOvercrowdFraction {
			continue
		}
		score := c*10 + overcrowdFraction*5 + float64(len(lines))*2
		if score > bestScore {
			bestScore = score
			best = st
		}
	}

	if best == nil {
		return false
	}
	best.IsInterchange = true
	gs.Interchanges--
	return true
}

