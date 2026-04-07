package systems

import (
	"log"
	"math"
	"minimetro-go/components"
	"minimetro-go/config"
	"minimetro-go/state"
	"sort"
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

// Solver is the autonomous AI player. When Enabled, it calls tick() every
// runInterval ms of simulation time and makes exactly one topology-changing
// action per tick, prioritised by urgency.
type Solver struct {
	Enabled     bool
	lastRunMs   float64
	runInterval float64
	ghost       ghostLineState
}

func NewSolver(ih *InputHandler) *Solver {
	return &Solver{
		Enabled:     false,
		runInterval: 300.0,
	}
}

// Update is called every frame from main.go, after GameSystem.Update.
func (s *Solver) Update(gs *state.GameState, _ *GraphManager, nowMs float64) {
	if !s.Enabled || gs.Paused || gs.GameOver {
		return
	}
	if nowMs-s.lastRunMs < s.runInterval {
		return
	}
	s.lastRunMs = nowMs
	s.tick(gs, nowMs)
}

// tick executes at most one action, in strict priority order.
func (s *Solver) tick(gs *state.GameState, nowMs float64) {
	// Always advance ghost state machine first (no topology change, no early return).
	s.updateGhost(gs, nowMs)

	// --- Priority 1: EMERGENCY (> 85 % overcrowd) ---
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

	// --- Priority 4: add carriages to loaded trains ---
	if gs.Carriages > 0 {
		if s.tryAddCarriage(gs) {
			log.Printf("[Solver] Added carriage to busy train")
			return
		}
	}

	// --- Priority 5: close loops on long lines ---
	if s.tryCloseLoop(gs) {
		log.Printf("[Solver] Closed loop on a line")
		return
	}

	// --- Priority 6: upgrade interchange on moderately overcrowded station ---
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
	gs.TrainIDCounter++
	gs.Trains = append(gs.Trains, train)
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

func (s *Solver) tryConnectIsolated(gs *state.GameState, station *components.Station) bool {
	// First: extend an existing active line.
	line, idx, ok := s.bestLineForStation(gs, station)
	if ok {
		s.attachStation(gs, line, station, idx)
		return true
	}

	// Second: start a new line if a spare slot and a spare train are available.
	if gs.AvailableTrains == 0 {
		return false
	}
	spare := s.firstSpareLine(gs)
	if spare == nil {
		return false
	}
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
	// Auto-spawn first train (line just became active).
	gs.AvailableTrains--
	cityCfg := config.Cities[gs.SelectedCity]
	train := components.NewTrain(gs.TrainIDCounter, spare, cityCfg.TrainCapacity, config.TrainMaxSpeed)
	gs.TrainIDCounter++
	gs.Trains = append(gs.Trains, train)
	spare.Trains = append(spare.Trains, train)
	return true
}

// bestPartnerForNewLine finds the best second station when starting a brand-new
// line for station. Prefers a different type, prefers proximity, excludes stations
// that would require more bridges than available.
func (s *Solver) bestPartnerForNewLine(gs *state.GameState, station *components.Station) *components.Station {
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
		typeDiff := 0.0
		if candidate.Type != station.Type {
			typeDiff = 30.0
		}
		score := typeDiff - dist*0.05
		if score > bestScore {
			bestScore = score
			best = candidate
		}
	}
	return best
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
	gs.TrainIDCounter++
	gs.Trains = append(gs.Trains, train)
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
	gs.TrainIDCounter++
	gs.Trains = append(gs.Trains, train)
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
// Loop closing
// ---------------------------------------------------------------------------

func (s *Solver) tryCloseLoop(gs *state.GameState) bool {
	for i := 0; i < gs.AvailableLines && i < len(gs.Lines); i++ {
		line := gs.Lines[i]
		if !line.Active || line.MarkedForDeletion {
			continue
		}
		n := len(line.Stations)
		if n < 4 {
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
		gs.Bridges -= cost
		line.AddStation(line.Stations[0], -1, func() { gs.GraphDirty = true })
		return true
	}
	return false
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

// bestLineForStation finds the best active line to extend to station, evaluating
// both prepend and append positions and scoring by line hygiene.
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
		// Skip closed loops (cannot be extended).
		if line.Stations[0] == line.Stations[n-1] {
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

		// Evaluate prepend (idx=0) and append (idx=-1).
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
			// Subtract distance penalty so closer stations are preferred when scores tie.
			dist := math.Hypot(station.X-neighbor.X, station.Y-neighbor.Y)
			score := hs - dist*0.01
			candidates = append(candidates, candidate{line, insertIdx, score})
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

// attachStation adds station to line at insertIdx, paying bridge cost and
// auto-spawning a train if the line just became active.
func (s *Solver) attachStation(gs *state.GameState, line *components.Line, station *components.Station, insertIdx int) {
	n := len(line.Stations)
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

	wasActive := line.Active
	line.AddStation(station, insertIdx, func() { gs.GraphDirty = true })

	if !wasActive && line.Active && gs.AvailableTrains > 0 && len(line.Trains) == 0 {
		gs.AvailableTrains--
		cityCfg := config.Cities[gs.SelectedCity]
		train := components.NewTrain(gs.TrainIDCounter, line, cityCfg.TrainCapacity, config.TrainMaxSpeed)
		gs.TrainIDCounter++
		gs.Trains = append(gs.Trains, train)
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

