package systems

import (
	"log"
	"math"
	"math/rand"
	"minimetro-go/components"
	"minimetro-go/config"
	"minimetro-go/state"
	"minimetro-go/systems/graph"
)

type Game struct {
	Initialized  bool
	GraphManager *graph.GraphManager

	// Pre-allocated buffers for hot-path update functions; reused every frame
	// to avoid heap allocations inside the 60 Hz simulation loop.
	trainSnapshot  []*components.Train
	gracedStations map[*components.Station]bool
	incomingTrains map[*components.Station][]*components.Train
}

func NewGame() *Game {
	return &Game{
		Initialized:    false,
		GraphManager:   graph.NewGraphManager(),
		gracedStations: make(map[*components.Station]bool),
		incomingTrains: make(map[*components.Station][]*components.Train),
	}
}

func (g *Game) InitGame(gs *state.GameState, screenWidth, screenHeight float64) {
	selectedCity := gs.SelectedCity
	gs.Reset()
	gs.SelectedCity = selectedCity

	cityCfg := config.Cities[selectedCity]
	gs.Bridges = cityCfg.Bridges
	gs.MaxLines = cityCfg.MaxLines
	gs.AvailableLines = 3

	for i := 0; i < gs.MaxLines; i++ {
		colorHex := config.LineColors[i%len(config.LineColors)]
		gs.Lines = append(gs.Lines, components.NewLine(colorHex, i))
	}

	g.createRivers(gs, screenWidth, screenHeight)
	g.createInitialStations(gs, screenWidth, screenHeight)

	g.Initialized = true
	g.GraphManager.MarkDirty()
}

func (g *Game) createRivers(gs *state.GameState, width, height float64) {
	rivers := config.Rivers[gs.SelectedCity]
	for _, riverData := range rivers {
		r := &components.River{}
		for _, p := range riverData {
			r.Points = append(r.Points, config.PointF{X: p.X * width, Y: p.Y * height})
		}
		gs.Rivers = append(gs.Rivers, r)
	}
}

func (g *Game) createInitialStations(gs *state.GameState, width, height float64) {
	types := config.BasicTypes()
	centerX := width / 2
	centerY := height / 2
	spread := math.Min(width, height) * config.InitialStationSpreadFraction
	minDistance := config.StationMinDistance

	for _, stationType := range types {
		attempts := 0
		maxAttempts := 200

		for attempts < maxAttempts {
			angle := rand.Float64() * math.Pi * 2
			radius := spread * (0.5 + rand.Float64()*0.5)
			x := centerX + math.Cos(angle)*radius
			y := centerY + math.Sin(angle)*radius

			tooClose := false
			for _, s := range gs.Stations {
				if math.Hypot(s.X-x, s.Y-y) < minDistance {
					tooClose = true
					break
				}
			}

			inRiver := false
			for _, r := range gs.Rivers {
				if r.Contains(x, y) {
					inRiver = true
					break
				}
			}

			if !tooClose && !inRiver {
				gs.AddStation(components.NewStation(gs.StationIDCounter, x, y, stationType))
				break
			}

			attempts++
		}

		if attempts >= maxAttempts {
			g.placeFallbackStation(gs, centerX, centerY, spread, minDistance, stationType)
		}
	}
}

// placeFallbackStation is called when the normal placement loop in
// createInitialStations exhausts its attempts. It widens the search radius
// (factor 0.3–1.5× spread instead of 0.5–1.0×) and ignores the river
// constraint as a last resort, so every station type is placed even on
// heavily obstructed maps.
//
// Returns true if a valid position was found and the station was added.
// Returns false only when 500 attempts with the widened radius all fail
// (degenerate case: minDistance larger than the available canvas).
func (g *Game) placeFallbackStation(gs *state.GameState, centerX, centerY, spread, minDistance float64, stationType config.StationType) bool {
	for fallback := 0; fallback < 500; fallback++ {
		angle := rand.Float64() * math.Pi * 2
		radius := spread * (0.3 + rand.Float64()*1.2)
		x := centerX + math.Cos(angle)*radius
		y := centerY + math.Sin(angle)*radius
		tooClose := false
		for _, s := range gs.Stations {
			if math.Hypot(s.X-x, s.Y-y) < minDistance {
				tooClose = true
				break
			}
		}
		if !tooClose {
			log.Printf("[Game] Initial station fallback placement: type=%s pos=(%.0f,%.0f)", stationType, x, y)
			gs.AddStation(components.NewStation(gs.StationIDCounter, x, y, stationType))
			return true
		}
	}
	return false
}

func (g *Game) Update(gs *state.GameState, deltaTime, screenWidth, screenHeight float64, nowMs float64) string {
	if !g.Initialized || gs.GameOver || gs.Paused {
		return ""
	}

	elapsed := nowMs - gs.WeekStartTime
	gs.Day = (int(elapsed/(config.WeekDuration/7)) % 7)

	if elapsed > config.WeekDuration {
		gs.Week++
		gs.WeekStartTime = nowMs
		gs.AvailableTrains++
		gs.CameraZoom = math.Max(0.55, 1.0-float64(gs.Week-1)*0.03)
		log.Printf("[Game] Week %d started — trains=%d stations=%d score=%d", gs.Week, gs.AvailableTrains, len(gs.Stations), gs.Score)
		if !gs.UpgradesEnabled {
			return ""
		}
		return "show_upgrades"
	}

	// Update order is load-bearing — do not reorder:
	//  1. Spawn passengers/stations so new arrivals exist before overcrowd ticks.
	//  2. Overcrowd: progress advances against the just-spawned passenger counts.
	//  3. Reservations: assign incoming trains before boarding so trains don't
	//     overfill (depends on train positions updated in the previous frame).
	//  4. Trains: move, board, alight — consumes reservations set in step 3.
	//  5. Cleanup: remove deleted lines after all train state has been resolved.
	g.updateSpawning(gs, screenWidth, screenHeight, nowMs)
	g.updateOvercrowding(gs, deltaTime)
	g.updatePassengerReservations(gs, nowMs)
	g.updateTrains(gs, deltaTime, nowMs)
	g.cleanupDeletedLines(gs)

	if g.CheckGameOver(gs, nowMs) {
		gs.GameOver = true
		gs.Paused = true
		return "game_over"
	}

	return ""
}

func (g *Game) updateOvercrowding(gs *state.GameState, deltaTime float64) {
	cityCfg := config.Cities[gs.SelectedCity]

	// Pre-compute the set of stations with an incoming moving train that can
	// actually relieve them (has capacity OR will drop off a passenger there).
	// A full train passing through grants no real relief; excluding it prevents
	// exploit loops where the RL agent circles loaded trains to suppress overcrowd.
	clear(g.gracedStations)
	for _, t := range gs.Trains {
		if t.Line.Active && len(t.Line.Stations) > 1 && t.State == components.TrainMoving &&
			t.NextStationIndex < len(t.Line.Stations) {
			nextSt := t.Line.Stations[t.NextStationIndex]
			hasCapacity := len(t.Passengers)+t.ReservedSeats < t.TotalCapacity()
			willAlight := false
			for _, p := range t.Passengers {
				if p.Destination == nextSt.Type {
					willAlight = true
					break
				}
			}
			if hasCapacity || willAlight {
				g.gracedStations[nextSt] = true
			}
		}
	}

	for _, s := range gs.Stations {
		cap := s.Capacity(cityCfg.StationCapacity)
		s.OvercrowdIsGrace = g.gracedStations[s]

		if len(s.Passengers) > cap {
			s.OvercrowdProgress += deltaTime * gs.Speed
		} else if s.OvercrowdProgress > 0 {
			s.OvercrowdProgress -= deltaTime * gs.Speed
			if s.OvercrowdProgress < 0 {
				s.OvercrowdProgress = 0
			}
		}
	}
}

func (g *Game) updatePassengerReservations(gs *state.GameState, nowMs float64) {
	// Reset all reservations before recalculating
	for _, t := range gs.Trains {
		t.ReservedSeats = 0
	}
	for _, p := range gs.Passengers {
		p.ReservedTrain = nil
	}

	// Pre-compute: station → trains arriving next tick.
	// Reduces the per-passenger train scan from O(T) to O(1) lookup.
	// Reset to length-zero rather than clearing the map so that backing arrays
	// are reused across frames, avoiding per-frame heap allocations.
	for k := range g.incomingTrains {
		g.incomingTrains[k] = g.incomingTrains[k][:0]
	}
	for _, t := range gs.Trains {
		if t.Line.Active && len(t.Line.Stations) > 1 && t.State == components.TrainMoving &&
			t.NextStationIndex < len(t.Line.Stations) {
			nextSt := t.Line.Stations[t.NextStationIndex]
			g.incomingTrains[nextSt] = append(g.incomingTrains[nextSt], t)
		}
	}

	for _, p := range gs.Passengers {
		if p.CurrentStation == nil {
			continue
		}

		var bestTrain *components.Train
		for _, t := range g.incomingTrains[p.CurrentStation] {
			if t.TotalCapacity()-len(t.Passengers)-t.ReservedSeats > 0 {
				upcoming := t.GetUpcomingStops(p.CurrentStation, true)
				if canBoard(g.GraphManager, gs, p, upcoming, nowMs) {
					bestTrain = t
					break
				}
			}
		}

		if bestTrain != nil {
			bestTrain.ReservedSeats++
			p.ReservedTrain = bestTrain
			continue
		}

		// No incoming train: consider repathing
		overcrowded := p.CurrentStation.OvercrowdProgress > float64(config.OvercrowdTime)*config.OvercrowdRepathThreshold
		cooldownElapsed := nowMs-p.LastRouteCalculation > config.RepathCooldownMs
		waitedTooLong := nowMs-p.WaitStartTime > config.PassengerWaitPatience && cooldownElapsed

		if (overcrowded && cooldownElapsed) || waitedTooLong {
			p.Path = graph.FindPath(g.GraphManager, gs, p.CurrentStation, p.Destination)
			p.PathIndex = 1
			p.LastRouteCalculation = nowMs
		}
	}
}

func (g *Game) updateTrains(gs *state.GameState, deltaTime, nowMs float64) {
	// Snapshot the slice before iterating: UpdateTrain may call gs.RemoveTrain,
	// which calls slices.Delete and zeroes the tail of gs.Trains. Iterating the
	// original range would then dereference nil entries at the zeroed positions.
	g.trainSnapshot = append(g.trainSnapshot[:0], gs.Trains...)
	for _, train := range g.trainSnapshot {
		if train != nil {
			g.UpdateTrain(train, gs, deltaTime, nowMs)
		}
	}
}

func (g *Game) cleanupDeletedLines(gs *state.GameState) {
	for i, line := range gs.Lines {
		if !line.MarkedForDeletion || !line.Active {
			continue
		}

		// Note: gs.AvailableLines is NOT incremented here. The slot being freed
		// was already within the allocation (ghost lines use existing spare slots;
		// there is no line-delete UI that would have decremented AvailableLines).
		// Incrementing would push the count past len(gs.Lines) and crash the UI.
		gs.AvailableTrains += len(line.Trains)
		for _, t := range line.Trains {
			gs.Carriages += t.CarriageCount
			t.CarriageCount = 0

			// Attach train to a dummy line so it can finish evacuating passengers
			dummyLine := components.NewLine("#000000", line.Index)
			dummyLine.Color = line.Color
			dummyLine.Active = false
			dummyLine.MarkedForDeletion = true
			cpy := make([]*components.Station, len(line.Stations))
			copy(cpy, line.Stations)
			dummyLine.Stations = cpy
			t.Line = dummyLine
		}
		line.ClearLine(func() { gs.GraphDirty = true })
		clearedLine := components.NewLine("#000000", line.Index)
		clearedLine.Color = line.Color
		gs.Lines[i] = clearedLine
	}
}

func (g *Game) UpdateTrain(t *components.Train, gs *state.GameState, deltaTime, nowMs float64) {
	if (!t.Line.Active && !t.Line.MarkedForDeletion) || len(t.Line.Stations) < 2 {
		return
	}

	if t.State == components.TrainWaiting {
		// Line was deleted while the train was docked: eject passengers to the
		// current station immediately rather than waiting for the next arrival.
		if t.Line.MarkedForDeletion {
			var currentStation *components.Station
			if t.CurrentStationIndex < len(t.Line.Stations) {
				currentStation = t.Line.Stations[t.CurrentStationIndex]
			}
			for _, p := range t.Passengers {
				p.OnTrain = nil
				if currentStation != nil {
					p.CurrentStation = currentStation
					p.WaitStartTime = nowMs
					if !currentStation.IsInterchange {
						p.WaitStartTime += float64(config.RegularTransferTime - config.InterchangeTransferTime)
					}
					currentStation.AddPassenger(p, nowMs)
				} else {
					gs.RemovePassenger(p)
				}
			}
			t.Passengers = nil
			gs.RemoveTrain(t)
			return
		}
		t.WaitTimer -= deltaTime
		if t.WaitTimer <= 0 {
			t.State = components.TrainMoving
			g.DetermineNextStation(t)
		}
		return
	}

	if t.CurrentStationIndex >= len(t.Line.Stations) {
		t.CurrentStationIndex = len(t.Line.Stations) - 1
		t.Progress = 0
	}
	if t.NextStationIndex >= len(t.Line.Stations) {
		t.NextStationIndex = t.CurrentStationIndex
		g.DetermineNextStation(t)
	}

	currentStation := t.Line.Stations[t.CurrentStationIndex]
	nextStation := t.Line.Stations[t.NextStationIndex]

	if currentStation == nil || nextStation == nil {
		t.State = components.TrainWaiting
		t.WaitTimer = 100
		return
	}

	distance := t.PathLength
	if distance <= 0 {
		distance = math.Hypot(nextStation.X-currentStation.X, nextStation.Y-currentStation.Y)
	}

	accelDist := math.Min(distance*0.4, 60)
	decelStart := math.Max(distance-accelDist, distance*0.6)

	if t.Progress < accelDist {
		t.Speed = math.Min(t.MaxSpeed, t.Speed+config.TrainAcceleration*deltaTime)
	} else if t.Progress > decelStart {
		t.Speed = math.Max(0.2*t.MaxSpeed, t.Speed-config.TrainAcceleration*deltaTime)
	} else {
		t.Speed = t.MaxSpeed
	}

	t.Progress += t.Speed * gs.Speed * (deltaTime / 16.67)

	if t.Progress >= distance {
		t.Progress = distance
		t.X = nextStation.X
		t.Y = nextStation.Y
		t.CurrentStationIndex = t.NextStationIndex

		if t.Line.MarkedForDeletion {
			var alighted []*components.Passenger
			n := 0
			for _, p := range t.Passengers {
				if g.shouldAlightPassenger(t, p, nextStation, gs, nowMs) {
					alighted = append(alighted, p)
				} else {
					p.OnTrain = nil
					p.WaitStartTime = nowMs
					if !nextStation.IsInterchange {
						p.WaitStartTime += float64(config.RegularTransferTime - config.InterchangeTransferTime)
					}
					nextStation.AddPassenger(p, nowMs)
					t.Passengers[n] = p
					n++
				}
			}
			t.Passengers = t.Passengers[:n]

			for _, p := range alighted {
				if p.OnTrain == t {
					p.OnTrain = nil
					gs.RemovePassenger(p)
				}
			}
			t.Passengers = nil

			gs.RemoveTrain(t)
			return
		}

		g.processPassengers(t, nextStation, gs, nowMs)
		t.Speed = 0
	} else {
		// interpolate position
		if len(t.PathPts) > 0 {
			t.X, t.Y = g.getPosOnPath(t, t.Progress)
		} else {
			dx := nextStation.X - currentStation.X
			dy := nextStation.Y - currentStation.Y
			d := math.Hypot(dx, dy)
			if d > 0 {
				frac := t.Progress / d
				t.X = currentStation.X + dx*frac
				t.Y = currentStation.Y + dy*frac
			}
		}
	}
}

func (g *Game) DetermineNextStation(t *components.Train) {
	if t.CurrentStationIndex >= len(t.Line.Stations) {
		t.CurrentStationIndex = 0
	}

	if t.IsLoop() {
		t.NextStationIndex = (t.CurrentStationIndex + 1) % (len(t.Line.Stations) - 1)
	} else {
		if t.CurrentStationIndex >= len(t.Line.Stations)-1 {
			t.Direction = -1
		} else if t.CurrentStationIndex == 0 {
			t.Direction = 1
		}
		t.NextStationIndex = t.CurrentStationIndex + t.Direction
	}

	t.Progress = 0.0

	// compute waypoints
	s1 := t.Line.Stations[t.CurrentStationIndex]
	s2 := t.Line.Stations[t.NextStationIndex]
	t.PathPts = components.GetTrainWaypoints(s1, s2, 0) // No offset logic for simple port, could be added later
	total := 0.0
	for i := 0; i < len(t.PathPts)-1; i++ {
		a, b := t.PathPts[i], t.PathPts[i+1]
		total += math.Hypot(b[0]-a[0], b[1]-a[1])
	}
	t.PathLength = total
	if t.PathLength == 0 {
		t.PathLength = 1.0
	}
}

func (g *Game) getPosOnPath(t *components.Train, dist float64) (float64, float64) {
	pts := t.PathPts
	if len(pts) == 0 {
		return t.X, t.Y
	}
	remaining := math.Max(0.0, dist)
	for i := 0; i < len(pts)-1; i++ {
		a, b := pts[i], pts[i+1]
		segLen := math.Hypot(b[0]-a[0], b[1]-a[1])
		if remaining <= segLen || i == len(pts)-2 {
			frac := 1.0
			if segLen > 1e-6 {
				frac = remaining / segLen
			}
			frac = math.Min(frac, 1.0)
			return a[0] + frac*(b[0]-a[0]), a[1] + frac*(b[1]-a[1])
		}
		remaining -= segLen
	}
	return pts[len(pts)-1][0], pts[len(pts)-1][1]
}

func (g *Game) processPassengers(t *components.Train, station *components.Station, gs *state.GameState, nowMs float64) {
	passengersChanged := 0

	var alighted []*components.Passenger
	n := 0
	for _, p := range t.Passengers {
		if g.shouldAlightPassenger(t, p, station, gs, nowMs) {
			alighted = append(alighted, p)
			passengersChanged++
		} else {
			t.Passengers[n] = p
			n++
		}
	}
	// Clear the tail for garbage collection
	for i := n; i < len(t.Passengers); i++ {
		t.Passengers[i] = nil
	}
	t.Passengers = t.Passengers[:n]

	for _, p := range alighted {
		if p.OnTrain == t {
			p.OnTrain = nil
			gs.RemovePassenger(p)
		}
	}

	availableSpace := t.TotalCapacity() - len(t.Passengers)
	if availableSpace > 0 && !t.Line.MarkedForDeletion {
		upcomingStops := t.GetUpcomingStops(station, true)

		for _, p := range station.Passengers {
			if p.CurrentStation != station {
				p.CurrentStation = station
				p.WaitStartTime = nowMs
			}
		}

		var waiting []*components.Passenger
		for _, p := range station.Passengers {
			if canBoard(g.GraphManager, gs, p, upcomingStops, nowMs) {
				waiting = append(waiting, p)
			}
		}

		limit := availableSpace
		if len(waiting) < limit {
			limit = len(waiting)
		}
		toBoard := waiting[:limit]
		for _, p := range toBoard {
			passengersChanged++
			station.RemovePassenger(p, nowMs)
			t.Passengers = append(t.Passengers, p)
			p.OnTrain = t
			p.CurrentStation = nil
		}
	}

	t.State = components.TrainWaiting
	baseTime := config.PassengerBoardTime
	if station.IsInterchange {
		baseTime = config.InterchangeTransferTime
	}
	t.WaitTimer = float64(passengersChanged * baseTime)
	if t.WaitTimer <= 0 {
		t.WaitTimer = 300
	}
}

func (g *Game) shouldAlightPassenger(t *components.Train, p *components.Passenger, station *components.Station, gs *state.GameState, nowMs float64) bool {
	if p.Destination == station.Type {
		gs.Score++
		gs.PassengersDelivered++
		station.DeliveryAnimation = &components.Animation{StartTime: nowMs, Duration: 500}
		return true
	}

	if p.Path != nil && len(p.Path) > p.PathIndex && p.Path[p.PathIndex] == station {
		p.PathIndex++
		if p.PathIndex >= len(p.Path) {
			gs.Score++
			gs.PassengersDelivered++
			station.DeliveryAnimation = &components.Animation{StartTime: nowMs, Duration: 500}
			return true
		}

		nextStop := p.Path[p.PathIndex]
		upcoming := t.GetUpcomingStops(station, true)

		hasNext := false
		for _, u := range upcoming {
			if u == nextStop {
				hasNext = true
				break
			}
		}

		if !hasNext {
			p.OnTrain = nil
			p.CurrentStation = station
			p.WaitStartTime = nowMs
			if !station.IsInterchange {
				p.WaitStartTime += float64(config.RegularTransferTime - config.InterchangeTransferTime)
			}
			station.AddPassenger(p, nowMs)
			return true
		}
	}
	return false
}

func canBoard(gm *graph.GraphManager, gs *state.GameState, p *components.Passenger, upcomingStops []*components.Station, nowMs float64) bool {
	// A passenger should board any train that visits a station matching their destination
	// type, regardless of which specific station their cached path was computed for.
	for _, u := range upcomingStops {
		if u.Type == p.Destination {
			return true
		}
	}

	if p.Path == nil || p.PathIndex >= len(p.Path) {
		p.Path = graph.FindPath(gm, gs, p.CurrentStation, p.Destination)
		p.PathIndex = 1
		p.LastRouteCalculation = nowMs
		if p.Path == nil || p.PathIndex >= len(p.Path) {
			return false
		}
	}

	nextStop := p.Path[p.PathIndex]
	for _, u := range upcomingStops {
		if u == nextStop {
			return true
		}
	}
	return false
}

func (g *Game) CheckGameOver(gs *state.GameState, nowMs float64) bool {
	for _, station := range gs.Stations {
		limit := float64(config.OvercrowdTime)
		if station.OvercrowdIsGrace {
			limit += config.OvercrowdGraceExtra
		}
		if station.OvercrowdProgress > limit {
			return true
		}
	}
	return false
}
