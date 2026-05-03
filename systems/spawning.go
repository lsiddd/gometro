package systems

import (
	"log"
	"math"
	"minimetro-go/components"
	"minimetro-go/config"
	"minimetro-go/state"
	"minimetro-go/systems/graph"
)

func (g *Game) updateSpawning(gs *state.GameState, screenWidth, screenHeight, nowMs float64) {
	difficultyMultiplier := math.Pow(config.DifficultyScaleFactor, float64(gs.Week-1))
	curriculumFactor := gs.SpawnRateFactor
	if curriculumFactor <= 0 {
		curriculumFactor = 1.0
	}
	currentSpawnRate := config.BaseSpawnRate * difficultyMultiplier * curriculumFactor
	currentStationSpawnRate := config.BaseStationSpawnRate * difficultyMultiplier * curriculumFactor

	if nowMs-gs.LastSpawnTime > currentSpawnRate/gs.Speed {
		g.SpawnPassenger(gs, nowMs)
		gs.LastSpawnTime = nowMs
	}

	canSpawnStation := gs.SpawnStationsEnabled
	if gs.StationSpawnLimit > 0 && len(gs.Stations) >= gs.StationSpawnLimit {
		canSpawnStation = false
	}
	if canSpawnStation && nowMs-gs.LastStationSpawnTime > currentStationSpawnRate/gs.Speed {
		g.SpawnStation(gs, screenWidth, screenHeight)
		gs.LastStationSpawnTime = nowMs
	}
}

func (g *Game) SpawnPassenger(gs *state.GameState, nowMs float64) {
	if len(gs.Stations) < 2 {
		return
	}
	rng := gs.Rand()
	station := gs.Stations[rng.Intn(len(gs.Stations))]

	availTypes := make(map[config.StationType]bool)
	for _, s := range gs.Stations {
		if s.Type != station.Type {
			availTypes[s.Type] = true
		}
	}

	if len(availTypes) > 0 {
		var list []config.StationType
		for t := range availTypes {
			list = append(list, t)
		}
		dest := list[rng.Intn(len(list))]
		passenger := components.NewPassenger(station, dest, nowMs)
		passenger.Path = graph.FindPath(g.GraphManager, gs, station, dest)
		passenger.PathIndex = 1
		station.AddPassenger(passenger, nowMs)
		gs.AddPassenger(passenger)
	}
}

func (g *Game) SpawnStation(gs *state.GameState, screenWidth, screenHeight float64) {
	stationType := g.getNewStationType(gs, gs.LastStationSpawnTime)
	specialTypes := config.SpecialTypes()
	isSpecial := false
	for _, sp := range specialTypes {
		if sp == stationType {
			isSpecial = true
			break
		}
	}

	margin := 80.0
	rng := gs.Rand()

	tryPlace := func(minDist float64, attempts int) bool {
		for i := 0; i < attempts; i++ {
			var x, y float64
			if isSpecial {
				edge := rng.Intn(4)
				borderDepth := 0.25
				switch edge {
				case 0:
					x = margin + rng.Float64()*(screenWidth-2*margin)
					y = margin + rng.Float64()*screenHeight*borderDepth
				case 1:
					x = screenWidth - margin - rng.Float64()*screenWidth*borderDepth
					y = margin + rng.Float64()*(screenHeight-2*margin)
				case 2:
					x = margin + rng.Float64()*(screenWidth-2*margin)
					y = screenHeight - margin - rng.Float64()*screenHeight*borderDepth
				case 3:
					x = margin + rng.Float64()*screenWidth*borderDepth
					y = margin + rng.Float64()*(screenHeight-2*margin)
				}
			} else {
				x = margin + rng.Float64()*(screenWidth-margin*2)
				y = margin + rng.Float64()*(screenHeight-margin*2)
			}

			tooClose := false
			for _, s := range gs.Stations {
				if math.Hypot(s.X-x, s.Y-y) < minDist {
					tooClose = true
					break
				}
			}

			inRiver := false
			for _, river := range gs.Rivers {
				if river.Contains(x, y) {
					inRiver = true
					break
				}
			}

			if !tooClose && !inRiver {
				st := components.NewStation(gs.StationIDCounter, x, y, stationType)
				gs.AddStation(st)
				log.Printf("[Game] Station spawned: id=%d type=%s pos=(%.0f,%.0f) dist=%.0f total=%d",
					st.ID, stationType, x, y, minDist, len(gs.Stations))
				return true
			}
		}
		return false
	}

	if tryPlace(config.StationMinDistance, 500) {
		return
	}
	// Fallback: progressively relax minDistance when the map is dense.
	// Mirrors the placeFallbackStation behaviour used during init.
	for _, dist := range []float64{80.0, 50.0} {
		if tryPlace(dist, 200) {
			return
		}
	}
	log.Printf("[Game] SpawnStation: map saturated, skipping spawn for type=%s", stationType)
}

func (g *Game) getNewStationType(gs *state.GameState, nowMs float64) config.StationType {
	gameTime := nowMs - gs.GameStartTime
	minutesPlayed := gameTime / 60000.0

	specialTypes := config.SpecialTypes()
	rng := gs.Rand()
	if minutesPlayed > 2 && rng.Float64() < 0.12 {
		usedSpecial := make(map[config.StationType]bool)
		for _, s := range gs.Stations {
			for _, sp := range specialTypes {
				if s.Type == sp {
					usedSpecial[sp] = true
				}
			}
		}

		var available []config.StationType
		for _, sp := range specialTypes {
			if !usedSpecial[sp] {
				available = append(available, sp)
			}
		}

		if len(available) > 0 {
			return available[rng.Intn(len(available))]
		}
	}

	c, sq, t := 0, 0, 0
	for _, st := range gs.Stations {
		if st.Type == config.Circle {
			c++
		} else if st.Type == config.Triangle {
			t++
		} else if st.Type == config.Square {
			sq++
		}
	}

	total := c + t + sq
	probCircle := 0.5
	if total > 0 {
		probCircle = float64(sq+t) / float64(2*total)
	}

	val := rng.Float64()
	if val < probCircle {
		return config.Circle
	}

	if rng.Float64() < 0.75 {
		return config.Triangle
	}
	return config.Square
}
