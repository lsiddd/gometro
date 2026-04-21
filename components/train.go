package components

type TrainState int

const (
	TrainWaiting TrainState = iota
	TrainMoving
)

type Train struct {
	ID                  int
	Line                *Line
	Passengers          []*Passenger
	Capacity            int
	CurrentStationIndex int
	NextStationIndex    int
	Direction           int
	X                   float64
	Y                   float64
	Progress            float64
	State               TrainState
	WaitTimer           float64
	Speed               float64
	MaxSpeed            float64
	PathPts             [][2]float64
	PathLength          float64
	CarriageCount       int
	ReservedSeats       int
}

func NewTrain(id int, line *Line, capacity int, maxSpeed float64) *Train {
	t := &Train{
		ID:                  id,
		Line:                line,
		Passengers:          make([]*Passenger, 0),
		Capacity:            capacity,
		CurrentStationIndex: 0,
		NextStationIndex:    1,
		Direction:           1,
		State:               TrainWaiting,
		WaitTimer:           500,
		MaxSpeed:            maxSpeed,
	}
	if len(line.Stations) > 0 {
		t.X = line.Stations[0].X
		t.Y = line.Stations[0].Y
	}
	return t
}

// IsLoop reports whether the train's line forms a closed loop. It reads the
// current line topology on every call, so it is always accurate even after
// stations are added or removed without an explicit revalidation step.
func (t *Train) IsLoop() bool {
	if len(t.Line.Stations) >= 3 {
		return t.Line.Stations[0] == t.Line.Stations[len(t.Line.Stations)-1]
	}
	return false
}

func (t *Train) HasCarriage() bool {
	return t.CarriageCount > 0
}

func (t *Train) TotalCapacity() int {
	return t.Capacity * (1 + t.CarriageCount)
}

func (t *Train) GetUpcomingStops(currentStation *Station, singleDirectionOnly bool) []*Station {
	lineStations := t.Line.Stations
	numStations := len(lineStations)
	isLoop := t.IsLoop()
	if isLoop {
		numStations-- // exclude the loop-closure duplicate
	}

	if numStations <= 1 {
		return nil
	}

	currentIndex := t.CurrentStationIndex

	if isLoop {
		// Circular iteration produces unique stations — no map needed.
		result := make([]*Station, 0, numStations-1)
		for i := 1; i < numStations; i++ {
			result = append(result, lineStations[(currentIndex+i)%numStations])
		}
		return result
	}

	currentDirection := t.Direction
	if currentIndex == 0 {
		currentDirection = 1
	} else if currentIndex >= len(lineStations)-1 {
		currentDirection = -1
	}

	if singleDirectionOnly {
		// Zero-allocation fast path: return a sub-slice of the line's station list.
		// Callers only read this slice and use it within the same update tick.
		if currentDirection == 1 {
			if currentIndex+1 >= len(lineStations) {
				return nil
			}
			return lineStations[currentIndex+1:]
		}
		if currentIndex <= 0 {
			return nil
		}
		return lineStations[:currentIndex]
	}

	// singleDirectionOnly=false: union of both directional sets; dedup required.
	// Iterate slices in order and check a seen-map to preserve insertion order
	// (ranging over a map is non-deterministic in Go, which breaks RL reproducibility).
	seen := make(map[*Station]bool, len(lineStations))
	result := make([]*Station, 0, len(lineStations))
	appendUnique := func(s *Station) {
		if !seen[s] {
			seen[s] = true
			result = append(result, s)
		}
	}
	if currentDirection == 1 {
		for i := currentIndex + 1; i < len(lineStations); i++ {
			appendUnique(lineStations[i])
		}
		for i := len(lineStations) - 2; i >= 0; i-- {
			appendUnique(lineStations[i])
		}
	} else {
		for i := currentIndex - 1; i >= 0; i-- {
			appendUnique(lineStations[i])
		}
		for i := 1; i < len(lineStations); i++ {
			appendUnique(lineStations[i])
		}
	}
	return result
}
