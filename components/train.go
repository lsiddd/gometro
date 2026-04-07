package components

type TrainState string

const (
	TrainWaiting TrainState = "WAITING"
	TrainMoving  TrainState = "MOVING"
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
	IsLoop              bool
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
	t.CheckLoopStatus()
	return t
}

func (t *Train) CheckLoopStatus() {
	if len(t.Line.Stations) >= 3 {
		t.IsLoop = t.Line.Stations[0] == t.Line.Stations[len(t.Line.Stations)-1]
	} else {
		t.IsLoop = false
	}
}

func (t *Train) HasCarriage() bool {
	return t.CarriageCount > 0
}

func (t *Train) TotalCapacity() int {
	return t.Capacity * (1 + t.CarriageCount)
}

func (t *Train) GetUpcomingStops(currentStation *Station, singleDirectionOnly bool) []*Station {
	upcoming := make(map[*Station]bool)
	lineStations := t.Line.Stations
	numStations := len(lineStations)
	if t.IsLoop {
		numStations--
	}

	if numStations <= 1 {
		return nil
	}

	currentIndex := t.CurrentStationIndex

	currentDirection := t.Direction
	if !t.IsLoop {
		if currentIndex == 0 {
			currentDirection = 1
		}
		if currentIndex >= len(lineStations)-1 {
			currentDirection = -1
		}
	}

	if t.IsLoop {
		for i := 1; i < numStations; i++ {
			index := (currentIndex + i) % numStations
			upcoming[lineStations[index]] = true
		}
	} else {
		if currentDirection == 1 {
			for i := currentIndex + 1; i < len(lineStations); i++ {
				upcoming[lineStations[i]] = true
			}
		} else {
			for i := currentIndex - 1; i >= 0; i-- {
				upcoming[lineStations[i]] = true
			}
		}

		if !singleDirectionOnly {
			if currentDirection == 1 {
				for i := len(lineStations) - 2; i >= 0; i-- {
					upcoming[lineStations[i]] = true
				}
			} else {
				for i := 1; i < len(lineStations); i++ {
					upcoming[lineStations[i]] = true
				}
			}
		}
	}

	var result []*Station
	for s := range upcoming {
		result = append(result, s)
	}
	return result
}
