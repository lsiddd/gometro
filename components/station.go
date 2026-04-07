package components



type Animation struct {
	StartTime float64
	Duration  float64
}

type Station struct {
	ID                 int
	X                  float64
	Y                  float64
	Type               string
	Passengers         []*Passenger
	IsInterchange     bool
	OvercrowdProgress float64
	OvercrowdIsGrace  bool

	ConnectionAnimation *Animation
	DeliveryAnimation   *Animation
	AnimateUpgrade      *Animation
}

func NewStation(id int, x, y float64, stationType string) *Station {
	return &Station{
		ID:         id,
		X:          x,
		Y:          y,
		Type:       stationType,
		Passengers: make([]*Passenger, 0),
	}
}

func (s *Station) Capacity(cityBase int) int {
	if s.IsInterchange {
		return 18
	}
	return cityBase
}

func (s *Station) AddPassenger(p *Passenger, currentTime float64) {
	if len(s.Passengers) < 100 {
		s.Passengers = append(s.Passengers, p)
	}
}

func (s *Station) RemovePassenger(p *Passenger, currentTime float64) {
	for i, passenger := range s.Passengers {
		if passenger == p {
			copy(s.Passengers[i:], s.Passengers[i+1:])
			s.Passengers[len(s.Passengers)-1] = nil
			s.Passengers = s.Passengers[:len(s.Passengers)-1]
			break
		}
	}
}
