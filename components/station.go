package components

import (
	"minimetro-go/config"
	"slices"
)

type Animation struct {
	StartTime float64
	Duration  float64
}

type Station struct {
	ID                 int
	X                  float64
	Y                  float64
	Type               config.StationType
	Passengers         []*Passenger
	IsInterchange     bool
	OvercrowdProgress float64
	OvercrowdIsGrace  bool

	DeliveryAnimation *Animation
}

func NewStation(id int, x, y float64, stationType config.StationType) *Station {
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
		return cityBase * 3
	}
	return cityBase
}

func (s *Station) AddPassenger(p *Passenger, currentTime float64) {
	if len(s.Passengers) < config.MaxPassengerQueueCap {
		s.Passengers = append(s.Passengers, p)
	}
}

func (s *Station) RemovePassenger(p *Passenger, currentTime float64) {
	if i := slices.Index(s.Passengers, p); i >= 0 {
		s.Passengers = slices.Delete(s.Passengers, i, i+1)
	}
}
