package components

import "testing"

func TestStationCapacity_Normal(t *testing.T) {
	s := NewStation(0, 0, 0, "circle")
	if got := s.Capacity(6); got != 6 {
		t.Errorf("normal station capacity: want 6, got %d", got)
	}
}

func TestStationCapacity_Interchange_TripleCityBase(t *testing.T) {
	// Spec: interchange increases limit to 18 passengers (6 * 3) for London
	s := NewStation(0, 0, 0, "circle")
	s.IsInterchange = true
	if got := s.Capacity(6); got != 18 {
		t.Errorf("interchange capacity: want 18, got %d", got)
	}
}

func TestStationCapacity_Paris_LowerBase(t *testing.T) {
	// Paris base is 4 — overcrowding is faster
	s := NewStation(0, 0, 0, "circle")
	if got := s.Capacity(4); got != 4 {
		t.Errorf("paris station capacity: want 4, got %d", got)
	}
}

func TestStationCapacity_Paris_Interchange(t *testing.T) {
	s := NewStation(0, 0, 0, "circle")
	s.IsInterchange = true
	if got := s.Capacity(4); got != 12 {
		t.Errorf("paris interchange capacity: want 12, got %d", got)
	}
}

func TestStation_AddPassenger(t *testing.T) {
	s := NewStation(0, 0, 0, "circle")
	p := &Passenger{Destination: "triangle"}
	s.AddPassenger(p, 0)
	if len(s.Passengers) != 1 {
		t.Errorf("expected 1 passenger, got %d", len(s.Passengers))
	}
}

func TestStation_RemovePassenger(t *testing.T) {
	s := NewStation(0, 0, 0, "circle")
	p := &Passenger{Destination: "triangle"}
	s.AddPassenger(p, 0)
	s.RemovePassenger(p, 0)
	if len(s.Passengers) != 0 {
		t.Errorf("expected 0 passengers after removal, got %d", len(s.Passengers))
	}
}

func TestStation_RemovePassenger_NotPresent(t *testing.T) {
	s := NewStation(0, 0, 0, "circle")
	p := &Passenger{Destination: "triangle"}
	// RemovePassenger on empty station should not panic
	s.RemovePassenger(p, 0)
	if len(s.Passengers) != 0 {
		t.Errorf("expected 0 passengers, got %d", len(s.Passengers))
	}
}

func TestStation_CapacityNotExceeded_AddHardLimit(t *testing.T) {
	// Station hard-caps at 100 to prevent OOM during runaway loads
	s := NewStation(0, 0, 0, "circle")
	for i := 0; i < 105; i++ {
		s.AddPassenger(&Passenger{Destination: "triangle"}, 0)
	}
	if len(s.Passengers) > 100 {
		t.Errorf("station exceeded hard cap of 100 passengers: got %d", len(s.Passengers))
	}
}
