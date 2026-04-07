package components

import "strconv"

type Line struct {
	Index             int
	Color             [3]int
	Stations          []*Station
	Trains            []*Train
	Active            bool
	MarkedForDeletion bool
	OriginalStart     *Station
	OriginalEnd       *Station
}

func NewLine(colorHex string, index int) *Line {
	r, g, b := hexToRGB(colorHex)
	return &Line{
		Index:    index,
		Color:    [3]int{r, g, b},
		Stations: make([]*Station, 0),
		Trains:   make([]*Train, 0),
	}
}

func hexToRGB(hex string) (int, int, int) {
	if len(hex) > 0 && hex[0] == '#' {
		hex = hex[1:]
	}
	r, _ := strconv.ParseInt(hex[0:2], 16, 32)
	g, _ := strconv.ParseInt(hex[2:4], 16, 32)
	b, _ := strconv.ParseInt(hex[4:6], 16, 32)
	return int(r), int(g), int(b)
}

func (l *Line) HasSegment(s1, s2 *Station) bool {
	for i := 0; i < len(l.Stations)-1; i++ {
		if (l.Stations[i] == s1 && l.Stations[i+1] == s2) || (l.Stations[i] == s2 && l.Stations[i+1] == s1) {
			return true
		}
	}
	// loop
	if len(l.Stations) > 2 && l.Stations[0] == l.Stations[len(l.Stations)-1] {
		if (l.Stations[0] == s1 && l.Stations[len(l.Stations)-2] == s2) || (l.Stations[0] == s2 && l.Stations[len(l.Stations)-2] == s1) {
			return true
		}
	}
	return false
}

func (l *Line) AddStation(station *Station, insertIndex int, markDirtyCallback func()) bool {
	if len(l.Stations) == 0 {
		l.OriginalStart = station
	}

	if len(l.Stations) > 0 {
		if insertIndex == 0 && station == l.Stations[0] {
			return false
		}
		isAppend := insertIndex == -1 || insertIndex >= len(l.Stations)
		if isAppend && station == l.Stations[len(l.Stations)-1] {
			return false
		}
	}

	isClosingLoop := len(l.Stations) >= 2 && station == l.Stations[0] && (insertIndex == -1 || insertIndex == len(l.Stations))

	hasStation := false
	for _, s := range l.Stations {
		if s == station {
			hasStation = true
			break
		}
	}
	if hasStation && !isClosingLoop {
		return false
	}

	if insertIndex == 0 {
		l.Stations = append([]*Station{station}, l.Stations...)
		l.OriginalStart = station
	} else if insertIndex > 0 && insertIndex < len(l.Stations) {
		l.Stations = append(l.Stations[:insertIndex], append([]*Station{station}, l.Stations[insertIndex:]...)...)
	} else {
		l.Stations = append(l.Stations, station)
		if !isClosingLoop {
			l.OriginalEnd = station
		}
	}

	l.Active = len(l.Stations) >= 2
	if markDirtyCallback != nil {
		markDirtyCallback()
	}

	return true
}

func (l *Line) ClearLine(markDirtyCallback func()) {
	// Refund bridging will be handled inside Game logic, since we need GameState
	// Here we just clean self
	l.Stations = nil
	l.Trains = nil
	l.Active = false
	l.MarkedForDeletion = false
	l.OriginalStart = nil
	l.OriginalEnd = nil
	if markDirtyCallback != nil {
		markDirtyCallback()
	}
}

