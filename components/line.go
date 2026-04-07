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

// RemoveEndStation removes station from an endpoint of the line. For loops it
// handles de-closing by removing the repeated entry. Sets MarkedForDeletion when
// the line falls below two stations.
func (l *Line) RemoveEndStation(station *Station, markDirtyCallback func()) {
	if len(l.Stations) < 2 {
		return
	}
	isLoop := len(l.Stations) > 2 && l.Stations[0] == l.Stations[len(l.Stations)-1]

	idx := -1
	for i, s := range l.Stations {
		if s == station {
			idx = i
			break
		}
	}
	if idx == -1 {
		return
	}
	lastIdx := len(l.Stations) - 1

	if isLoop {
		if station == l.OriginalStart && (idx == 0 || idx == lastIdx) {
			l.Stations = l.Stations[:lastIdx]
			if len(l.Stations) > 0 {
				l.OriginalEnd = l.Stations[len(l.Stations)-1]
			}
		} else if station == l.OriginalEnd {
			l.Stations = l.Stations[:lastIdx]
			for i, j := 0, len(l.Stations)-1; i < j; i, j = i+1, j-1 {
				l.Stations[i], l.Stations[j] = l.Stations[j], l.Stations[i]
			}
			if len(l.Stations) > 0 {
				l.OriginalStart = l.Stations[0]
				l.OriginalEnd = l.Stations[len(l.Stations)-1]
			}
		}
	} else {
		if idx == 0 {
			l.Stations = l.Stations[1:]
			if len(l.Stations) > 0 {
				l.OriginalStart = l.Stations[0]
			}
		} else if idx == lastIdx {
			l.Stations = l.Stations[:lastIdx]
			if len(l.Stations) > 0 {
				l.OriginalEnd = l.Stations[len(l.Stations)-1]
			}
		}
	}

	if len(l.Stations) < 2 {
		l.MarkedForDeletion = true
	} else {
		l.Active = true
	}
	if markDirtyCallback != nil {
		markDirtyCallback()
	}
}

// RemoveStation removes all occurrences of station from the line. If the station
// is an endpoint, delegates to RemoveEndStation. For mid-line removals on loops
// it repairs the closing entry. Sets MarkedForDeletion when the line falls below
// two stations.
func (l *Line) RemoveStation(station *Station, markDirtyCallback func()) {
	var indices []int
	for i, s := range l.Stations {
		if s == station {
			indices = append(indices, i)
		}
	}
	if len(indices) == 0 {
		return
	}

	isLoop := len(l.Stations) > 2 && l.Stations[0] == l.Stations[len(l.Stations)-1]
	isEndpoint := !isLoop && (indices[0] == 0 || indices[0] == len(l.Stations)-1)

	if isEndpoint {
		l.RemoveEndStation(station, markDirtyCallback)
		return
	}

	if len(indices) > 1 {
		l.Stations = l.Stations[:len(l.Stations)-1]
		l.Stations = append(l.Stations[:indices[0]], l.Stations[indices[0]+1:]...)
		if len(l.Stations) > 0 {
			l.Stations = append(l.Stations, l.Stations[0])
		}
	} else {
		l.Stations = append(l.Stations[:indices[0]], l.Stations[indices[0]+1:]...)
	}

	l.Active = len(l.Stations) >= 2
	if !l.Active {
		l.MarkedForDeletion = true
	}
	if markDirtyCallback != nil {
		markDirtyCallback()
	}
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

