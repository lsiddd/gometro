package systems

import (
	"log"
	"math"
	"minimetro-go/components"
	"minimetro-go/config"
	"minimetro-go/state"
)

type Point struct {
	X float64
	Y float64
}

type DraggedSegment struct {
	Line  *components.Line
	Index int
}

type PreviewLine struct {
	Start     Point
	End       Point
	Color     [3]int
	IsSegment bool
	S1        *components.Station
	S2        *components.Station
	Target    Point
}

type InputHandler struct {
	IsDrawingLine       bool
	IsRemovingSegment   bool
	DraggedSegment     *DraggedSegment
	CurrentPath         []*components.Station

	DraggedTrainResource bool
	DraggedCarriage      bool
	DraggedInterchange   bool
	DraggedExistingTrain *components.Train

	HoveredStation *components.Station
	HoveredTrain   *components.Train
	PreviewLine    *PreviewLine

	MousePos         Point
	MousePressed     bool
	RightPressed     bool

	// Exposed edge-trigger flags — valid for the current frame, read by UI before PrevMousePressed is updated
	LeftJustPressed  bool
	LeftJustReleased bool
	RightJustPressed bool
}

func NewInputHandler() *InputHandler {
	return &InputHandler{
		CurrentPath: make([]*components.Station, 0),
	}
}

func (ih *InputHandler) Update(gs *state.GameState, x, y float64, leftDown, rightDown, shiftDown, leftJustPressed, leftJustReleased, rightJustPressed bool) {
	ih.MousePos = Point{X: x, Y: y}

	ih.LeftJustPressed = leftJustPressed
	ih.LeftJustReleased = leftJustReleased
	ih.RightJustPressed = rightJustPressed

	ih.MousePressed = leftDown
	ih.RightPressed = rightDown

	ih.updateHoveredStation(gs)
	ih.updatePreviewLine(gs)

	if ih.LeftJustPressed {
		ih.handleMouseDown(gs, 1, shiftDown)
	} else if ih.RightJustPressed {
		ih.handleMouseDown(gs, 3, shiftDown)
	}

	if ih.LeftJustReleased {
		ih.handleMouseUp(gs, 1)
	}
}

func (ih *InputHandler) updateHoveredStation(gs *state.GameState) {
	ih.HoveredStation = nil
	for _, s := range gs.Stations {
		dist := math.Hypot(ih.MousePos.X-s.X, ih.MousePos.Y-s.Y)
		if dist <= config.StationRadius+15 {
			ih.HoveredStation = s
			break
		}
	}
}

func (ih *InputHandler) updatePreviewLine(gs *state.GameState) {
	if ih.IsDrawingLine && len(ih.CurrentPath) > 0 {
		startStation := ih.CurrentPath[0]
		if ih.IsRemovingSegment {
			if math.Hypot(ih.MousePos.X-startStation.X, ih.MousePos.Y-startStation.Y) > config.StationRadius*2.5 {
				ih.IsRemovingSegment = false
			}
		}

		if ih.IsRemovingSegment {
			ih.PreviewLine = nil
		} else if gs.SelectedLine < len(gs.Lines) {
			color := gs.Lines[gs.SelectedLine].Color
			ih.PreviewLine = &PreviewLine{
				Start: Point{X: startStation.X, Y: startStation.Y},
				End: ih.MousePos,
				Color: color,
				IsSegment: false,
			}
		}
	} else if ih.DraggedSegment != nil {
		s1 := ih.DraggedSegment.Line.Stations[ih.DraggedSegment.Index]
		s2 := ih.DraggedSegment.Line.Stations[ih.DraggedSegment.Index+1]
		
		target := ih.MousePos
		if ih.HoveredStation != nil {
			target = Point{X: ih.HoveredStation.X, Y: ih.HoveredStation.Y}
		}

		ih.PreviewLine = &PreviewLine{
			IsSegment: true,
			S1: s1,
			S2: s2,
			Target: target,
			Color: ih.DraggedSegment.Line.Color,
		}
	} else {
		ih.PreviewLine = nil
	}
}

func (ih *InputHandler) distanceToSegment(pt Point, s1, s2 *components.Station) float64 {
	A := pt.X - s1.X
	B := pt.Y - s1.Y
	C := s2.X - s1.X
	D := s2.Y - s1.Y

	dot := A*C + B*D
	lenSq := C*C + D*D
	param := -1.0
	if lenSq != 0 {
		param = dot / lenSq
	}

	var xx, yy float64
	if param < 0 {
		xx, yy = s1.X, s1.Y
	} else if param > 1 {
		xx, yy = s2.X, s2.Y
	} else {
		xx = s1.X + param*C
		yy = s1.Y + param*D
	}

	dx := pt.X - xx
	dy := pt.Y - yy
	return math.Sqrt(dx*dx + dy*dy)
}

func (ih *InputHandler) handleMouseDown(gs *state.GameState, button int, shiftDown bool) bool {
	if gs.GameOver {
		return false
	}

	if button == 1 {
		train := ih.getTrainAtPos(gs, ih.MousePos)
		if train != nil {
			ih.DraggedExistingTrain = train
			return true
		}

		if gs.SelectedLine >= len(gs.Lines) {
			return false
		}
		line := gs.Lines[gs.SelectedLine]

		station := ih.getStationAtPos(gs, ih.MousePos)
		if station != nil {
			if shiftDown {
				for _, s := range line.Stations {
					if s == station {
						ih.removeStationFromLine(gs, line, station)
						return true
					}
				}
			}

			isAlreadyLoop := len(line.Stations) > 2 && line.Stations[0] == line.Stations[len(line.Stations)-1]
			if isAlreadyLoop {
				has := false
				for _, s := range line.Stations {
					if s == station {
						has = true
						break
					}
				}
				if has { return false }
			}

			ih.IsDrawingLine = true

			stationIndex := -1
			for i, s := range line.Stations {
				if s == station {
					stationIndex = i
					break
				}
			}

			if stationIndex != -1 {
				if stationIndex == 0 || stationIndex == len(line.Stations)-1 {
					ih.CurrentPath = []*components.Station{station}
					ih.IsRemovingSegment = true
				} else {
					ih.IsDrawingLine = false
				}
			} else {
				ih.CurrentPath = []*components.Station{station}
				ih.IsRemovingSegment = false
			}
			return true
		}

		for _, lineObj := range gs.Lines {
			if !lineObj.Active {
				continue
			}
			for i := 0; i < len(lineObj.Stations)-1; i++ {
				s1 := lineObj.Stations[i]
				s2 := lineObj.Stations[i+1]
				if ih.distanceToSegment(ih.MousePos, s1, s2) < 20 {
					ih.DraggedSegment = &DraggedSegment{Line: lineObj, Index: i}
					ih.IsDrawingLine = false
					return true
				}
			}
		}

	} else if button == 3 {
		station := ih.getStationAtPos(gs, ih.MousePos)
		if station != nil {
			for _, line := range gs.Lines {
				has := false
				for _, s := range line.Stations {
					if s == station {
						has = true; break
					}
				}
				if has {
					ih.removeStationFromLine(gs, line, station)
				}
			}
			return true
		}
	}

	return false
}

func (ih *InputHandler) handleMouseUp(gs *state.GameState, button int) bool {
	if button != 1 {
		return false
	}

	if ih.DraggedTrainResource {
		targetLine := ih.getLineAtPos(gs, ih.MousePos)
		if targetLine != nil && targetLine.Active && len(targetLine.Trains) < config.MaxTrainsPerLine {
			gs.AvailableTrains--
			cityCfg := config.Cities[gs.SelectedCity]
			newTrain := components.NewTrain(gs.TrainIDCounter, targetLine, cityCfg.TrainCapacity, config.TrainMaxSpeed)
			gs.TrainIDCounter++
			gs.Trains = append(gs.Trains, newTrain)
			targetLine.Trains = append(targetLine.Trains, newTrain)
		}
		ih.Reset()
		return true
	}

	if ih.DraggedCarriage {
		targetTrain := ih.getTrainAtPos(gs, ih.MousePos)
		if targetTrain != nil && targetTrain.CarriageCount < config.MaxCarriagesPerTrain {
			targetTrain.CarriageCount++
			gs.Carriages--
		}
		ih.Reset()
		return true
	}

	if ih.DraggedInterchange {
		targetStation := ih.getStationAtPos(gs, ih.MousePos)
		if targetStation != nil && !targetStation.IsInterchange {
			targetStation.IsInterchange = true
			gs.Interchanges--
		}
		ih.Reset()
		return true
	}

	if ih.DraggedExistingTrain != nil {
		targetLine := ih.getLineAtPos(gs, ih.MousePos)
		srcLine := ih.DraggedExistingTrain.Line
		if targetLine != nil && targetLine.Active && targetLine != srcLine && len(targetLine.Trains) < config.MaxTrainsPerLine {
			// Remove from old line
			for i, t := range srcLine.Trains {
				if t == ih.DraggedExistingTrain {
					srcLine.Trains = append(srcLine.Trains[:i], srcLine.Trains[i+1:]...)
					break
				}
			}
			// Assign to new line
			ih.DraggedExistingTrain.Line = targetLine
			targetLine.Trains = append(targetLine.Trains, ih.DraggedExistingTrain)
			ih.DraggedExistingTrain.CurrentStationIndex = 0
			if len(targetLine.Stations) > 1 {
				ih.DraggedExistingTrain.NextStationIndex = 1
			} else {
				ih.DraggedExistingTrain.NextStationIndex = 0
			}
			ih.DraggedExistingTrain.Direction = 1
			ih.DraggedExistingTrain.Progress = 0
			ih.DraggedExistingTrain.State = components.TrainWaiting
			ih.DraggedExistingTrain.WaitTimer = 500
			if len(targetLine.Stations) > 0 {
				ih.DraggedExistingTrain.X = targetLine.Stations[0].X
				ih.DraggedExistingTrain.Y = targetLine.Stations[0].Y
			}
			ih.DraggedExistingTrain.CheckLoopStatus()
		}
		ih.Reset()
		return true
	}

	if gs.SelectedLine >= len(gs.Lines) {
		ih.Reset()
		return false
	}
	line := gs.Lines[gs.SelectedLine]
	targetStation := ih.getStationAtPos(gs, ih.MousePos)

	if ih.IsRemovingSegment && len(ih.CurrentPath) > 0 {
		startStation := ih.CurrentPath[0]
		if targetStation == nil || targetStation == startStation {
			// Find neighbor to refund bridge
			if len(line.Stations) > 0 {
				s1 := line.Stations[0]
				var neighbor *components.Station
				if startStation == s1 && len(line.Stations) > 1 {
					neighbor = line.Stations[1]
				} else if len(line.Stations) > 1 && startStation == line.Stations[len(line.Stations)-1] {
					neighbor = line.Stations[len(line.Stations)-2]
				}

				if neighbor != nil && CheckRiverCrossing(gs, startStation, neighbor) {
					gs.Bridges++
				}
			}
			ih.removeEndStation(gs, line, startStation)
		}
	} else if ih.DraggedSegment != nil && targetStation != nil {
		segment := ih.DraggedSegment
		lineObj := segment.Line
		index := segment.Index

		hasSt := false
		for _, s := range lineObj.Stations {
			if s == targetStation {
				hasSt = true; break
			}
		}
		if !hasSt {
			s1 := lineObj.Stations[index]
			s2 := lineObj.Stations[index+1]
			needs1 := CheckRiverCrossing(gs, s1, targetStation)
			needs2 := CheckRiverCrossing(gs, targetStation, s2)
			hadB := CheckRiverCrossing(gs, s1, s2)
			b1 := 0; b2 := 0; hb := 0
			if needs1 { b1 = 1 }
			if needs2 { b2 = 1 }
			if hadB { hb = 1 }
			cost := b1 + b2 - hb

			if gs.Bridges >= cost {
				gs.Bridges -= cost
				lineObj.AddStation(targetStation, index+1, func() { gs.GraphDirty = true })
			}
		}
	} else if ih.IsDrawingLine && len(ih.CurrentPath) > 0 && targetStation != nil {
		startStation := ih.CurrentPath[0]

		if targetStation != startStation {
			stationsBefore := len(line.Stations)
			needsBridge := CheckRiverCrossing(gs, startStation, targetStation)

			if !(needsBridge && gs.Bridges <= 0) {
				success := false

				if len(line.Stations) == 0 {
					line.AddStation(startStation, -1, nil)
					success = line.AddStation(targetStation, -1, nil)
				} else if line.Stations[0] == startStation {
					success = line.AddStation(targetStation, 0, nil)
				} else if line.Stations[len(line.Stations)-1] == startStation {
					success = line.AddStation(targetStation, -1, nil)
				} else if line.Stations[0] == targetStation {
					success = line.AddStation(startStation, 0, nil)
				} else if line.Stations[len(line.Stations)-1] == targetStation {
					success = line.AddStation(startStation, -1, nil)
				}

				if success {
					log.Printf("[Input] Line %d extended: %d stations", line.Index, len(line.Stations))
					if needsBridge {
						gs.Bridges--
						log.Printf("[Input] Bridge used (remaining: %d)", gs.Bridges)
					}
					if stationsBefore < 2 && len(line.Stations) >= 2 {
						if gs.AvailableTrains > 0 && len(line.Trains) == 0 {
							gs.AvailableTrains--
							cityCfg := config.Cities[gs.SelectedCity]
							newTrain := components.NewTrain(gs.TrainIDCounter, line, cityCfg.TrainCapacity, config.TrainMaxSpeed)
							gs.TrainIDCounter++
							gs.Trains = append(gs.Trains, newTrain)
							line.Trains = append(line.Trains, newTrain)
							log.Printf("[Input] Train %d auto-spawned on line %d", newTrain.ID, line.Index)
						} else if len(line.Trains) == 0 {
							// NO TRAINS AVAILABLE: Rollback the station addition and bridge usage
							log.Printf("[Input] Ghost train line prevented: out of available trains!")
							ih.removeEndStation(gs, line, targetStation)
							if needsBridge {
								gs.Bridges++ // refund
							}
						}
					}
					gs.GraphDirty = true
				}
			}
		}
	}

	ih.Reset()
	return true
}

func (ih *InputHandler) Reset() {
	ih.IsDrawingLine = false
	ih.IsRemovingSegment = false
	ih.DraggedSegment = nil
	ih.CurrentPath = nil
	ih.PreviewLine = nil
	ih.DraggedTrainResource = false
	ih.DraggedCarriage = false
	ih.DraggedInterchange = false
	ih.DraggedExistingTrain = nil
}

func (ih *InputHandler) getStationAtPos(gs *state.GameState, pos Point) *components.Station {
	for _, station := range gs.Stations {
		if math.Hypot(pos.X-station.X, pos.Y-station.Y) <= config.StationRadius+15 {
			return station
		}
	}
	return nil
}

func (ih *InputHandler) getTrainAtPos(gs *state.GameState, pos Point) *components.Train {
	for _, train := range gs.Trains {
		if math.Hypot(pos.X-train.X, pos.Y-train.Y) <= 15 {
			return train
		}
	}
	return nil
}

func (ih *InputHandler) getLineAtPos(gs *state.GameState, pos Point) *components.Line {
	for _, line := range gs.Lines {
		if !line.Active {
			continue
		}
		for i := 0; i < len(line.Stations)-1; i++ {
			s1 := line.Stations[i]
			s2 := line.Stations[i+1]
			if ih.distanceToSegment(pos, s1, s2) < 20 {
				return line
			}
		}
	}
	return nil
}

func CheckRiverCrossing(gs *state.GameState, s1, s2 *components.Station) bool {
	if s1 == nil || s2 == nil {
		return false
	}
	for _, river := range gs.Rivers {
		for i := 0; i < len(river.Points); i++ {
			p1 := river.Points[i]
			p2 := river.Points[(i+1)%len(river.Points)]
			if segmentsIntersect(s1.X, s1.Y, s2.X, s2.Y, p1.X, p1.Y, p2.X, p2.Y) {
				return true
			}
		}
	}
	return false
}

func segmentsIntersect(x1, y1, x2, y2, x3, y3, x4, y4 float64) bool {
	denom := (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
	if denom == 0 {
		return false
	}
	t := ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
	u := -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
	return t > 0 && t < 1 && u > 0 && u < 1
}

func (ih *InputHandler) removeEndStation(gs *state.GameState, line *components.Line, station *components.Station) {
	if len(line.Stations) < 2 {
		return
	}
	isLoop := len(line.Stations) > 2 && line.Stations[0] == line.Stations[len(line.Stations)-1]

	idx := -1
	for i, s := range line.Stations {
		if s == station {
			idx = i; break
		}
	}
	if idx == -1 {
		return
	}
	lastIdx := len(line.Stations) - 1

	if isLoop {
		if station == line.OriginalStart && (idx == 0 || idx == lastIdx) {
			line.Stations = line.Stations[:lastIdx]
			if len(line.Stations) > 0 { line.OriginalEnd = line.Stations[len(line.Stations)-1] }
		} else if station == line.OriginalEnd {
			line.Stations = line.Stations[:lastIdx]
			// Reverse
			for i, j := 0, len(line.Stations)-1; i < j; i, j = i+1, j-1 {
				line.Stations[i], line.Stations[j] = line.Stations[j], line.Stations[i]
			}
			if len(line.Stations) > 0 {
				line.OriginalStart = line.Stations[0]
				line.OriginalEnd = line.Stations[len(line.Stations)-1]
			}
		}
	} else {
		if idx == 0 {
			line.Stations = line.Stations[1:]
			if len(line.Stations) > 0 { line.OriginalStart = line.Stations[0] }
		} else if idx == lastIdx {
			line.Stations = line.Stations[:lastIdx]
			if len(line.Stations) > 0 { line.OriginalEnd = line.Stations[len(line.Stations)-1] }
		}
	}

	if len(line.Stations) < 2 {
		line.MarkedForDeletion = true
	} else {
		line.Active = true
	}
	gs.GraphDirty = true
}

func (ih *InputHandler) removeStationFromLine(gs *state.GameState, line *components.Line, station *components.Station) {
	var indices []int
	for i, s := range line.Stations {
		if s == station {
			indices = append(indices, i)
		}
	}
	if len(indices) == 0 {
		return
	}

	isLoop := len(line.Stations) > 2 && line.Stations[0] == line.Stations[len(line.Stations)-1]
	isEndpoint := !isLoop && (indices[0] == 0 || indices[0] == len(line.Stations)-1)

	if isEndpoint {
		ih.removeEndStation(gs, line, station)
		return
	}

	if len(indices) > 1 {
		line.Stations = line.Stations[:len(line.Stations)-1]
		line.Stations = append(line.Stations[:indices[0]], line.Stations[indices[0]+1:]...)
		if len(line.Stations) > 0 {
			line.Stations = append(line.Stations, line.Stations[0])
		}
	} else {
		line.Stations = append(line.Stations[:indices[0]], line.Stations[indices[0]+1:]...)
	}

	line.Active = len(line.Stations) >= 2
	if !line.Active {
		line.MarkedForDeletion = true
	}
	gs.GraphDirty = true
}
