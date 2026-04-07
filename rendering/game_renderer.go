package rendering

import (
	"fmt"
	"image/color"
	"math"
	"minimetro-go/components"
	"minimetro-go/config"
	"minimetro-go/state"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

// whiteImage is a 3×3 white image used as source texture for DrawTriangles.
var whiteImage *ebiten.Image

func init() {
	whiteImage = ebiten.NewImage(3, 3)
	whiteImage.Fill(color.White)
}

// fillPolygon draws a filled arbitrary polygon using fan triangulation from vertex 0.
// Works well for convex and near-convex polygons.
func fillPolygon(screen *ebiten.Image, pts [][2]float32, clr color.RGBA) {
	n := len(pts)
	if n < 3 {
		return
	}
	r := float32(clr.R) / 255
	g := float32(clr.G) / 255
	b := float32(clr.B) / 255
	a := float32(clr.A) / 255

	verts := make([]ebiten.Vertex, n)
	for i, p := range pts {
		verts[i] = ebiten.Vertex{
			DstX: p[0], DstY: p[1],
			SrcX: 1, SrcY: 1,
			ColorR: r, ColorG: g, ColorB: b, ColorA: a,
		}
	}
	indices := make([]uint16, 0, (n-2)*3)
	for i := 1; i < n-1; i++ {
		indices = append(indices, 0, uint16(i), uint16(i+1))
	}
	screen.DrawTriangles(verts, indices, whiteImage, nil)
}

type GameRenderer struct{}

func NewGameRenderer() *GameRenderer {
	return &GameRenderer{}
}

func (gr *GameRenderer) Render(screen *ebiten.Image, gs *state.GameState) {
	screen.Fill(color.RGBA{244, 241, 233, 255})

	for _, river := range gs.Rivers {
		gr.drawRiver(screen, river)
	}
	for _, line := range gs.Lines {
		gr.drawLine(screen, line, gs.Lines)
	}
	cityBaseCap := config.Cities[gs.SelectedCity].StationCapacity
	for _, station := range gs.Stations {
		gr.drawStation(screen, station, gs.SimTimeMs, cityBaseCap)
	}
	for _, train := range gs.Trains {
		gr.drawTrain(screen, train)
	}
}

func (gr *GameRenderer) drawRiver(screen *ebiten.Image, river *components.River) {
	if len(river.Points) < 3 {
		return
	}
	pts := make([][2]float32, len(river.Points))
	for i, p := range river.Points {
		pts[i] = [2]float32{float32(p.X), float32(p.Y)}
	}
	// Soft blue-gray, no outline — blends with the off-white background.
	fillPolygon(screen, pts, color.RGBA{176, 196, 214, 200})
}

// animProgress returns the animation's normalised progress [0,1] at nowMs,
// or -1 if the animation is nil or has already expired.
func animProgress(anim *components.Animation, nowMs float64) float64 {
	if anim == nil {
		return -1
	}
	elapsed := nowMs - anim.StartTime
	if elapsed < 0 || elapsed >= anim.Duration {
		return -1
	}
	return elapsed / anim.Duration
}

func toRGBA(c [3]int, alpha uint8) color.RGBA {
	return color.RGBA{uint8(c[0]), uint8(c[1]), uint8(c[2]), alpha}
}

func (gr *GameRenderer) drawLine(screen *ebiten.Image, line *components.Line, allLines []*components.Line) {
	if (len(line.Stations) < 2 && !line.MarkedForDeletion) || !line.Active {
		return
	}

	for i := 0; i < len(line.Stations)-1; i++ {
		s1 := line.Stations[i]
		s2 := line.Stations[i+1]

		var shared []*components.Line
		for _, l := range allLines {
			if l.Active && l.HasSegment(s1, s2) {
				shared = append(shared, l)
			}
		}

		n := len(shared)
		myRank := 0
		for j, l := range shared {
			if l.Index == line.Index {
				myRank = j
				break
			}
		}

		offset := (float64(myRank) - float64(n-1)/2.0) * (config.LineWidth + 3.0)
		pts := components.GetTrainWaypoints(s1, s2, offset)

		lineColor := line.Color
		if line.MarkedForDeletion {
			lineColor[0] = int(float64(lineColor[0])*0.5 + 64)
			lineColor[1] = int(float64(lineColor[1])*0.5 + 64)
			lineColor[2] = int(float64(lineColor[2])*0.5 + 64)
		}

		dashed := line.MarkedForDeletion
		w := float32(config.LineWidth)

		gr.drawMetroPath(screen, pts, toRGBA(lineColor, 255), w, dashed)
	}
}

func (gr *GameRenderer) drawMetroPath(screen *ebiten.Image, pts [][2]float64, clr color.Color, width float32, dashed bool) {
	if len(pts) < 2 {
		return
	}

	if dashed {
		for i := 0; i < len(pts)-1; i++ {
			gr.drawDashedLine(screen, pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], clr, width)
		}
	} else {
		for i := 0; i < len(pts)-1; i++ {
			vector.StrokeLine(screen, float32(pts[i][0]), float32(pts[i][1]), float32(pts[i+1][0]), float32(pts[i+1][1]), width, clr, true)
		}
	}
}

func (gr *GameRenderer) drawDashedLine(screen *ebiten.Image, x1, y1, x2, y2 float64, clr color.Color, width float32) {
	dx := x2 - x1
	dy := y2 - y1
	dist := math.Hypot(dx, dy)
	if dist == 0 {
		return
	}
	dashLen := 8.0
	gapLen := 4.0
	totalLen := dashLen + gapLen
	numDashes := int(dist / totalLen)

	for i := 0; i <= numDashes; i++ {
		startRatio := (float64(i) * totalLen) / dist
		endRatio := math.Min((float64(i)*totalLen+dashLen)/dist, 1.0)
		if startRatio >= 1.0 {
			break
		}
		sx := x1 + dx*startRatio
		sy := y1 + dy*startRatio
		ex := x1 + dx*endRatio
		ey := y1 + dy*endRatio
		vector.StrokeLine(screen, float32(sx), float32(sy), float32(ex), float32(ey), width, clr, true)
	}
}

func (gr *GameRenderer) drawStation(screen *ebiten.Image, station *components.Station, nowMs float64, cityBaseCap int) {
	fillClr := color.RGBA{244, 241, 233, 255}
	borderClr := color.RGBA{60, 60, 60, 255}
	radius := float32(config.StationRadius)
	x, y := float32(station.X), float32(station.Y)

	if station.OvercrowdProgress > 0 {
		if station.OvercrowdProgress <= float64(config.OvercrowdTime) {
			pulse := float32(config.StationRadius + 8 + math.Sin(nowMs/config.OvercrowdPulseHz)*5)
			bg := color.RGBA{211, 47, 47, 50}
			fg := color.RGBA{211, 47, 47, 255}

			if station.OvercrowdIsGrace {
				bg = color.RGBA{255, 165, 0, 50}
				fg = color.RGBA{255, 165, 0, 255}
			}
			vector.FillCircle(screen, x, y, pulse, bg, true)
			progress := station.OvercrowdProgress / float64(config.OvercrowdTime)
			vector.StrokeCircle(screen, x, y, float32(config.StationRadius+8), float32(4)*float32(progress), fg, true)
		}
	}

	if station.Type == config.Circle {
		vector.FillCircle(screen, x, y, radius, fillClr, true)
		vector.StrokeCircle(screen, x, y, radius, 3.5, borderClr, true)
	} else if station.Type == config.Triangle {
		h := radius * 1.7
		pts := [][2]float32{
			{x, y - h*0.6},
			{x - h*0.5, y + h*0.4},
			{x + h*0.5, y + h*0.4},
		}
		fillPolygon(screen, pts, fillClr)
		for i := 0; i < len(pts); i++ {
			p1 := pts[i]
			p2 := pts[(i+1)%len(pts)]
			vector.StrokeLine(screen, p1[0], p1[1], p2[0], p2[1], 3.5, borderClr, true)
		}
	} else if station.Type == config.Square {
		size := radius * 1.6
		vector.FillRect(screen, x-size/2, y-size/2, size, size, fillClr, true)
		vector.StrokeRect(screen, x-size/2, y-size/2, size, size, 3.5, borderClr, true)
	}

	if station.IsInterchange {
		vector.StrokeCircle(screen, x, y, radius+8, 5, borderClr, true)
	}

	// Delivery flash: fades from opaque to transparent over the animation duration.
	if p := animProgress(station.DeliveryAnimation, nowMs); p >= 0 {
		alpha := uint8((1.0 - p) * 220)
		vector.FillCircle(screen, x, y, radius*2.5, color.RGBA{255, 220, 50, alpha}, true)
	}

	// Passengers inside station
	passRadius := float32(config.PassengerSize) / 2.0
	passengerArea := config.StationRadius * 0.9
	cols := 3
	slotSize := float64(config.PassengerSize + 2)
	startX := station.X - (float64(cols)/2-0.5)*slotSize
	startY := station.Y - passengerArea/2 + float64(config.PassengerSize)/2

	for i, passenger := range station.Passengers {
		if i < station.Capacity(cityBaseCap) {
			row := i / cols
			col := i % cols
			px := startX + float64(col)*slotSize
			py := startY + float64(row)*slotSize
			gr.drawPassenger(screen, passenger, float32(px), float32(py), passRadius, borderClr)
		} else if i == station.Capacity(cityBaseCap) {
			gr.drawPassenger(screen, passenger, float32(station.X), float32(station.Y+config.StationRadius+15), passRadius, borderClr)
		}
	}
}

func (gr *GameRenderer) drawPassenger(screen *ebiten.Image, passenger *components.Passenger, px, py, radius float32, clr color.Color) {
	drawShape(screen, passenger.Destination, px, py, radius, clr)
}

func drawShape(screen *ebiten.Image, shapeType config.StationType, px, py, radius float32, clr color.Color) {
	strokeW := float32(math.Max(1.0, float64(radius)*0.5))
	switch shapeType {
	case config.Circle:
		vector.FillCircle(screen, px, py, radius, clr, true)
	case config.Square:
		size := radius * 2
		vector.FillRect(screen, px-size/2, py-size/2, size, size, clr, true)
	case config.Triangle:
		pts := [][2]float32{
			{px, py - radius},
			{px - radius, py + radius},
			{px + radius, py + radius},
		}
		for i := range pts {
			p1, p2 := pts[i], pts[(i+1)%len(pts)]
			vector.StrokeLine(screen, p1[0], p1[1], p2[0], p2[1], strokeW, clr, true)
		}
	case config.Pentagon:
		drawPolygon(screen, px, py, radius, 5, clr, strokeW)
	case config.Diamond:
		pts := [][2]float32{
			{px, py - radius},
			{px + radius*0.75, py},
			{px, py + radius},
			{px - radius*0.75, py},
		}
		for i := range pts {
			p1, p2 := pts[i], pts[(i+1)%len(pts)]
			vector.StrokeLine(screen, p1[0], p1[1], p2[0], p2[1], strokeW, clr, true)
		}
	case config.Star:
		drawStar(screen, px, py, radius, clr, strokeW)
	case config.Cross:
		vector.StrokeLine(screen, px-radius, py, px+radius, py, strokeW, clr, true)
		vector.StrokeLine(screen, px, py-radius, px, py+radius, strokeW, clr, true)
	default:
		vector.FillCircle(screen, px, py, radius, clr, true)
	}
}

func drawPolygon(screen *ebiten.Image, cx, cy, radius float32, sides int, clr color.Color, strokeW float32) {
	pts := make([][2]float32, sides)
	for i := range pts {
		a := float64(i)*2*math.Pi/float64(sides) - math.Pi/2
		pts[i] = [2]float32{cx + radius*float32(math.Cos(a)), cy + radius*float32(math.Sin(a))}
	}
	for i := range pts {
		p1, p2 := pts[i], pts[(i+1)%len(pts)]
		vector.StrokeLine(screen, p1[0], p1[1], p2[0], p2[1], strokeW, clr, true)
	}
}

func drawStar(screen *ebiten.Image, cx, cy, radius float32, clr color.Color, strokeW float32) {
	inner := radius * 0.45
	pts := make([][2]float32, 10)
	for i := range pts {
		r := radius
		if i%2 == 1 {
			r = inner
		}
		a := float64(i)*math.Pi/5 - math.Pi/2
		pts[i] = [2]float32{cx + r*float32(math.Cos(a)), cy + r*float32(math.Sin(a))}
	}
	for i := range pts {
		p1, p2 := pts[i], pts[(i+1)%len(pts)]
		vector.StrokeLine(screen, p1[0], p1[1], p2[0], p2[1], strokeW, clr, true)
	}
}

func (gr *GameRenderer) drawTrain(screen *ebiten.Image, train *components.Train) {
	if !train.Line.Active || len(train.Line.Stations) < 2 {
		return
	}

	height := float32(config.TrainHeight)
	width := float32(config.TrainWidth)

	currentSt := train.Line.Stations[train.CurrentStationIndex]
	var nextSt *components.Station
	if train.NextStationIndex < len(train.Line.Stations) {
		nextSt = train.Line.Stations[train.NextStationIndex]
	}
	if currentSt == nil || nextSt == nil {
		nextSt = currentSt
	}

	var angle float64
	if len(train.PathPts) >= 2 {
		pts := train.PathPts
		remaining := math.Max(0.0, train.Progress)
		found := false
		for i := 0; i < len(pts)-1; i++ {
			a, b := pts[i], pts[i+1]
			segLen := math.Hypot(b[0]-a[0], b[1]-a[1])
			if remaining <= segLen || i == len(pts)-2 {
				angle = math.Atan2(b[1]-a[1], b[0]-a[0])
				found = true
				break
			}
			remaining -= segLen
		}
		if !found {
			angle = math.Atan2(pts[len(pts)-1][1]-pts[len(pts)-2][1], pts[len(pts)-1][0]-pts[len(pts)-2][0])
		}
	} else if nextSt != currentSt {
		angle = math.Atan2(nextSt.Y-currentSt.Y, nextSt.X-currentSt.X)
	}

	nUnits := 1 + train.CarriageCount
	img, totalWidth := getTrainBodyImage(train.Line.Color, nUnits)

	var op ebiten.DrawImageOptions
	op.GeoM.Translate(-float64(totalWidth)/2, -float64(height)/2)
	op.GeoM.Rotate(angle)
	op.GeoM.Translate(train.X, train.Y)
	screen.DrawImage(img, &op)

	cosA, sinA := math.Cos(angle), math.Sin(angle)
	maxVisible := train.TotalCapacity()
	if maxVisible > 12 {
		maxVisible = 12
	}
	cols := 2
	slotW := float64(width) / float64(cols+1)
	slotH := float64(height) / 3.0

	for i, p := range train.Passengers {
		if i >= maxVisible {
			break
		}
		unit := i / (cols * 2)
		slotInUnit := i % (cols * 2)
		col := slotInUnit % cols
		row := slotInUnit / cols

		unitOffsetX := float64(unit) * float64(width+5)
		localX := unitOffsetX + slotW*float64(col+1) - float64(totalWidth)/2
		localY := slotH*float64(row+1) - float64(height)/2

		screenX := train.X + localX*cosA - localY*sinA
		screenY := train.Y + localX*sinA + localY*cosA

		gr.drawPassenger(screen, p, float32(screenX), float32(screenY), 2.5, color.White)
	}
}

var trainBodyCache = make(map[string]*ebiten.Image)

func getTrainBodyImage(clr [3]int, nUnits int) (*ebiten.Image, float32) {
	key := fmt.Sprintf("%d-%d-%d-%d", clr[0], clr[1], clr[2], nUnits)
	width, height := float32(config.TrainWidth), float32(config.TrainHeight)
	gap := float32(5)
	totalWidth := float32(nUnits)*width + float32(nUnits-1)*gap

	if img, ok := trainBodyCache[key]; ok {
		return img, totalWidth
	}

	img := ebiten.NewImage(int(totalWidth)+2, int(height)+2)
	img.Fill(color.Transparent)

	lineColor := toRGBA(clr, 255)

	for u := 0; u < nUnits; u++ {
		ux := float32(u) * (width + gap)
		vector.FillRect(img, ux, 0, width, height, lineColor, true)
		// Subtle separator between carriages, no hard border
		if u > 0 {
			vector.StrokeLine(img, ux-gap, height/2, ux, height/2, 1, color.RGBA{0, 0, 0, 40}, true)
		}
	}

	trainBodyCache[key] = img
	return img, totalWidth
}
