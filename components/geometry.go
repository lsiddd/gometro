package components

import (
	"fmt"
	"math"
)

// ComputeMetroWaypoints returns the intermediate waypoints for a metro line segment
// drawn in a Manhattan-style diagonal-then-straight path.
func ComputeMetroWaypoints(p1X, p1Y, p2X, p2Y float64) [][2]float64 {
	dx := p2X - p1X
	dy := p2Y - p1Y
	adx, ady := math.Abs(dx), math.Abs(dy)

	if adx < 2 || ady < 2 || math.Abs(adx-ady) < 2 {
		return [][2]float64{{p1X, p1Y}, {p2X, p2Y}}
	}

	sx, sy := 1.0, 1.0
	if dx < 0 {
		sx = -1.0
	}
	if dy < 0 {
		sy = -1.0
	}

	diag := math.Min(adx, ady)
	elbowX := p1X + diag*sx
	elbowY := p1Y + diag*sy
	return [][2]float64{{p1X, p1Y}, {elbowX, elbowY}, {p2X, p2Y}}
}

// UnitVec returns the unit vector for the given direction.
func UnitVec(dx, dy float64) (float64, float64) {
	d := math.Sqrt(dx*dx + dy*dy)
	if d < 1e-6 {
		return 0, 0
	}
	return dx / d, dy / d
}

// PerpUnit returns the unit perpendicular (left-hand normal) of the given vector.
func PerpUnit(dx, dy float64) (float64, float64) {
	d := math.Sqrt(dx*dx + dy*dy)
	if d < 1e-6 {
		return 0, -1
	}
	return -dy / d, dx / d
}

// LineIntersect2D finds the intersection of two infinite lines defined by a point
// and direction vector each. Returns an error when the lines are parallel.
func LineIntersect2D(px1, py1, dx1, dy1, px2, py2, dx2, dy2 float64) (float64, float64, error) {
	denom := dx1*dy2 - dy1*dx2
	if math.Abs(denom) < 1e-9 {
		return 0, 0, fmt.Errorf("parallel")
	}
	t := ((px2-px1)*dy2 - (py2-py1)*dx2) / denom
	return px1 + t*dx1, py1 + t*dy1, nil
}

// OffsetPath shifts a 2- or 3-point path laterally by offset units (perpendicular to
// the path direction). Used to render parallel metro lines without overlap.
func OffsetPath(pts [][2]float64, offset float64) [][2]float64 {
	if len(pts) == 2 {
		p0, p2 := pts[0], pts[1]
		nx, ny := PerpUnit(p2[0]-p0[0], p2[1]-p0[1])
		return [][2]float64{
			{p0[0] + offset*nx, p0[1] + offset*ny},
			{p2[0] + offset*nx, p2[1] + offset*ny},
		}
	}

	p0, p1, p2 := pts[0], pts[1], pts[2]
	d01x, d01y := UnitVec(p1[0]-p0[0], p1[1]-p0[1])
	d12x, d12y := UnitVec(p2[0]-p1[0], p2[1]-p1[1])

	n01x, n01y := -d01y, d01x
	n12x, n12y := -d12y, d12x

	p0_off := [2]float64{p0[0] + offset*n01x, p0[1] + offset*n01y}
	p2_off := [2]float64{p2[0] + offset*n12x, p2[1] + offset*n12y}

	ax, ay := p1[0]+offset*n01x, p1[1]+offset*n01y
	bx, by := p1[0]+offset*n12x, p1[1]+offset*n12y

	ix, iy, err := LineIntersect2D(ax, ay, d01x, d01y, bx, by, d12x, d12y)
	if err != nil {
		ix, iy = (ax+bx)/2.0, (ay+by)/2.0
	}

	return [][2]float64{p0_off, {ix, iy}, p2_off}
}

// GetTrainWaypoints returns a Bezier-smoothed path for a train travelling between
// two stations, optionally offset laterally for multi-line rendering.
func GetTrainWaypoints(s1, s2 *Station, offset float64) [][2]float64 {
	base := ComputeMetroWaypoints(s1.X, s1.Y, s2.X, s2.Y)
	pts := base
	if math.Abs(offset) > 0.01 {
		pts = OffsetPath(base, offset)
	}

	if len(pts) == 2 {
		return pts
	}

	p0, p1, p2 := pts[0], pts[1], pts[2]
	r := 14.0
	d01x, d01y := UnitVec(p1[0]-p0[0], p1[1]-p0[1])
	d12x, d12y := UnitVec(p2[0]-p1[0], p2[1]-p1[1])
	tax, tay := p1[0]-d01x*r, p1[1]-d01y*r
	tbx, tby := p1[0]+d12x*r, p1[1]+d12y*r

	var bezierPts [][2]float64
	bezierPts = append(bezierPts, p0)
	nB := 10
	for i := 0; i <= nB; i++ {
		t := float64(i) / float64(nB)
		x := math.Pow(1-t, 2)*tax + 2*(1-t)*t*p1[0] + math.Pow(t, 2)*tbx
		y := math.Pow(1-t, 2)*tay + 2*(1-t)*t*p1[1] + math.Pow(t, 2)*tby
		bezierPts = append(bezierPts, [2]float64{x, y})
	}
	bezierPts = append(bezierPts, p2)
	return bezierPts
}
