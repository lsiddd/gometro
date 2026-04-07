package components

import "minimetro-go/config"

type River struct {
	Points []config.PointF
}

// Ray-casting algorithm to check if point is inside river polygon
func (r *River) Contains(x, y float64) bool {
	inside := false
	j := len(r.Points) - 1
	for i := 0; i < len(r.Points); i++ {
		xi := r.Points[i].X
		yi := r.Points[i].Y
		xj := r.Points[j].X
		yj := r.Points[j].Y

		intersect := ((yi > y) != (yj > y)) && (x < (xj-xi)*(y-yi)/(yj-yi)+xi)
		if intersect {
			inside = !inside
		}
		j = i
	}
	return inside
}
