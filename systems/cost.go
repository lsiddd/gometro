package systems

import (
	"math"
	"minimetro-go/components"
	"minimetro-go/config"
	"minimetro-go/state"
)

// Cost function weights and bonuses. These govern the tradeoffs the SA
// optimizer makes when choosing between topologies.
const (
	// Angle / geometry
	anglePenaltyWeight = 150.0 // per unit of cos(θ) at each sharp interior angle
	// Increased from 100: soft turns are a first-class objective.

	// Type hygiene
	hygienePenaltyBase = 10.0 // base penalty, doubled exponentially per run length

	// Loop preference — loops eliminate terminal starvation and distribute trains evenly.
	loopBonusValue    = 120.0 // subtracted from line cost when the line forms a loop
	shortLoopBonus    = 60.0  // additional bonus for loops with ≤ 4 unique stations
	// Increased from 50: loops are strongly preferred over point-to-point lines.

	// Geographic compactness
	arcLengthSoftCap  = 550.0 // pixels; arc length below this threshold costs nothing
	arcLengthPenaltyW = 0.18  // penalty per pixel exceeding the soft cap
	// Tightened from 700/0.12: discourage single very long lines that create huge cycle times.

	// Station count — lines with too many stops create enormous cycle times.
	stationCountSoftCap  = 5    // unique stations; beyond this, penalise each additional one
	stationCountPenaltyW = 20.0 // per station beyond soft cap

	// Inter-line crossings outside stations slow trains and create routing ambiguity.
	crossingPenaltyW = 50.0

	// Isolation / overcrowding
	isolatedPenalty = 200.0 // per station not reachable by any active line
	overcrowdWeight = 0.8   // per ms of overcrowd progress across all stations
)

// AnglePenalty returns the total angle penalty for a line.
//
// For each interior station B between A and C, the penalty is proportional to
// cos(angle ABC). An obtuse angle (cos < 0, i.e., > 90°) carries no penalty.
// A sharp U-turn (cos close to 1) receives the full weight — it forces the
// train to decelerate sharply, reducing throughput.
func AnglePenalty(line *components.Line) float64 {
	stations := line.Stations
	n := len(stations)
	if n < 3 {
		return 0
	}
	total := 0.0
	for i := 1; i < n-1; i++ {
		a, b, c := stations[i-1], stations[i], stations[i+1]
		// Skip the loop-closure duplicate station.
		if a == c || a == b || b == c {
			continue
		}
		// Use vectors FROM b: back toward a and forward toward c.
		// This computes the interior angle at b (angle ABC).
		// A straight line gives cos = −1 (no penalty); a U-turn gives cos = +1 (max penalty).
		v1x, v1y := a.X-b.X, a.Y-b.Y // b → a
		v2x, v2y := c.X-b.X, c.Y-b.Y // b → c
		d1 := math.Hypot(v1x, v1y)
		d2 := math.Hypot(v2x, v2y)
		if d1 < 1e-6 || d2 < 1e-6 {
			continue
		}
		cosTheta := (v1x*v2x + v1y*v2y) / (d1 * d2)
		// Only penalise angles ≤ 90° (cos ≥ 0).
		if cosTheta > 0 {
			total += cosTheta * anglePenaltyWeight
		}
	}
	return total
}

// HygienePenalty returns the penalty for consecutive same-type stations.
//
// Runs of identical station types reduce type coverage and slow passenger
// routing. Each additional station in a same-type run doubles the penalty,
// making three circles in a row far more expensive than two.
func HygienePenalty(line *components.Line) float64 {
	stations := line.Stations
	n := len(stations)
	if n < 2 {
		return 0
	}
	total := 0.0
	runLen := 1
	for i := 1; i < n; i++ {
		// Skip the loop-closure repeated entry.
		if stations[i] == stations[i-1] {
			continue
		}
		if stations[i].Type == stations[i-1].Type {
			runLen++
			total += math.Pow(2, float64(runLen)) * hygienePenaltyBase
		} else {
			runLen = 1
		}
	}
	return total
}

// LoopBonus returns a bonus (negative cost delta) when the line forms a closed
// loop. Loops provide cadence stability, eliminate terminal-starvation bouncing,
// and distribute trains more evenly. Short loops (≤ 4 stations) receive an
// extra bonus because their small cycle time makes travel times predictable.
func LoopBonus(line *components.Line) float64 {
	n := len(line.Stations)
	if n < 3 || line.Stations[0] != line.Stations[n-1] {
		return 0
	}
	bonus := loopBonusValue
	uniqueN := n - 1 // subtract the duplicate closing station
	if uniqueN <= 4 {
		bonus += shortLoopBonus
	}
	return bonus
}

// ArcLengthPenalty penalises geographically long lines. The total Euclidean
// length of the line is free up to arcLengthSoftCap pixels; every pixel beyond
// that incurs arcLengthPenaltyW cost. This discourages routing a single line
// across the entire map when a second line would serve passengers better.
func ArcLengthPenalty(line *components.Line) float64 {
	stations := line.Stations
	n := len(stations)
	if n < 2 {
		return 0
	}
	total := 0.0
	for i := 0; i < n-1; i++ {
		a, b := stations[i], stations[i+1]
		if a == b { // skip loop-closure duplicate
			continue
		}
		total += math.Hypot(b.X-a.X, b.Y-a.Y)
	}
	if total <= arcLengthSoftCap {
		return 0
	}
	return (total - arcLengthSoftCap) * arcLengthPenaltyW
}

// StationCountPenalty penalises lines with more than stationCountSoftCap unique
// stations. Very long lines create enormous cycle times — a train must traverse
// 8+ stops before returning to pick up new passengers at the start, which
// causes local collapse. Two shorter lines serve the same area better.
func StationCountPenalty(line *components.Line) float64 {
	stations := line.Stations
	n := len(stations)
	isLoop := n >= 3 && stations[0] == stations[n-1]
	unique := n
	if isLoop {
		unique = n - 1
	}
	if unique <= stationCountSoftCap {
		return 0
	}
	return float64(unique-stationCountSoftCap) * stationCountPenaltyW
}

// segmentsIntersectStrict returns true when segments AB and CD have a proper
// interior crossing — neither endpoint of one lies on the other. This is used
// to detect line crossings outside of shared stations.
func segmentsIntersectStrict(ax, ay, bx, by, cx, cy, dx, dy float64) bool {
	cross := func(ox, oy, px, py, qx, qy float64) float64 {
		return (px-ox)*(qy-oy) - (py-oy)*(qx-ox)
	}
	d1 := cross(cx, cy, dx, dy, ax, ay)
	d2 := cross(cx, cy, dx, dy, bx, by)
	d3 := cross(ax, ay, bx, by, cx, cy)
	d4 := cross(ax, ay, bx, by, dx, dy)
	return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
		((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))
}

// NetworkCrossingPenalty returns a penalty proportional to the number of
// strict interior crossings between segments belonging to different lines.
// Crossings that occur at shared stations are exempt (those are interchanges
// and are desirable). This encourages a grid-like topology where lines meet
// only at stations rather than crossing mid-segment.
func NetworkCrossingPenalty(gs *state.GameState) float64 {
	type seg struct {
		x1, y1, x2, y2 float64
		lineIdx         int
	}

	var segs []seg
	for i := 0; i < gs.AvailableLines && i < len(gs.Lines); i++ {
		l := gs.Lines[i]
		if !l.Active || l.MarkedForDeletion {
			continue
		}
		for j := 0; j < len(l.Stations)-1; j++ {
			a, b := l.Stations[j], l.Stations[j+1]
			if a == b {
				continue
			}
			segs = append(segs, seg{a.X, a.Y, b.X, b.Y, i})
		}
	}

	penalty := 0.0
	for i := 0; i < len(segs); i++ {
		for j := i + 1; j < len(segs); j++ {
			if segs[i].lineIdx == segs[j].lineIdx {
				continue
			}
			if segmentsIntersectStrict(
				segs[i].x1, segs[i].y1, segs[i].x2, segs[i].y2,
				segs[j].x1, segs[j].y1, segs[j].x2, segs[j].y2,
			) {
				penalty += crossingPenaltyW
			}
		}
	}
	return penalty
}

// LineCost computes the geometric cost for a single active line.
// Lower values indicate a healthier topology:
//   - smooth angles (soft turns preferred)
//   - type variety (alternating station types)
//   - loops preferred over open lines
//   - compact geographic extent
//   - few stations per line (to avoid long cycle times)
func LineCost(line *components.Line) float64 {
	if !line.Active || line.MarkedForDeletion {
		return 0
	}
	return AnglePenalty(line) +
		HygienePenalty(line) +
		ArcLengthPenalty(line) +
		StationCountPenalty(line) -
		LoopBonus(line)
}

// NetworkCost computes the total topology quality score for a game state.
// It is the primary objective function used by the SA optimizer — lower is
// always better.
//
// Four components contribute:
//  1. Geometric line cost (angles + hygiene − loop bonuses + station count) across all lines.
//  2. Isolated-station penalty for every station unreachable by any active line.
//  3. Overcrowding pressure: the sum of overcrowd timers biased by station count.
//  4. Inter-line crossing penalty for segments that cross outside of stations.
func NetworkCost(gs *state.GameState) float64 {
	total := 0.0

	// 1. Line topology quality.
	for i := 0; i < gs.AvailableLines && i < len(gs.Lines); i++ {
		total += LineCost(gs.Lines[i])
	}

	// 2. Isolation penalty.
	onLine := make(map[*components.Station]bool, len(gs.Stations))
	for i := 0; i < gs.AvailableLines && i < len(gs.Lines); i++ {
		for _, st := range gs.Lines[i].Stations {
			onLine[st] = true
		}
	}
	for _, st := range gs.Stations {
		if !onLine[st] {
			total += isolatedPenalty
		}
	}

	// 3. Overcrowding pressure.
	for _, st := range gs.Stations {
		if st.OvercrowdProgress > 0 {
			fraction := st.OvercrowdProgress / float64(config.OvercrowdTime)
			total += fraction * overcrowdWeight * float64(config.OvercrowdTime)
		}
	}

	// 4. Inter-line crossings outside stations.
	total += NetworkCrossingPenalty(gs)

	return total
}
