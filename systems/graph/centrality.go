package graph

import (
	"minimetro-go/components"
	"minimetro-go/state"
)

// BetweennessCentrality computes the normalised betweenness centrality for
// every station in the current passenger-routing graph.
//
// A station's centrality measures how often it appears on the shortest paths
// between all other station pairs. High-centrality stations are natural
// bottlenecks: upgrading them to interchanges (tripling capacity) yields the
// greatest relief per resource spent.
//
// The implementation follows Brandes' BFS-based algorithm for unweighted graphs,
// consistent with the passenger routing logic in pathfinding.go.
//
// Result values are normalised to [0, 1] using (n−1)(n−2)/2 so they are
// comparable across maps with different station counts.
func BetweennessCentrality(gs *state.GameState, gm *GraphManager) map[*components.Station]float64 {
	graph := gm.GetGraph(gs)
	stations := gs.Stations
	n := len(stations)

	counts := make(map[*components.Station]float64, n)
	for _, s := range stations {
		counts[s] = 0
	}

	if n < 3 {
		return counts // not enough nodes for non-trivial centrality
	}

	// Build a station → index map for O(1) position lookup.
	idx := make(map[*components.Station]int, n)
	for i, s := range stations {
		idx[s] = i
	}

	// Allocate working slices once outside the source loop and reset per iteration,
	// replacing the four per-source map allocations in the original implementation.
	sigma := make([]float64, n)
	dist := make([]int, n)
	delta := make([]float64, n)
	pred := make([][]int, n)  // pred[i] = indices of predecessors of station i
	order := make([]int, 0, n)
	queue := make([]*components.Station, 0, n)
	rawCounts := make([]float64, n)

	for srcIdx, src := range stations {
		// Reset per-source state; reuse backing arrays to avoid heap pressure.
		for i := 0; i < n; i++ {
			sigma[i] = 0
			dist[i] = -1
			delta[i] = 0
			pred[i] = pred[i][:0] // keep capacity, reset length
		}
		order = order[:0]
		queue = queue[:0]
		qHead := 0

		sigma[srcIdx] = 1
		dist[srcIdx] = 0
		queue = append(queue, src)

		for qHead < len(queue) {
			v := queue[qHead]
			qHead++
			vi := idx[v]
			order = append(order, vi)

			for _, edge := range graph[v] {
				w := edge.To
				wi := idx[w]
				if dist[wi] < 0 { // first visit
					queue = append(queue, w)
					dist[wi] = dist[vi] + 1
				}
				if dist[wi] == dist[vi]+1 { // shortest-path edge
					sigma[wi] += sigma[vi]
					pred[wi] = append(pred[wi], vi)
				}
			}
		}

		// Back-propagate dependency scores (Brandes' accumulation step).
		for i := len(order) - 1; i >= 0; i-- {
			wi := order[i]
			for _, vi := range pred[wi] {
				if sigma[wi] > 0 {
					delta[vi] += (sigma[vi] / sigma[wi]) * (1 + delta[wi])
				}
			}
			if wi != srcIdx {
				rawCounts[wi] += delta[wi]
			}
		}
	}

	// Normalise. Because we run BFS from every source and edges are bidirectional,
	// Brandes on an undirected graph accumulates each unordered pair {s,t} twice
	// (once from s, once from t). The correct normalisation factor is therefore
	// (n−1)(n−2) — the directed formula — which absorbs the factor-of-2 from the
	// double-counting and keeps all values in [0, 1].
	norm := float64(n-1) * float64(n-2)
	for i, s := range stations {
		if norm > 0 {
			counts[s] = rawCounts[i] / norm
		}
	}

	return counts
}
