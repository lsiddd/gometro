package systems

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

	for _, src := range stations {
		// σ[v] = number of shortest paths from src to v.
		sigma := make(map[*components.Station]float64, n)
		// dist[v] = BFS distance from src to v (−1 = unvisited).
		dist := make(map[*components.Station]int, n)
		// pred[v] = predecessors of v on shortest paths from src.
		pred := make(map[*components.Station][]*components.Station, n)

		for _, s := range stations {
			sigma[s] = 0
			dist[s] = -1
		}
		sigma[src] = 1
		dist[src] = 0

		queue := make([]*components.Station, 0, n)
		queue = append(queue, src)

		// BFS traversal order (for back-propagation).
		order := make([]*components.Station, 0, n)

		for len(queue) > 0 {
			v := queue[0]
			queue = queue[1:]
			order = append(order, v)

			for _, edge := range graph[v] {
				w := edge.To
				if dist[w] < 0 { // first visit
					queue = append(queue, w)
					dist[w] = dist[v] + 1
				}
				if dist[w] == dist[v]+1 { // shortest-path edge
					sigma[w] += sigma[v]
					pred[w] = append(pred[w], v)
				}
			}
		}

		// Back-propagate dependency scores (Brandes' accumulation step).
		delta := make(map[*components.Station]float64, n)
		for i := len(order) - 1; i >= 0; i-- {
			w := order[i]
			for _, v := range pred[w] {
				if sigma[w] > 0 {
					delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
				}
			}
			if w != src {
				counts[w] += delta[w]
			}
		}
	}

	// Normalise. Because we run BFS from every source and edges are bidirectional,
	// Brandes on an undirected graph accumulates each unordered pair {s,t} twice
	// (once from s, once from t). The correct normalisation factor is therefore
	// (n−1)(n−2) — the directed formula — which absorbs the factor-of-2 from the
	// double-counting and keeps all values in [0, 1].
	norm := float64(n-1) * float64(n-2)
	if norm > 0 {
		for s := range counts {
			counts[s] /= norm
		}
	}

	return counts
}
