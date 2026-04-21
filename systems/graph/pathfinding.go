package graph

import (
	"container/heap"
	"minimetro-go/components"
	"minimetro-go/config"
	"minimetro-go/state"
)

// pqItem is a node entry in the Dijkstra priority queue.
type pqItem struct {
	station   *components.Station
	depth     int
	transfers int
	lastLine  *components.Line
	score     float64
}

// minHeap implements heap.Interface ordered by ascending score.
type minHeap []*pqItem

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i].score < h[j].score }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)         { *h = append(*h, x.(*pqItem)) }
func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	old[n-1] = nil
	*h = old[:n-1]
	return x
}

type Edge struct {
	To   *components.Station
	Line *components.Line
}

type GraphManager struct {
	graph       map[*components.Station][]Edge
	isDirty     bool
	centrality  map[*components.Station]float64 // nil when stale
}

func NewGraphManager() *GraphManager {
	return &GraphManager{
		isDirty: true,
	}
}

func (gm *GraphManager) MarkDirty() {
	gm.isDirty = true
}

func (gm *GraphManager) GetGraph(gameState *state.GameState) map[*components.Station][]Edge {
	if gameState.GraphDirty || gm.graph == nil {
		gm.graph = gm.buildGraph(gameState)
		gm.centrality = nil // topology changed — centrality must be recomputed
		gameState.GraphDirty = false
		// Topology changed: cached paths may now be invalid.
		// Nil them so each passenger recomputes lazily on next boarding attempt.
		for _, p := range gameState.Passengers {
			p.Path = nil
			p.PathIndex = 0
		}
	}
	return gm.graph
}

// CachedCentrality returns the betweenness centrality map, computing it only
// when the topology has changed since the last call. This avoids recomputing
// the O(V·E) Brandes algorithm every frame when the graph is stable.
func (gm *GraphManager) CachedCentrality(gs *state.GameState) map[*components.Station]float64 {
	if gm.centrality == nil {
		gm.centrality = BetweennessCentrality(gs, gm)
	}
	return gm.centrality
}

func (gm *GraphManager) buildGraph(gameState *state.GameState) map[*components.Station][]Edge {
	graph := make(map[*components.Station][]Edge, len(gameState.Stations))

	for _, s := range gameState.Stations {
		graph[s] = make([]Edge, 0, 4) // Pre-allocate small capacity
	}

	for _, line := range gameState.Lines {
		if !line.Active || len(line.Stations) < 2 {
			continue
		}
		stations := line.Stations

		numIter := len(stations) - 1
		for i := 0; i < numIter; i++ {
			current := stations[i]
			next := stations[i+1]

			// Avoid adding the same edge multiple times if stations alternate weirdly
			// For Mini Metro clones, multiple lines can bridge the same segment. We just add them all.
			graph[current] = append(graph[current], Edge{To: next, Line: line})
			graph[next] = append(graph[next], Edge{To: current, Line: line})
		}
	}
	return graph
}

// FindPath finds the minimum-cost path from startStation to any station of
// destinationType using Dijkstra's algorithm. Cost = depth + transfers*TransferPenalty
// + overcrowd penalty. Parent pointers are used for O(nodes) path reconstruction.
func FindPath(gm *GraphManager, gameState *state.GameState, startStation *components.Station, destinationType config.StationType) []*components.Station {
	if startStation == nil {
		return nil
	}
	if startStation.Type == destinationType {
		return []*components.Station{startStation}
	}

	graph := gm.GetGraph(gameState)

	n := len(gameState.Stations)
	visited := make(map[*components.Station]float64, n)
	parent := make(map[*components.Station]*components.Station, n)

	h := make(minHeap, 0, n)
	heap.Init(&h)
	heap.Push(&h, &pqItem{station: startStation, depth: 1, score: 0})
	visited[startStation] = 0

	var bestEnd *components.Station

	for h.Len() > 0 {
		curr := heap.Pop(&h).(*pqItem)

		// Dijkstra: skip stale entries (node already settled at lower cost).
		if prevScore, ok := visited[curr.station]; ok && curr.score > prevScore {
			continue
		}

		// First pop of a destination station is optimal.
		if curr.station.Type == destinationType {
			bestEnd = curr.station
			break
		}

		for _, edge := range graph[curr.station] {
			neighbor := edge.To
			line := edge.Line

			newTransfers := curr.transfers
			if curr.lastLine != nil && curr.lastLine != line {
				newTransfers++
			}

			newDepth := curr.depth + 1
			// Limit path length to prevent runaway expansion in complex cyclic graphs.
			if newDepth > 20 {
				continue
			}

			score := float64(newDepth) + float64(newTransfers)*config.TransferPenalty
			if neighbor.OvercrowdProgress > 0 && neighbor.Type != destinationType {
				score += (neighbor.OvercrowdProgress / config.OvercrowdTime) * config.OvercrowdScoreFactor
			}

			if prevScore, ok := visited[neighbor]; !ok || score < prevScore {
				visited[neighbor] = score
				parent[neighbor] = curr.station
				heap.Push(&h, &pqItem{
					station:   neighbor,
					depth:     newDepth,
					transfers: newTransfers,
					lastLine:  line,
					score:     score,
				})
			}
		}
	}

	if bestEnd == nil {
		return nil
	}

	// Reconstruct path by backtracking through parent pointers, then reverse.
	var path []*components.Station
	for s := bestEnd; s != startStation; s = parent[s] {
		path = append(path, s)
		if parent[s] == nil {
			// Disconnected parent chain — should not happen in a valid graph.
			return nil
		}
	}
	path = append(path, startStation)
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	return path
}
