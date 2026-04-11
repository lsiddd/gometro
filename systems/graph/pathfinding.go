package graph

import (
	"minimetro-go/components"
	"minimetro-go/config"
	"minimetro-go/state"
)

type Edge struct {
	To   *components.Station
	Line *components.Line
}

type GraphManager struct {
	graph   map[*components.Station][]Edge
	isDirty bool
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

// Simple BFS that prioritizes fewer transfers (simplified from A* for Go without heavy heap deps).
// Path reconstruction uses parent pointers rather than copying the path slice at every BFS
// expansion, reducing allocations from O(nodes×depth) to O(nodes).
func FindPath(gm *GraphManager, gameState *state.GameState, startStation *components.Station, destinationType config.StationType) []*components.Station {
	if startStation == nil {
		return nil
	}
	if startStation.Type == destinationType {
		return []*components.Station{startStation}
	}

	graph := gm.GetGraph(gameState)

	type State struct {
		Station   *components.Station
		Depth     int // path length from start (startStation = 1)
		Transfers int
		LastLine  *components.Line
	}

	n := len(gameState.Stations)
	queue := make([]State, 0, n)
	queue = append(queue, State{Station: startStation, Depth: 1, Transfers: 0, LastLine: nil})

	visited := make(map[*components.Station]float64, n)
	parent := make(map[*components.Station]*components.Station, n)
	visited[startStation] = 0

	var bestEnd *components.Station
	bestScore := float64(999999)

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]

		if curr.Station.Type == destinationType {
			score := float64(curr.Depth) + float64(curr.Transfers)*config.TransferPenalty
			if score < bestScore {
				bestScore = score
				bestEnd = curr.Station
			}
			continue
		}

		for _, edge := range graph[curr.Station] {
			neighbor := edge.To
			line := edge.Line

			newTransfers := curr.Transfers
			if curr.LastLine != nil && curr.LastLine != line {
				newTransfers++
			}

			newDepth := curr.Depth + 1
			score := float64(newDepth) + float64(newTransfers)*config.TransferPenalty
			if neighbor.OvercrowdProgress > 0 && neighbor.Type != destinationType {
				score += (neighbor.OvercrowdProgress / config.OvercrowdTime) * config.OvercrowdScoreFactor
			}

			// Limit path length to 20 to prevent runaway BFS in complex cyclic graphs.
			if prevScore, ok := visited[neighbor]; !ok || prevScore > score || (prevScore == score && newDepth < 20) {
				visited[neighbor] = score
				parent[neighbor] = curr.Station
				queue = append(queue, State{
					Station:   neighbor,
					Depth:     newDepth,
					Transfers: newTransfers,
					LastLine:  line,
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
