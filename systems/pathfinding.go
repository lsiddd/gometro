package systems

import (
	"minimetro-go/components"
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

// Simple BFS that prioritizes fewer transfers (simplified from A* for Go without heavy heap deps)
func FindPath(gm *GraphManager, gameState *state.GameState, startStation *components.Station, destinationType string) []*components.Station {
	if startStation == nil {
		return nil
	}
	if startStation.Type == destinationType {
		return []*components.Station{startStation}
	}

	graph := gm.GetGraph(gameState)

	type State struct {
		Station   *components.Station
		Path      []*components.Station
		Transfers int
		LastLine  *components.Line
	}

	queue := []State{{
		Station:   startStation,
		Path:      []*components.Station{startStation},
		Transfers: 0,
		LastLine:  nil,
	}}

	visited := make(map[*components.Station]float64) // keep track of min score to this station
	visited[startStation] = 0

	var bestPath []*components.Station
	bestScore := float64(999999)

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]

		if curr.Station.Type == destinationType {
			score := float64(len(curr.Path)) + float64(curr.Transfers)*2.5
			if score < bestScore {
				bestScore = score
				bestPath = curr.Path
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

			score := float64(len(curr.Path)) + float64(newTransfers)*2.5
			if neighbor.OvercrowdProgress > 0 && neighbor.Type != destinationType {
				score += (neighbor.OvercrowdProgress / 45000.0) * 4.0
			}

			// Limit path length to 20 to prevent runaway BFS in complex cyclic graphs
			if prevScore, ok := visited[neighbor]; !ok || prevScore > score || (prevScore == score && len(curr.Path) < 20) {
				visited[neighbor] = score
				
				// Capacity is important to cap memory allocs when branching
				newPath := make([]*components.Station, len(curr.Path), len(curr.Path) + 1)
				copy(newPath, curr.Path)
				newPath = append(newPath, neighbor)

				queue = append(queue, State{
					Station:   neighbor,
					Path:      newPath,
					Transfers: newTransfers,
					LastLine:  line,
				})
			}
		}
	}

	return bestPath
}
