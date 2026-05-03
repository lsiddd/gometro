package systems

import (
	"math/rand"
	"minimetro-go/state"
)

const (
	UpgradeNewLine     = "new_line"
	UpgradeCarriage    = "carriage"
	UpgradeBridge      = "bridge"
	UpgradeInterchange = "interchange"
)

type weightedUpgrade struct {
	key    string
	weight float64
}

// GenerateUpgradeChoices builds a weighted pool of upgrades and draws 2 distinct
// options. The locomotive bonus is always granted before this is called.
func (g *Game) GenerateUpgradeChoices(gs *state.GameState) []string {
	linesAtMax := gs.AvailableLines >= gs.MaxLines
	hasRivers := len(gs.Rivers) > 0
	week := gs.Week

	var pool []weightedUpgrade

	// New Line is common early, fades out late, and is absent when at cap.
	if !linesAtMax {
		w := 5.0
		if week >= 4 {
			w = 2.0
		}
		if week >= 6 {
			w = 1.0
		}
		pool = append(pool, weightedUpgrade{UpgradeNewLine, w})
	}

	// Carriage scales up as the network fills.
	{
		w := 1.5
		if week >= 4 {
			w = 3.5
		}
		if week >= 6 {
			w = 5.0
		}
		pool = append(pool, weightedUpgrade{UpgradeCarriage, w})
	}

	// Bridge/Tunnel is only available on maps with rivers and gets doubled
	// weight when the stock is empty.
	if hasRivers {
		w := 3.5
		if week >= 5 {
			w = 2.0
		}
		if gs.Bridges == 0 {
			w *= 2.0
		}
		pool = append(pool, weightedUpgrade{UpgradeBridge, w})
	}

	// Interchange is unavailable on week 1 and gains weight in late game.
	if week >= 2 {
		w := 1.0
		if week >= 5 {
			w = 3.5
		}
		pool = append(pool, weightedUpgrade{UpgradeInterchange, w})
	}

	for len(pool) < 2 {
		pool = append(pool, weightedUpgrade{UpgradeCarriage, 1.0})
	}

	return pickWeightedUpgrades(gs.Rand(), pool, 2)
}

func pickWeightedUpgrades(rng *rand.Rand, pool []weightedUpgrade, n int) []string {
	result := make([]string, 0, n)
	remaining := make([]weightedUpgrade, len(pool))
	copy(remaining, pool)

	for i := 0; i < n && len(remaining) > 0; i++ {
		total := 0.0
		for _, w := range remaining {
			total += w.weight
		}
		r := rng.Float64() * total
		cumulative := 0.0
		chosen := len(remaining) - 1
		for j, w := range remaining {
			cumulative += w.weight
			if r <= cumulative {
				chosen = j
				break
			}
		}
		result = append(result, remaining[chosen].key)
		remaining = append(remaining[:chosen], remaining[chosen+1:]...)
	}
	return result
}

// ApplyUpgrade mutates the game state to grant the chosen weekly upgrade.
func ApplyUpgrade(gs *state.GameState, choice string) {
	switch choice {
	case UpgradeNewLine:
		if gs.AvailableLines < gs.MaxLines {
			gs.AvailableLines++
		}
	case UpgradeCarriage:
		gs.Carriages++
	case UpgradeBridge:
		gs.Bridges += 2
	case UpgradeInterchange:
		gs.Interchanges++
	}
}
