package main

import (
	"minimetro-go/state"
	"testing"
)

// TestTickDelta verifies that the effective frame time scales correctly with
// the fast-forward setting. At 60 Hz the base frame is 1000/60 ≈ 16.67 ms;
// FF levels 1 and 2 multiply by 2× and 4× respectively.
func TestTickDelta_AllFastForwardLevels(t *testing.T) {
	const baseMs = 1000.0 / 60.0
	cases := []struct {
		ff   int
		want float64
	}{
		{0, baseMs},
		{1, baseMs * 2},
		{2, baseMs * 4},
	}

	for _, tc := range cases {
		m := &MinimetroGame{GameState: state.NewGameState()}
		m.GameState.FastForward = tc.ff
		got := m.tickDelta()
		if got != tc.want {
			t.Errorf("FastForward=%d: want %.6f ms, got %.6f ms", tc.ff, tc.want, got)
		}
	}
}
