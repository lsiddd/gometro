package main

import (
	"flag"
	"log"
	"minimetro-go/rendering"
	"minimetro-go/rl"
	"minimetro-go/state"
	"minimetro-go/systems"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
)

type MinimetroGame struct {
	GameState    *state.GameState
	GameSystem   *systems.Game
	Renderer     *rendering.GameRenderer
	UI           *systems.UI
	InputHandler *systems.InputHandler
	Solver       *systems.Solver
	RLClient     *rl.Client // non-nil when --rl-client flag is set
	screenW      int
	screenH      int
}

func (m *MinimetroGame) Update() error {
	m.UI.Width = float64(m.screenW)
	m.UI.Height = float64(m.screenH)

	effectiveDelta := m.tickDelta()
	m.GameState.SimTimeMs += effectiveDelta

	m.handleInput()
	m.UI.Update(m.GameState, m.InputHandler, m.GameSystem)

	if !m.UI.ShowStartScreen && !m.UI.ShowGameOverModal && !m.UI.ShowUpgradeModal {
		m.tickSimulation(effectiveDelta)
		m.tickAI()
	}

	return nil
}

// tickDelta returns the effective milliseconds for this frame, scaled by the
// current fast-forward factor. Uses a fixed 60 Hz base so physics are stable
// regardless of actual TPS (which can fluctuate or be 0 on some platforms).
func (m *MinimetroGame) tickDelta() float64 {
	const baseMs = 1000.0 / 60.0
	ffFactors := [3]float64{1, 2, 4}
	return baseMs * ffFactors[m.GameState.FastForward]
}

// handleInput polls Ebitengine for the current input state, updates the
// InputHandler, and logs click events for debugging.
func (m *MinimetroGame) handleInput() {
	x, y := ebiten.CursorPosition()
	worldX, worldY := float64(x), float64(y)

	m.InputHandler.Update(
		m.GameState, worldX, worldY,
		ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft),
		ebiten.IsMouseButtonPressed(ebiten.MouseButtonRight),
		ebiten.IsKeyPressed(ebiten.KeyShift) || ebiten.IsKeyPressed(ebiten.KeyShiftLeft),
		inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft),
		inpututil.IsMouseButtonJustReleased(ebiten.MouseButtonLeft),
		inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonRight),
	)

	if m.InputHandler.LeftJustPressed {
		log.Printf("[Input] LEFT DOWN  pos=(%.0f,%.0f) startScreen=%v gameOver=%v",
			worldX, worldY, m.UI.ShowStartScreen, m.UI.ShowGameOverModal)
	}
	if m.InputHandler.LeftJustReleased {
		log.Printf("[Input] LEFT UP    pos=(%.0f,%.0f)", worldX, worldY)
	}
}

// tickSimulation advances the game engine by one frame and handles the events
// it returns ("show_upgrades" or "game_over").
func (m *MinimetroGame) tickSimulation(effectiveDelta float64) {
	result := m.GameSystem.Update(m.GameState, effectiveDelta, m.UI.Width, m.UI.Height, m.GameState.SimTimeMs)
	switch result {
	case "show_upgrades":
		log.Printf("[Game] Week %d complete — showing upgrade modal", m.GameState.Week)
		m.UI.UpgradeChoices = m.GameSystem.GenerateUpgradeChoices(m.GameState)
		if m.RLClient != nil {
			m.RLClient.Update(m.GameState, m.GameSystem, m.GameState.SimTimeMs, m.UI.UpgradeChoices)
		} else if m.Solver.Enabled {
			chosen := m.Solver.ChooseUpgrade(m.GameState, m.UI.UpgradeChoices)
			systems.ApplyUpgrade(m.GameState, chosen)
		} else {
			m.UI.ShowUpgradeModal = true
			m.GameState.Paused = true
		}
	case "game_over":
		log.Printf("[Game] GAME OVER — score=%d passengers=%d", m.GameState.Score, m.GameState.PassengersDelivered)
		m.UI.ShowGameOverModal = true
	}
}

// tickAI runs whichever agent is active: the remote RL client or the local
// heuristic solver. The two are mutually exclusive (solver is disabled when
// RLClient is set).
func (m *MinimetroGame) tickAI() {
	if m.RLClient != nil {
		m.RLClient.Update(m.GameState, m.GameSystem, m.GameState.SimTimeMs, nil)
	} else {
		m.Solver.Update(m.GameState, m.GameSystem.GraphManager, m.GameState.SimTimeMs)
	}
}

func (m *MinimetroGame) Draw(screen *ebiten.Image) {
	if m.GameSystem.Initialized && !m.UI.ShowStartScreen {
		m.Renderer.Render(screen, m.GameState)
	}
	m.UI.Draw(screen, m.GameState, m.InputHandler)
}

func (m *MinimetroGame) Layout(outsideWidth, outsideHeight int) (int, int) {
	m.screenW = outsideWidth
	m.screenH = outsideHeight
	return outsideWidth, outsideHeight
}

func main() {
	rlClientURL := flag.String("rl-client", "", "if set, use the RL agent at this inference server URL (e.g. http://localhost:9000)")
	flag.Parse()

	ebiten.SetWindowSize(1200, 800)
	ebiten.SetWindowTitle("Mini Metro - Go Edition")
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)

	ih := systems.NewInputHandler()
	solver := systems.NewSolver()
	ui := systems.NewUI()

	g := &MinimetroGame{
		GameState:    state.NewGameState(),
		GameSystem:   systems.NewGame(),
		Renderer:     rendering.NewGameRenderer(),
		UI:           ui,
		InputHandler: ih,
		Solver:       solver,
	}

	if *rlClientURL != "" {
		log.Printf("[main] RL client mode enabled — inference server: %s", *rlClientURL)
		g.RLClient = rl.NewClient(*rlClientURL)
		g.Solver.Enabled = false // RL agent takes over
	}

	// Wire AI toggle: UI signals the intent; main.go owns the solver state.
	ui.OnAIToggle = func() {
		g.Solver.Enabled = !g.Solver.Enabled
		ui.AIEnabled = g.Solver.Enabled
		log.Printf("[main] AI solver toggled: %v", g.Solver.Enabled)
	}
	ui.AIEnabled = solver.Enabled

	if err := ebiten.RunGame(g); err != nil {
		log.Fatal(err)
	}
}
