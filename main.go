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

	// Ebitengine Update() runs at a fixed 60Hz by default. Using actual TPS() can fluctuate or be 0 causing logic bugs.
	deltaMs := 1000.0 / 60.0

	ffFactors := [3]float64{1, 2, 4}
	effectiveDelta := deltaMs * ffFactors[m.GameState.FastForward]
	m.GameState.SimTimeMs += effectiveDelta

	// Input capturing
	x, y := ebiten.CursorPosition()
	leftDown := ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft)
	rightDown := ebiten.IsMouseButtonPressed(ebiten.MouseButtonRight)
	shiftDown := ebiten.IsKeyPressed(ebiten.KeyShift) || ebiten.IsKeyPressed(ebiten.KeyShiftLeft)
	leftJustPressed := inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft)
	leftJustReleased := inpututil.IsMouseButtonJustReleased(ebiten.MouseButtonLeft)
	rightJustPressed := inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonRight)

	// Since we are applying camera zoom and translation, let's convert screen space to world space:
	// Our renderer centers the map if zoom < 0.995, but since we didn't implement scale yet in renderer
	// we will just pass standard coordinates.
	worldX, worldY := float64(x), float64(y)

	m.InputHandler.Update(m.GameState, worldX, worldY, leftDown, rightDown, shiftDown, leftJustPressed, leftJustReleased, rightJustPressed)

	if m.InputHandler.LeftJustPressed {
		log.Printf("[Input] LEFT DOWN  pos=(%.0f,%.0f) startScreen=%v gameOver=%v", worldX, worldY, m.UI.ShowStartScreen, m.UI.ShowGameOverModal)
	}
	if m.InputHandler.LeftJustReleased {
		log.Printf("[Input] LEFT UP    pos=(%.0f,%.0f)", worldX, worldY)
	}

	m.UI.Update(m.GameState, m.InputHandler, m.GameSystem)

	if !m.UI.ShowStartScreen && !m.UI.ShowGameOverModal && !m.UI.ShowUpgradeModal {
		result := m.GameSystem.Update(m.GameState, effectiveDelta, m.UI.Width, m.UI.Height, m.GameState.SimTimeMs)
		if result == "show_upgrades" {
			log.Printf("[Game] Week %d complete — showing upgrade modal", m.GameState.Week)
			m.UI.UpgradeChoices = m.GameSystem.GenerateUpgradeChoices(m.GameState)
			if m.RLClient != nil {
				// RL agent chooses upgrade; no modal shown.
				m.RLClient.Update(m.GameState, m.GameSystem, m.GameState.SimTimeMs, m.UI.UpgradeChoices)
			} else if m.Solver.Enabled {
				chosen := m.Solver.ChooseUpgrade(m.GameState, m.UI.UpgradeChoices)
				systems.ApplyUpgrade(m.GameState, chosen)
			} else {
				m.UI.ShowUpgradeModal = true
				m.GameState.Paused = true
			}
		} else if result == "game_over" {
			log.Printf("[Game] GAME OVER — score=%d passengers=%d", m.GameState.Score, m.GameState.PassengersDelivered)
			m.UI.ShowGameOverModal = true
		}

		if m.RLClient != nil {
			m.RLClient.Update(m.GameState, m.GameSystem, m.GameState.SimTimeMs, nil)
		} else {
			m.Solver.Update(m.GameState, m.GameSystem.GraphManager, m.GameState.SimTimeMs)
		}
	}

	return nil
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
	ui.Solver = solver

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

	if err := ebiten.RunGame(g); err != nil {
		log.Fatal(err)
	}
}
