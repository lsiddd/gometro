package systems

import (
	"bytes"
	"fmt"
	"image/color"
	"log"
	"math"
	"minimetro-go/state"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
	textv2 "github.com/hajimehoshi/ebiten/v2/text/v2"
	"golang.org/x/image/font/gofont/goregular"
)

// ── Font ─────────────────────────────────────────────────────────────────────

var uiFontSrc *textv2.GoTextFaceSource

func init() {
	src, err := textv2.NewGoTextFaceSource(bytes.NewReader(goregular.TTF))
	if err != nil {
		panic("ui: failed to parse Go Regular font: " + err.Error())
	}
	uiFontSrc = src
}

func uiFont(size float64) *textv2.GoTextFace {
	return &textv2.GoTextFace{Source: uiFontSrc, Size: size}
}

func drawText(screen *ebiten.Image, str string, size float64, x, y float64, clr color.RGBA) {
	op := &textv2.DrawOptions{}
	op.GeoM.Translate(x, y)
	op.ColorScale.ScaleWithColor(clr)
	textv2.Draw(screen, str, uiFont(size), op)
}

// ── Layout constants ──────────────────────────────────────────────────────────

const (
	hudH     = 62.0
	ink      = 0xFF // alpha for text
	darkGray = 0x3C // 60
)

var (
	clrInk  = color.RGBA{darkGray, darkGray, darkGray, ink}
	clrHud  = color.RGBA{244, 241, 233, 230}
	clrBg   = color.RGBA{244, 241, 233, 255}
)

// ── City buttons ──────────────────────────────────────────────────────────────

var cities = []struct {
	Key  string
	Name string
}{
	{"london", "London"},
	{"paris", "Paris"},
	{"newyork", "New York"},
	{"tokyo", "Tokyo"},
}

type cityButtonRect struct {
	Key        string
	X, Y, W, H float64
}

// ── UI struct ─────────────────────────────────────────────────────────────────

type UI struct {
	ShowUpgradeModal  bool
	UpgradeChoices    []string
	ShowGameOverModal bool
	ShowStartScreen   bool

	Width  float64
	Height float64

	Solver *Solver

	cityButtons                         []cityButtonRect
	startBtnX, startBtnY, startBtnW, startBtnH float64
}

func NewUI() *UI {
	return &UI{
		ShowStartScreen: true,
	}
}

func (ui *UI) Draw(screen *ebiten.Image, gs *state.GameState, ih *InputHandler) {
	if ui.ShowStartScreen {
		ui.drawStartScreen(screen, gs)
	} else if ui.ShowGameOverModal {
		ui.drawGameOver(screen, gs)
	} else {
		ui.drawGameUI(screen, gs, ih)
	}
}

func (ui *UI) Update(gs *state.GameState, ih *InputHandler, game *Game) {
	if ui.ShowStartScreen {
		if ih.LeftJustPressed {
			log.Printf("[UI] Click at (%.0f, %.0f) on start screen", ih.MousePos.X, ih.MousePos.Y)

			for _, btn := range ui.cityButtons {
				if pointInRect(ih.MousePos, btn.X, btn.Y, btn.W, btn.H) {
					log.Printf("[UI] City selected: %s", btn.Key)
					gs.SelectedCity = btn.Key
					return
				}
			}

			if pointInRect(ih.MousePos, ui.startBtnX, ui.startBtnY, ui.startBtnW, ui.startBtnH) {
				log.Printf("[UI] START clicked — initialising game (city=%s, screen=%.0fx%.0f)", gs.SelectedCity, ui.Width, ui.Height)
				ui.ShowStartScreen = false
				game.InitGame(gs, ui.Width, ui.Height)
				log.Printf("[UI] Game initialised: %d stations, %d lines", len(gs.Stations), len(gs.Lines))
			}
		}
	} else if ui.ShowGameOverModal {
		if ih.LeftJustPressed {
			log.Printf("[UI] Click at (%.0f, %.0f) on game-over modal", ih.MousePos.X, ih.MousePos.Y)
			btnX := ui.Width/2 - 75
			btnY := ui.Height/2 + 60
			if pointInRect(ih.MousePos, btnX, btnY, 150, 40) {
				log.Printf("[UI] Play Again clicked")
				ui.ShowGameOverModal = false
				ui.ShowStartScreen = true
			}
		}
	} else if ui.ShowUpgradeModal {
		gs.Paused = false
		ui.ShowUpgradeModal = false
	} else {
		if ih.LeftJustPressed {
			if pointInRect(ih.MousePos, ui.Width-140, 8, 60, 30) {
				gs.FastForward = (gs.FastForward + 1) % 3
				return
			}
			if ui.Solver != nil && pointInRect(ih.MousePos, ui.Width-70, 8, 60, 30) {
				ui.Solver.Enabled = !ui.Solver.Enabled
				log.Printf("[UI] AI solver toggled: %v", ui.Solver.Enabled)
				return
			}
			for i := 0; i < gs.AvailableLines; i++ {
				lx := ui.Width/2 - float64(gs.AvailableLines*30) + float64(i*60) + 30
				ly := ui.Height - 50
				if math.Hypot(ih.MousePos.X-lx, ih.MousePos.Y-ly) < 28 {
					log.Printf("[UI] Line %d selected", i)
					gs.SelectedLine = i
				}
			}
		}
	}
}

func pointInRect(p Point, x, y, w, h float64) bool {
	return p.X >= x && p.X <= x+w && p.Y >= y && p.Y <= y+h
}

// ── Start screen ──────────────────────────────────────────────────────────────

func (ui *UI) drawStartScreen(screen *ebiten.Image, gs *state.GameState) {
	screen.Fill(clrBg)

	cx := ui.Width / 2
	cy := ui.Height / 2

	drawText(screen, "MINI METRO", 32, cx-68, cy-160, clrInk)
	drawText(screen, "Design the subway system for a growing city.", 13, cx-148, cy-118, color.RGBA{100, 100, 100, 255})
	drawText(screen, "Select city", 13, cx-38, cy-84, clrInk)

	ui.cityButtons = ui.cityButtons[:0]
	btnW, btnH := 120.0, 40.0
	for i, c := range cities {
		col := float64(i % 2)
		row := float64(i / 2)
		bx := cx - 130 + col*140
		by := cy - 64 + row*52

		ui.cityButtons = append(ui.cityButtons, cityButtonRect{Key: c.Key, X: bx, Y: by, W: btnW, H: btnH})

		selected := gs.SelectedCity == c.Key
		if selected {
			vector.FillRect(screen, float32(bx), float32(by), float32(btnW), float32(btnH), color.RGBA{60, 60, 60, 255}, true)
			drawText(screen, c.Name, 14, bx+14, by+10, color.RGBA{244, 241, 233, 255})
		} else {
			vector.FillRect(screen, float32(bx), float32(by), float32(btnW), float32(btnH), color.RGBA{220, 218, 212, 255}, true)
			drawText(screen, c.Name, 14, bx+14, by+10, clrInk)
		}
	}

	ui.startBtnX = cx - 70
	ui.startBtnY = cy + 100
	ui.startBtnW = 140
	ui.startBtnH = 46
	vector.FillRect(screen, float32(ui.startBtnX), float32(ui.startBtnY), float32(ui.startBtnW), float32(ui.startBtnH), color.RGBA{80, 160, 90, 255}, true)
	drawText(screen, "START", 16, ui.startBtnX+42, ui.startBtnY+12, color.RGBA{244, 241, 233, 255})
}

// ── In-game HUD ───────────────────────────────────────────────────────────────

func (ui *UI) drawGameUI(screen *ebiten.Image, gs *state.GameState, ih *InputHandler) {
	// HUD background — semi-transparent strip
	vector.FillRect(screen, 0, 0, float32(ui.Width), hudH, clrHud, true)

	// Score (prominent, centred)
	scoreStr := fmt.Sprintf("%d", gs.Score)
	drawText(screen, scoreStr, 26, ui.Width/2-float64(len(scoreStr))*8, 10, clrInk)
	drawText(screen, "SCORE", 10, ui.Width/2-18, 38, color.RGBA{140, 138, 132, 255})

	// Left block — week / day
	drawText(screen, fmt.Sprintf("Week %d  ·  Day %d", gs.Week, gs.Day), 13, 16, 12, clrInk)
	drawText(screen, fmt.Sprintf("%d passengers", gs.PassengersDelivered), 12, 16, 32, color.RGBA{120, 118, 112, 255})

	// Right block — resources
	resStr := fmt.Sprintf("Trains %d  ·  Carriages %d  ·  Bridges %d", gs.AvailableTrains, gs.Carriages, gs.Bridges)
	drawText(screen, resStr, 12, ui.Width-360, 12, color.RGBA{120, 118, 112, 255})
	drawText(screen, fmt.Sprintf("%s  ·  %d stations", gs.SelectedCity, len(gs.Stations)), 12, ui.Width-360, 32, color.RGBA{120, 118, 112, 255})

	// Fast-forward button — pill with speed indicator dots
	{
		ffLabels := [3]string{"▶", "▶▶", "▶▶▶"}
		ffBg := [3]color.RGBA{
			{210, 208, 202, 255},
			{140, 180, 220, 255},
			{80, 140, 210, 255},
		}
		btnX, btnY, btnW, btnH := float32(ui.Width-140), float32(10), float32(54), float32(28)
		vector.FillRect(screen, btnX, btnY, btnW, btnH, ffBg[gs.FastForward], true)
		drawText(screen, ffLabels[gs.FastForward], 13, float64(btnX)+10, float64(btnY)+6, color.RGBA{244, 241, 233, 255})
	}

	// AI toggle button
	if ui.Solver != nil {
		btnX, btnY, btnW, btnH := float32(ui.Width-76), float32(10), float32(54), float32(28)
		var bg color.RGBA
		var label string
		if ui.Solver.Enabled {
			bg = color.RGBA{80, 160, 90, 255}
			label = "AI ON"
		} else {
			bg = color.RGBA{180, 178, 172, 255}
			label = "AI OFF"
		}
		vector.FillRect(screen, btnX, btnY, btnW, btnH, bg, true)
		drawText(screen, label, 12, float64(btnX)+8, float64(btnY)+7, color.RGBA{244, 241, 233, 255})
	}

	// Line selector circles (bottom-centre)
	for i := 0; i < gs.AvailableLines; i++ {
		lx := float32(ui.Width/2 - float64(gs.AvailableLines*30) + float64(i*60) + 30)
		ly := float32(ui.Height - 50)

		c := gs.Lines[i].Color
		clr := color.RGBA{uint8(c[0]), uint8(c[1]), uint8(c[2]), 255}

		if i == gs.SelectedLine {
			// White halo to indicate selection
			vector.FillCircle(screen, lx, ly, 28, color.RGBA{244, 241, 233, 255}, true)
			vector.FillCircle(screen, lx, ly, 23, clr, true)
		} else {
			vector.FillCircle(screen, lx, ly, 18, clr, true)
		}
	}

	// Line preview while dragging
	if ih.PreviewLine != nil {
		p := ih.PreviewLine
		clr := color.RGBA{uint8(p.Color[0]), uint8(p.Color[1]), uint8(p.Color[2]), 150}
		if p.IsSegment {
			vector.StrokeLine(screen, float32(p.S1.X), float32(p.S1.Y), float32(p.Target.X), float32(p.Target.Y), 6, clr, true)
			vector.StrokeLine(screen, float32(p.Target.X), float32(p.Target.Y), float32(p.S2.X), float32(p.S2.Y), 6, clr, true)
		} else {
			vector.StrokeLine(screen, float32(p.Start.X), float32(p.Start.Y), float32(p.End.X), float32(p.End.Y), 6, clr, true)
		}
	}
}

// ── Game over ─────────────────────────────────────────────────────────────────

func (ui *UI) drawGameOver(screen *ebiten.Image, gs *state.GameState) {
	vector.FillRect(screen, 0, 0, float32(ui.Width), float32(ui.Height), color.RGBA{0, 0, 0, 160}, true)

	modX, modY := float32(ui.Width/2-220), float32(ui.Height/2-140)
	vector.FillRect(screen, modX, modY, 440, 280, clrBg, true)

	cx := float64(modX) + 220
	drawText(screen, "Metro Closed", 28, cx-82, float64(modY)+36, clrInk)
	drawText(screen, fmt.Sprintf("%d passengers delivered", gs.PassengersDelivered), 15, cx-110, float64(modY)+100, clrInk)
	drawText(screen, fmt.Sprintf("Score: %d", gs.Score), 20, cx-52, float64(modY)+132, clrInk)

	btnX, btnY := float64(modX)+145, float64(modY)+196
	vector.FillRect(screen, float32(btnX), float32(btnY), 150, 42, color.RGBA{60, 60, 60, 255}, true)
	drawText(screen, "Play Again", 15, btnX+28, btnY+10, color.RGBA{244, 241, 233, 255})
}
