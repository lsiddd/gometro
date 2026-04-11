package rl

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
	"minimetro-go/state"
	"minimetro-go/systems"
)

// Client drives the live game using a remote Python inference server.
// It mirrors Solver.Update's signature so the main game loop can swap them
// interchangeably: replace s.Solver.Update(...) with s.RLClient.Update(...).
//
// The Python inference server must expose:
//
//	POST /act  body: {obs: [...], mask: [...]}  →  {action: int}
type Client struct {
	serverURL    string
	runInterval  float64
	lastRunMs    float64

	// Upgrade modal state (mirrors RLEnv).
	inUpgradeModal bool
	upgradeChoices []string

	// Synthetic env shell just for observation/mask building (no simulation).
	fakeEnv *RLEnv

	httpClient *http.Client
}

// inferenceTimeout caps how long a single /act request may block the game loop.
// Must be comfortably below runInterval (300 ms) so a slow server degrades
// gracefully rather than freezing the game.
const inferenceTimeout = 250 * time.Millisecond

// NewClient creates an RLClient that calls serverURL for action decisions.
// Typical serverURL: "http://localhost:9000".
func NewClient(serverURL string) *Client {
	return &Client{
		serverURL:   serverURL,
		runInterval: 300.0, // ms, matches Solver default
		httpClient:  &http.Client{Timeout: inferenceTimeout},
	}
}

// Update is called every game frame (60 Hz) by the main loop. It mirrors
// Solver.Update so the two are drop-in replacements.
func (c *Client) Update(gs *state.GameState, game *systems.Game, nowMs float64, upgradeChoices []string) {
	if gs.Paused || gs.GameOver {
		return
	}

	// Handle upgrade modal: if the game just surfaced an upgrade choice, ask
	// the inference server and apply it immediately.
	if len(upgradeChoices) > 0 && !c.inUpgradeModal {
		c.inUpgradeModal = true
		c.upgradeChoices = upgradeChoices
	}

	if c.inUpgradeModal {
		fakeEnv := c.buildFakeEnv(gs, game)
		actionVal := c.queryServer(fakeEnv)
		if actionVal == nil {
			// Server unreachable this frame — keep modal open and retry next frame.
			return
		}
		choiceIdx := 0 // default: first option
		if len(actionVal) == 4 && actionVal[0] == ActChooseUpgrade {
			choiceIdx = actionVal[3]
		}
		if choiceIdx < len(c.upgradeChoices) {
			systems.ApplyUpgrade(gs, c.upgradeChoices[choiceIdx])
		}
		c.inUpgradeModal = false
		c.upgradeChoices = nil
		return
	}

	if nowMs-c.lastRunMs < c.runInterval {
		return
	}
	c.lastRunMs = nowMs

	fakeEnv := c.buildFakeEnv(gs, game)
	actionVal := c.queryServer(fakeEnv)
	ApplyRLAction(fakeEnv, actionVal)
}

// buildFakeEnv constructs a minimal RLEnv shell around the live game state so
// that BuildObservation and BuildActionMask can run without owning the state.
func (c *Client) buildFakeEnv(gs *state.GameState, game *systems.Game) *RLEnv {
	env := &RLEnv{
		gs:             gs,
		game:           game,
		inUpgradeModal: c.inUpgradeModal,
		upgradeChoices: c.upgradeChoices,
	}
	return env
}

type actReq struct {
	Obs  []float32 `json:"obs"`
	Mask []bool    `json:"mask"`
}

type actResp struct {
	Action []int `json:"action"`
}

func (c *Client) queryServer(env *RLEnv) []int {
	obs := BuildObservation(env)
	mask := BuildActionMaskMulti(env)

	body, err := json.Marshal(actReq{Obs: obs, Mask: mask})
	if err != nil {
		log.Printf("[RLClient] marshal error: %v", err)
		return nil
	}

	resp, err := c.httpClient.Post(
		fmt.Sprintf("%s/act", c.serverURL),
		"application/json",
		bytes.NewReader(body),
	)
	if err != nil {
		log.Printf("[RLClient] inference server unreachable: %v", err)
		return nil
	}
	defer resp.Body.Close()

	var ar actResp
	if err := json.NewDecoder(resp.Body).Decode(&ar); err != nil {
		log.Printf("[RLClient] decode error: %v", err)
		return nil
	}
	return ar.Action
}
