package rl

import (
	"context"
	"log"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "minimetro-go/rl/proto"
	"minimetro-go/state"
	"minimetro-go/systems"
)

// Client drives the live game using a remote Python inference server over gRPC.
// It mirrors Solver.Update's signature so the main game loop can swap them
// interchangeably.
//
// The Python inference server must implement the Inference gRPC service
// (see rl/proto/minimetro.proto).
type Client struct {
	conn        *grpc.ClientConn
	stub        pb.InferenceClient
	runInterval float64
	lastRunMs   float64

	// Upgrade modal state (mirrors RLEnv).
	inUpgradeModal bool
	upgradeChoices []string

	// Synthetic env shell just for observation/mask building (no simulation).
	fakeEnv *RLEnv
}

// inferenceTimeout caps how long a single Act RPC may block the game loop.
const inferenceTimeout = 250 * time.Millisecond

// NewClient dials addr and returns a Client backed by the Inference gRPC service.
// addr format: "host:port" (e.g. "localhost:9000").
func NewClient(addr string) *Client {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("[RLClient] dial %s: %v", addr, err)
	}
	return &Client{
		conn:        conn,
		stub:        pb.NewInferenceClient(conn),
		runInterval: 300.0, // ms, matches Solver default
	}
}

// Update is called every game frame (60 Hz) by the main loop.
func (c *Client) Update(gs *state.GameState, game *systems.Game, nowMs float64, upgradeChoices []string) {
	if gs.Paused || gs.GameOver {
		return
	}

	if len(upgradeChoices) > 0 && !c.inUpgradeModal {
		c.inUpgradeModal = true
		c.upgradeChoices = upgradeChoices
	}

	if c.inUpgradeModal {
		fakeEnv := c.buildFakeEnv(gs, game)
		actionVal := c.queryServer(fakeEnv)
		if actionVal == nil {
			return
		}
		choiceIdx := 0
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

func (c *Client) buildFakeEnv(gs *state.GameState, game *systems.Game) *RLEnv {
	return &RLEnv{
		gs:             gs,
		game:           game,
		inUpgradeModal: c.inUpgradeModal,
		upgradeChoices: c.upgradeChoices,
	}
}

func (c *Client) queryServer(env *RLEnv) []int {
	obs := BuildObservation(env)
	mask := BuildActionMaskMulti(env)

	ctx, cancel := context.WithTimeout(context.Background(), inferenceTimeout)
	defer cancel()

	resp, err := c.stub.Act(ctx, &pb.ActRequest{Obs: obs, Mask: mask})
	if err != nil {
		log.Printf("[RLClient] inference server unreachable: %v", err)
		return nil
	}

	out := make([]int, len(resp.Action))
	for i, v := range resp.Action {
		out[i] = int(v)
	}
	return out
}
