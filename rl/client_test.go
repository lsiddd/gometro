package rl

import (
	"context"
	"net"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "minimetro-go/rl/proto"
	"minimetro-go/state"
	"minimetro-go/systems"
)

// ── test inference server ─────────────────────────────────────────────────────

type stubInferenceServicer struct {
	pb.UnimplementedInferenceServer
	respond func(obs []float32, mask []bool) []int32
}

func (s *stubInferenceServicer) Act(_ context.Context, req *pb.ActRequest) (*pb.ActionResponse, error) {
	return &pb.ActionResponse{Action: s.respond(req.Obs, req.Mask)}, nil
}

// newTestInferenceServer starts an in-process gRPC Inference server and returns
// its address. The server is stopped automatically when t finishes.
func newTestInferenceServer(t *testing.T, respond func(obs []float32, mask []bool) []int32) string {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	srv := grpc.NewServer()
	pb.RegisterInferenceServer(srv, &stubInferenceServicer{respond: respond})
	go srv.Serve(lis) //nolint:errcheck
	t.Cleanup(srv.Stop)
	return lis.Addr().String()
}

func initializedEnv(t *testing.T) *RLEnv {
	t.Helper()
	env := NewRLEnv()
	env.Reset("london", 1.0)
	return env
}

// ── NewClient ─────────────────────────────────────────────────────────────────

func TestNewClient_DefaultRunInterval(t *testing.T) {
	addr := newTestInferenceServer(t, func(_ []float32, _ []bool) []int32 { return nil })
	c := NewClient(addr)
	defer c.conn.Close()
	if c.runInterval != 300.0 {
		t.Errorf("runInterval: want 300, got %f", c.runInterval)
	}
}

// ── buildFakeEnv ──────────────────────────────────────────────────────────────

func TestBuildFakeEnv_WiresGameState(t *testing.T) {
	addr := newTestInferenceServer(t, func(_ []float32, _ []bool) []int32 { return nil })
	c := NewClient(addr)
	defer c.conn.Close()

	gs := state.NewGameState()
	game := systems.NewGame()
	env := c.buildFakeEnv(gs, game)

	if env.gs != gs {
		t.Error("buildFakeEnv must wire gs from the caller")
	}
	if env.game != game {
		t.Error("buildFakeEnv must wire game from the caller")
	}
}

func TestBuildFakeEnv_ForwardsUpgradeModalState(t *testing.T) {
	addr := newTestInferenceServer(t, func(_ []float32, _ []bool) []int32 { return nil })
	c := NewClient(addr)
	defer c.conn.Close()
	c.inUpgradeModal = true
	c.upgradeChoices = []string{"carriage", "bridge"}

	gs := state.NewGameState()
	game := systems.NewGame()
	env := c.buildFakeEnv(gs, game)

	if !env.inUpgradeModal {
		t.Error("buildFakeEnv must forward inUpgradeModal = true")
	}
	if len(env.upgradeChoices) != 2 {
		t.Errorf("upgradeChoices: want 2, got %d", len(env.upgradeChoices))
	}
}

// ── queryServer ───────────────────────────────────────────────────────────────

func TestQueryServer_Success_ReturnsAction(t *testing.T) {
	expected := []int32{0, 0, 0, 1}
	addr := newTestInferenceServer(t, func(_ []float32, _ []bool) []int32 { return expected })
	c := NewClient(addr)
	defer c.conn.Close()

	env := initializedEnv(t)
	got := c.queryServer(env)

	if len(got) != len(expected) {
		t.Fatalf("action length: want %d, got %d", len(expected), len(got))
	}
	for i := range expected {
		if got[i] != int(expected[i]) {
			t.Errorf("action[%d]: want %d, got %d", i, expected[i], got[i])
		}
	}
}

func TestQueryServer_SendsObsAndMask(t *testing.T) {
	var receivedObs []float32
	var receivedMask []bool

	addr := newTestInferenceServer(t, func(obs []float32, mask []bool) []int32 {
		receivedObs = obs
		receivedMask = mask
		return []int32{0, 0, 0, 0}
	})
	c := NewClient(addr)
	defer c.conn.Close()

	env := initializedEnv(t)
	c.queryServer(env)

	if len(receivedObs) != ObsDim {
		t.Errorf("obs sent to server: want %d elements, got %d", ObsDim, len(receivedObs))
	}
	if len(receivedMask) == 0 {
		t.Error("mask sent to server must not be empty")
	}
}

func TestQueryServer_ServerUnreachable_ReturnsNil(t *testing.T) {
	// Use a real address with nothing listening so the RPC fails within timeout.
	conn, err := grpc.NewClient("127.0.0.1:1",
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	c := &Client{
		conn:        conn,
		stub:        pb.NewInferenceClient(conn),
		runInterval: 300.0,
	}
	defer c.conn.Close()

	env := initializedEnv(t)
	result := c.queryServer(env)
	if result != nil {
		t.Errorf("unreachable server should return nil, got %v", result)
	}
}

func TestQueryServer_EmptyActionInResponse_ReturnsEmptySlice(t *testing.T) {
	addr := newTestInferenceServer(t, func(_ []float32, _ []bool) []int32 { return []int32{} })
	c := NewClient(addr)
	defer c.conn.Close()

	env := initializedEnv(t)
	result := c.queryServer(env)
	if result == nil {
		t.Error("valid response with empty action should not return nil")
	}
}
