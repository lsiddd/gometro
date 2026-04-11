package rl

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"minimetro-go/state"
	"minimetro-go/systems"
)

// initializedEnv returns an RLEnv that has been reset and is safe to use for
// building observations and masks without a running HTTP server.
func initializedEnv(t *testing.T) *RLEnv {
	t.Helper()
	env := NewRLEnv()
	env.Reset("london")
	return env
}

// ── NewClient / buildFakeEnv ─────────────────────────────────────────────────

func TestNewClient_SetsServerURL(t *testing.T) {
	c := NewClient("http://localhost:9000")
	if c.serverURL != "http://localhost:9000" {
		t.Errorf("serverURL: want http://localhost:9000, got %s", c.serverURL)
	}
}

func TestNewClient_DefaultRunInterval(t *testing.T) {
	c := NewClient("http://localhost:9000")
	if c.runInterval != 300.0 {
		t.Errorf("runInterval: want 300, got %f", c.runInterval)
	}
}

func TestBuildFakeEnv_WiresGameState(t *testing.T) {
	c := NewClient("http://localhost:9000")
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
	c := NewClient("http://localhost:9000")
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
	expected := []int{0, 0, 0, 1}
	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/act" || r.Method != http.MethodPost {
			http.Error(w, "unexpected request", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(actResp{Action: expected})
	}))
	defer fake.Close()

	c := NewClient(fake.URL)
	env := initializedEnv(t)
	got := c.queryServer(env)

	if len(got) != len(expected) {
		t.Fatalf("action length: want %d, got %d", len(expected), len(got))
	}
	for i := range expected {
		if got[i] != expected[i] {
			t.Errorf("action[%d]: want %d, got %d", i, expected[i], got[i])
		}
	}
}

func TestQueryServer_SendsObsAndMask(t *testing.T) {
	var receivedObs []float32
	var receivedMask []bool

	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req actReq
		json.NewDecoder(r.Body).Decode(&req)
		receivedObs = req.Obs
		receivedMask = req.Mask
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(actResp{Action: []int{0, 0, 0, 0}})
	}))
	defer fake.Close()

	c := NewClient(fake.URL)
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
	// Port 1 is reserved and will always refuse connections.
	c := NewClient("http://127.0.0.1:1")
	env := initializedEnv(t)
	result := c.queryServer(env)
	if result != nil {
		t.Errorf("unreachable server should return nil, got %v", result)
	}
}

func TestQueryServer_InvalidJSONResponse_ReturnsNil(t *testing.T) {
	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("not valid json{{"))
	}))
	defer fake.Close()

	c := NewClient(fake.URL)
	env := initializedEnv(t)
	result := c.queryServer(env)
	if result != nil {
		t.Errorf("invalid JSON response should return nil, got %v", result)
	}
}

func TestQueryServer_EmptyActionInResponse_ReturnsEmptySlice(t *testing.T) {
	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(actResp{Action: []int{}})
	}))
	defer fake.Close()

	c := NewClient(fake.URL)
	env := initializedEnv(t)
	result := c.queryServer(env)
	// An empty (but valid) action slice is returned as-is; caller handles it.
	if result == nil {
		t.Error("valid JSON with empty action should not return nil")
	}
}
