package rl

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// ── helpers ───────────────────────────────────────────────────────────────────

func newTestServer() *Server {
	return NewServer()
}

func postJSON(t *testing.T, s *Server, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.ServeHTTP(w, req)
	return w
}

// resetServer calls POST /reset on s and returns the parsed response.
func resetServer(t *testing.T, s *Server, city string) resetResp {
	t.Helper()
	w := postJSON(t, s, "/reset", map[string]string{"city": city})
	if w.Code != http.StatusOK {
		t.Fatalf("/reset returned %d: %s", w.Code, w.Body.String())
	}
	var resp resetResp
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("/reset decode: %v", err)
	}
	return resp
}

// ── /info ─────────────────────────────────────────────────────────────────────

func TestHandleInfo_ReturnsCorrectDimensions(t *testing.T) {
	s := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/info", nil)
	w := httptest.NewRecorder()
	s.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var resp infoResp
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.ObsDim != ObsDim {
		t.Errorf("obs_dim: want %d, got %d", ObsDim, resp.ObsDim)
	}
	if resp.NumStations != MaxStationSlots {
		t.Errorf("num_stations: want %d, got %d", MaxStationSlots, resp.NumStations)
	}
	if resp.NumLines != MaxLineSlots {
		t.Errorf("num_lines: want %d, got %d", MaxLineSlots, resp.NumLines)
	}
	if len(resp.ActionDims) != 4 {
		t.Errorf("action_dims length: want 4, got %d", len(resp.ActionDims))
	}
}

func TestHandleInfo_ContentTypeJSON(t *testing.T) {
	s := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/info", nil)
	w := httptest.NewRecorder()
	s.ServeHTTP(w, req)

	ct := w.Header().Get("Content-Type")
	if !strings.HasPrefix(ct, "application/json") {
		t.Errorf("Content-Type: want application/json, got %q", ct)
	}
}

// ── /reset ────────────────────────────────────────────────────────────────────

func TestHandleReset_ReturnsObsAndMask(t *testing.T) {
	s := newTestServer()
	resp := resetServer(t, s, "london")

	if len(resp.Obs) != ObsDim {
		t.Errorf("obs length: want %d, got %d", ObsDim, len(resp.Obs))
	}
	if len(resp.Mask) == 0 {
		t.Error("mask must not be empty")
	}
}

func TestHandleReset_EmptyCity_DefaultsToLondon(t *testing.T) {
	s := newTestServer()
	// Sending empty city should fall back to "london" without error.
	w := postJSON(t, s, "/reset", map[string]string{"city": ""})
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 with empty city, got %d: %s", w.Code, w.Body.String())
	}
	var resp resetResp
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp.Obs) != ObsDim {
		t.Errorf("obs length after default-city reset: want %d, got %d", ObsDim, len(resp.Obs))
	}
}

func TestHandleReset_WrongMethod_Returns405(t *testing.T) {
	s := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/reset", nil)
	w := httptest.NewRecorder()
	s.ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405 on GET /reset, got %d", w.Code)
	}
}

func TestHandleReset_InvalidJSON_Returns400(t *testing.T) {
	s := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/reset", strings.NewReader("{bad json"))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400 on bad JSON, got %d", w.Code)
	}
}

// ── /step ─────────────────────────────────────────────────────────────────────

func TestHandleStep_ReturnsObsRewardDoneMask(t *testing.T) {
	s := newTestServer()
	resetServer(t, s, "london")

	// ActNoOp = 0; safe no-op action that advances the simulation.
	w := postJSON(t, s, "/step", map[string]any{"action": []int{0, 0, 0, 0}})
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 from /step, got %d: %s", w.Code, w.Body.String())
	}

	var resp stepResp
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp.Obs) != ObsDim {
		t.Errorf("obs length: want %d, got %d", ObsDim, len(resp.Obs))
	}
	if len(resp.Mask) == 0 {
		t.Error("mask must not be empty after step")
	}
	// reward can be any float; done can be true or false — just check it decodes.
}

func TestHandleStep_WrongMethod_Returns405(t *testing.T) {
	s := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/step", nil)
	w := httptest.NewRecorder()
	s.ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405 on GET /step, got %d", w.Code)
	}
}

func TestHandleStep_InvalidJSON_Returns400(t *testing.T) {
	s := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/step", strings.NewReader("not-json"))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400 on bad JSON, got %d", w.Code)
	}
}

// ── /solver_act ───────────────────────────────────────────────────────────────

func TestHandleSolverAct_GET_ReturnsAction(t *testing.T) {
	s := newTestServer()
	resetServer(t, s, "london")

	req := httptest.NewRequest(http.MethodGet, "/solver_act", nil)
	w := httptest.NewRecorder()
	s.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 from GET /solver_act, got %d: %s", w.Code, w.Body.String())
	}
	var resp solverActResp
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp.Action) != 4 {
		t.Errorf("solver_act action length: want 4, got %d", len(resp.Action))
	}
}

func TestHandleSolverAct_POST_ReturnsAction(t *testing.T) {
	s := newTestServer()
	resetServer(t, s, "london")

	w := postJSON(t, s, "/solver_act", map[string]any{})
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 from POST /solver_act, got %d", w.Code)
	}
}

func TestHandleSolverAct_WrongMethod_Returns405(t *testing.T) {
	s := newTestServer()
	req := httptest.NewRequest(http.MethodDelete, "/solver_act", nil)
	w := httptest.NewRecorder()
	s.ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405 on DELETE /solver_act, got %d", w.Code)
	}
}

// ── writeJSON / requireMethod / decodeBody helpers ────────────────────────────

func TestWriteJSON_EncodesValue(t *testing.T) {
	w := httptest.NewRecorder()
	writeJSON(w, map[string]int{"x": 42})

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
	var out map[string]int
	if err := json.NewDecoder(w.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out["x"] != 42 {
		t.Errorf("x: want 42, got %d", out["x"])
	}
}

func TestRequireMethod_AllowsMatchingMethod(t *testing.T) {
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/", nil)
	if !requireMethod(w, req, http.MethodPost) {
		t.Error("requireMethod should return true for a matching method")
	}
	if w.Code != http.StatusOK {
		t.Errorf("no error should be written for matching method; code=%d", w.Code)
	}
}

func TestRequireMethod_Rejects405(t *testing.T) {
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	if requireMethod(w, req, http.MethodPost) {
		t.Error("requireMethod should return false for non-matching method")
	}
	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", w.Code)
	}
}

func TestDecodeBody_Success(t *testing.T) {
	type payload struct {
		City string `json:"city"`
	}
	body, _ := json.Marshal(payload{City: "paris"})
	req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(body))
	w := httptest.NewRecorder()

	v, ok := decodeBody[payload](w, req)
	if !ok {
		t.Fatal("decodeBody should succeed on valid JSON")
	}
	if v.City != "paris" {
		t.Errorf("city: want paris, got %s", v.City)
	}
}

func TestDecodeBody_InvalidJSON_Returns400(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader("{bad"))
	w := httptest.NewRecorder()

	type payload struct{ City string }
	_, ok := decodeBody[payload](w, req)
	if ok {
		t.Error("decodeBody should return false on invalid JSON")
	}
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}
