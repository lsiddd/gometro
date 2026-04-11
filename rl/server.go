package rl

import (
	"bytes"
	"encoding/json"
	"log"
	"net/http"
)

// Server wraps a single RLEnv and exposes it via a JSON HTTP API that mirrors
// the gymnasium interface.
//
// Endpoints:
//
//	GET  /info    → {obs_dim, num_actions}
//	POST /reset   → {city: string}          → {obs, mask}
//	POST /step    → {action: int}            → {obs, reward, done, mask, info}
type Server struct {
	env *RLEnv
	mux *http.ServeMux
}

func NewServer() *Server {
	s := &Server{env: NewRLEnv()}
	s.mux = http.NewServeMux()
	s.mux.HandleFunc("/info", s.handleInfo)
	s.mux.HandleFunc("/reset", s.handleReset)
	s.mux.HandleFunc("/step", s.handleStep)
	s.mux.HandleFunc("/solver_act", s.handleSolverAct)
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

// ── /info ─────────────────────────────────────────────────────────────────────

type infoResp struct {
	ObsDim      int   `json:"obs_dim"`
	ActionDims  []int `json:"action_dims"`
	GlobalDim   int   `json:"global_dim"`
	StationDim  int   `json:"station_dim"`
	NumStations int   `json:"num_stations"`
	LineDim     int   `json:"line_dim"`
	NumLines    int   `json:"num_lines"`
}

func (s *Server) handleInfo(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, infoResp{
		ObsDim:      ObsDim,
		ActionDims:  []int{NumActionCats, MaxLineSlots, MaxStationSlots, NumOptions},
		GlobalDim:   GlobalDim,
		StationDim:  StationDim,
		NumStations: MaxStationSlots,
		LineDim:     LineDim,
		NumLines:    MaxLineSlots,
	})
}

// ── /reset ────────────────────────────────────────────────────────────────────

type resetReq struct {
	City string `json:"city"`
}

type resetResp struct {
	Obs  []float32 `json:"obs"`
	Mask []bool    `json:"mask"`
}

func (s *Server) handleReset(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	req, ok := decodeBody[resetReq](w, r)
	if !ok {
		return
	}
	city := req.City
	if city == "" {
		city = "london"
	}
	obs, mask := s.env.Reset(city)
	writeJSON(w, resetResp{Obs: obs, Mask: mask})
}

// ── /step ─────────────────────────────────────────────────────────────────────

type stepReq struct {
	Action []int `json:"action"`
}

type stepResp struct {
	Obs    []float32      `json:"obs"`
	Reward float64        `json:"reward"`
	Done   bool           `json:"done"`
	Mask   []bool         `json:"mask"`
	Info   map[string]any `json:"info"`
}

func (s *Server) handleStep(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	req, ok := decodeBody[stepReq](w, r)
	if !ok {
		return
	}
	obs, reward, done, mask := s.env.Step(req.Action)
	writeJSON(w, stepResp{
		Obs:    obs,
		Reward: reward,
		Done:   done,
		Mask:   mask,
		Info:   s.env.Info(),
	})
	if done {
		log.Printf("[RL] Episode ended — score=%d passengers=%d",
			s.env.gs.Score, s.env.gs.PassengersDelivered)
	}
}

// ── /solver_act ───────────────────────────────────────────────────────────────

type solverActResp struct {
	Action []int `json:"action"`
}

// handleSolverAct returns the action the heuristic solver would take in the
// current state WITHOUT advancing the simulation. Used by pretrain.py to collect
// behavioral-cloning demonstrations.
func (s *Server) handleSolverAct(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet, http.MethodPost) {
		return
	}
	action := s.env.InferSolverAction()
	writeJSON(w, solverActResp{Action: action})
}

// ── helpers ───────────────────────────────────────────────────────────────────

// requireMethod rejects the request with 405 if the HTTP method does not match
// any of the allowed values. Returns false when the request was rejected so the
// caller can return immediately.
func requireMethod(w http.ResponseWriter, r *http.Request, methods ...string) bool {
	for _, m := range methods {
		if r.Method == m {
			return true
		}
	}
	http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	return false
}

// decodeBody decodes a JSON request body into T. On failure it writes a 400
// response and returns false so the caller can return immediately.
func decodeBody[T any](w http.ResponseWriter, r *http.Request) (T, bool) {
	var v T
	if err := json.NewDecoder(r.Body).Decode(&v); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return v, false
	}
	return v, true
}

// writeJSON encodes v into a temporary buffer before writing so that a
// serialisation error results in a 500 response rather than a partial body
// with a 200 status (which would silently corrupt the Python client).
func writeJSON(w http.ResponseWriter, v any) {
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(v); err != nil {
		log.Printf("[RL] JSON encode error: %v", err)
		http.Error(w, "internal encoding error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(buf.Bytes())
}
