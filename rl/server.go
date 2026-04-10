package rl

import (
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
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

// ── /info ─────────────────────────────────────────────────────────────────────

type infoResp struct {
	ObsDim     int `json:"obs_dim"`
	NumActions int `json:"num_actions"`
}

func (s *Server) handleInfo(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, infoResp{ObsDim: ObsDim, NumActions: NumActions})
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
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req resetReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
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
	Action int `json:"action"`
}

type stepResp struct {
	Obs    []float32      `json:"obs"`
	Reward float64        `json:"reward"`
	Done   bool           `json:"done"`
	Mask   []bool         `json:"mask"`
	Info   map[string]any `json:"info"`
}

func (s *Server) handleStep(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req stepReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
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

// ── helpers ───────────────────────────────────────────────────────────────────

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("[RL] JSON encode error: %v", err)
	}
}
