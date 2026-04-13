package rl

import (
	"context"
	"io"
	"log"

	pb "minimetro-go/rl/proto"
)

// GRPCService implements pb.RLEnvServer.
// Each instance owns one RLEnv (same invariant as the HTTP Server).
type GRPCService struct {
	pb.UnimplementedRLEnvServer
	env *RLEnv
}

// NewGRPCService allocates a GRPCService backed by a fresh RLEnv.
func NewGRPCService() *GRPCService {
	return &GRPCService{env: NewRLEnv()}
}

// ── Info ─────────────────────────────────────────────────────────────────────

func (s *GRPCService) Info(_ context.Context, _ *pb.Empty) (*pb.InfoResponse, error) {
	return &pb.InfoResponse{
		ObsDim:      int32(ObsDim),
		ActionDims:  []int32{NumActionCats, MaxLineSlots, MaxStationSlots, NumOptions},
		GlobalDim:   int32(GlobalDim),
		StationDim:  int32(StationDim),
		NumStations: int32(MaxStationSlots),
		LineDim:     int32(LineDim),
		NumLines:    int32(MaxLineSlots),
	}, nil
}

// ── Reset ─────────────────────────────────────────────────────────────────────

func (s *GRPCService) Reset(_ context.Context, req *pb.ResetRequest) (*pb.ResetResponse, error) {
	city := req.City
	if city == "" {
		city = "london"
	}
	spawnRateFactor := float64(req.SpawnRateFactor)
	if spawnRateFactor <= 0 {
		spawnRateFactor = 1.0
	}
	obs, mask := s.env.Reset(city, spawnRateFactor)
	return &pb.ResetResponse{
		Obs:  obs,
		Mask: boolSlice(mask),
	}, nil
}

// ── RunEpisode ────────────────────────────────────────────────────────────────

// RunEpisode is the hot path for RL training.  The client streams ActionRequests
// and receives StepResponses without the overhead of a new RPC per step.
func (s *GRPCService) RunEpisode(stream pb.RLEnv_RunEpisodeServer) error {
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}

		action := int32SliceToInt(req.Action)
		obs, reward, done, mask := s.env.Step(action)
		gs := s.env.gs

		resp := &pb.StepResponse{
			Obs:                 obs,
			Mask:                boolSlice(mask),
			Reward:              reward,
			Done:                done,
			Score:               int32(gs.Score),
			PassengersDelivered: int32(gs.PassengersDelivered),
			Week:                int32(gs.Week),
			Stations:            int32(len(gs.Stations)),
			GameOver:            gs.GameOver,
			InUpgradeModal:      s.env.inUpgradeModal,
		}
		if err := stream.Send(resp); err != nil {
			return err
		}
		if done {
			log.Printf("[RL] Episode ended — score=%d passengers=%d",
				gs.Score, gs.PassengersDelivered)
		}
	}
}

// ── SolverAct ─────────────────────────────────────────────────────────────────

func (s *GRPCService) SolverAct(_ context.Context, _ *pb.Empty) (*pb.ActionResponse, error) {
	action := s.env.InferSolverAction()
	return &pb.ActionResponse{Action: intSliceToInt32(action)}, nil
}

// ── helpers ───────────────────────────────────────────────────────────────────

func boolSlice(src []bool) []bool {
	// pb.StepResponse.Mask is []bool — return as-is; protobuf handles packing.
	return src
}

func int32SliceToInt(src []int32) []int {
	out := make([]int, len(src))
	for i, v := range src {
		out[i] = int(v)
	}
	return out
}

func intSliceToInt32(src []int) []int32 {
	out := make([]int32, len(src))
	for i, v := range src {
		out[i] = int32(v)
	}
	return out
}
