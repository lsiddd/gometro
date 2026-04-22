package rl

import (
	"context"
	"testing"

	pb "minimetro-go/rl/proto"
)

func TestResetPreservesSpawnRateFactor(t *testing.T) {
	env := NewRLEnv()
	env.Reset("london", 3.0)

	if env.gs.SpawnRateFactor != 3.0 {
		t.Fatalf("spawn factor: want 3.0, got %f", env.gs.SpawnRateFactor)
	}
}

func TestVectorAutoResetUsesCurrentSpawnRateFactor(t *testing.T) {
	svc := NewGRPCService()
	_, err := svc.ResetVector(context.Background(), &pb.VectorResetRequest{
		NumEnvs:         1,
		City:            "london",
		SpawnRateFactor: 4.0,
	})
	if err != nil {
		t.Fatalf("ResetVector: %v", err)
	}

	svc.setSpawnRateFactor(2.0)
	env := svc.vecEnvs[0]
	env.gs.GameOver = true

	obs, _, done, _ := env.Step([]int{ActNoOp, 0, 0, 0})
	if !done {
		t.Fatal("expected forced game-over step to be done")
	}
	if len(obs) != ObsDim {
		t.Fatalf("terminal obs length: want %d, got %d", ObsDim, len(obs))
	}

	_, mask := env.Reset(env.gs.SelectedCity, svc.spawnRateFactor())
	if env.gs.SpawnRateFactor != 2.0 {
		t.Fatalf("auto-reset spawn factor: want 2.0, got %f", env.gs.SpawnRateFactor)
	}
	if len(mask) != MaskSize {
		t.Fatalf("mask length: want %d, got %d", MaskSize, len(mask))
	}
}
