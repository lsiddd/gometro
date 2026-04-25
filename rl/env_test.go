package rl

import (
	"context"
	"testing"

	"minimetro-go/config"
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

func TestComputeRewardKeepsPerStepScaleBalanced(t *testing.T) {
	env := NewRLEnv()
	env.Reset("london", 1.0)

	env.prevDelivered = 0
	env.prevWeek = 1
	env.gs.PassengersDelivered = 2
	env.gs.Week = 2
	env.gs.Stations[0].OvercrowdProgress = 0.5 * float64(config.OvercrowdTime)
	env.gs.Stations[1].OvercrowdProgress = 0.9 * float64(config.OvercrowdTime)

	reward := env.computeReward(false)

	want := (2.0 * rewardPerPassenger) -
		((0.5 + 0.9) * rewardOvercrowdCoeff) -
		rewardDangerPenalty +
		rewardWeekBonus
	if reward != want {
		t.Fatalf("reward: want %.3f, got %.3f", want, reward)
	}
	if env.lastReward.Danger != rewardDangerPenalty {
		t.Fatalf("danger penalty: want %.3f, got %.3f", rewardDangerPenalty, env.lastReward.Danger)
	}
}
