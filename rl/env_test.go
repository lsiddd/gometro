package rl

import (
	"context"
	"math"
	"testing"

	"minimetro-go/components"
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

func TestResetAppliesComplexityConfig(t *testing.T) {
	env := NewRLEnv()
	env.SetComplexity(0)
	env.Reset("london", 1.0)

	if len(env.gs.Rivers) != 0 {
		t.Fatalf("rivers should be disabled at complexity 0, got %d", len(env.gs.Rivers))
	}
	if env.gs.Bridges != 0 {
		t.Fatalf("bridges should be removed when rivers are disabled, got %d", env.gs.Bridges)
	}
	if env.gs.UpgradesEnabled {
		t.Fatal("upgrades should be disabled at complexity 0")
	}
	if env.gs.StationSpawnLimit != 6 {
		t.Fatalf("station spawn limit: want 6, got %d", env.gs.StationSpawnLimit)
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

func TestVectorResetUsesCurrentComplexityLevel(t *testing.T) {
	svc := NewGRPCService()
	svc.setComplexityLevel(0)
	_, err := svc.ResetVector(context.Background(), &pb.VectorResetRequest{
		NumEnvs:         1,
		City:            "london",
		SpawnRateFactor: 1.0,
	})
	if err != nil {
		t.Fatalf("ResetVector: %v", err)
	}

	env := svc.vecEnvs[0]
	if env.gs.UpgradesEnabled {
		t.Fatal("vector env should use disabled upgrades at complexity 0")
	}
	if len(env.gs.Rivers) != 0 {
		t.Fatalf("vector env rivers: want 0, got %d", len(env.gs.Rivers))
	}
}

func TestComputeRewardKeepsPerStepScaleBalanced(t *testing.T) {
	env := NewRLEnv()
	env.Reset("london", 1.0)

	env.prevDelivered = 0
	env.prevWeek = 1
	env.prevQueuePressure = 0.25
	env.prevOvercrowdPressure = 0.25
	env.gs.PassengersDelivered = 2
	env.gs.Week = 2
	env.gs.Stations[0].Passengers = []*components.Passenger{
		components.NewPassenger(env.gs.Stations[0], config.Triangle, 0),
		components.NewPassenger(env.gs.Stations[0], config.Square, 0),
		components.NewPassenger(env.gs.Stations[0], config.Circle, 0),
	}
	env.gs.Stations[0].OvercrowdProgress = 0.5 * float64(config.OvercrowdTime)
	env.gs.Stations[1].OvercrowdProgress = 0.9 * float64(config.OvercrowdTime)

	reward := env.computeReward(false)

	queuePressure := 3.0 / float64(config.Cities["london"].StationCapacity)
	overcrowdPressure := 0.5*0.5 + 0.9*0.9
	want := (2.0 * rewardPerPassenger) -
		(queuePressure * rewardQueueCoeff) -
		((queuePressure - 0.25) * rewardQueueDeltaCoeff) -
		(overcrowdPressure * rewardOvercrowdCoeff) -
		((overcrowdPressure - 0.25) * rewardOvercrowdDeltaCoeff) -
		rewardDangerPenalty +
		rewardWeekBonus
	if math.Abs(reward-want) > 1e-9 {
		t.Fatalf("reward: want %.3f, got %.3f", want, reward)
	}
	if env.lastReward.Queue <= 0 {
		t.Fatal("queue penalty should be positive when passengers are waiting")
	}
	if env.lastReward.OvercrowdDelta <= 0 {
		t.Fatal("overcrowd delta penalty should be positive when risk increases")
	}
	if env.lastReward.Danger != rewardDangerPenalty {
		t.Fatalf("danger penalty: want %.3f, got %.3f", rewardDangerPenalty, env.lastReward.Danger)
	}
}

func TestComputeNoOpPenaltyOnlyInCriticalRepeatedNoOp(t *testing.T) {
	env := NewRLEnv()
	env.Reset("london", 1.0)

	env.consecutiveNoOp = 1
	if got := env.computeNoOpPenalty(); got != 0 {
		t.Fatalf("first noop should not be penalized, got %.3f", got)
	}

	env.consecutiveNoOp = 2
	if got := env.computeNoOpPenalty(); got != 0 {
		t.Fatalf("safe state noop should not be penalized, got %.3f", got)
	}

	env.gs.Stations[0].OvercrowdProgress = 0.9 * float64(config.OvercrowdTime)
	got := env.computeNoOpPenalty()
	if got != rewardNoOpCriticalPenalty {
		t.Fatalf("critical repeated noop penalty: want %.3f, got %.3f", rewardNoOpCriticalPenalty, got)
	}

	env.consecutiveNoOp = 20
	got = env.computeNoOpPenalty()
	maxPenalty := rewardNoOpCriticalPenalty * 3.0
	if got != maxPenalty {
		t.Fatalf("noop penalty should cap at %.3f, got %.3f", maxPenalty, got)
	}
}
