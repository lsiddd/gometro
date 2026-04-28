package rl

import (
	"testing"

	"minimetro-go/components"
	"minimetro-go/config"
)

func TestBuildObservationIncludesTrainDynamicsAndSeparatedTopology(t *testing.T) {
	env := NewRLEnv()
	env.Reset("london", 1.0)

	if len(env.gs.Stations) < 3 {
		t.Fatalf("expected at least 3 initial stations, got %d", len(env.gs.Stations))
	}
	line := env.gs.Lines[0]
	line.AddStation(env.gs.Stations[0], -1, nil)
	line.AddStation(env.gs.Stations[1], -1, nil)
	line.AddStation(env.gs.Stations[2], -1, nil)

	train := components.NewTrain(env.gs.TrainIDCounter, line, config.Cities["london"].TrainCapacity, config.TrainMaxSpeed)
	train.State = components.TrainMoving
	train.CurrentStationIndex = 1
	train.NextStationIndex = 2
	train.Direction = 1
	train.Progress = 25
	train.PathLength = 100
	train.Speed = train.MaxSpeed / 2
	train.WaitTimer = 250
	train.CarriageCount = 1
	line.Trains = []*components.Train{train}
	env.gs.Trains = []*components.Train{train}

	obs := BuildObservation(env)
	if len(obs) != ObsDim {
		t.Fatalf("obs length: want %d, got %d", ObsDim, len(obs))
	}

	lineBase := GlobalDim + MaxStationSlots*StationDim
	if got := obs[lineBase+7]; got != 0.25 {
		t.Fatalf("avg train progress: want 0.25, got %.3f", got)
	}
	if got := obs[lineBase+8]; got != 1 {
		t.Fatalf("moving ratio: want 1, got %.3f", got)
	}
	if got := obs[lineBase+11]; got != 0.5 {
		t.Fatalf("avg speed: want 0.5, got %.3f", got)
	}

	membershipBase := lineBase + MaxLineSlots*LineDim
	roleBase := membershipBase + MaxLineSlots*MaxStationSlots
	if got := obs[membershipBase+0]; got != 1 {
		t.Fatalf("membership for first station: want 1, got %.3f", got)
	}
	if got := obs[roleBase+0]; got != 0.5 {
		t.Fatalf("role for first station: want head 0.5, got %.3f", got)
	}
	if got := obs[roleBase+1]; got != 0.25 {
		t.Fatalf("role for middle station: want 0.25, got %.3f", got)
	}
	if got := obs[roleBase+2]; got != 1 {
		t.Fatalf("role for tail station: want 1, got %.3f", got)
	}
}
