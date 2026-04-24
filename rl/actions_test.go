package rl

import "testing"

func activateFirstLine(t *testing.T, env *RLEnv) {
	t.Helper()
	if len(env.gs.Stations) < 2 {
		t.Fatal("test setup requires at least two stations")
	}
	line := env.gs.Lines[0]
	if !line.AddStation(env.gs.Stations[0], -1, func() { env.gs.GraphDirty = true }) {
		t.Fatal("add first station")
	}
	if !line.AddStation(env.gs.Stations[1], -1, func() { env.gs.GraphDirty = true }) {
		t.Fatal("add second station")
	}
}

func TestActionMaskOnlyAdvertisesResourceActionsWithTargets(t *testing.T) {
	env := NewRLEnv()
	env.Reset("london", 1.0)

	mask := BuildActionMaskMulti(env)
	if mask[ActDeployTrain] {
		t.Fatal("deploy train should be masked until an active line exists")
	}
	if mask[ActAddCarriage] {
		t.Fatal("add carriage should be masked until a train can receive one")
	}
	if mask[ActUpgradeInterchange] {
		t.Fatal("upgrade interchange should be masked when no interchange upgrade is available")
	}

	activateFirstLine(t, env)
	mask = BuildActionMaskMulti(env)
	if !mask[ActDeployTrain] {
		t.Fatal("deploy train should be available for an active line with train capacity")
	}

	if !ApplyRLAction(env, []int{ActDeployTrain, 0, 0, 0}) {
		t.Fatal("deploy train action failed")
	}
	env.gs.Carriages = 1
	mask = BuildActionMaskMulti(env)
	if !mask[ActAddCarriage] {
		t.Fatal("add carriage should be available when a train can receive one")
	}

	env.gs.Interchanges = 1
	mask = BuildActionMaskMulti(env)
	if !mask[ActUpgradeInterchange] {
		t.Fatal("upgrade interchange should be available when a station can receive it")
	}
	for _, st := range env.gs.Stations {
		st.IsInterchange = true
	}
	mask = BuildActionMaskMulti(env)
	if mask[ActUpgradeInterchange] {
		t.Fatal("upgrade interchange should be masked when all stations are already interchanges")
	}
}

func TestResourceActionsUseInvalidLineAsPreferenceOnly(t *testing.T) {
	env := NewRLEnv()
	env.Reset("london", 1.0)
	activateFirstLine(t, env)

	if !ApplyRLAction(env, []int{ActDeployTrain, 1, 0, 0}) {
		t.Fatal("deploy train should fall back from inactive preferred line to active line")
	}
	if got := len(env.gs.Lines[0].Trains); got != 1 {
		t.Fatalf("line 0 trains: want 1, got %d", got)
	}

	env.gs.Carriages = 1
	if !ApplyRLAction(env, []int{ActAddCarriage, 1, 0, 0}) {
		t.Fatal("add carriage should fall back from inactive preferred line to train-bearing line")
	}
	if got := env.gs.Lines[0].Trains[0].CarriageCount; got != 1 {
		t.Fatalf("carriage count: want 1, got %d", got)
	}
}

func TestApplyRLActionRejectsNegativeIndices(t *testing.T) {
	env := NewRLEnv()
	env.Reset("london", 1.0)

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("ApplyRLAction panicked on negative index: %v", r)
		}
	}()

	if ApplyRLAction(env, []int{ActAddEndpoint, 0, -1, 0}) {
		t.Fatal("negative station index should be rejected")
	}
	if ApplyRLAction(env, []int{ActDeployTrain, -1, 0, 0}) {
		t.Fatal("negative line index should be rejected")
	}
}

func TestConditionalMaskRestrictsParametersByActionCategory(t *testing.T) {
	env := NewRLEnv()
	env.Reset("london", 1.0)
	activateFirstLine(t, env)

	mask := BuildActionMaskMulti(env)
	if len(mask) != MaskSize {
		t.Fatalf("mask length: want %d, got %d", MaskSize, len(mask))
	}
	if !mask[ActDeployTrain] {
		t.Fatal("deploy train category should be available")
	}
	if !mask[CondLineOffset+ActDeployTrain*MaxLineSlots+0] {
		t.Fatal("deploy train should allow active line 0")
	}
	if mask[CondLineOffset+ActDeployTrain*MaxLineSlots+1] {
		t.Fatal("deploy train should not allow inactive line 1")
	}
	if !mask[CondStationOffset+((ActDeployTrain*MaxLineSlots+0)*MaxStationSlots)+0] {
		t.Fatal("deploy train should keep an ignored station parameter unmasked")
	}
	if mask[CondOptionOffset+ActDeployTrain*NumOptions+1] {
		t.Fatal("deploy train should not allow unused option 1")
	}

	if !mask[ActSwapEndpoint] {
		t.Fatal("swap endpoint category should be available")
	}
	for _, st := range env.gs.Lines[0].Stations {
		idx := -1
		for i, candidate := range env.gs.Stations {
			if candidate == st {
				idx = i
				break
			}
		}
		if idx >= 0 && mask[CondStationOffset+((ActSwapEndpoint*MaxLineSlots+0)*MaxStationSlots)+idx] {
			t.Fatalf("swap endpoint should not allow station already on line: index %d", idx)
		}
	}
}

func TestUpgradeModalConditionalMask(t *testing.T) {
	env := NewRLEnv()
	env.Reset("london", 1.0)
	env.inUpgradeModal = true
	env.upgradeChoices = []string{"new_line", "carriage"}

	mask := BuildActionMaskMulti(env)
	if !mask[ActChooseUpgrade] {
		t.Fatal("choose upgrade should be the available category")
	}
	if mask[ActNoOp] {
		t.Fatal("noop should be masked while choosing an upgrade")
	}
	if !mask[CondOptionOffset+ActChooseUpgrade*NumOptions] ||
		!mask[CondOptionOffset+ActChooseUpgrade*NumOptions+1] {
		t.Fatal("both upgrade options should be available")
	}
}
