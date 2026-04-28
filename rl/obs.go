package rl

import (
	"math"
	"minimetro-go/components"
	"minimetro-go/config"
	"minimetro-go/systems"
)

// Observation-vector layout constants. Python reads these via GET /info and
// validates them at startup; keep them in sync with python/constants.py.
const (
	GlobalDim  = 15 // global scalar features
	StationDim = 16 // per-station features (9 base + 7 passenger-demand dims)
	LineDim    = 14 // per-line features
)

// ObsDim is the total length of the observation vector returned by BuildObservation.
//
//	Global:   GlobalDim                        = 15
//	Stations: MaxStationSlots × StationDim     = 800
//	Lines:      MaxLineSlots × LineDim          = 98
//	Membership: MaxLineSlots × MaxStationSlots  = 350
//	Role:       MaxLineSlots × MaxStationSlots  = 350
//	Total:      1613
const ObsDim = GlobalDim + MaxStationSlots*StationDim + MaxLineSlots*LineDim + 2*MaxLineSlots*MaxStationSlots

// numStationTypes is the number of distinct station/passenger types.
const numStationTypes = 7

// upgradeTypeIndex maps an upgrade key to an integer label for the observation.
// Returns values in [1, 4]; 0 is reserved for "no choice" (modal inactive).
func upgradeTypeIndex(key string) float32 {
	switch key {
	case systems.UpgradeNewLine:
		return 1
	case systems.UpgradeCarriage:
		return 2
	case systems.UpgradeBridge:
		return 3
	case systems.UpgradeInterchange:
		return 4
	}
	return 0
}

// stationTypeIndex maps a StationType to an integer label in [0, 6].
// Returns -1 for unknown types so callers can skip them rather than
// misclassifying them as Circle (index 0).
func stationTypeIndex(t config.StationType) int {
	switch t {
	case config.Circle:
		return 0
	case config.Triangle:
		return 1
	case config.Square:
		return 2
	case config.Pentagon:
		return 3
	case config.Diamond:
		return 4
	case config.Star:
		return 5
	case config.Cross:
		return 6
	}
	return -1
}

// clamp01 clamps v to [0, 1].
func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// BuildObservation serialises the current game state into a fixed-size float32
// vector of length ObsDim. All values are normalised to [0, 1].
func BuildObservation(env *RLEnv) []float32 {
	gs := env.gs
	obs := make([]float32, ObsDim)
	i := 0

	// ── Global features (15) ─────────────────────────────────────────────────
	obs[i] = clamp01(float32(gs.Week) / 10.0)
	i++
	obs[i] = clamp01(float32(gs.Day) / 7.0)
	i++
	obs[i] = clamp01(float32(gs.Score) / 10000.0)
	i++
	obs[i] = clamp01(float32(gs.PassengersDelivered) / 1000.0)
	i++
	obs[i] = clamp01(float32(gs.AvailableTrains) / 10.0)
	i++
	obs[i] = clamp01(float32(gs.Carriages) / 10.0)
	i++
	obs[i] = clamp01(float32(gs.Interchanges) / 10.0)
	i++
	obs[i] = float32(gs.AvailableLines) / float32(MaxLineSlots)
	i++
	obs[i] = float32(gs.MaxLines) / float32(MaxLineSlots)
	i++
	obs[i] = clamp01(float32(gs.Bridges) / 10.0)
	i++
	// Fraction of current week elapsed.
	weekElapsed := gs.SimTimeMs - gs.WeekStartTime
	obs[i] = clamp01(float32(weekElapsed) / config.WeekDuration)
	i++
	// Upgrade modal flag.
	if env.inUpgradeModal {
		obs[i] = 1
	}
	i++
	// Upgrade choice types: 0 = no modal / unknown, 1-4 = upgrade type.
	// When the modal is inactive both slots are 0; the flag above disambiguates.
	if env.inUpgradeModal && len(env.upgradeChoices) >= 2 {
		obs[i] = upgradeTypeIndex(env.upgradeChoices[0]) / 4.0
		i++
		obs[i] = upgradeTypeIndex(env.upgradeChoices[1]) / 4.0
		i++
	} else {
		obs[i] = 0
		i++
		obs[i] = 0
		i++
	}
	obs[i] = 0 // reserved
	i++

	// ── Centrality ───────────────────────────────────────────────────────────
	// CachedCentrality reuses the last computed result when the topology has not
	// changed, avoiding a redundant O(V·E) Brandes pass every frame.
	env.game.GraphManager.GetGraph(gs)
	centrality := env.game.GraphManager.CachedCentrality(gs)

	// Normalise centrality to [0, 1] by dividing by the max observed value.
	maxCentrality := float32(0)
	for _, v := range centrality {
		fv := float32(v)
		if fv > maxCentrality {
			maxCentrality = fv
		}
	}

	// ── Station features (50 × 16) ───────────────────────────────────────────
	// Per-station layout (base+0 .. base+15):
	//   0: x            1: y             2: type
	//   3: pax ratio    4: overcrowd     5: is_interchange
	//   6: is_valid     7: lines through 8: centrality
	//   9-15: passenger demand by type (Circle..Cross) / capacity

	// Build station-ID → line-count map once (O(L×S)) rather than recomputing
	// it inside the per-station loop (was O(S×L×S)).
	stationLineCnt := make(map[int]int, len(gs.Stations))
	for l := 0; l < gs.AvailableLines && l < len(gs.Lines); l++ {
		line := gs.Lines[l]
		if !line.Active || line.MarkedForDeletion {
			continue
		}
		seen := make(map[int]bool, len(line.Stations))
		for _, ls := range line.Stations {
			if !seen[ls.ID] {
				stationLineCnt[ls.ID]++
				seen[ls.ID] = true
			}
		}
	}

	cityCfg := config.Cities[gs.SelectedCity]
	for s := 0; s < MaxStationSlots; s++ {
		base := i + s*StationDim
		if s >= len(gs.Stations) {
			continue // padding zeros
		}
		st := gs.Stations[s]
		cap := st.Capacity(cityCfg.StationCapacity)
		fcap := float32(cap)

		linesThr := stationLineCnt[st.ID]

		// Passenger demand breakdown by destination type.
		var demand [numStationTypes]float32
		for _, p := range st.Passengers {
			idx := stationTypeIndex(p.Destination)
			if idx >= 0 && idx < numStationTypes {
				demand[idx]++
			}
		}

		centVal := float32(0)
		if maxCentrality > 0 {
			centVal = float32(centrality[st]) / maxCentrality
		}

		typeIdx := stationTypeIndex(st.Type)
		if typeIdx < 0 {
			typeIdx = 0
		}

		obs[base+0] = float32(st.X) / systems.SimScreenWidth
		obs[base+1] = float32(st.Y) / systems.SimScreenHeight
		obs[base+2] = float32(typeIdx) / float32(numStationTypes-1)
		obs[base+3] = clamp01(float32(len(st.Passengers)) / fcap)
		effectiveOCLimit := float32(config.OvercrowdTime)
		if st.OvercrowdIsGrace {
			effectiveOCLimit += config.OvercrowdGraceExtra
		}
		obs[base+4] = clamp01(float32(st.OvercrowdProgress) / effectiveOCLimit)
		if st.IsInterchange {
			obs[base+5] = 1
		}
		obs[base+6] = 1 // is_valid
		obs[base+7] = float32(linesThr) / float32(MaxLineSlots)
		obs[base+8] = clamp01(centVal)
		for t := 0; t < numStationTypes; t++ {
			obs[base+9+t] = clamp01(demand[t] / fcap)
		}
		// Compile-time guard: StationDim must cover indices 0..15 (9 base + 7 demand).
		_ = [StationDim - 16]struct{}{} // fails to compile if StationDim != 16
	}
	i += MaxStationSlots * StationDim

	// ── Line features (7 × 14) ───────────────────────────────────────────────
	// Per-line layout (base+0 .. base+13):
	//   0: active          1: available       2: station count
	//   3: train count     4: load ratio      5: is loop
	//   6: arc length      7: avg progress    8: moving train ratio
	//   9: forward ratio  10: avg wait       11: avg speed
	//  12: avg carriages  13: avg station index
	// Compile-time guard: LineDim must cover indices 0..13.
	_ = [LineDim - 14]struct{}{}
	for l := 0; l < MaxLineSlots; l++ {
		base := i + l*LineDim
		if l >= len(gs.Lines) {
			continue
		}
		line := gs.Lines[l]

		arcLen := 0.0
		stations := line.Stations
		for k := 0; k+1 < len(stations); k++ {
			dx := stations[k+1].X - stations[k].X
			dy := stations[k+1].Y - stations[k].Y
			arcLen += float64(dx*dx + dy*dy)
		}
		if arcLen > 0 {
			arcLen = math.Sqrt(arcLen)
		}

		nSt := len(stations)
		totalPax, totalCap := 0, 0
		var totalProgress, moving, forward, wait, speed, carriages, stationIndex float64
		for _, t := range line.Trains {
			totalPax += len(t.Passengers)
			totalCap += t.TotalCapacity()
			if t.PathLength > 0 {
				totalProgress += t.Progress / t.PathLength
			}
			if t.State == components.TrainMoving {
				moving++
			}
			if t.Direction >= 0 {
				forward++
			}
			wait += t.WaitTimer / 1000.0
			if t.MaxSpeed > 0 {
				speed += t.Speed / t.MaxSpeed
			}
			carriages += float64(t.CarriageCount) / float64(config.MaxCarriagesPerTrain)
			if nSt > 1 {
				stationIndex += float64(t.CurrentStationIndex) / float64(nSt-1)
			}
		}
		loadRatio := float32(0)
		if totalCap > 0 {
			loadRatio = clamp01(float32(totalPax) / float32(totalCap))
		}

		isLoop := line.Active && nSt > 2 && stations[0] == stations[nSt-1]

		if line.Active {
			obs[base+0] = 1
		}
		if l < gs.AvailableLines {
			obs[base+1] = 1
		}
		obs[base+2] = float32(nSt) / float32(MaxStationSlots)
		obs[base+3] = float32(len(line.Trains)) / float32(config.MaxTrainsPerLine)
		obs[base+4] = loadRatio
		if isLoop {
			obs[base+5] = 1
		}
		obs[base+6] = clamp01(float32(arcLen) / 1000.0)
		if len(line.Trains) > 0 {
			denom := float32(len(line.Trains))
			obs[base+7] = clamp01(float32(totalProgress) / denom)
			obs[base+8] = clamp01(float32(moving) / denom)
			obs[base+9] = clamp01(float32(forward) / denom)
			obs[base+10] = clamp01(float32(wait) / denom)
			obs[base+11] = clamp01(float32(speed) / denom)
			obs[base+12] = clamp01(float32(carriages) / denom)
			obs[base+13] = clamp01(float32(stationIndex) / denom)
		}
	}
	i += MaxLineSlots * LineDim

	// ── Topology membership (7 × 50) ──────────────────────────────────────────
	for l := 0; l < MaxLineSlots; l++ {
		var stations []*components.Station
		if l < len(gs.Lines) && gs.Lines[l].Active {
			stations = gs.Lines[l].Stations
		}

		for s := 0; s < MaxStationSlots; s++ {
			var st *components.Station
			if s < len(gs.Stations) {
				st = gs.Stations[s]
			}

			val := float32(0)
			if st != nil && len(stations) > 0 {
				for _, ls := range stations {
					if ls == st {
						val = 1
						break
					}
				}
			}

			obs[i] = val
			i++
		}
	}

	// ── Topology role (7 × 50) ────────────────────────────────────────────────
	// 0: absent, 0.25: middle, 0.5: head, 0.75: loop point, 1.0: tail.
	for l := 0; l < MaxLineSlots; l++ {
		var stations []*components.Station
		if l < len(gs.Lines) && gs.Lines[l].Active {
			stations = gs.Lines[l].Stations
		}

		for s := 0; s < MaxStationSlots; s++ {
			var st *components.Station
			if s < len(gs.Stations) {
				st = gs.Stations[s]
			}

			val := float32(0)
			if st != nil && len(stations) > 0 {
				isHead, isTail, isMiddle := false, false, false
				for idx, ls := range stations {
					if ls == st {
						if idx == 0 {
							isHead = true
						} else if idx == len(stations)-1 {
							isTail = true
						} else {
							isMiddle = true
						}
					}
				}
				if isHead && isTail {
					val = 0.75
				} else if isHead {
					val = 0.5
				} else if isTail {
					val = 1.0
				} else if isMiddle {
					val = 0.25
				}
			}

			obs[i] = val
			i++
		}
	}

	if i != ObsDim {
		panic("BuildObservation: wrote wrong number of elements")
	}

	return obs
}
