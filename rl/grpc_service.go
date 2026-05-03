package rl

import (
	"context"
	"fmt"
	"io"
	"log"
	"runtime/debug"
	"sync"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	pb "minimetro-go/rl/proto"
)

// GRPCService implements pb.RLEnvServer.
type GRPCService struct {
	pb.UnimplementedRLEnvServer
	pb.UnimplementedControlServer
	env                    *RLEnv   // Legacy single environment
	vecEnvs                []*RLEnv // Vectorized execution batch
	mu                     sync.RWMutex
	currentSpawnRateFactor float64
	currentComplexityLevel int
	currentRewardConfig    RewardConfig
	vecSeeds               []int64
	vecEpisodes            []int64
	lastDifficultyLog      time.Time
}

func NewGRPCService() *GRPCService {
	return &GRPCService{
		env:                    NewRLEnv(),
		currentSpawnRateFactor: 1.0,
		currentComplexityLevel: 4,
		currentRewardConfig:    DefaultRewardConfig(),
	}
}

func (s *GRPCService) setSpawnRateFactor(sf float64) {
	if sf <= 0 {
		sf = 1.0
	}
	s.mu.Lock()
	s.currentSpawnRateFactor = sf
	shouldLog := time.Since(s.lastDifficultyLog) > 2*time.Second
	if shouldLog {
		s.lastDifficultyLog = time.Now()
	}
	s.mu.Unlock()
	if shouldLog {
		log.Printf("[rl] difficulty set spawn_rate_factor=%.3f", sf)
	}
}

func (s *GRPCService) spawnRateFactor() float64 {
	s.mu.RLock()
	sf := s.currentSpawnRateFactor
	s.mu.RUnlock()
	if sf <= 0 {
		return 1.0
	}
	return sf
}

func (s *GRPCService) setComplexityLevel(level int) {
	if level < 0 {
		level = 0
	}
	if level > 4 {
		level = 4
	}
	s.mu.Lock()
	s.currentComplexityLevel = level
	s.env.SetComplexity(level)
	for _, env := range s.vecEnvs {
		env.SetComplexity(level)
	}
	s.mu.Unlock()
	log.Printf("[rl] complexity set level=%d", level)
}

func (s *GRPCService) complexityLevel() int {
	s.mu.RLock()
	level := s.currentComplexityLevel
	s.mu.RUnlock()
	return level
}

func (s *GRPCService) setRewardConfig(cfg RewardConfig) error {
	if err := ValidateRewardConfig(cfg); err != nil {
		return err
	}
	s.mu.Lock()
	s.currentRewardConfig = cfg
	if err := s.env.SetRewardConfig(cfg); err != nil {
		s.mu.Unlock()
		return err
	}
	for _, env := range s.vecEnvs {
		if err := env.SetRewardConfig(cfg); err != nil {
			s.mu.Unlock()
			return err
		}
	}
	s.mu.Unlock()
	log.Printf("[rl] reward_config set per_passenger=%.3f week_bonus=%.3f terminal=%.3f", cfg.PerPassenger, cfg.WeekBonus, cfg.TerminalPenalty)
	return nil
}

func (s *GRPCService) rewardConfig() RewardConfig {
	s.mu.RLock()
	cfg := s.currentRewardConfig
	s.mu.RUnlock()
	return cfg
}

func (s *GRPCService) SetDifficulty(_ context.Context, req *pb.ResetRequest) (*pb.Empty, error) {
	s.setSpawnRateFactor(float64(req.SpawnRateFactor))
	return &pb.Empty{}, nil
}

func (s *GRPCService) SetComplexity(_ context.Context, req *pb.ResetRequest) (*pb.Empty, error) {
	s.setComplexityLevel(int(req.SpawnRateFactor))
	return &pb.Empty{}, nil
}

func (s *GRPCService) SetRewardConfig(_ context.Context, req *pb.RewardConfigRequest) (*pb.Empty, error) {
	cfg := rewardConfigFromProto(req)
	if err := s.setRewardConfig(cfg); err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	return &pb.Empty{}, nil
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

// ── Single Episode Logic (Legacy) ─────────────────────────────────────────────

func (s *GRPCService) Reset(_ context.Context, req *pb.ResetRequest) (*pb.ResetResponse, error) {
	city := req.City
	if city == "" {
		city = "london"
	}
	sf := float64(req.SpawnRateFactor)
	if sf <= 0 {
		sf = 1.0
	}
	s.setSpawnRateFactor(sf)
	s.env.SetComplexity(s.complexityLevel())
	if err := s.env.SetRewardConfig(s.rewardConfig()); err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	log.Printf("[rl] Reset city=%s spawn_rate_factor=%.3f complexity=%d", city, sf, s.complexityLevel())
	obs, mask := s.env.ResetWithSeed(city, sf, req.GetSeed())
	return &pb.ResetResponse{Obs: obs, Mask: boolSlice(mask)}, nil
}

func (s *GRPCService) RunEpisode(stream pb.RLEnv_RunEpisodeServer) (err error) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("[rl] RunEpisode panic: %v\n%s", r, debug.Stack())
			err = status.Errorf(codes.Internal, "RunEpisode panic: %v", r)
		}
	}()
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}

		obs, reward, done, mask := s.env.Step(int32SliceToInt(req.Action))
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
	}
}

// ── Vectorized Logic ──────────────────────────────────────────────────────────

func (s *GRPCService) ResetVector(_ context.Context, req *pb.VectorResetRequest) (*pb.VectorResetResponse, error) {
	city := req.City
	if city == "" {
		city = "london"
	}
	sf := float64(req.SpawnRateFactor)
	if sf <= 0 {
		sf = 1.0
	}
	s.setSpawnRateFactor(sf)
	complexity := s.complexityLevel()
	log.Printf("[rl] ResetVector n=%d city=%s spawn_rate_factor=%.3f complexity=%d", int(req.NumEnvs), city, sf, complexity)

	n := int(req.NumEnvs)
	if n <= 0 {
		return nil, status.Error(codes.InvalidArgument, "num_envs must be positive")
	}
	if len(s.vecEnvs) != n {
		s.vecEnvs = make([]*RLEnv, n)
		for i := 0; i < n; i++ {
			s.vecEnvs[i] = NewRLEnv()
		}
	}
	if len(s.vecSeeds) != n {
		s.vecSeeds = make([]int64, n)
		s.vecEpisodes = make([]int64, n)
	}
	baseSeed := req.GetSeed()
	for i := 0; i < n; i++ {
		if baseSeed != 0 {
			s.vecSeeds[i] = baseSeed + int64(i)
		} else {
			s.vecSeeds[i] = 0
		}
		s.vecEpisodes[i] = 0
	}
	rewardCfg := s.rewardConfig()

	allObs := make([]float32, n*ObsDim)
	allMask := make([]bool, n*MaskSize)

	var wg sync.WaitGroup
	var resetErr error
	var resetErrMu sync.Mutex
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(idx int) {
			defer wg.Done()
			s.vecEnvs[idx].SetComplexity(complexity)
			if err := s.vecEnvs[idx].SetRewardConfig(rewardCfg); err != nil {
				resetErrMu.Lock()
				if resetErr == nil {
					resetErr = fmt.Errorf("env %d reward config: %w", idx, err)
				}
				resetErrMu.Unlock()
				return
			}
			obs, mask := s.vecEnvs[idx].ResetWithSeed(city, sf, s.vecSeeds[idx])
			copy(allObs[idx*ObsDim:(idx+1)*ObsDim], obs)
			copy(allMask[idx*MaskSize:(idx+1)*MaskSize], mask)
		}(i)
	}
	wg.Wait()
	if resetErr != nil {
		return nil, status.Error(codes.Internal, resetErr.Error())
	}

	return &pb.VectorResetResponse{
		Obs:  allObs,
		Mask: allMask,
	}, nil
}

func (s *GRPCService) RunVectorEpisode(stream pb.RLEnv_RunVectorEpisodeServer) error {
	n := len(s.vecEnvs)
	if n == 0 {
		return status.Error(codes.FailedPrecondition, "ResetVector must be called before RunVectorEpisode")
	}
	log.Printf("[rl] RunVectorEpisode start n=%d", n)
	lastTrace := time.Now()
	stepCount := 0
	doneCount := 0
	var simAccum time.Duration
	var sendAccum time.Duration
	var simMax time.Duration
	var sendMax time.Duration
	allObs := make([]float32, n*ObsDim)
	allMask := make([]bool, n*MaskSize)
	reward := make([]float64, n)
	done := make([]bool, n)
	termObs := make([]float32, n*ObsDim)
	score := make([]int32, n)
	pax := make([]int32, n)
	week := make([]int32, n)
	stations := make([]int32, n)
	gameOver := make([]bool, n)
	inModal := make([]bool, n)

	for {
		req, err := stream.Recv()
		if err == io.EOF {
			log.Printf("[rl] RunVectorEpisode eof n=%d steps=%d", n, stepCount)
			return nil
		}
		if err != nil {
			log.Printf("[rl] RunVectorEpisode recv_error n=%d steps=%d err=%v", n, stepCount, err)
			return err
		}

		actions := req.Actions
		if len(actions) != n*4 {
			return status.Errorf(codes.InvalidArgument, "actions length = %d, want %d", len(actions), n*4)
		}
		stepCount++

		clear(done)
		clear(gameOver)
		clear(inModal)
		terminalObsOut := []float32(nil)
		var termMu sync.Mutex
		var panicMu sync.Mutex
		var panicErr error

		var wg sync.WaitGroup
		wg.Add(n)
		simStart := time.Now()
		for i := 0; i < n; i++ {
			go func(idx int) {
				defer wg.Done()
				defer func() {
					if r := recover(); r != nil {
						err := fmt.Errorf("env %d panic: %v", idx, r)
						log.Printf("[rl] RunVectorEpisode panic %v\n%s", err, debug.Stack())
						panicMu.Lock()
						if panicErr == nil {
							panicErr = err
						}
						panicMu.Unlock()
					}
				}()
				env := s.vecEnvs[idx]

				offset := idx * 4
				act := [4]int{
					int(actions[offset]),
					int(actions[offset+1]),
					int(actions[offset+2]),
					int(actions[offset+3]),
				}
				obs, r, d, mask := env.Step(act[:])

				reward[idx] = r
				done[idx] = d
				score[idx] = int32(env.gs.Score)
				pax[idx] = int32(env.gs.PassengersDelivered)
				week[idx] = int32(env.gs.Week)
				stations[idx] = int32(len(env.gs.Stations))
				gameOver[idx] = env.gs.GameOver
				inModal[idx] = env.inUpgradeModal

				if d {
					termMu.Lock()
					if terminalObsOut == nil {
						clear(termObs)
						terminalObsOut = termObs
					}
					copy(termObs[idx*ObsDim:(idx+1)*ObsDim], obs)
					termMu.Unlock()
					// Native AutoReset
					env.SetComplexity(s.complexityLevel())
					if err := env.SetRewardConfig(s.rewardConfig()); err != nil {
						panic(err)
					}
					if idx < len(s.vecEpisodes) {
						s.vecEpisodes[idx]++
					}
					nextSeed := int64(0)
					if idx < len(s.vecSeeds) && s.vecSeeds[idx] != 0 {
						nextSeed = s.vecSeeds[idx] + s.vecEpisodes[idx]*1_000_003
					}
					obs, mask = env.ResetWithSeed(env.gs.SelectedCity, s.spawnRateFactor(), nextSeed)
				}

				copy(allObs[idx*ObsDim:(idx+1)*ObsDim], obs)
				copy(allMask[idx*MaskSize:(idx+1)*MaskSize], mask)
			}(i)
		}
		wg.Wait()
		if panicErr != nil {
			return status.Error(codes.Internal, panicErr.Error())
		}
		for _, d := range done {
			if d {
				doneCount++
			}
		}
		simElapsed := time.Since(simStart)
		simAccum += simElapsed
		if simElapsed > simMax {
			simMax = simElapsed
		}

		resp := &pb.VectorStepResponse{
			Obs:                 allObs,
			Mask:                allMask,
			Reward:              reward,
			Done:                done,
			TerminalObs:         terminalObsOut,
			Score:               score,
			PassengersDelivered: pax,
			Week:                week,
			Stations:            stations,
			GameOver:            gameOver,
			InUpgradeModal:      inModal,
		}
		sendStart := time.Now()
		if err := stream.Send(resp); err != nil {
			log.Printf("[rl] RunVectorEpisode send_error n=%d steps=%d err=%v", n, stepCount, err)
			return err
		}
		sendElapsed := time.Since(sendStart)
		sendAccum += sendElapsed
		if sendElapsed > sendMax {
			sendMax = sendElapsed
		}
		if time.Since(lastTrace) >= 10*time.Second {
			avgSim := simAccum / time.Duration(maxInt(stepCount, 1))
			avgSend := sendAccum / time.Duration(maxInt(stepCount, 1))
			log.Printf(
				"[rl] vector_trace n=%d steps=%d dones=%d avg_sim=%s max_sim=%s avg_send=%s max_send=%s sf=%.3f",
				n, stepCount, doneCount, avgSim, simMax, avgSend, sendMax, s.spawnRateFactor(),
			)
			lastTrace = time.Now()
		}
	}
}

func boolSlice(src []bool) []bool { return src }
func int32SliceToInt(src []int32) []int {
	out := make([]int, len(src))
	for i, v := range src {
		out[i] = int(v)
	}
	return out
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func rewardConfigFromProto(req *pb.RewardConfigRequest) RewardConfig {
	return RewardConfig{
		PerPassenger:        req.GetPerPassenger(),
		QueueCoeff:          req.GetQueueCoeff(),
		QueueDeltaCoeff:     req.GetQueueDeltaCoeff(),
		OvercrowdCoeff:      req.GetOvercrowdCoeff(),
		OvercrowdDeltaCoeff: req.GetOvercrowdDeltaCoeff(),
		DangerThresh:        req.GetDangerThresh(),
		DangerPenalty:       req.GetDangerPenalty(),
		NoOpCriticalPenalty: req.GetNoopCriticalPenalty(),
		WeekBonus:           req.GetWeekBonus(),
		TerminalPenalty:     req.GetTerminalPenalty(),
		InvalidAction:       req.GetInvalidAction(),
	}
}

func rewardConfigToProto(cfg RewardConfig) *pb.RewardConfigRequest {
	return &pb.RewardConfigRequest{
		PerPassenger:        cfg.PerPassenger,
		QueueCoeff:          cfg.QueueCoeff,
		QueueDeltaCoeff:     cfg.QueueDeltaCoeff,
		OvercrowdCoeff:      cfg.OvercrowdCoeff,
		OvercrowdDeltaCoeff: cfg.OvercrowdDeltaCoeff,
		DangerThresh:        cfg.DangerThresh,
		DangerPenalty:       cfg.DangerPenalty,
		NoopCriticalPenalty: cfg.NoOpCriticalPenalty,
		WeekBonus:           cfg.WeekBonus,
		TerminalPenalty:     cfg.TerminalPenalty,
		InvalidAction:       cfg.InvalidAction,
	}
}
