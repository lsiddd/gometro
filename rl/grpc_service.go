package rl

import (
	"context"
	"fmt"
	"io"
	"log"
	"runtime/debug"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	pb "minimetro-go/rl/proto"
)

// GRPCService implements pb.RLEnvServer.
type GRPCService struct {
	pb.UnimplementedRLEnvServer
	env                    *RLEnv   // Legacy single environment
	vecEnvs                []*RLEnv // Vectorized execution batch
	mu                     sync.RWMutex
	currentSpawnRateFactor float64
	lastDifficultyLog      time.Time
}

func NewGRPCService() *GRPCService {
	return &GRPCService{env: NewRLEnv(), currentSpawnRateFactor: 1.0}
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

// RegisterControlService exposes a tiny manually-registered control endpoint.
// It avoids regenerating the protobuf stubs just to let the Python curriculum
// update the spawn-rate factor used by native vector auto-reset.
func RegisterControlService(reg grpc.ServiceRegistrar, svc *GRPCService) {
	reg.RegisterService(&grpc.ServiceDesc{
		ServiceName: "rl.Control",
		HandlerType: (*controlServer)(nil),
		Methods: []grpc.MethodDesc{
			{
				MethodName: "SetDifficulty",
				Handler:    controlSetDifficultyHandler,
			},
		},
		Streams:  []grpc.StreamDesc{},
		Metadata: "rl/proto/minimetro.proto",
	}, svc)
}

type controlServer interface{}

func controlSetDifficultyHandler(
	srv any,
	ctx context.Context,
	dec func(any) error,
	interceptor grpc.UnaryServerInterceptor,
) (any, error) {
	in := new(pb.ResetRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		srv.(*GRPCService).setSpawnRateFactor(float64(in.SpawnRateFactor))
		return &pb.Empty{}, nil
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/rl.Control/SetDifficulty",
	}
	handler := func(ctx context.Context, req any) (any, error) {
		srv.(*GRPCService).setSpawnRateFactor(float64(req.(*pb.ResetRequest).SpawnRateFactor))
		return &pb.Empty{}, nil
	}
	return interceptor(ctx, in, info, handler)
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
	log.Printf("[rl] Reset city=%s spawn_rate_factor=%.3f", city, sf)
	obs, mask := s.env.Reset(city, sf)
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
	log.Printf("[rl] ResetVector n=%d city=%s spawn_rate_factor=%.3f", int(req.NumEnvs), city, sf)

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

	allObs := make([]float32, n*ObsDim)
	allMask := make([]bool, n*MaskSize)

	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(idx int) {
			defer wg.Done()
			obs, mask := s.vecEnvs[idx].Reset(city, sf)
			copy(allObs[idx*ObsDim:(idx+1)*ObsDim], obs)
			copy(allMask[idx*MaskSize:(idx+1)*MaskSize], mask)
		}(i)
	}
	wg.Wait()

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
					obs, mask = env.Reset(env.gs.SelectedCity, s.spawnRateFactor())
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
