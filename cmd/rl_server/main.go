// rl_server is a headless binary that exposes a single Mini Metro game
// environment over gRPC. It is intended for use as the simulation back-end
// during RL training from Python.
//
// Usage:
//
//	./rl_server [--port 8765]
//
// Each worker process should run its own rl_server instance on a distinct
// port (see python/train.py). The gRPC service implements the RLEnv proto
// defined in rl/proto/minimetro.proto.
package main

import (
	"flag"
	"fmt"
	"log"
	"net"

	"google.golang.org/grpc"
	"minimetro-go/rl"
	pb "minimetro-go/rl/proto"
)

func main() {
	port := flag.Int("port", 8765, "gRPC port to listen on")
	flag.Parse()

	addr := fmt.Sprintf(":%d", *port)
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		log.Fatalf("[rl_server] listen %s: %v", addr, err)
	}

	srv := grpc.NewServer()
	rlSvc := rl.NewGRPCService()
	pb.RegisterRLEnvServer(srv, rlSvc)
	rl.RegisterControlService(srv, rlSvc)

	log.Printf("[rl_server] gRPC listening on %s", addr)
	log.Fatal(srv.Serve(lis))
}
