// rl_server is a headless binary that exposes a single Mini Metro game
// environment over HTTP/JSON. It is intended for use as the simulation
// back-end during RL training from Python.
//
// Usage:
//
//	./rl_server [--port 8765]
//
// The server serialises all requests; each worker process should run its
// own rl_server instance on a distinct port (see python/train.py).
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"

	"minimetro-go/rl"
)

func main() {
	port := flag.Int("port", 8765, "HTTP port to listen on")
	flag.Parse()

	srv := rl.NewServer()
	addr := fmt.Sprintf(":%d", *port)
	log.Printf("[rl_server] listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, srv))
}
