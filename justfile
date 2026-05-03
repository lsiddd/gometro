binary     := "rl_server"
python_dir := "python"
ckpt_dir   := python_dir / "checkpoints"
tb_port    := "6006"
tb_pid     := "/tmp/minimetro_tb.pid"

n_envs     := "12"
city       := "london"
checkpoint := ""

evo_run_dir := "evolution_runs/reward_search_001"
evo_smoke_dir := "evolution_runs/reward_search_smoke"

# compile all Go binaries
build:
	go build -o minimetro .
	go build -tags headless -o {{binary}} ./cmd/rl_server/

# run the playable game
run:
	go run .

# start a gRPC inference server for live play
infer model="checkpoints/best_model.zip" port="9000":
	cd {{python_dir}} && uv run python infer.py --model "{{model}}" --port {{port}}

# regenerate protobuf bindings in-place
proto:
	protoc \
		--go_out=. --go_opt=paths=source_relative \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \
		rl/proto/minimetro.proto
	cd {{python_dir}} && uv run python -m grpc_tools.protoc \
		-I .. \
		--python_out=. \
		--grpc_python_out=. \
		../rl/proto/minimetro.proto

# build, start TensorBoard, then train from scratch
train: build _kill-servers
	#!/usr/bin/env bash
	set -euo pipefail
	just _start-tb
	echo ">>> Training started  (http://localhost:{{tb_port}} for TensorBoard)"
	cd {{python_dir}} && uv run python train.py \
		--n-envs {{n_envs}} \
		--city {{city}} || true
	just _stop-tb

# quick local validation for the evolutionary reward search pipeline
evolve-smoke run_dir=evo_smoke_dir: build _kill-servers
	cd {{python_dir}} && uv run python evolve_rewards.py \
		--run-dir {{run_dir}} \
		--population 3 \
		--generations 1 \
		--elites 1 \
		--random-individuals 1 \
		--train-timesteps 10000 \
		--learn-chunk 10000 \
		--eval-episodes 2 \
		--eval-max-steps 500 \
		--eval-seeds 101 \
		--n-envs 2 \
		--skip-baseline \
		--retrain-top 0

# run the default robust evolutionary search; override args as needed:
# just evolve evolution_runs/my_search 12 10 500000 32
evolve run_dir=evo_run_dir population="12" generations="10" train_steps="500000" eval_episodes="32": build _kill-servers
	@echo ">>> Evolution run: {{python_dir}}/{{run_dir}}"
	@echo ">>> Dashboard:     {{python_dir}}/{{run_dir}}/dashboard.html"
	@echo ">>> Leaderboard:   {{python_dir}}/{{run_dir}}/leaderboard.json"
	@echo ">>> Events:        {{python_dir}}/{{run_dir}}/events.jsonl"
	cd {{python_dir}} && uv run python evolve_rewards.py \
		--run-dir {{run_dir}} \
		--population {{population}} \
		--generations {{generations}} \
		--elites 3 \
		--random-individuals 2 \
		--sigma 0.25 \
		--train-timesteps {{train_steps}} \
		--learn-chunk {{train_steps}} \
		--n-envs 8 \
		--eval-episodes {{eval_episodes}} \
		--eval-max-steps 4000 \
		--eval-cities london \
		--eval-complexities 4 \
		--eval-spawn-factors 1.0 \
		--eval-seeds 101,202,303,404 \
		--retrain-top 3 \
		--retrain-timesteps 20000000 \
		--retrain-eval-episodes 64 \
		--retrain-eval-cities london,paris,newyork \
		--retrain-eval-spawn-factors 1.25,1.0 \
		--retrain-eval-seeds 1001,1002,1003,1004

# resume an interrupted evolutionary run without redoing finished evals
evolve-resume run_dir=evo_run_dir: build _kill-servers
	cd {{python_dir}} && uv run python evolve_rewards.py \
		--run-dir {{run_dir}} \
		--resume-run

# print the current leaderboard for an evolutionary run
evolve-summary run_dir=evo_run_dir:
	cd {{python_dir}} && uv run python -c 'import json; from pathlib import Path; run=Path("{{run_dir}}"); paths=[run/"leaderboard.json", run/"final_summary.json"]; path=next((p for p in paths if p.exists()), None); assert path is not None, f"no summary found under {run}"; data=json.loads(path.read_text()); print(path); print(json.dumps(data[:10] if path.name=="leaderboard.json" else data.get("best"), indent=2))'

# show the local dashboard path for an evolutionary run
evolve-dashboard run_dir=evo_run_dir:
	@echo "{{python_dir}}/{{run_dir}}/dashboard.html"

# remove Python bytecode/cache directories
clean-pycache:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

# remove generated run artifacts, checkpoints, logs, TensorBoard data, and built binaries
# Pass yes explicitly: just clean-artifacts yes
clean-artifacts confirm="no":
	#!/usr/bin/env bash
	set -euo pipefail
	if [ "{{confirm}}" != "yes" ]; then
		echo "This removes checkpoints, tb_logs, evolution_runs, logs, and built binaries."
		echo "Run: just clean-artifacts yes"
		exit 1
	fi
	rm -rf \
		minimetro \
		{{binary}} \
		{{python_dir}}/checkpoints \
		{{python_dir}}/tb_logs \
		{{python_dir}}/evolution_runs \
		{{python_dir}}/*.log \
		/tmp/rl_server_vector_*.log

# remove Python caches and generated run artifacts
clean-all-artifacts confirm="no": clean-pycache
	just clean-artifacts {{confirm}}

# build, start TensorBoard, then resume from latest checkpoint
# usage: just resume   OR   just resume checkpoints/pretrain_bc
resume ckpt_arg="": build _kill-servers
	#!/usr/bin/env bash
	set -euo pipefail
	just _start-tb
	if [ -n "{{ckpt_arg}}" ]; then
		ckpt="{{ckpt_arg}}"
	elif [ -f "{{ckpt_dir}}/minimetro_latest.zip" ]; then
		ckpt="checkpoints/minimetro_latest.zip"
	else
		ckpt=$(ls -t {{ckpt_dir}}/*.zip 2>/dev/null | head -1 | sed 's|{{python_dir}}/||')
		if [ -z "$ckpt" ]; then
			echo "No checkpoint found in {{ckpt_dir}}"
			just _stop-tb
			exit 1
		fi
	fi
	echo ">>> Resuming from $ckpt  (http://localhost:{{tb_port}} for TensorBoard)"
	cd {{python_dir}} && uv run python train.py \
		--n-envs {{n_envs}} \
		--city {{city}} \
		--resume "$ckpt" || true
	just _stop-tb

# collect solver demos and run behavioral cloning pre-training
pretrain episodes="50" city="london": build _kill-servers
	#!/usr/bin/env bash
	set -euo pipefail
	just _start-tb
	echo ">>> Pre-training from solver demos (http://localhost:{{tb_port}} for TensorBoard)"
	cd {{python_dir}} && uv run python pretrain.py \
		--episodes {{episodes}} \
		--city {{city}}
	just _stop-tb

# start TensorBoard only (foreground)
tensorboard:
	cd {{python_dir}} && uv run tensorboard --logdir tb_logs --port {{tb_port}}

# ── internal helpers ──────────────────────────────────────────────────────────

[private]
_kill-servers:
	#!/usr/bin/env bash
	pkill -f 'rl_server' 2>/dev/null || true
	sleep 0.3

[private]
_start-tb:
	#!/usr/bin/env bash
	echo ">>> Starting TensorBoard on http://localhost:{{tb_port}} ..."
	cd {{python_dir}} && uv run tensorboard \
		--logdir tb_logs --port {{tb_port}} \
		>/dev/null 2>&1 & echo $! > {{tb_pid}}
	sleep 1

[private]
_stop-tb:
	#!/usr/bin/env bash
	if [ -f {{tb_pid}} ]; then
		kill $(cat {{tb_pid}}) 2>/dev/null || true
		rm -f {{tb_pid}}
	fi
