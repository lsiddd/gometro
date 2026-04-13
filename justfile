binary     := "rl_server"
python_dir := "python"
ckpt_dir   := python_dir / "checkpoints"
tb_port    := "6006"
tb_pid     := "/tmp/minimetro_tb.pid"

n_envs     := "12"
city       := "london"
checkpoint := ""

# compile all Go binaries
build:
	go build -o minimetro .
	go build -tags headless -o {{binary}} ./cmd/rl_server/

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
