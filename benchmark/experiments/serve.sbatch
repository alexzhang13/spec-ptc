#!/bin/bash
#SBATCH -p team4
#SBATCH --gres=gpu:2
#SBATCH -J spec-ptc-serve
#SBATCH --mem=300G
#SBATCH -o infra/serve.%j.log
set -x
ROOT=/shared/home/altzhang-de4f8c/spec-ptc
PY=$ROOT/.venv-vllm/bin/python
MAIN_MODEL=${MAIN_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}
SUB_MODEL=${SUB_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
HOST=$(scontrol show node $(hostname -s) 2>/dev/null | grep -oP "NodeAddr=\S+" | cut -d= -f2); HOST=${HOST:-$(hostname -I | tr " " "\n" | grep "^10\." | head -1)}
# node4 has no CUDA toolkit: keep every flashinfer JIT path off
export PATH=$ROOT/.venv-vllm/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NCCL_NVLS_ENABLE=0   # node4 Fabric Manager is broken for NVLS multicast
KCFG='{"enable_flashinfer_autotune": false, "enable_cutedsl_warmup": false, "enable_jit_warmup": false}' 

# 30B-A3B MoE fits one 80GB H100 in bf16 — single GPU avoids NCCL entirely
CUDA_VISIBLE_DEVICES=0 $PY -m vllm.entrypoints.openai.api_server \
  --model $MAIN_MODEL --port 8100 --gpu-memory-utilization 0.92 \
  --max-model-len 32768 --enforce-eager --kernel-config "$KCFG" &
CUDA_VISIBLE_DEVICES=1 $PY -m vllm.entrypoints.openai.api_server \
  --model $SUB_MODEL --port 8101 --gpu-memory-utilization 0.85 \
  --max-model-len 16384 --enforce-eager --kernel-config "$KCFG" &

cat > $ROOT/.endpoints.env <<EON
SPEC_MAIN_URL=http://$HOST:8100/v1
SPEC_MAIN_MODEL=$MAIN_MODEL
SPEC_SUB_URL=http://$HOST:8101/v1
SPEC_SUB_MODEL=$SUB_MODEL
EON
wait
