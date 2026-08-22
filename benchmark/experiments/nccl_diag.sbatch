#!/bin/bash
#SBATCH -p team4
#SBATCH -w node4
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH -J nccl-diag
#SBATCH -o infra/nccl_diag.%j.log
cd /shared/home/altzhang-de4f8c/spec-ptc
PY=.venv-vllm/bin/python

pick_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    sort -t, -k2 -n | head -2 | cut -d, -f1 | paste -sd,
}
export CUDA_VISIBLE_DEVICES=$(pick_gpus)
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi topo -m 2>/dev/null | head -12

cat > /tmp/allreduce_test.py <<'PYEOF'
import os, sys, torch, torch.distributed as dist
rank = int(sys.argv[1])
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29517",
                  RANK=str(rank), WORLD_SIZE="2")
dist.init_process_group("nccl", rank=rank, world_size=2)
torch.cuda.set_device(rank)
x = torch.ones(1024, device=f"cuda:{rank}")
dist.all_reduce(x)
torch.cuda.synchronize()
print(f"rank {rank}: allreduce OK, sum={x[0].item()}", flush=True)
PYEOF

run_test() {
  echo "=== attempt: $1 ==="
  ( $PY /tmp/allreduce_test.py 0 & $PY /tmp/allreduce_test.py 1 & wait ) 2>&1 | \
    grep -E "OK|Error|error|NCCL INFO (Bootstrap|P2P|SHM|CUMEM|Failed|internal)|misc|transport" | head -12
}

export NCCL_DEBUG=WARN
run_test "defaults (NCCL_DEBUG=WARN)"

export NCCL_CUMEM_ENABLE=0
run_test "NCCL_CUMEM_ENABLE=0"

unset NCCL_CUMEM_ENABLE
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
run_test "P2P+SHM disabled (socket only)"

unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE
export NCCL_NVLS_ENABLE=0
run_test "NCCL_NVLS_ENABLE=0 only (NVLink P2P intact)"
echo DIAG_DONE
