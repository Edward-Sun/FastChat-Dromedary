set -x
set -e

export CUDA_VISIBLE_DEVICES=5

CONTROLLER_IP=0.0.0.0
CONTROLLER_PORT=21001
WORKER_IP=$(hostname -I | cut -d' ' -f1)
WORKER_PORT=21002

export MODEL_DIR="/usr1/data/zhiqings/dromedary_ckpt"
CKPT_NAME="4-shards"

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=3,5,6,7
export SLURM_NNODES=1
export SLURM_PROCID=0
export MASTER_ADDR="0.0.0.0"
export MASTER_PORT=9901
export OMP_NUM_THREADS=4
export GPUS_PER_NODE=4

torchrun --nproc_per_node $GPUS_PER_NODE \
  --nnodes $SLURM_NNODES \
  --node_rank $SLURM_PROCID \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  -m fastchat.serve.dromedary_model_worker \
  --model-name dromedary \
  --model-path $MODEL_DIR/$CKPT_NAME \
  --tokenizer-path $MODEL_DIR/tokenizer.model
  --controller-address http://${CONTROLLER_IP}:${CONTROLLER_PORT} \
  --worker-address http://${WORKER_IP}:${WORKER_PORT} \
  --host 0.0.0.0 \
  --port ${WORKER_PORT}
