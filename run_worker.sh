set -x
set -e

export CUDA_VISIBLE_DEVICES=5

CONTROLLER_IP=0.0.0.0
CONTROLLER_PORT=21001
WORKER_IP=$(hostname -I | cut -d' ' -f1)
WORKER_PORT=21002
python3 -m fastchat.serve.dromedary_model_worker \
  --model-path lmsys/fastchat-t5-3b-v1.0 \
  --controller-address http://${CONTROLLER_IP}:${CONTROLLER_PORT} \
  --worker-address http://${WORKER_IP}:${WORKER_PORT} \
  --host 0.0.0.0 \
  --port ${WORKER_PORT}
  
# --model-name alpaca-13b \
