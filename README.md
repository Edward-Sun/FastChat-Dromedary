# FastChat for Dromedary

This is a patch for deploying **Dromedary** on **FastChat** with model parallel infenrence.

## FastChat

Below is some information copied from the original [FastChat](https://github.com/lm-sys/FastChat/) README.md.

| [**Demo**](https://chat.lmsys.org/) | [**Arena**](https://arena.lmsys.org) | [**Discord**](https://discord.gg/h6kCZb72G7) | [**Twitter**](https://twitter.com/lmsysorg) |

An open platform for training, serving, and evaluating large language model based chatbots.

## Dromedary

[Dromedary](https://github.com/IBM/Dromedary) is an open-source self-aligned language model trained with minimal human supervision.  For comprehensive details and insights, we kindly direct you to our [project page](https://mitibmdemos.draco.res.ibm.com/dromedary) and [paper](https://arxiv.org/abs/2305.03047).

## Usage

### 1. Run controller

```bash
python3 -m fastchat.serve.controller --host 0.0.0.0 --port 21001 &
```

### 2. Run torchrun workers

```bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=5

CONTROLLER_IP=0.0.0.0
CONTROLLER_PORT=21001
WORKER_IP=$(hostname -I | cut -d' ' -f1)
WORKER_PORT=21002

MP="?"
export MODEL_DIR="path/to/your/ckpt/"
CKPT_NAME="$MP-shards"

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=3,6,7
export SLURM_NNODES=1
export SLURM_PROCID=0
export MASTER_ADDR="0.0.0.0"
export MASTER_PORT=9901
export OMP_NUM_THREADS=$MP
export GPUS_PER_NODE=$MP

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
```

### 3. Run web server

```bash
python3 -m fastchat.serve.gradio_web_server --share
```

### Citation

Please cite the following paper if you use the data or code in this repo.

```
@misc{sun2023principledriven,
      title={Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision},
      author={Zhiqing Sun and Yikang Shen and Qinhong Zhou and Hongxin Zhang and Zhenfang Chen and David Cox and Yiming Yang and Chuang Gan},
      year={2023},
      eprint={2305.03047},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```
@misc{vicuna2023,
    title = {Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90\%* ChatGPT Quality},
    url = {https://lmsys.org/blog/2023-03-30-vicuna/},
    author = {Chiang, Wei-Lin and Li, Zhuohan and Lin, Zi and Sheng, Ying and Wu, Zhanghao and Zhang, Hao and Zheng, Lianmin and Zhuang, Siyuan and Zhuang, Yonghao and Gonzalez, Joseph E. and Stoica, Ion and Xing, Eric P.},
    month = {March},
    year = {2023}
}
```