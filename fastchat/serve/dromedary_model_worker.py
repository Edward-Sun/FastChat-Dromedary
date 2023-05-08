"""
A model worker executes the model.
"""
import argparse
import asyncio
import dataclasses
import logging
import json
import os
import time
from typing import List, Union, Tuple
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import requests

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LlamaTokenizer,
        AutoModel,
    )
except ImportError:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LLaMATokenizer,
        AutoModel,
    )
import torch
import uvicorn

from fastchat.constants import WORKER_HEART_BEAT_INTERVAL
from fastchat.serve.inference import load_model, generate_stream, add_model_args
from fastchat.serve.serve_chatglm import chatglm_generate_stream
from fastchat.serve.serve_dromedary import dromedary_generate_stream
from fastchat.utils import build_logger, server_error_msg, pretty_print_semaphore
from fastchat.conversation import get_default_conv_template


from fairscale.nn.model_parallel import initialize as mpu
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama_dromedary import ModelArgs, Transformer, Tokenizer, LLaMA

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("dromedary_model_worker", f"dromedary_model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class DromeadryModelWorker:
    def __init__(
        self,
        controller_addr,
        worker_addr,
        worker_id,
        no_register,
        generator,
        model_name,
        device,
        num_gpus,
        max_gpu_memory,
        load_8bit=False,
        cpu_offloading=False,
        tokenizer_path=None,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_name = model_name
        self.device = device

        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")

        del num_gpus
        del max_gpu_memory
        # self.model, self.tokenizer = load_model(
        #     model_path, device, num_gpus, max_gpu_memory, load_8bit, cpu_offloading
        # )
        if load_8bit:
            raise NotImplementedError("8-bit model is not supported for Dromedary")
        if cpu_offloading:
            raise NotImplementedError("CPU offloading is not supported for Dromedary")

        self.model = generator.generate
        self.tokenizer = generator.tokenizer

        # if hasattr(self.model.config, "max_sequence_length"):
        #     self.context_len = self.model.config.max_sequence_length
        # elif hasattr(self.model.config, "max_position_embeddings"):
        #     self.context_len = self.model.config.max_position_embeddings
        # else:
        self.context_len = 2048

        # is_chatglm = "chatglm" in str(type(self.model)).lower()
        # if is_chatglm:
        #     self.generate_stream_func = chatglm_generate_stream
        # else:
        #     self.generate_stream_func = generate_stream
        self.generate_stream_func = dromedary_generate_stream

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Models: {[self.model_name]}. "
            f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
            f"global_counter: {global_counter}"
        )

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=5,
                )
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if (
            model_semaphore is None
            or model_semaphore._value is None
            or model_semaphore._waiters is None
        ):
            return 0
        else:
            return (
                args.limit_model_concurrency
                - model_semaphore._value
                + len(model_semaphore._waiters)
            )

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def generate_stream_gate(self, params):
        try:
            for output in self.generate_stream_func(
                self.model,
                self.tokenizer,
                params,
                self.device,
                self.context_len,
                args.stream_interval,
            ):
                ret = {
                    "text": output,
                    "error_code": 0,
                }
                yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.OutOfMemoryError:
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"

    def generate_completion(self, params):
        raise NotImplementedError

    def get_embeddings(self, params):
        raise NotImplementedError


app = FastAPI()


def release_model_semaphore():
    model_semaphore.release()


async def acquire_model_semaphore():
    global model_semaphore, global_counter
    global_counter += 1
    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()


def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_model_semaphore()
    generator = worker.generate_stream_gate(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate_completion")
async def api_generate_completion(request: Request):
    params = await request.json()
    await acquire_model_semaphore()
    completion = worker.generate_completion(params)
    background_tasks = create_background_tasks()
    return JSONResponse(content=completion, background=background_tasks)


@app.post("/worker_get_embeddings")
async def api_get_embeddings(request: Request):
    params = await request.json()
    await acquire_model_semaphore()
    embedding = worker.get_embeddings(params)
    background_tasks = create_background_tasks()
    return JSONResponse(content=embedding, background=background_tasks)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    global_rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size, pipeline_length=1)
    print("Model parallelism:", mpu.get_model_parallel_world_size())
    print("Global rank:", global_rank, "World size:", world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return global_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    global_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    max_shared_seq_len: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[global_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)

    model_args.vocab_size = tokenizer.n_words
    if model_args.qkv_dim != 0:
        print("Original n_heads:", model_args.n_heads)
        model_args.n_heads = (model_args.n_heads * model_args.qkv_dim) // model_args.dim
        print("New n_heads:", model_args.n_heads)
    model_args.max_shared_seq_len = max_shared_seq_len
    model_args.use_prefix_cache = True

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model.half()

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    add_model_args(parser)
    parser.add_argument("--model-name", type=str, required=True, help="Optional display name")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    global_rank, world_size = setup_model_parallel()

    max_seq_len = 2048
    max_batch_size = 1
    max_shared_seq_len = 0

    t0 = time.time()
    generator = load(
        args.model_path, args.tokenizer_path,
        global_rank, world_size,
        max_seq_len,
        max_batch_size,
        max_shared_seq_len,
    )
    t1 = time.time()
    loading_time = t1-t0
    global_rank = torch.distributed.get_rank()
    print("Model loading time on %d: " % global_rank, loading_time)

    def run_fake_evaluate(stop):
        while True:
            prompt = "Fake prompt"
            # sync the process with torch.distributed.barrier
            # TODO(zhiqings): find a better way to sync the processes, and avoid timeout in barrier
            # torch.distributed.barrier()
            fake_tensor = torch.zeros(1, dtype=torch.long, device="cuda")
            torch.distributed.recv(fake_tensor, src=0)

            # sync the prompt string across all processes
            prompt_tensor = torch.zeros(4096, dtype=torch.long, device="cuda") + generator.tokenizer.pad_id
            tokenized_prompt = generator.tokenizer.encode(prompt, bos=True, eos=False)
            prompt_tensor[:len(tokenized_prompt)] = torch.tensor(tokenized_prompt, dtype=torch.long, device="cuda")
            torch.distributed.broadcast(prompt_tensor, 0)
            t = prompt_tensor.tolist()
            try:
                t = t[: t.index(generator.tokenizer.pad_id)]
            except ValueError:
                pass
            prompt = generator.tokenizer.decode(t)

            temperature_tensor = torch.tensor([0.0], dtype=torch.float, device="cuda")
            torch.distributed.broadcast(temperature_tensor, 0)

            top_p_tensor = torch.tensor([0.0], dtype=torch.float, device="cuda")
            torch.distributed.broadcast(top_p_tensor, 0)

            max_new_tokens_tensor = torch.tensor([0], dtype=torch.long, device="cuda")
            torch.distributed.broadcast(max_new_tokens_tensor, 0)

            # time.sleep(0.1 * global_rank)
            output = generator.generate(
                [prompt],
                max_gen_len=max_new_tokens_tensor[0],
                temperature=temperature_tensor[0],
                top_p=top_p_tensor[0],
                stop=stop,
                unitoken_frequency_penalty=0.5,
            )[0]

    conv_template = get_default_conv_template(args.model_name)

    if global_rank != 0:
        run_fake_evaluate(stop=conv_template.stop_str)

    worker = DromeadryModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.no_register,
        generator,
        args.model_name,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
