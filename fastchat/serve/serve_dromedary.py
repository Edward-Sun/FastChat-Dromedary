import gc
import os
import queue
import threading

import torch


@torch.inference_mode()
def dromedary_generate_stream(
    model, tokenizer, params, device, context_len=2048, stream_interval=2, top_p=0.95,
):
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    echo = params.get("echo", True)
    assert echo is False, "echo is not supported in stream mode"

    world_size = int(os.environ.get("WORLD_SIZE", -1))
    # sync the process with torch.distributed.send
    for i in range(world_size):
        if i != 0:
            torch.distributed.send(torch.tensor([1]), dst=i)

    # sync the prompt string across all processes, max_len=4096
    prompt_tensor = torch.zeros(4096, dtype=torch.long, device="cuda") + tokenizer.pad_id
    tokenized_prompt = tokenizer.encode(prompt, bos=True, eos=False)

    prompt_tensor[:len(tokenized_prompt)] = torch.tensor(tokenized_prompt, dtype=torch.long, device="cuda")
    torch.distributed.broadcast(prompt_tensor, 0)
    t = prompt_tensor.tolist()
    try:
        t = t[: t.index(tokenizer.pad_id)]
    except ValueError:
        pass
    prompt = tokenizer.decode(t)

    temperature_tensor = torch.tensor([temperature], dtype=torch.float, device="cuda")
    torch.distributed.broadcast(temperature_tensor, 0)

    top_p_tensor = torch.tensor([top_p], dtype=torch.float, device="cuda")
    torch.distributed.broadcast(top_p_tensor, 0)

    max_new_tokens_tensor = torch.tensor([max_new_tokens], dtype=torch.long, device="cuda")
    torch.distributed.broadcast(max_new_tokens_tensor, 0)

    def generate_output(prompt, max_gen_len, temperature, top_p, stream_queue):
        output = model(
            [prompt],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            stop="### User",
            unitoken_frequency_penalty=0.5,
            stream_queue=stream_queue,
        )[0]

    stream_queue = queue.Queue()
    generate_thread = threading.Thread(target=generate_output, args=(
        prompt, max_new_tokens_tensor[0], temperature_tensor[0], top_p_tensor[0], stream_queue))
    generate_thread.start()

    i = 0
    while True:
        words = stream_queue.get()
        if words is None:
            break
        output = tokenizer.decode(words[0])

        i += 1
        if i % stream_interval == 0 or i == max_new_tokens - 1:
            rfind_start = 0
            if stop_str:
                pos = output.rfind(stop_str, rfind_start)
                if pos != -1:
                    output = output[:pos - 1]

            if stop_str == "### User":
                if output.endswith("\n\n###") or output.endswith("\n\n##") or output.endswith("\n\n#"):
                    output = output.rsplit("\n\n", 1)[0]

            yield output.strip()

    gc.collect()
    torch.cuda.empty_cache()
