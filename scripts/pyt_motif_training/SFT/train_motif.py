import argparse
import os
from functools import partial

import torch
from time import perf_counter
from datasets import load_dataset
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from kernels import get_kernel

activation = get_kernel("motif-technologies/activation")

def collate_fn(samples, tokenizer):
    inp, attn_mask = [], []
    for x in samples:
        message = x['messages']
        chat = tokenizer.apply_chat_template(message, tokenize=False)
        single_batch = tokenizer(
            chat,
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        inp.append(single_batch["input_ids"])
        attn_mask.append(single_batch["attention_mask"])

    return torch.concatenate(inp, dim=0), torch.concatenate(attn_mask, dim=0)

def model_patcher(model) -> torch.nn.Module:
    for child_name, child_module in model.named_children():
        if any([target in child_name for target in ["act_fn"]]):
            layer = activation.layers.PolyNorm(eps=child_module.eps)
            layer.weight.data = child_module.weight.data.clone()
            layer.bias.data = child_module.bias.data.clone()
            setattr(model, child_name, layer)
        elif any([target in child_name for target in ["subln", "input_layernorm", "post_attention_layernorm", "norm"]]):
            layer = activation.layers.RMSNorm(child_module.weight.shape[-1], eps=child_module.variance_epsilon)
            layer.weight.data = child_module.weight.data.clone()
            setattr(model, child_name, layer)
        else:
            model_patcher(child_module)
    
    return model


def main(args):
    accelerator = Accelerator()

    # this demo will use 100 samples of origin data
    # downloading the dataset will consume about 2.5 gb of your storage
    train_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:100]")
    total_iters = len(train_dataset) // (args.batchsize * accelerator.state.num_processes)

    # loading model
    model = AutoModelForCausalLM.from_pretrained(
        "Motif-Technologies/Motif-2.6b",
        trust_remote_code=True,
        _attn_implementation="flash_attention_2",
        device_map="cpu",
    ).to(torch.bfloat16)

    if args.use_kernels:
        model = model_patcher(model).to(torch.bfloat16)
    model = model.train()

    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Motif-Technologies/Motif-2.6b",
        trust_remote_code=True,
    )

    # defining dataloader, optimizer and scheduler
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        drop_last=True,
        pin_memory=False,
        shuffle=False,
        num_workers=accelerator.num_processes,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = LinearLR(optimizer=optimizer, total_iters=total_iters, last_epoch=-1)

    # wrap everything with accelerator
    # use accelerate config when running!
    # adopted from train_llama.py

    optimizer, lr_scheduler, dataloader, model = accelerator.prepare(
        optimizer, lr_scheduler, dataloader, model
    )

    # train loop starts
    if accelerator.is_main_process:
        logger.info("=====TRAIN START=====")

    for epoch in range(args.epochs):
        for idx, batch in enumerate(dataloader):
            if accelerator.is_main_process:
                st = perf_counter()
            loss = model(
                input_ids=batch[0],
                labels=batch[0],
                attention_mask=batch[1],
            ).loss

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            if accelerator.is_main_process:
                total_time = perf_counter() - st
                logger.info(
                        f"TRAIN | {epoch + 1}/{args.epochs} epochs | {(batch[0].numel() / total_time):<10.2f} TPS | {(idx + 1):<3}/{total_iters} steps | loss: {loss.detach().item():.5f} | lr: {lr_scheduler.get_lr()[0]}"
                )

            accelerator.wait_for_everyone()
    accelerator.end_training()

    logger.info("=====TRAIN COMPLETE=====")


if __name__ == "__main__":
    # define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=1)
    parser.add_argument("--batchsize", "-b", type=int, default=4)
    parser.add_argument("--lr", "-l", type=float, default=1e-5)
    parser.add_argument("--use-kernels", "-u", action='store_true')
    args = parser.parse_args()

    main(args)
