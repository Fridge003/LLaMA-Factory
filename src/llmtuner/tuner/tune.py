from typing import Any, Dict, List, Optional
import os
import resource

import torch
from torch.distributed import get_rank

from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers import TrainerCallback

from llmtuner.extras.callbacks import LogCallback
from llmtuner.extras.logging import get_logger
from llmtuner.tuner.core import get_train_args, get_infer_args, load_model_and_tokenizer
from llmtuner.tuner.pt import run_pt
from llmtuner.tuner.sft import run_sft
from llmtuner.tuner.rm import run_rm
from llmtuner.tuner.ppo import run_ppo
from llmtuner.tuner.dpo import run_dpo



logger = get_logger(__name__)

class MemoryProfileCallback(TrainerCallback):

    def __init__(self, profiling_step: int, log_path: str):
        self.profiling_step = profiling_step
        self.log_path = log_path
        self.step = 0
        self.rank = get_rank()
        try:
            os.remove(log_path)
        except Exception as e:
            pass

    def write_log(self, msg: str):
        with open(self.log_path, 'a') as f:
            f.write(f"RANK {self.rank}-----{msg}")     
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.write_log(f"Current CUDA memory usage before training: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB\n")
        self.write_log(f"Max CUDA memory usage before training: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB\n")
        self.write_log(f"Max CPU memory usage before training: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f} MB\n")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.step == self.profiling_step and self.rank == 0:
            self.write_log(f"Current CUDA memory usage after step {self.step}: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB\n")
            self.write_log(f"Max CUDA memory usage after step {self.step}: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB\n")
            self.write_log(f"Max CPU memory usage after step {self.step}: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f} MB\n")
            self.write_log(torch.cuda.memory_summary())
        torch.cuda.empty_cache()
        self.step += 1


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, data_args, training_args, finetuning_args, generating_args, memory_profiling_args = get_train_args(args)
    memory_profiling_step, memory_log_path = int(memory_profiling_args[1]), memory_profiling_args[3]
    callbacks = [LogCallback()] if callbacks is None else callbacks
    callbacks.append(MemoryProfileCallback(memory_profiling_step, memory_log_path))

    if finetuning_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "rm":
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "ppo":
        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError("Unknown task.")


def export_model(args: Optional[Dict[str, Any]] = None, max_shard_size: Optional[str] = "10GB"):
    model_args, _, finetuning_args, _ = get_infer_args(args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)
    model.config.use_cache = True
    model.save_pretrained(model_args.export_dir, max_shard_size=max_shard_size)
    try:
        tokenizer.padding_side = "left" # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)
    except:
        logger.warning("Cannot save tokenizer, please copy the files manually.")


if __name__ == "__main__":
    run_exp()
