# MIT License

# Copyright (c) 2021 Konstantin Dobler

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# KLib is a deep learning utility library.
from typing import Any, Hashable
import os
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.distributed import rank_zero_only
import wandb
import torch
import click
import re


class CustomWandbLogger(WandbLogger):
    @rank_zero_only
    def finalize(self, status: str) -> None:
        """Overwrite to enavle saving without directory structure"""
        # upload all checkpoints from saving dir
        if self._log_model:
            save_glob = os.path.join(self.save_dir, "*.ckpt")
            wandb.save(save_glob, os.path.dirname(save_glob))


class kdict(dict):
    """Wrapper around the native dict class that allows access via dot syntax and JS-like behavior for KeyErrors."""

    def __getattr__(self, key: Hashable) -> Any:
        try:
            return self[key]
        except KeyError:
            return None

    def __setattr__(self, key: Hashable, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: Hashable) -> None:
        del self[key]

    def __dir__(self):
        return self.keys()


def process_click_args(ctx: click.Context, cmd_args: dict) -> int:
    cmd_args = kdict(cmd_args)
    if cmd_args.offline:
        os.environ["WANDB_MODE"] = "dryrun"

    ######### Handle CUDA & GPU stuff #########
    num_gpus = len(cmd_args.gpus.split(',')
                   ) if cmd_args.gpus else 0
    if num_gpus > 0:
        # Assume machine has not more than 100 GPUs
        if not re.compile("^([0-9]|[1-9][0-9])(,([0-9]|[1-9][0-9]))*$").fullmatch(cmd_args.gpus):
            ctx.fail(
                f"invalid GPU string specified: \"{cmd_args.gpus}\". Expected format is a comma-seperated list of integers corresponding to GPU indices (execute nvidia-smi for more info)")
        if not torch.cuda.is_available():
            ctx.fail("GPUs were requested but machine has no CUDA!")
        os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.gpus
        torch.cudnn.enabled = True
        torch.cudnn.benchmark = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cmd_args.gpus = num_gpus

    return cmd_args
