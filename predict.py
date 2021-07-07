from pathlib import Path
from transformers.utils.dummy_pt_objects import MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING
from klib import CustomWandbLogger, process_click_args, int_sequence, UnlimitedNargsOption
import click
import os
import torch
import pytorch_lightning as pl
from src.dataloading import BasicDataModule
from src.model import TransformerClassifier
import wandb
WANDB_PROJECT = "paper-classification"
WANDB_ENTITY = "paper-judging"


@click.command()
@click.pass_context
@click.option('--workers', '-w', help="Number of workers", default=4, type=int)
@click.option('--epochs', '-e', help="Number of epochs to train", default=10, type=int)
@click.option('--batch-size', '-b', help="Batch size per GPU", default=8, type=int)
@click.option('--offline', help="Disbale wandb online syncing", is_flag=True)
@click.option('--seed', help="Specify seed", type=int, default=None)
@click.option('--gpu', '-g', type=int, help="Specify the GPU to use. If omitted, use the CPU.")
@click.option('--input', '-i', required=True, multiple=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--checkpoint', '-c', type=click.Path(exists=True, dir_okay=False))
@click.option('--use-wandb', '--wandb')
# 122bvd7r
def main(ctx, **cmd_args):
    cmd_args = process_click_args(ctx, cmd_args)
    if cmd_args.use_wandb:
        print("Downloading pkl from wandb")
        model_file = wandb.restore(cmd_args.checkpoint or "network-snapshot",
                                   run_path=f"paper-judging/paper-classification/{cmd_args.use_wandb}")
        cmd_args.checkpoint = model_file.name
    cmd_args.seed = pl.seed_everything(workers=True, seed=cmd_args.seed)
    print(cmd_args)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    device = torch.device(
        f"cuda:{cmd_args.gpu}") if cmd_args.gpu else torch.device("cpu")
    model = TransformerClassifier.load_from_checkpoint(
        cmd_args.checkpoint, map_location=device)
    for input in cmd_args.input:
        with input.open('r') as f:
            abstract = f.read()
        logits = model(abstract)

        accepted = torch.sigmoid(logits) > 0.5
        print(input, accepted, logits)


if __name__ == '__main__':
    # Needed against multiprocessing error in Colab
    __spec__ = None
    main()
