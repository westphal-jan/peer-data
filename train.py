from datetime import datetime

from pytorch_lightning.utilities.distributed import rank_zero_only
from klib.misc import push_file_to_wandb
from pathlib import Path
from klib import CustomWandbLogger, process_click_args, int_sequence, UnlimitedNargsOption
import click
import os
import wandb
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
from src.dataloading import BasicDataModule
from src.model import TransformerClassifier
import subprocess
from dotenv import load_dotenv

WANDB_PROJECT = "paper-classification"
WANDB_ENTITY = "paper-judging"


def get_wandb_api_key():
    try:
        api_key = os.environ.get("WANDB_API_KEY")
    except Exception as e:
        print(e)
        api_key = None
    return api_key


def try_wandb_login():
    WAND_API_KEY = get_wandb_api_key()
    if WAND_API_KEY:
        try:
            subprocess.run(["wandb", "login", WAND_API_KEY], check=True)
            return True
        except Exception as e:
            print(e)
            return False
    else:
        print("WARNING: No wandb API key found, this run will NOT be logged to wandb.")
        input("Press any key to continue...")
        return False


def start_wandb_logging(cfg, model, project):
    if try_wandb_login():
        wandb.init(project=project, entity=WANDB_ENTITY,
                   name=cfg.run_name)
        wandb.config.update(cfg)
        wandb.watch(model, log="all")


@click.command()
@click.pass_context
@click.option('--workers', '-w', help="Number of workers", default=4, type=int)
@click.option('--epochs', '-e', help="Number of epochs to train", default=10, type=int)
@click.option('--batch-size', '-b', help="Batch size per GPU", default=8, type=int)
@click.option('--offline', help="Disable wandb online syncing", is_flag=True)
@click.option('--seed', help="Specify seed", type=int, default=None)
@click.option('--gpus', '-g', type=int_sequence, cls=UnlimitedNargsOption, help="Specify the GPU indices to use. If `-1`, try to use all available GPUs. If omitted, use the CPU.")
@click.option('--datasets', '-d', help="Datasets to train on", required=True, multiple=True, type=click.Path(exists=True, writable=True, file_okay=False))
@click.option('--results-dir', '-r', type=click.Path(writable=True, file_okay=False), default=Path("./results"))
@click.option('--run-name', '-n', default=datetime.now().strftime('%d-%m-%Y_%H_%M_%S'))
def main(ctx, **cmd_args):
    cmd_args = process_click_args(ctx, cmd_args)

    manual_seed_specified = cmd_args.seed is not None
    cmd_args.seed = pl.seed_everything(workers=True, seed=cmd_args.seed)
    cmd_args.results_dir = cmd_args.results_dir / cmd_args.run_name

    print(cmd_args)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    os.makedirs(cmd_args.results_dir, exist_ok=True)
    dm = BasicDataModule(
        data_dirs=cmd_args.datasets, workers=cmd_args.workers, batch_size=cmd_args.batch_size)
    model = TransformerClassifier()

    load_dotenv()
    if rank_zero_only.rank == 0:
        start_wandb_logging(cmd_args, model, WANDB_PROJECT)
    wandb_logger = CustomWandbLogger(name=cmd_args.run_name, project=WANDB_PROJECT, experiment=wandb.run,
                                     entity=WANDB_ENTITY, job_type='train', log_model=False)
    checkpoint_callback = ModelCheckpoint(
        dirpath=cmd_args.results_dir, save_last=True, every_n_val_epochs=1)
    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=cmd_args.epochs,
                         progress_bar_refresh_rate=1,
                         gpus=cmd_args.gpus,
                         accelerator=cmd_args.accelerator,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback],
                         benchmark=not manual_seed_specified,
                         deterministic=manual_seed_specified,
                         replace_sampler_ddp=False)

    trainer.fit(model, dm)
    if rank_zero_only.rank == 0:
        push_file_to_wandb(f"{str(cmd_args.results_dir)}/*.ckpt")
    trainer.test(model=model, datamodule=dm, ckpt_path=None)


if __name__ == '__main__':
    # Needed against multiprocessing error in Colab
    __spec__ = None
    main()
