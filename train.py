import torch
from datetime import datetime
from glob import glob

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
import re
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
        wandb.watch(model, log="all", log_freq=200)


def postfixify_run_dir(results_dir: Path, run_name: str):
    if not os.path.isdir(results_dir / run_name):
        return results_dir / run_name
    postfix_number = 1
    while os.path.isdir(results_dir / f"{run_name}-{postfix_number}"):
        postfix_number += 1
    return results_dir / f"{run_name}-{postfix_number}"


on_disk_agus = ['back-translations', 'insert-distilbert', 'substitute-distilbert',
                'hyper-insert-distilbert', 'hyper-substitute-distilbert']

on_disk_agus = on_disk_agus + ['sentence-gpt', 'insert-glove', 'substitute-glove']

@click.command()
@click.pass_context
@click.option('--workers', '-w', help="Number of workers", default=16, type=int)
@click.option('--epochs', '-e', help="Number of epochs to train", default=10, type=int)
@click.option('--batch-size', '-b', help="Batch size per GPU", default=8, type=int)
@click.option('--lr', help="Initial learning rate", default=2e-5, type=float)
@click.option('--weight-decay', '--wd', help="Weight decay", default=0.01, type=float)
@click.option('--dropout-p', help="Dropout probability for classifier layers", default=0., type=float)
@click.option('--offline', help="Disable wandb online syncing", is_flag=True)
@click.option('--seed', help="Specify seed", type=int, default=None)
@click.option('--gpus', '-g', type=int_sequence, cls=UnlimitedNargsOption, help="Specify the GPU indices to use. If `-1`, try to use all available GPUs. If omitted, use the CPU.")
@click.option('--datasets', '-d', help="Datasets to train on", required=True, multiple=True, type=click.Path(exists=True, writable=True, file_okay=False))
@click.option('--results-dir', '-r', type=click.Path(writable=True, file_okay=False), default=Path("./results"))
@click.option('--run-name', '-n', default=datetime.now().strftime('%d-%m-%Y_%H_%M_%S'))
@click.option('--fast-dev', '--fd', is_flag=True)
@click.option('--aug-datasets', '-a', multiple=True, type=click.Choice(on_disk_agus + ['none']), default=on_disk_agus,
              help="specify the additional augmented datasets to use for training (e.g. -a=back-translations -a=insert-distilbert")
@click.option('--dynamic-augmentations', '-da', multiple=True, type=click.Choice(['wordnet', 'insert-glove', 'substitute-glove', 'insert-word2vec', 'substitute-word2vec']),
              help="specify the additional 'on-the-fly' augmentations (e.g. -da=wordnet -da=insert-glove")
@click.option('--no-oversampling', is_flag=True)
@click.option('--accepted-class-weight', type=float, help="weight of accepted class for binary cross entropy loss", default=1.)
@click.option('--simple-linear-layer', is_flag=True)
@click.option('--f1-loss', is_flag=True)
@click.option('--test', is_flag=True)
@click.option('--wandb-project', default=WANDB_PROJECT)
def main(ctx, **cmd_args):
    cmd_args = process_click_args(ctx, cmd_args)

    manual_seed_specified = cmd_args.seed is not None
    cmd_args.seed = pl.seed_everything(workers=True, seed=cmd_args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    model = TransformerClassifier(
        lr=cmd_args.lr, accepted_class_weight=cmd_args.accepted_class_weight, weight_decay=cmd_args.weight_decay, dropout_p=cmd_args.dropout_p,
        simple_linear_layer=cmd_args.simple_linear_layer, f1_loss=cmd_args.f1_loss)
    load_dotenv()
    if rank_zero_only.rank == 0:
        start_wandb_logging(cmd_args, model, cmd_args.wandb_project)
        cmd_args.results_dir = postfixify_run_dir(
            cmd_args.results_dir, cmd_args.run_name)
        assert not os.path.exists(cmd_args.results_dir)
        os.makedirs(cmd_args.results_dir, exist_ok=True)
        print(cmd_args)

    if len(cmd_args.aug_datasets) == 1 and cmd_args.aug_datasets[0] == 'none':
        cmd_args.aug_datasets = []
    dm = BasicDataModule(
        data_dirs=cmd_args.datasets, workers=cmd_args.workers, batch_size=cmd_args.batch_size, ddp=cmd_args.accelerator == "ddp", augmentation_datasets=cmd_args.aug_datasets, dynamic_augmentations=cmd_args.dynamic_augmentations, no_oversampling=cmd_args.no_oversampling)

    wandb_logger = CustomWandbLogger(name=cmd_args.run_name, project=cmd_args.wandb_project, experiment=wandb.run,
                                     entity=WANDB_ENTITY, job_type='train', log_model=False)
    checkpoint_callback = ModelCheckpoint(
        dirpath=cmd_args.results_dir, every_n_val_epochs=1, filename="model-snaphot-best-epoch-{epoch}-loss-{val/loss:.2f}-f1-{val/f1:.2f}", monitor='val/loss', mode='min', auto_insert_metric_name=False)
    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=cmd_args.epochs,
                         progress_bar_refresh_rate=1,
                         gpus=cmd_args.gpus,
                         accelerator=cmd_args.accelerator,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback],
                         benchmark=not manual_seed_specified,
                         deterministic=manual_seed_specified,
                         gradient_clip_val=1,
                         #  stochastic_weight_avg=True, # leads to error in test phase : AttributeError: 'TransformerClassifier' object has no attribute '_parameters'
                         fast_dev_run=cmd_args.fast_dev)

    trainer.fit(model, dm)
    if rank_zero_only.rank == 0:
        push_file_to_wandb(
            f"{str(cmd_args.results_dir)}/*.ckpt")
    if cmd_args.test:
        torch.distributed.destroy_process_group()
        if trainer.global_rank == 0:
            best_model = TransformerClassifier.load_from_checkpoint(
                checkpoint_callback.best_model_path)
            trainer = pl.Trainer(
                         gpus=1,
                         logger=wandb_logger,
                         benchmark=not manual_seed_specified,
                         deterministic=manual_seed_specified,
                         fast_dev_run=cmd_args.fast_dev)
            trainer.test(model=best_model,
                         datamodule=dm, ckpt_path=None)


if __name__ == '__main__':
    main()
