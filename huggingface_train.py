from datetime import datetime
import json
import glob
from klib.cmd_line_parsing import process_click_args
from pathlib import Path
import click
from dotenv import load_dotenv

import wandb
from train import WANDB_ENTITY, try_wandb_login
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

class PaperDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def read_dataset(data_dir: Path):
    file_paths = glob.glob(f"{data_dir}/*.json")
    texts = []
    labels = []
    for i, file_path in enumerate(file_paths):
        with open(file_path) as f:
            paper_json = json.load(f)
            accepted = paper_json["review"]["accepted"]
            abstract = paper_json["review"]["abstract"]
            
            texts.append(abstract)
            labels.append(int(accepted))
    return texts, labels

@click.command()
@click.pass_context
@click.option('--datasets', '-d', help="Datasets to train on", default=[Path("data/original")], multiple=True, type=click.Path(exists=True, writable=True, file_okay=False))
@click.option('--run-name', '-n', default=datetime.now().strftime('%d-%m-%Y_%H_%M_%S'))
def main(ctx, **args):
    load_dotenv()
    args = process_click_args(ctx, args)
    if try_wandb_login():
        wandb.init(project="huggingface", entity=WANDB_ENTITY, name=args.run_name)

    data_dir = Path(args.datasets[0])
    train_texts, train_labels = read_dataset(data_dir)

    num_accepted = len(list(filter(lambda x: x == 1, train_labels)))
    num_not_accepted = len(list(filter(lambda x: x == 0, train_labels)))

    print(num_accepted, num_not_accepted)

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = PaperDataset(train_encodings, train_labels)
    val_dataset = PaperDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        eval_steps=50,
        evaluation_strategy="steps",
        logging_first_step=True,
    )

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    trainer.evaluate()
    trainer.train()

if __name__ == "__main__":
    main()
