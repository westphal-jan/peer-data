from typing import List
import nlpaug.augmenter.word as naw
import nltk
import nlpaug.augmenter.sentence as nas
import pytorch_lightning as pl
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset,  random_split, Subset, ConcatDataset
import glob
import json
import torch
from catalyst.data.sampler import DistributedSamplerWrapper
import numpy as np
import random
from tqdm import tqdm


class BasicDataModule(pl.LightningDataModule):
    def __init__(self, data_dirs: str, batch_size: int, workers: int, ddp: bool = False, fast_debug: bool = False, augmentation_datasets: List[str] = []):
        super().__init__()
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.workers = workers
        self.fast_debug = fast_debug
        self.ddp = ddp
        self.augmentation_datasets = augmentation_datasets

    def setup(self, stage):
        self._file_paths = glob.glob(f"{self.data_dirs[0]}/*.json")
        complete_data = PaperDataset(self._file_paths)
        print(len(self._file_paths))

        # Get random index split for train/val/test.
        idx = list(range(len(self._file_paths)))
        random.shuffle(idx)
        total_len = len(idx)
        train_len, val_len = int(0.8*total_len), int(0.1*total_len)
        train_idx = idx[:train_len]
        val_idx = idx[train_len:(train_len + val_len)]
        test_idx = idx[(train_len + val_len):]

        print(
            f"Perform train/val/test split: train {train_len}, val {val_len}, test {len(test_idx)}")

        self.train_set, self.val_set, self.test_set = Subset(complete_data, train_idx), Subset(
            complete_data, val_idx), Subset(complete_data, test_idx)

        # We need to be careful and only train on augmentations of abstracts that are in the train set.
        for aug in self.augmentation_datasets:
            print("Using augmentation dataset", aug)
            backtranslation_paths = glob.glob(
                f"./data/{aug}/*.json")
            # print(backtranslation_paths[:10], self._file_paths[:10])
            aug_data = PaperDataset(backtranslation_paths)
            aug_train_set = Subset(aug_data, train_idx)
            self.train_set = ConcatDataset([self.train_set, aug_train_set])

        
        print("Train set len", len(self.train_set))

    def train_dataloader(self) -> DataLoader:
        labels = [label for abstract, label in self.train_set]

        # Sanity check
        for i, (text, label) in enumerate(self.train_set):
            if not label == labels[i]:
                print(label, labels[i])

        # Do oversampling of minority class
        number_rejected, number_accepted = np.bincount(labels)
        print("Accepted:", number_accepted, "Rejected:", number_rejected)

        sampler = None

        if number_rejected > number_accepted:
            class_weights = (1, number_rejected / number_accepted)
            minority_class = "Accepted papers"
        else:
            class_weights = (number_accepted / number_rejected, 1)
            minority_class = "Rejected papers"

        sample_weights = [class_weights[label] for label in labels]
        print(
            f"Oversampling minority class ({minority_class}) with ratio: (Rejected) {class_weights[0]}:{class_weights[1]} (Accepted)")
        sampler = data.WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights))

        if self.ddp:
            sampler = DistributedSamplerWrapper(sampler)
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.workers, sampler=sampler, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True)


# # Synonym Augmenter
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')


class PaperDataset(Dataset):
    def __init__(self, file_paths, augment=True) -> None:
        super().__init__()
        self._file_paths = file_paths
        self.augment = augment
        self.papers = []
        for i, file_path in tqdm(enumerate(self._file_paths), leave=False):
            with open(file_path) as f:
                paper_json = json.load(f)
                accepted = paper_json["review"]["accepted"]
                abstract = paper_json["review"]["abstract"]
                self.papers.append(
                    {"accepted": int(accepted), "abstract": abstract})

    def __len__(self):
        return len(self._file_paths)

    def _augment(self, text):
        wordnet_synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_min=5,
                                             aug_max=50, aug_p=0.1)

        try:
            text = wordnet_synonym_aug.augment(
                text).replace(' - ', '-')
        except Exception as e:
            return text

        # print(augmented_text, text)
        return text

    def __getitem__(self, index):
        abstract = self.papers[index]['abstract']
        if self.augment:
            abstract = self._augment(abstract)
        return abstract, torch.tensor(self.papers[index]['accepted'])
