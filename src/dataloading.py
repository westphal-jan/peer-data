import pytorch_lightning as pl
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset, random_split
import glob
import json
import torch
from catalyst.data.sampler import BalanceClassSampler, DistributedSamplerWrapper
import numpy as np
class BasicDataModule(pl.LightningDataModule):
    def __init__(self, data_dirs: str, batch_size: int, workers: int, ddp: bool = False, fast_debug: bool = False):
        super().__init__()
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.workers = workers
        self.fast_debug = fast_debug
        self.ddp = ddp

    def setup(self, stage):
        self._file_paths = glob.glob(f"{self.data_dirs[0]}/*.json")
        complete_data = PaperDataset(self._file_paths)
        print(len(self._file_paths))
        total_len = len(complete_data)
        train_len, val_len = int(0.8*total_len), int(0.1*total_len)
        test_len = total_len - (train_len + val_len)
        print(
            f"Perform train/val/test split: train {train_len}, val {val_len}, test {test_len}")
        self.train_set, self.val_set, self.test_set = random_split(complete_data,
                                                                   [train_len, val_len, test_len])

    def train_dataloader(self) -> DataLoader:
        labels = [ label for abstract, label in self.train_set]
        for i, (text, label) in enumerate(self.train_set):
            if not label == labels[i]:
                print(label, labels[i])
        # print(len(labels), labels)
        # sampler = BalanceClassSampler(
        #     labels=label_callback(self.train_set))
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



class PaperDataset(Dataset):
    def __init__(self, file_paths) -> None:
        super().__init__()
        self._file_paths = file_paths
        self.papers = []
        for i, file_path in enumerate(self._file_paths):
            with open(file_path) as f:
                paper_json = json.load(f)
                accepted = paper_json["review"]["accepted"]
                abstract = paper_json["review"]["abstract"]
                self.papers.append({"accepted": int(accepted), "abstract": abstract})

    def __len__(self):
        return len(self._file_paths)

    def __getitem__(self, index):
        return self.papers[index]['abstract'], torch.tensor(self.papers[index]['accepted'])

def label_callback(dataset: Subset[PaperDataset]):
    labels = []
    indices = dataset.indices

    for i, file_path in enumerate(dataset.dataset._file_paths):
        if i in indices:
            with open(file_path) as f:
                paper_json = json.load(f)
                accepted = paper_json["review"]["accepted"]
                labels.append(int(accepted))
    return labels
