import pytorch_lightning as pl
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset, random_split
import glob
import json
import torch
from catalyst.data.sampler import BalanceClassSampler, DistributedSamplerWrapper

class BasicDataModule(pl.LightningDataModule):
    def __init__(self, data_dirs: str, batch_size: int, workers: int, fast_debug: bool = False):
        super().__init__()
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.workers = workers
        self.fast_debug = fast_debug

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
        labels = label_callback(self.train_set)
        print(len(labels), labels)
        sampler = BalanceClassSampler(
            labels=label_callback(self.train_set), mode='upsampling')
        ddp_sampler = DistributedSamplerWrapper(sampler)
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.workers, sampler=ddp_sampler, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True)



class PaperDataset(Dataset):
    def __init__(self, file_paths) -> None:
        super().__init__()
        self._file_paths = file_paths

    def __len__(self):
        return len(self._file_paths)

    def __getitem__(self, index):

        with open(self._file_paths[index]) as f:
            paper_json = json.load(f)

        abstract = paper_json["review"]["abstract"]
        accepted = paper_json["review"]["accepted"]
        return abstract, torch.tensor(int(accepted))

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
