import glob
import numpy as np


file_paths = glob.glob(f"./data/original/*.json")
print(len(file_paths))
idx = np.arange(len(file_paths))

rnd = np.random.RandomState(42)
rnd.shuffle(idx)

total_len = len(idx)
train_len, val_len = int(0.8*total_len), int(0.1*total_len)
train_file_paths = file_paths[:train_len]
val_file_paths = file_paths[train_len:(train_len + val_len)]
test_file_paths = file_paths[(train_len + val_len):]

print(len(train_file_paths), len(val_file_paths), len(test_file_paths))
for dataset, paths in [("train", train_file_paths), ("val", val_file_paths), ("test", test_file_paths)]:
    filenames = list(map(lambda x: x.split("/")[-1], paths))
    with open(f"data/{dataset}.txt", "w+") as f:
        f.write("\n".join(filenames))
