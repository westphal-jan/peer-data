import glob
import numpy as np
import json

file_paths = glob.glob(f"./data/original/*.json")
print(len(file_paths))
idx = np.arange(len(file_paths))

rnd = np.random.RandomState(2)
rnd.shuffle(file_paths)

total_len = len(idx)
train_len, val_len = int(0.7*total_len), int(0.15*total_len)
train_file_paths = file_paths[:train_len]
val_file_paths = file_paths[train_len:(train_len + val_len)]
test_file_paths = file_paths[(train_len + val_len):]

print(len(train_file_paths), len(val_file_paths), len(test_file_paths))
for dataset, paths in [("train", train_file_paths), ("val", val_file_paths), ("test", test_file_paths)]:
    filenames = list(map(lambda x: x.split("/")[-1], paths))
    with open(f"data/{dataset}.txt", "w+") as f:
        f.write("\n".join(filenames))


def read_dataset(data_dir, restrict_file=None):
    file_paths = glob.glob(f"{data_dir}/*.json")
    if restrict_file:
        with open(restrict_file, "r") as f:
            filter_file_names = f.read().splitlines()
            file_paths = [p for p in file_paths if p.split("/")[-1] in filter_file_names]

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

data_dir = "data/original"

train_texts, train_labels = read_dataset(data_dir, restrict_file="data/train.txt")
val_texts, val_labels = read_dataset(data_dir, restrict_file="data/val.txt")
test_texts, test_labels = read_dataset(data_dir, restrict_file="data/test.txt")

def label_distribution(labels):
    num_rejected, num_accepted = labels.count(0), labels.count(1)
    print(num_rejected, num_rejected / len(labels), num_accepted, num_accepted / len(labels))

label_distribution(train_labels)
label_distribution(val_labels)
label_distribution(test_labels)
