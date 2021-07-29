import glob
import nlpaug.augmenter.word as naw
import json
import copy
from tqdm import tqdm
import os

BATCH_SIZE = 16
START_QUARTER = 2
map = {0: 0, 1:1, 2:3, 3:4}
back_translation_aug = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de',
    to_model_name='facebook/wmt19-de-en',
    device=f"cuda:{map[START_QUARTER]}",
    max_length=512,
    batch_size=BATCH_SIZE
)

paths = glob.glob(f"./data/original/*.json")
paths = paths[int(len(paths) / 4)*START_QUARTER:]
batch = []
for path in tqdm(paths):
    filename = path.split('/')[-1]
    if os.path.exists('./data/back-translations/' + filename):
        continue

    with open(path, 'r') as f:
        paper_json = json.load(f)
        batch.append((paper_json, filename))
    if len(batch) == BATCH_SIZE:
        abstracts = [paper["review"]["abstract"] for paper, _ in batch]
        backtranslated_abstracts = back_translation_aug.augment(abstracts)
        for i, (paper, filename) in enumerate(batch):
            paper['review']['abstract'] = backtranslated_abstracts[i]
            with open('./data/back-translations/' + filename, 'w') as f:
                json.dump(paper, f)
        batch = []
if len(batch) > 0:
    abstracts = [paper["review"]["abstract"] for paper, _ in batch]
    backtranslated_abstracts = back_translation_aug.augment(abstracts)
    for i, (paper, filename) in enumerate(batch):
        paper['review']['abstract'] = backtranslated_abstracts[i]
        with open('./data/back-translations/' + filename, 'w') as f:
            json.dump(paper, f)
    # print(abstract, '\n\n', backtrans)
