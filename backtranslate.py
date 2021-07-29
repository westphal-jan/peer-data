import glob
import nlpaug.augmenter.word as naw
import json
import copy
from tqdm import tqdm
back_translation_aug = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de',
    to_model_name='facebook/wmt19-de-en',
    device="cpu",
    max_length=512
)

paths = glob.glob(f"./data/original/*.json")

for path in tqdm(paths):
    with open(path, 'r') as f:
        paper_json = json.load(f)
        new_json = copy.deepcopy(paper_json)
        abstract = paper_json["review"]["abstract"]
        backtrans = back_translation_aug.augment(abstract)
        new_json['review']['abstract'] = backtrans
    filename = path.split('/')[-1]
    print(abstract, '\n\n', backtrans)
    with open('./data/back-translation/' + filename, 'w') as f:
        json.dump(new_json, f, ensure_ascii=False)
