import glob
import nlpaug.augmenter.word as naw
import json
import copy
from tqdm import tqdm
import os
import click
import multiprocessing as mp


def process_chunk(paths, gpu_idx, augmentation_name, batch_size):
    augmentation = get_augment(augmentation_name, gpu_idx, batch_size)
    batch = []
    out_dir = f'./data/{augmentation_name}/'
    os.makedirs(out_dir, exist_ok=True)
    for path in tqdm(paths, pos=gpu_idx):
        filename = path.split('/')[-1]
        if os.path.exists(out_dir + filename):
            continue

        with open(path, 'r') as f:
            paper_json = json.load(f)
            batch.append((paper_json, filename))
        if len(batch) == batch_size:
            abstracts = [paper["review"]["abstract"] for paper, _ in batch]

            augmented_abstracts = augmentation.augment(abstracts)

            for i, (paper, filename) in enumerate(batch):
                paper['review']['abstract'] = augmented_abstracts[i]
                with open(out_dir + filename, 'w') as f:
                    json.dump(paper, f)
            batch = []
    if len(batch) > 0:
        abstracts = [paper["review"]["abstract"] for paper, _ in batch]
        augmented_abstracts = augmentation.augment(abstracts)
        for i, (paper, filename) in enumerate(batch):
            paper['review']['abstract'] = augmented_abstracts[i]
            with open(out_dir + filename, 'w') as f:
                json.dump(paper, f)


def get_augment(augmentation_name, gpu_idx, batch_size):
    if augmentation_name == "substitute-distilbert":
        return naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', aug_p=0.1, batch_size=batch_size, device=f"cuda:{gpu_idx}")
    if augmentation_name == "insert-distilbert":
        return naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', aug_p=0.1, action='insert', batch_size=batch_size, device=f"cuda:{gpu_idx}")
    if augmentation_name == "back-translations":
        return naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en',
                                      device=f"cuda:{gpu_idx}",
                                      max_length=512,
                                      batch_size=batch_size
                                      )


def chunker_list(seq, size):
    return list((seq[i::size] for i in range(size)))


@click.command()
@click.pass_context
@click.option('--batch-size', '-b', help="Batch size per GPU", default=32, type=int)
@click.option('--gpus', help="comma sepreated list of gpus to use")
@click.option('--augmentation')
def main(ctx, batch_size, gpus, augmentation):
    gpu_idxs = [int(gpu) for gpu in gpus.split(',')]

    paths = glob.glob(f"./data/original/*.json")
    split_paths = chunker_list(paths, len(gpu_idxs))
    d_len = len(split_paths)
    with mp.Pool(len(gpu_idxs)) as pool:
        pool.starmap(process_chunk, zip(split_paths, gpu_idxs, [
                     augmentation]*d_len, [batch_size]*d_len))

        # print(abstract, '\n\n', backtrans)


if __name__ == "__main__":
    main()
