import glob
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import json
import copy
from tqdm import tqdm
import os
import click
import multiprocessing as mp


def process_chunk(paths, gpu_idx, pos_idx, augmentation_name, batch_size):
    augmentation = get_augment(augmentation_name, gpu_idx, batch_size)
    batch = []
    out_dir = f'./data/{augmentation_name}/'
    os.makedirs(out_dir, exist_ok=True)
    for path in tqdm(paths, position=pos_idx):
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
    extra_kwargs = dict(batch_size=batch_size, device=f"cuda:{gpu_idx}")
    if augmentation_name == "substitute-distilbert":
        return naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', aug_p=0.1, aug_max=None, **extra_kwargs)
    if augmentation_name == "hyper-substitute-distilbert":
        return naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', aug_p=0.3, aug_max=None, **extra_kwargs)
    if augmentation_name == "insert-distilbert":
        return naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', aug_p=0.1, aug_max=None, action='insert', **extra_kwargs)
    if augmentation_name == "hyper-insert-distilbert":
        return naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', aug_p=0.3, aug_max=None, action='insert', **extra_kwargs)
    if augmentation_name == "sentence-gpt":
        return nas.ContextualWordEmbsForSentenceAug(model_path='distilgpt2', max_length=512, **extra_kwargs)
    if augmentation_name == "back-translations":
        return naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en',
                                      max_length=512,
                                      **extra_kwargs
                                      )


def split_list_into_chunks(seq, num_chunks):
    return list((seq[i::num_chunks] for i in range(num_chunks)))


@click.command()
@click.pass_context
@click.option('--batch-size', '-b', help="Batch size per GPU", default=32, type=int)
@click.option('--gpus', help="comma sepreated list of gpus to use")
@click.option('--augmentation')
def main(ctx, batch_size, gpus, augmentation):
    gpu_idxs = [int(gpu) for gpu in gpus.split(',')]

    paths = glob.glob(f"./data/original/*.json")
    split_paths = split_list_into_chunks(paths, len(gpu_idxs))
    d_len = len(split_paths)
    with mp.Pool(len(gpu_idxs)) as pool:
        pool.starmap(process_chunk, zip(split_paths, gpu_idxs, list(range(len(gpu_idxs))), [
                     augmentation]*d_len, [batch_size]*d_len))

        # print(abstract, '\n\n', backtrans)


if __name__ == "__main__":
    main()
