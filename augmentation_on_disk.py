import glob
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import json
import copy
from tqdm import tqdm
import os
import click
import multiprocessing as mp
import numpy as np
import math

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

def augment(to_augment, augmentation, batch_size, pos_idx):
    num_batches = math.ceil(len(to_augment) / batch_size)
    print("Num batches:", num_batches)
    batches = np.array_split(to_augment, num_batches)
    print(len(batches))

    augmented = []
    for _to_augment in tqdm(batches, position=pos_idx):
        _augmented = augmentation.augment(_to_augment.tolist())
        augmented.extend(_augmented)
    return augmented


def section_process_chunk(paths, gpu_idx, pos_idx, augmentation_name, appendix, batch_size):
    augmentation = get_augment(augmentation_name, gpu_idx, batch_size)
    out_dir = f'./data/{augmentation_name}-{appendix}/' if appendix else f"./data/{augmentation_name}/"
    os.makedirs(out_dir, exist_ok=True)

    submissions = []
    sections = []
    for path in tqdm(paths, position=pos_idx):
        with open(path, 'r') as f:
            paper_json = json.load(f)
            submissions.append(paper_json)
            _sections = paper_json["pdf"]["metadata"]["sections"]
            assert _sections
            _sections = list(map(lambda x: x["text"], _sections))
            sections.append(_sections)
    
    flattened_sections = np.hstack(sections).tolist()
    # flattened_sections = [" ".join(s.split(" ")[:400]) for s in flattened_sections]
    print(len(flattened_sections))
    num_sections = list(map(len, sections))
    idx = np.cumsum(num_sections)

    augmented = augment(flattened_sections, augmentation, batch_size, gpu_idx)
    augmented_sections = np.split(augmented, idx)
    augmented_sections = [i.tolist() for i in augmented_sections if len(i)]
    
    for path, submission, aug_sections in zip(paths, submissions, augmented_sections):
        _sections = submission["pdf"]["metadata"]["sections"]
        assert len(_sections) == len(aug_sections)
        for i in range(len(_sections)):
            _sections[i]["text"] = aug_sections[i]
        submission["pdf"]["metadata"]["sections"] = _sections

        filename = path.split("/")[-1]
        with open(out_dir + filename, 'w') as f:
            json.dump(submission, f)


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
    raise ValueError(f"Augmentation {augmentation_name} is not defined")

def filter_by_outcome(file_paths, outcome):
    assert outcome != None
    filtered_paths = []
    for file_path in file_paths:
        with open(file_path) as f:
            paper_json = json.load(f)
            accepted = paper_json["review"]["accepted"]
            if accepted == outcome:
                filtered_paths.append(file_path)
    return filtered_paths

def chunker_list(seq, size):
    return list((seq[i::size] for i in range(size)))

#------------------------------------------------------------------------------------------------------------------------------------------------------
# Note that in order for nlpaug to work truncation needs to be turned on in the backtranslation model as our texts are rather long.
# Therefore change the line:
# + tokenized_texts = tokenizer(data, padding=True, return_tensors='pt')
# (https://github.com/makcedward/nlpaug/blob/5480074c61978e735f21af165f9ace73e8fa99bd/nlpaug/model/lang_models/machine_translation_transformers.py#L47)
# to:
# + tokenized_texts = tokenizer(data, padding=True, truncation=True, return_tensors='pt')
#------------------------------------------------------------------------------------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--batch-size', '-b', help="Batch size per GPU", default=32, type=int)
@click.option('--gpus', help="comma sepreated list of gpus to use")
@click.option('--augmentation')
@click.option('--appendix', default="", type=str)
@click.option('--filter-outcome', help="Filter paper by there acceptance state. True:Accepted, False:Rejected, Default:Both", default=None, type=bool)
@click.option('--filter-file-path', default=None, type=str)
def main(ctx, batch_size, gpus, augmentation, appendix, filter_outcome, filter_file_path):
    gpu_idxs = [int(gpu) for gpu in gpus.split(',')]

    paths = glob.glob(f"./data/original/*.json")

    if filter_file_path:
        with open(filter_file_path, "r") as f:
            filter_files = f.read().splitlines()
            filter_files = set(filter_files)
        filtered_paths = [p for p in paths if p.split("/")[-1] in filter_files]
        print(f"File: Filtered from {len(paths)} to {len(filtered_paths)}")
        paths = filtered_paths

    if filter_outcome != None:
        filtered_paths = filter_by_outcome(paths, filter_outcome)
        print(f"Outcome: Filtered from {len(paths)} to {len(filtered_paths)}")
        paths = filtered_paths

    split_paths = chunker_list(paths, len(gpu_idxs))
    d_len = len(split_paths)
    with mp.Pool(len(gpu_idxs)) as pool:
        pool.starmap(section_process_chunk, zip(split_paths, gpu_idxs, list(range(len(gpu_idxs))), [
                     augmentation]*d_len, [appendix]*d_len, [batch_size]*d_len))

        # print(abstract, '\n\n', backtrans)


if __name__ == "__main__":
    main()
