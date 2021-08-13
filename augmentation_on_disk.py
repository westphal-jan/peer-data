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

def sentence_gpt_truncate(abstracts):
    trunc_abstracts = []
    for abstract in abstracts:
        split_sen = abstract.split('.')
        if len(split_sen) >= 3:
            trunc_abstracts.append(".".join(split_sen[:3]))
            continue
        else:
            trunc_abstracts.append(abstract)
    return trunc_abstracts

def process_chunk(paths, gpu_idx, pos_idx, augmentation_name, out_dir, batch_size):
    augmentation = get_augment(augmentation_name, gpu_idx, batch_size)
    batch = []
    for path in tqdm(paths, position=pos_idx):
        filename = path.split('/')[-1]
        if os.path.exists(out_dir + filename):
            continue

        with open(path, 'r') as f:
            paper_json = json.load(f)
            batch.append((paper_json, filename))
        if len(batch) == batch_size:
            abstracts = [paper["review"]["abstract"] for paper, _ in batch]
            if augmentation_name == 'sentence-gpt':
                abstracts = sentence_gpt_truncate(abstracts)

            augmented_abstracts = augmentation.augment(abstracts)

            for i, (paper, filename) in enumerate(batch):
                paper['review']['abstract'] = augmented_abstracts[i]
                with open(out_dir + filename, 'w') as f:
                    json.dump(paper, f)
            batch = []
    if len(batch) > 0:
        abstracts = [paper["review"]["abstract"] for paper, _ in batch]
        if augmentation_name == 'sentence-gpt':
            abstracts = sentence_gpt_truncate(abstracts)

        augmented_abstracts = augmentation.augment(abstracts)
        for i, (paper, filename) in enumerate(batch):
            paper['review']['abstract'] = augmented_abstracts[i]
            with open(out_dir + filename, 'w') as f:
                json.dump(paper, f)

def augment(to_augment, augmentation, batch_size, pos_idx):
    num_batches = math.ceil(len(to_augment) / batch_size)
    batches = np.array_split(to_augment, num_batches)

    augmented = []
    for _to_augment in tqdm(batches, position=pos_idx):
        _to_augment = _to_augment.tolist()
        # Use to reduced processed data
        # _to_augment = [" ".join(x.split()[:512]) for x in _to_augment]
        _augmented = augmentation.augment(_to_augment)
        augmented.extend(_augmented)
    return augmented

def section_process_chunk(paths, gpu_idx, pos_idx, augmentation_name, out_dir, batch_size):
    augmentation = get_augment(augmentation_name, gpu_idx, batch_size)

    for path in tqdm(paths, position=pos_idx):
        with open(path, 'r') as f:
            submission = json.load(f)
        
        sections = submission["pdf"]["metadata"]["sections"]
        assert sections
        text_sections = list(map(lambda x: x["text"], sections))
        augmented_text_sections = augmentation.augment(text_sections)
        
        for i in range(len(sections)):
            sections[i]["text"] = augmented_text_sections[i]
        submission["pdf"]["metadata"]["sections"] = sections

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
    if augmentation_name == "substitute-glove":
        return naw.WordEmbsAug(model_type='glove', model_path="./embeddings/glove.6B.50d.txt", action='substitute', aug_max=None, aug_p=0.5)
    if augmentation_name == "insert-glove":
        return naw.WordEmbsAug(model_type='glove',  model_path="./embeddings/glove.6B.50d.txt",
                               action='insert', aug_max=None, aug_p=0.5)
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

def split_list_into_chunks(seq, num_chunks):
    return list((seq[i::num_chunks] for i in range(num_chunks)))

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
@click.option('--reuse', default=True, type=bool)
@click.option('--sections', help="Set to augment sections otherwise augment abstract", is_flag=True)
def main(ctx, batch_size, gpus, augmentation, appendix, filter_outcome, filter_file_path, reuse, sections):
    gpu_idxs = [int(gpu) for gpu in gpus.split(',')]

    paths = glob.glob(f"./data/original/*.json")

    description = augmentation
    if appendix:
        description += f"-{appendix}"
    if filter_outcome != None:
        description += "-" + ("accepted" if filter_outcome else "rejected")
    out_dir = f"data/{description}/"
    os.makedirs(out_dir, exist_ok=True)

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

    if reuse:
        filtered_paths = [p for p in paths if not os.path.isfile(out_dir + p.split("/")[-1])]
        print(f"Reuse: Filtered from {len(paths)} to {len(filtered_paths)}")
        paths = filtered_paths

    if len(paths) == 0:
        print("No more files to augment")
        return

    split_paths = split_list_into_chunks(paths, len(gpu_idxs))
    d_len = len(split_paths)

    if sections:
        print("Augmenting sections")
        func = section_process_chunk
    else:
        print("Augmenting abstracts")
        func = process_chunk
    
    with mp.Pool(len(gpu_idxs)) as pool:
        pool.starmap(func, zip(split_paths, gpu_idxs, list(range(len(gpu_idxs))), [
                     augmentation]*d_len, [out_dir]*d_len, [batch_size]*d_len))

        # print(abstract, '\n\n', backtrans)


if __name__ == "__main__":
    main()
