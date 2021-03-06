import os
import json
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.stem import PorterStemmer


filepaths = []
for path, subdirs, files in os.walk("data"):
    for name in files:
        filepath = os.path.join(path, name)
        filepaths.append(filepath)

data = []
for filepath in filepaths:
    with open(filepath, "r") as f:
        submission = json.loads(f.read())
        data.append(submission)

example = data[0]
print(example["id"])
print(example["review"].keys())
print(example["pdf"]["metadata"].keys())

def analyze_text_sections(sections: List[str]):
    num_words = []
    num_sentences = []
    vocab = set()
    for section in sections:
        words = [w for w in section.split() if w]
        num_words.append(len(words))
        vocab.update(words)
        sentences = [s for s in section.split(".") if s]
        num_sentences.append(len(sentences))
    print("Avg #Words", np.average(num_words))
    print("Avg #Sentences", np.average(num_sentences))
    print("Vocab Size:", len(vocab))

def normalize(sections: List[str]):
    porter = PorterStemmer()

    num_words = []
    num_sentences = []
    all_tokens = set()
    all_words = set()
    for i, section in tqdm(enumerate(sections)):
        sentences = sent_tokenize(section)
        words = word_tokenize(section)
        tokens = [word for word in words if word.isalpha()]
        tokens = [word.lower() for word in tokens]    
        tokens = [word for word in tokens if not word in stopwords.words("english")]
        tokens = [porter.stem(word) for word in tokens]

        num_words.append(len(words))
        all_words.update(words)
        all_tokens.update(tokens)
        num_sentences.append(len(sentences))
    
    print("#Words:", len(all_words))
    print("#Normalized:", len(all_tokens))
    print("Avg words:", np.average(num_words))
    print("Avg sentences:", np.average(num_sentences))

abstracts = []
for submission in data:
    abstract = submission["review"]["abstract"]
    abstracts.append(abstract)

# analyze_text_sections(abstracts)
print("Abstracts:")
normalize(abstracts)

valid_sections = []
invalid_sections = []
no_sections = []
valid_reviews = []
for submission in data:
    sections = submission["pdf"]["metadata"]["sections"]
    if sections == None:
        no_sections.append(submission["id"])
    elif len(sections) == 1 and sections[0]["heading"] == None:
        invalid_sections.append(submission["id"])
    else:
        valid_sections.append(sections)
    
    reviews = submission["review"]["reviews"]
    if len(reviews) > 0:
        valid_reviews.append(submission["id"])

print("Valid reviews:", len(valid_reviews))
print("No sections:", len(no_sections))
print("Invalid sections:", len(invalid_sections))
print("Submissions with valid sections:", len(valid_sections))

print(invalid_sections[0])

num_sections = list(map(len, valid_sections))
print("Avg #Sections:", np.average(num_sections))

sections = []
submission_texts = []

for submission_sections in valid_sections:
    submission_text = ""
    for s in submission_sections:
        sections.append(s["text"])
        submission_text += " " + s["text"]
    submission_texts.append(submission_text)


print("Sections:")
normalize(sections)

print("Submission:")
normalize(submission_texts)
