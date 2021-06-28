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
    # break

example = data[0]
print(example["id"])
print(example["review"].keys())
print(example["pdf"]["metadata"].keys())

def normalize(sections: List[str]):
    porter = PorterStemmer()

    num_words = []
    num_sentences = []
    all_words = set()
    for i, section in tqdm(enumerate(sections)):
        sentences = sent_tokenize(section)
        words = word_tokenize(section)
        num_words.append(len(words))
        all_words.update(words)
        num_sentences.append(len(sentences))
    
    tokens = [word for word in all_words if word.isalnum()]
    tokens = [word.lower() for word in tokens]    
    tokens = [word for word in tokens if not word in stopwords.words("english")]
    tokens = [porter.stem(word) for word in tokens]
    all_tokens = set(tokens)

    print("#Words:", len(all_words))
    print("#Normalized:", len(all_tokens))
    print("Avg words:", np.average(num_words))
    print("Avg sentences:", np.average(num_sentences))
    return num_words, num_sentences

abstracts = []
for submission in data:
    abstract = submission["review"]["abstract"]
    abstracts.append(abstract)

print("Abstracts:", len(abstracts))
num_words, num_sentences = normalize(abstracts)

# n_bins = 30
# plt.figure()
# plt.hist(num_words, weights=np.ones(len(data)) / len(data), bins=n_bins)
# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title(f"Number of words in abstracts")
# plt.xlabel("Number of words")
# plt.ylabel("Occurrences")
# plt.savefig(f"abstract_words.png", dpi=300)
# 
# fig = plt.figure()
# plt.hist(num_sentences, weights=np.ones(len(data)) / len(data), bins=n_bins)
# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.title("Number of sentences in abstract")
# plt.xlabel("Number of sentences")
# plt.ylabel("Occurrences")
# plt.savefig("abstract_sentences.png", dpi=300)

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

print("Sections:", len(sections))
normalize(sections)

print("Submission:", len(submission_texts))
normalize(submission_texts)
