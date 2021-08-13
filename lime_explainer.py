import json
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_path = "models/scibert_best/network-snapshot-latest"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
model.to(device)

model_name = 'allenai/scibert_scivocab_uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# with open("data/test/stylegan.json", "r") as f:
#     data = json.load(f)
#     sections = list(map(lambda x: x["text"], data["sections"]))

# print(len(sections))

class_names = ['positive','negative', 'neutral']

def predictor(texts):
    print(len(texts))
    outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
    probas = F.softmax(outputs.logits).detach().numpy()
    return probas

explainer = LimeTextExplainer(class_names=class_names)

str_to_predict = "Our goal is to predict whether a paper is going to be rejected or accepted at a conference, just from the contentsof that paper. For this task, we utilized thePeerReaddataset [1], which contains about 14k reviews, papersand their acceptance decision of up until 2017. We use two different approaches to embed text into a vectorspace: a Bag of Words embedding and pre-trained BERT model [2]. Both approaches have a feed-forwardneural network on top to determine the final acceptance prediction. For the majority of the implementation1of our networks, we used the packages Pytorch [3], Pytorch Lightning [4], Huggingface Transformers [5] andSentence-Transformers [6]. For simplified logging and visualization we used the package Weights & Biases [7]."
exp = explainer.explain_instance(str_to_predict, predictor, num_features=20, num_samples=100)
# exp.show_in_notebook(text=str_to_predict)
exp.save_to_file('lime.html')

# tokenized_sections = tokenizer(sections, truncation=True, padding="max_length", max_length=512)

# with torch.no_grad():
#     input = {key: torch.tensor(val).to(device) for key, val in tokenized_sections.items()}
#     output = model(**input)
#     print(output)
#     prediction = output.logits.mean(axis=0)
#     print(prediction)
#     prob = torch.nn.functional.softmax(prediction, dim=0)
#     print(prob)