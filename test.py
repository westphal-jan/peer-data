import json
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_path = "results/11-08-2021_22_30_28-scibert-sections-back_translation-weighted/network-snapshot-latest"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
model.to(device)

model_name = 'allenai/scibert_scivocab_uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open("data/test/stylegan.json", "r") as f:
    data = json.load(f)
    sections = list(map(lambda x: x["text"], data["sections"]))

print(len(sections))

tokenized_sections = tokenizer(sections, truncation=True, padding="max_length", max_length=512)

with torch.no_grad():
    input = {key: torch.tensor(val).to(device) for key, val in tokenized_sections.items()}
    output = model(**input)
    print(output)
    prediction = output.logits.mean(axis=0)
    print(prediction)
    prob = torch.nn.functional.softmax(prediction, dim=0)
    print(prob)

