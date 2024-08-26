from transformers import pipeline
from PIL import Image
import requests
import torch


device =  torch.device('cuda' if torch.cuda.is_available() else "cpu")
print("device", device)
# load pipe
image_classifier = pipeline(task="zero-shot-image-classification", model="google/siglip-base-patch16-224", device = device)

# load image
url = 'test/page_12.png'
image = Image.open(url)

# inference
candidate_labels = ["language English", "language Vietnamese", "language Hindi", "language Germany", "language Japan", "language Korean", "language Chinese"]
outputs = image_classifier(image, candidate_labels=candidate_labels)
outputs = [{"score": round(output["score"], 4), "label": output["label"] } for output in outputs]
print(outputs)