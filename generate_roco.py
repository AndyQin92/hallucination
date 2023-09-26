import torch
import pickle
from tqdm import tqdm
from collections import defaultdict
from diffusers import StableDiffusionPipeline

with open('./imgid2imgid.pickle', 'rb') as f1:
    imgid2imgid = pickle.load(f1)

with open('./imgid2caption.pickle', 'rb') as f2:
    imgid2caption = pickle.load(f2)

model_id = "Nihirc/Prompt2MedImage"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

for imgid, cap in tqdm(imgid2caption.items()):
    prompt = cap
    for i in range(4):
        image = pipe(prompt).images[0]  
        image.save(f"./roco_diff/val/{str(i)}_{imgid}.png")

