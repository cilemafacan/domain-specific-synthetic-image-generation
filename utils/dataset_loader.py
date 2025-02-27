import os
import torch
import numpy as np
import openslide as ops
from torch.utils.data import Dataset

class CustomDatasetFromSlide(Dataset):
    def __init__(self, dataset, slide_dir, transform=None):
        self.dataset = dataset["train"]
        self.transform = transform
        self.slide_dir = slide_dir
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        organ = self.dataset[idx]["organ"]
        x = self.dataset[idx]["x"]
        y = self.dataset[idx]["y"]
        patch_size = self.dataset[idx]["patch_size"]
        slide_name = self.dataset[idx]["slide_name"]
        slide_path = os.path.join(self.slide_dir, organ, slide_name)

        # Cache kontrolü
        if slide_path not in self.cache:
            self.cache[slide_path] = ops.OpenSlide(slide_path)
        
        slide = self.cache[slide_path]
        image = slide.read_region((x, y), 0, patch_size).convert("RGB")

        if self.transform:
            image = self.transform(image)
    
        embedding = self.dataset[idx]["image_embedding_vector"]
        embedding = torch.tensor(embedding, dtype=torch.float32)

        return {"image": image, "embedding": embedding, "organ": organ}
    
class CustomDatasetWithLatent(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        latent = self.dataset[idx]["latent"]
        embedding = self.dataset[idx]["embedding_vector"]
        return {"latent": latent, "embedding": embedding}
    
class CustomDatasetWithGenerated(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        generated = self.dataset[idx]["generated"]
        if self.transform:
            generated = self.transform(generated)
        return {"generated": generated}

class CustomDatasetFromSource(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        if self.transform:
            image = self.transform(image)
        embedding = self.dataset[idx]["image_embedding_vector"]
        embedding = torch.tensor(embedding, dtype=torch.float32)
        return {"image": image, "embedding": embedding}