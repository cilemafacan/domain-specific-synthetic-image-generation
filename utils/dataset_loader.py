import openslide as ops
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, slide_dir,  transform=None):
        self.data = data
        self.transform = transform
        self.slide_dir = slide_dir
        self.slide_cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        slide_name = sample["slide_name"]
        x = sample["x"]
        y = sample["y"]
        patch_size = sample["patch_size"]

        if slide_name not in self.slide_cache:
            self.slide_cache[slide_name] = ops.OpenSlide(f"{self.slide_dir}/{slide_name}")

        slide = self.slide_cache[slide_name]
        image = slide.read_region((x, y), 0, patch_size)
        image = image.convert("RGB")

        embedding = sample["embedding_vector"]
        if self.transform:
            image = self.transform(image)
        return image, embedding