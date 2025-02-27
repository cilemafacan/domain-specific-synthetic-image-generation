from diffusers import AutoencoderKL
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel, CLIPProcessor, CLIPModel


class LatentExtractor:
    def __init__(self, vae_name, device, transform=None):
        self.vae = AutoencoderKL.from_pretrained(vae_name, subfolder="vae")
        self.vae.requires_grad_(False)
        self.vae.to(device)
        self.transform = transform

    def extract_latent(self, image):
        if self.transform:
            image = self.transform(image)
            
        latent = self.vae.encode(image).latent_dist.sample()
        latent = latent * 0.18125  
        
        return latent.cpu().detach().numpy()

class TextEmbeddingExtractor:
    def __init__(self, text_encoder_name, device):

        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_name)
        self.text_encoder.requires_grad_(False)
        self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder_name)
        self.device = device
        self.text_encoder.to(self.device)

    def extract_text_embedding(self, text):
        
        text_input = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        text_embedding = self.text_encoder(**text_input).last_hidden_state
            
        return text_embedding.cpu().detach().numpy()

class ImageEmbeddingExtractor:
    def __init__(self, img_encoder_name, device):
       self.clip_processor = CLIPProcessor.from_pretrained(img_encoder_name)
       self.clip_model = CLIPModel.from_pretrained(img_encoder_name)
       self.clip_model.requires_grad_(False)
       self.device = device
       self.clip_model.to(self.device)

    def extract_image_embedding(self, image):
        clip_inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        image_embedding = self.clip_model.get_image_features(**clip_inputs)

        return image_embedding.cpu().detach().numpy()
    

                               