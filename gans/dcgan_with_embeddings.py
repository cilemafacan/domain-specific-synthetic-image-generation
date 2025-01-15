import torch
import torch.nn as nn
import tensorflow as tf
from huggingface_hub import from_pretrained_keras

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 32, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 64x64
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # 112x112
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 17, bias=False),  # 224x224
            nn.Sigmoid()
        ) 

    def forward(self, input):
        return self.main(input)

""" class Discriminator(nn.Module):
    def __init__(self, input_size=384):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        batch_size = input.size(0)
        input_flat = input.view(batch_size, -1)
        return self.main(input_flat) """


class Discriminator(nn.Module):
    def __init__(self, input_size=384):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        batch_size = input.size(0)
        input_flat = input.view(batch_size, -1)
        return self.main(input_flat)
    

""" class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
           
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
      
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
           
            nn.Conv2d(ndf * 16, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input) """
    
class PathFoundationModel():
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = from_pretrained_keras(model_name)
        self.infer = self.model.signatures["serving_default"]
    
    def inference(self, input_tensor):
        tf_input = tf.convert_to_tensor(input_tensor)
        transposed_input = tf.transpose(tf_input, perm=[0, 2, 3, 1])
        embedding = self.infer(transposed_input)["output_0"].numpy()
        embedding = torch.from_numpy(embedding)
        return embedding