import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
# from util import CustomDataset, thai_char_to_number
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
# from util import CustomDataset, thai_char_to_number
import matplotlib.pyplot as plt
import torch
# from tools.model import cVAE
import numpy as np
import cv2
from PIL import Image
import random

weights = [0.1] * 10 + [1] * 38

mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'ก': 10, 'ข': 11, 
           'ค': 12, 'ฆ': 13, 'ง': 14, 'จ': 15, 'ฉ': 16, 'ช': 17, 'ฌ': 18, 'ญ': 19, 'ฎ': 20, 'ฐ': 21, 'ฒ': 22, 
           'ณ': 23, 'ด': 24, 'ต': 25, 'ถ': 26, 'ท': 27, 'ธ': 28, 'น': 29, 'บ': 30, 'ป': 31, 'ผ': 32, 'พ': 33, 
           'ฟ': 34, 'ภ': 35, 'ม': 36, 'ย': 37, 'ร': 38, 'ล': 39, 'ว': 40, 'ศ': 41, 'ษ': 42, 'ส': 43, 'ห': 44, 'ฬ': 45, 'อ': 46, 'ฮ': 47, '-': '-', '':'-'}

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

device = torch.device("cuda")

def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)

class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size=40, class_size=48):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size
        self.latent = latent_size

        # encode
        self.fc1  = nn.Linear(feature_size + class_size, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, 400)
        self.fc4 = nn.Linear(400, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) 
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        print('.......')
        print(z[0].size())
        print(c[0])
        print('........')
        inputs = torch.cat([z, c], 1) 
        h3 = self.elu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 28*28*3), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def generate(self, c):
        '''
        Generate an image conditioned on the class label c.
        '''
        # Generate a random latent vector
        z = torch.randn(1, self.latent).to(c.device)
        print(z)
        print(c)
        return self.decode(z, c)

def generate(number, model, images):
    with torch.no_grad():
        tensor_list = [number]
        tensor = torch.tensor(tensor_list).to(device)
        hot_tensor = one_hot(tensor, 48)
        out_img,_,_ = model(images.to(device), hot_tensor)

        # print(out_img)

        picture_array  = out_img.view(1, 3, 28, 28).cpu().detach().numpy()
        picture_array = picture_array.squeeze(0)
        picture_array *= 255
        return picture_array.transpose(1,2,0).astype(np.uint8)
    
# dire = "experiment/same_class/datasets/"    
# out_dire = "experiment/same_class/generation/" 
# txt_files = [file for file in os.listdir(dire) if file.endswith('.txt')]


# for text_name in txt_files :
#     # break

#     aa = cv2.imread(dire + text_name.replace('.txt', '.jpg'))
#     aa_resized = cv2.resize(aa, (640, 640))

#     text_path = f'{dire}{text_name}'
#     with open(text_path, 'r') as text_file:
#         lines = text_file.readlines()
#     print(lines)
    

#     # print(modified_lines)
            
#     history = []
#     # print(len(lines))
#     crop_image = []
#     random_numbers = random.choices(range(48), weights=weights,k=len(lines))
#     modified_lines = []
#     for line, charaf in zip(lines, random_numbers):
#         parts = line.split()
#         if parts:  
#             history.append(int(parts[0]))
#             parts[0] = str(charaf)  
           
#     for idx, (line, cha) in enumerate(zip(lines,history)):
#         components = line.strip().split()
#         if len(components) == 5 and components[0] != "2กรุงเทพมหานคร":
#             components[0] = 0
#             class_label, x1, y1, x2, y2 = map(float, components)
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             if x1>640 :
#                 x1 = 640
#             if x2>640 :
#                 x2 = 640
#             if y1>640 :
#                 y1 = 640
#             if y2>640 :
#                 y2 = 640            
#             crop_image = aa_resized[y1:y2, x1:x2]
#             print(f'crop_image_size = {crop_image.shape}')
#             crop_image_pil = Image.fromarray(crop_image)
#             transformed_image = transform(crop_image_pil)
#             image_from_cvae = generate(cha, model, transformed_image.unsqueeze(0))
#             image_from_cvae_resized = cv2.resize(image_from_cvae, (x2 - x1, y2 - y1))
#             aa_resized[y1:y2, x1:x2] = image_from_cvae_resized

#     with open(f'{out_dire}generate_{text_name}', "w") as output_file:
#         for line in lines:
#             output_file.write(line)           

#     cv2.imwrite(f'{out_dire}generate_{text_name.replace(".txt", ".jpg")}', cv2.resize(aa_resized, (123,64)))

