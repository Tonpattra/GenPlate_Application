import torch
# from tools.model import cVAE
import numpy as np
from torch import nn, optim
import cv2
from torchvision import datasets, transforms
    
gen_list = [10,2,3,5]
mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'ก': 10, 'ข': 11, 
           'ค': 12, 'ฆ': 13, 'ง': 14, 'จ': 15, 'ฉ': 16, 'ช': 17, 'ฌ': 18, 'ญ': 19, 'ฎ': 20, 'ฐ': 21, 'ฒ': 22, 
           'ณ': 23, 'ด': 24, 'ต': 25, 'ถ': 26, 'ท': 27, 'ธ': 28, 'น': 29, 'บ': 30, 'ป': 31, 'ผ': 32, 'พ': 33, 
           'ฟ': 34, 'ภ': 35, 'ม': 36, 'ย': 37, 'ร': 38, 'ล': 39, 'ว': 40, 'ศ': 41, 'ษ': 42, 'ส': 43, 'ห': 44, 'ฬ': 45, 'อ': 46, 'ฮ': 47}

# Load annotation file
text_path = 'data_save/labels/license_plate_save.txt'
with open(text_path, 'r') as text_file:
    lines = text_file.readlines()

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
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
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
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load cVAE model
net = CVAE(28*28*3, 40, class_size=48).to(device)
checkpoint = torch.load("cVAE.pt", map_location=device)
net.load_state_dict(checkpoint["net"])
net.to(device)
net.eval()
 
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

aa = cv2.imread('data_save/license_plate/license_plate_save.jpg')
aa_resized = cv2.resize(aa, (640, 640))

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

print(len(lines))
for line, cha in zip(lines,gen_list):
    components = line.strip().split()
    # print(len(components))
    if len(components) == 5:
        class_label, x1, y1, x2, y2 = map(float, components)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(x1, y1, x2, y2)
        # Generate an image from cVAE model
        image_from_cvae = generate(cha, net)
        print(image_from_cvae.shape)
        image_from_cvae_resized = cv2.resize(image_from_cvae, (x2 - x1, y2 - y1))
        aa_resized[y1:y2, x1:x2] = image_from_cvae_resized
        print('the loop so done')

# Display annotated image
cv2.imshow('Annotated Image', cv2.resize(aa_resized, (123,64)))
cv2.waitKey(0)
cv2.destroyAllWindows()