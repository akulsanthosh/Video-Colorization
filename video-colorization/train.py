from os import listdir

import numpy as np
import torch
import torch.optim as optim
from cv2 import cv2
from torch import nn
import cupy

from torch.utils.data import DataLoader, Dataset
from model import load_model, preprocess_img, preprocess_img_ab
from network import Model
from discriminator import PixelDiscriminator


class MyDataset(Dataset):

    def __init__(self, root, transform_x=None, transform_y=None):
        self.transformX = transform_x
        self.transformY = transform_y

        x = []
        y = []

        for i in range(len(listdir(root))):
            image = cupy.asarray(cv2.imread(f"{root}/image-{i}.png"))
            original, image_processed = preprocess_img(image)

            x.append(image_processed)
            y.append(torch.swapaxes(preprocess_img_ab(image)[1], 0, 2))

        self.X = x
        self.Y = y

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):
        im = self.X[idx]
        label = self.Y[idx] - self.Y[idx - (idx > 0)]
        im_prev = im if idx == 0 else self.X[idx - 1]

        if self.transformX is not None:
            im = self.transformX(im)

        if self.transformY is not None:
            label = self.transformY(label)

        return (im_prev, im), label


def train(pix, generator, discriminator, batch_size=16):
    train_loader = torch.utils.data.DataLoader(MyDataset(root="../output/8/x"), batch_size=batch_size, shuffle=True,
                                               num_workers=0, drop_last=True)

    pix.cuda()
    generator.cuda()
    discriminator.cuda()

    generator_optimiser = optim.Adam(generator.parameters(), lr=0.3)
    discriminator_optimiser = optim.Adam(discriminator.parameters(), lr=0.3)

    generator_loss = nn.MSELoss()
    discriminator_loss = nn.MSELoss()

    FALSE = torch.zeros([batch_size, 1, 256, 256], dtype=torch.float).cuda()
    TRUE = torch.ones([batch_size, 1, 256, 256], dtype=torch.float).cuda()
    device = torch.device('cuda:0')

    for epoch in range(10):
        total_loss = 0
        for images, label in train_loader:
            img = images[0].to(device)
            prev = images[1].to(device)
            label = label.to(device)

            generator_optimiser.zero_grad()
            discriminator_optimiser.zero_grad()

            pix1 = pix(prev)
            pix2 = pix(img)

            output = generator(pix1, pix2).detach()

            output = (output - pix1).detach()

            output_fake = discriminator(output)

            disc_l = discriminator_loss(output_fake, FALSE)
            loss_gen = generator_loss(output, label) + disc_l
            loss_gen.backward()
            generator_optimiser.step()

            FALSE = FALSE.detach()
            output_real = discriminator(label)
            disc_l = disc_l.detach()
            loss_des = disc_l + discriminator_loss(output_real, TRUE)
            loss_des.backward()
            discriminator_optimiser.step()
            FALSE = FALSE.detach()

            total_loss += loss_gen

        print(f"Epoch {epoch} Loss {total_loss / len(train_loader)}")
        torch.save(generator.state_dict(), f'saves/{epoch}.pth')

if __name__ == "__main__":
    train(load_model(True), Model(), PixelDiscriminator(2))
