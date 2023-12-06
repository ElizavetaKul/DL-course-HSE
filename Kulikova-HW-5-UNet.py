import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.utils.tensorboard import SummaryWriter

from glob import glob
import numpy as np
from imageio import imread
from torch.utils.data import Dataset, DataLoader, random_split


torch.manual_seed(2023)

## DATASET

paths = glob("./stage1_train/*")

class DSB2018(Dataset):

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        img_path = glob(self.paths[idx] + "/images/*")[0]
        mask_imgs = glob(self.paths[idx] + "/masks/*")
        img = imread(img_path)[:, :, 0:3]
        img = np.moveaxis(img, -1, 0)
        img = img / 255.0
        masks = [imread(f) / 255.0 for f in mask_imgs]

        final_mask = np.zeros(masks[0].shape)
        for m in masks:
            final_mask = np.logical_or(final_mask, m)
        final_mask = final_mask.astype(np.float32)

        img, final_mask = torch.tensor(img), torch.tensor(final_mask).unsqueeze(
            0)

        img = nn.functional.interpolate(img.unsqueeze(0), (256, 256))
        final_mask = nn.functional.interpolate(final_mask.unsqueeze(0), (256, 256))
        # Now the shapes  are (B=1, C, W, H) We need to convert them back to FloatTensors and grab the first item in the "batch".
        # This will return a tuple of: (3, 256, 256), (1, 256, 256)
        return img.type(torch.FloatTensor)[0], final_mask.type(torch.FloatTensor)[0]

## MODEL

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer,
        # with the exception of the last block which does not include a max-pooling layer.

        # input img: 256x256x3
        # padding добавлен, чтобы сохранить размерность итогового выхода
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # output: 254x254x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # output: 252x252x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 126x126x64


        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out




dsb_data = DSB2018(paths)

train_split, test_split = torch.utils.data.random_split(dsb_data, [500, len(dsb_data)-500])
generator = torch.Generator().manual_seed(42)
test_split, val_split = random_split(train_split, [0.8, 0.2], generator=generator)

train_seg_loader = DataLoader(train_split, batch_size=1, shuffle=True)
test_seg_loader = DataLoader(test_split,  batch_size=1)
val_seg_loader = DataLoader(val_split,  batch_size=1)


loss_func = nn.BCEWithLogitsLoss()
num_epochs = 5
model = UNet(n_class=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    model.train()
    train_running_loss = 0
    for img, mask in train_seg_loader:
        optimizer.zero_grad()
        prediction = model(img)

        loss = loss_func(prediction, mask)
        train_running_loss += loss

        loss.backward()
        optimizer.step()
    train_loss = train_running_loss / len(train_split)

    model.eval()
    val_running_loss = 0
    with torch.no_grad():
        for img, mask in val_seg_loader:
            prediction = model(img)
            loss = loss_func(prediction, mask)

            val_running_loss += loss

        val_loss = val_running_loss / len(val_split)
    print(f'Epoch {epoch}. Train loss is {train_loss}')
    print(f'Epoch {epoch}. Val loss is {val_loss}')




# сохранение модели
torch.save(model.state_dict(), './unet-model.pt')


