from PIL import Image
import os
import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm 

class HigwWayDataset(Dataset):
    # def __init__(self, root_dir, transform=None):
    #     self.root_dir = root_dir
    #     self.rgb_dir = f"{root_dir}/rgb"
    #     self.seg_dir = f"{root_dir}/seg"
    #     self.data = []  # List to store loaded images
    #     self.labels = []
    #     self.transform = transform

    #     # Loop through all files in the directory
    #     for filename in tqdm(os.listdir(self.rgb_dir), desc="Loading dataset"):
    #         if filename.endswith('.png'):  # Consider only PNG files
    #             img_path = os.path.join(self.rgb_dir, filename)
    #             image = Image.open(img_path)
    #             label = np.load(os.path.join(self.seg_dir, filename.replace(".png", ".npy")))

    #             if self.transform:
    #                 image = self.transform(image)

    #             self.data.append(image)
    #             self.labels.append(label)
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.rgb_dir = f"{root_dir}/rgb"
        self.seg_dir = f"{root_dir}/seg"
        self.filenames = []
        self.transform = transform

        # Loop through all files in the directory
        for filename in os.listdir(self.rgb_dir):
            if filename.endswith('.png'):  # Consider only PNG files
                self.filenames.append(filename)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(os.path.join(self.rgb_dir, filename))
        label = np.load(os.path.join(self.seg_dir, filename.replace(".png", ".npy")))
        if self.transform:
            image = self.transform(image)

        return image, label

# transform = transforms.Compose([
#     transforms.Resize((512, 910)),  # Resize images
#     transforms.ToTensor()  # Convert images to PyTorch tensors
# ])

# higway_dataset = HigwWayDataset("./data/filtered_dataset", transform)
# batch_size = 4

# # data_loader = DataLoader(a, batch_size=batch_size, shuffle=True)
# train_size = int(0.8 * len(higway_dataset))  # 80% train
# test_size = len(higway_dataset) - train_size  # Remaining for test
# # Split the dataset
# train_set, test_set = random_split(higway_dataset, [train_size, test_size])

# # Create DataLoader for train and test sets
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# for x,y in train_loader:
#     # print(x.shape, y.shape)
#     # print(x)
#     # print(y)
#     for i in range(batch_size):
#         img = x[i].squeeze(0)
#         seg = y[i].squeeze(0)
#         from fastseg.image.colorize import colorize, blend
#         print(type(seg), seg.size())
#         # a = torchvision.transforms.functional.to_pil_image(img)
#         seg_img = colorize(seg.numpy()) # <---- input is numpy, output is PIL.Image
#         blended_img = blend(torchvision.transforms.functional.to_pil_image(img), seg_img, factor=0.55) # <---- input is PIL.Image in both arguments

#         # Concatenate images for simultaneous view
#         new_array = np.concatenate((np.asarray(blended_img), np.asarray(seg_img)), axis=1)

#         # Show image from PIL.Image class
#         combination = Image.fromarray(new_array)
#         combination.show()
#     break