import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np

class RoadSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

# Paths to images and masks
image_dir = "./data/images"
mask_dir = "./data/masks"

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])

dataset = RoadSegDataset(image_dir, mask_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = deeplabv3_resnet50(pretrained=False, num_classes=6)
model = model.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    model.train()
    total_loss = 0
    for imgs, masks in dataloader:
        imgs, masks = imgs.cuda(), masks.cuda()
        outputs = model(imgs)['out']
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "deeplabv3_roadseg.pth")
