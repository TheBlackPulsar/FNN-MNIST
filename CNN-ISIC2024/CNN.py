import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import h5py
from time import time
import numpy as np
from PIL import Image
import io
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch.optim.lr_scheduler as lr_scheduler

# Debug prints
DEBUG = False

# Define image size
IMG_SIZE = (128, 128)

# Define number of Epochs (30 already get a good accuracy but in this case of medical data 45 should be taken to really cover all cancer samples)
EPOCHS = 45

# Define number of negative cases. The number of postitive is always fully taken
NEGATIVE = 5000

# Dataset inputs
HDF5 = 'train-image.hdf5'
CSV = 'train-metadata.csv'

# Use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load train dataset from location
ds_full = pd.read_csv(CSV, usecols=range(2))

if DEBUG == True:
  print('Full dataset counting')
  print(ds_full.target.value_counts())

# Sorting the dataset
ds_group = ds_full.groupby(ds_full.target)
ds_pos = ds_group.get_group(1)
ds_neg = ds_group.get_group(0)

# Get just a part of the negative dataset
n = NEGATIVE
ds_part = ds_neg.iloc[:n]

# Combining datasets
ds_recombined = pd.concat([ds_pos, ds_part], axis=0)

# Shuffle dataset
ds_recombined = ds_recombined.sample(frac=1, random_state=42)

if DEBUG == True:
  print('Recombined dataset counting')
  print(ds_recombined.target.value_counts())

# Calculate class weights
class_counts = ds_recombined['target'].value_counts().to_dict()
total_samples = len(ds_recombined)
class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
weights = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(device)

# Split dataset
ds_train, ds_test = train_test_split(ds_recombined, test_size=0.2, random_state=42)

# Assigning data types
dtypes = {'isic_id': str, 'target': int}
ds_train = ds_test.astype(dtypes)
ds_test = ds_test.astype(dtypes)

# Manage image and patient data
class ISICDataset(Dataset):
    def __init__(self, hdf5_file, df, transform=None):
        self.hdf5_file = hdf5_file
        self.test_hd5 = h5py.File(hdf5_file, 'r')
        self.df = df
        self.labels = df
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t = str(self.df.iloc[idx]['isic_id'])
        p = self.test_hd5[t][()]
        img = Image.open(io.BytesIO(p))
        img = img.convert('RGB')
        img = np.array(img)
        img = (img - img.min()) / (img.max() - img.min() +1e-6) * 255
        if self.transform:
            label = torch.tensor(self.labels.iloc[idx, 1], dtype=torch.long)  # Column of label
            img = self.transform(image=img.astype(np.uint8))['image']
        return label, img.transpose(2, 0, 1)

transforms_val = A.Compose([
  A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
  A.Normalize(mean=0.5, std=0.5)
])

# Weighted focalloss function
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        self.reduction = reduction

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))

        target = target.view(-1, 1)

        logpt = nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

dataset_train = ISICDataset(HDF5, ds_train, transform=transforms_val)

# Convert class weights to a list for WeightedRandomSampler
sample_weights = [0] * len(dataset_train)
for idx, (_, row) in enumerate(ds_train.iterrows()):
    label = row['target']
    sample_weights[idx] = class_weights[label]

# Create WeightedRandomSampler
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(dataset_train, batch_size=32, sampler=sampler)

# Define CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = FocalLoss(gamma=2.0, alpha=weights, reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize CosineAnnealingLR
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Training the model
num_epochs = EPOCHS

for epoch in range(num_epochs):
    start = time()
    model.train()
    running_loss = 0.0
    for labels, images in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        scheduler.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Learning Rate: {scheduler.get_last_lr()}, Runtime of Epoch: {round((time()-start), 2)}')

dataset_test = ISICDataset(HDF5, ds_test, transform=transforms_val)

test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

# Evaluating the model
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for labels, images in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Calculate accuracy
#accuracy = accuracy_score(y_true, y_pred)
#print(f'Test Accuracy: {accuracy:.4f}')

# Calculate classification report
report = classification_report(y_true, y_pred)
print(report)