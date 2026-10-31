import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms

class MathDataset(Dataset):

  def __init__(self, hf_dataset, tokenizer, transform=None, max_length=128):
    self.ds = hf_dataset
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.transform = transform or transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):
    item = self.ds[idx]
    image = item['image'].convert('RGB')
    latex_str = item['latex']

    if self.transform:
      image_tensor = self.transform(image)

    target_tensor = self.tokenizer.encode(latex_str, max_length=self.max_length)
    return image_tensor, target_tensor


def get_dataloaders(tokenizer, batch_size=32):
  
  raw_ds = load_dataset('deepcopy/MathWriting-human', split='train[:5000]') 
  split_ds = raw_ds.train_test_split(test_size=0.2) 
  train_dataset = MathDataset(split_ds['train'], tokenizer)
  test_dataset = MathDataset(split_ds['test'], tokenizer)

  train_loader = DataLoader(
      train_dataset, batch_size=batch_size, shuffle=True
  )
  test_loader = DataLoader(
      test_dataset, batch_size=batch_size, shuffle=False
  )

  return train_loader, test_loader
from src.tokenizer import Tokenizer

class MathDataSet(Dataset):

  def __init__(self, hf_dataset, tokenizer, transform=None, max_length=128):
    self.ds = hf_dataset
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.transform = transform or transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
  def __len__(self):
    return len(self.ds)
  def __getitem__(self, idx):
    item = self.ds[idx]
    image = item['image'].convert('RGB')
    latex_string = item['latex']
    image_tensor = self.transform(image) #!
    target_tensor = self.tokenizer.encode(latex_string, max_length=self.max_length) 
    return image_tensor, target_tensor
