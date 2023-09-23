import os
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import multiprocessing

IMG_SIZE    = 128
BATCH_SIZE  = 16
NUM_WORKERS = int(multiprocessing.cpu_count() / 2)

def get_dataset(dir):
  # Oxford IIIT Pets Segmentation dataset loaded via torchvision.
  train_data_path = os.path.join(dir, 'OxfordPets', 'train')
  test_data_path = os.path.join(dir, 'OxfordPets', 'test')
  train_dataset = torchvision.datasets.OxfordIIITPet(root=train_data_path, split="trainval", target_types="segmentation", download=True)
  test_dataset = torchvision.datasets.OxfordIIITPet(root=test_data_path, split="test", target_types="segmentation", download=True)
  return train_data_path, train_dataset, test_data_path, test_dataset

#train_data_path, train_dataset, test_data_path, test_dataset = get_dataset(dir="./data")

class OxfordDataset(torchvision.datasets.OxfordIIITPet):
  def __init__(self,
               root: str,
        split: str,
        target_types="segmentation",
        download=False,transform=None):

    super().__init__(
            root=root,
            split=split,
            target_types=target_types,
            download=download,
            transform=transform,
        )
    self.transform = transform

  def __len__(self):
    return super().__len__()

  def __getitem__(self,index):
    (img, mask_img) = super().__getitem__(index) # img is already a tensor
    mask_img = self.transform(mask_img)
    mask_img = mask_img * 255
    mask_img = mask_img.to(torch.long)
    mask_img = mask_img - 1
    return (img, mask_img)

def get_common_transform():
  common_transform = transforms.Compose([
      transforms.Resize((IMG_SIZE,IMG_SIZE), interpolation=T.InterpolationMode.NEAREST),
      transforms.ToTensor(),  # normalizes by default and converts to tensor
  ])
  return common_transform


def get_train_data_transform(root):
  transformed_train_data = OxfordDataset(root=root,
                                         split="trainval",
                                         target_types="segmentation",
                                         download=False, transform = get_common_transform())
  return transformed_train_data


def get_test_data_transform(root):

  transformed_test_data = OxfordDataset(root=root,
                                        split="test",
                                        target_types="segmentation",
                                        download=False,
                                        transform = get_common_transform(),
                                       )
  return transformed_test_data


def get_dataloaders(train_data_path, test_data_path):

  transformed_train_data = get_train_data_transform(train_data_path)
  transformed_test_data = get_test_data_transform(test_data_path)

  # DATALOADERS
  train_dataloader = DataLoader(transformed_train_data, batch_size = BATCH_SIZE,
                                shuffle=True, num_workers=NUM_WORKERS,
                                pin_memory=True)
  test_dataloader = DataLoader(transformed_test_data, batch_size = BATCH_SIZE,
                                shuffle=False, num_workers=NUM_WORKERS,
                                pin_memory=True)
  return train_dataloader, test_dataloader


