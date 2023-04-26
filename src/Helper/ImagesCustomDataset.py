
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os

class ImagesCustomDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir_images, root_dir_label, transform=None):

        self.transform = transform
        volumes = list(filter(lambda x: True if 'volume-' in x else False, os.listdir(root_dir_images)))
        self.data = []
        self.targets = []
        for volume_id in volumes:
          img_dir = os.path.join(root_dir_images, volume_id)
          target_dir = os.path.join(root_dir_label, volume_id)
          samples = list(map(lambda x: os.path.join(img_dir, x), sorted(os.listdir(img_dir))))
          labels = list(map(lambda x: os.path.join(target_dir, x), sorted(os.listdir(target_dir))))

          self.data.extend(samples)
          self.targets.extend(labels)

    def __len__(self):
      return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        target = np.load(self.targets[idx])
        
        if self.transform:
          img=self.transform(img)

        return (img, target)