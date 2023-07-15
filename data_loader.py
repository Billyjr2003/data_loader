import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from random import random
import random
from shutil import copyfile
from PIL import Image
import pathlib
import matplotlib.pyplot as plt

data_set = 'data_set/'

subdirs = ['train/', 'test/']

for subdir in subdirs:
    label_dirs = ['arabic/','chines/','english/']
    for label_dir in label_dirs :
        new_dir = data_set + subdir + label_dir
        os.makedirs(new_dir, exist_ok=True)


def load_data(src_dir , val_ratio = 0.25):
    global data_set
    for file in os.listdir(src_dir):
        src = os.path.join(src_dir,file)
        if file.startswith('english'):
          for img in os.listdir(src):
            main = os.path.join(src,img)
            dst_dir = 'train/'
            if random() < val_ratio:
              dst_dir = 'test/'
            dst = data_set + dst_dir + 'english/'  + img
            copyfile(main, dst)
        if file.startswith('arabic'):
          for img in os.listdir(src):
            main = os.path.join(src,img)
            dst_dir = 'train/'
            if random() < val_ratio:
              dst_dir = 'test/'
            dst = data_set + dst_dir + 'arabic/'  + img
            copyfile(main, dst)
        if file.startswith('chines'):
          for img in os.listdir(src):
            main = os.path.join(src,img)
            dst_dir = 'train/'
            if random() < val_ratio:
              dst_dir = 'test/'
            dst = data_set + dst_dir + 'chines/'  + img
            copyfile(main, dst)



train_dir = 'data_set/train'
test_dir = 'data_set/test'

def find_classes(directory: str):
  classes = sorted(entry.name for entry in list(os.scandir(directory)) if entry.is_dir())

  if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

  class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
  return classes, class_to_idx


class ImageFolderCustom(Dataset):
    
    
    def __init__(self, targ_dir: str, transform=None):
        
        
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg")) 
        
        self.transform = transform
        
        self.classes, self.class_to_idx = find_classes(targ_dir)
        

   
    def load_image(self, index: int) -> Image.Image:
        
        image_path = self.paths[index]
        return Image.open(image_path)
        #img.show() 
    
   
    def __len__(self) -> int:
        
        return len(self.paths)
    
    
    def __getitem__(self, index: int) :
        
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name 
        class_idx = self.class_to_idx[class_name]

        
        if self.transform:
            return self.transform(img), class_idx 
        else:
            return img, class_idx 

    def get_file_name(self,index):
      text_name = pathlib.Path(self.paths[index]).stem
      return text_name
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])


test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_data_custom = ImageFolderCustom(targ_dir=train_dir, 
                                      transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=test_dir, 
                                     transform=test_transforms)


train_dataloader_custom = DataLoader(dataset=train_data_custom, 
                                     batch_size=16, 
                                     num_workers=0, 
                                     shuffle=True) 

test_dataloader_custom = DataLoader(dataset=test_data_custom, 
                                    batch_size=16, 
                                    num_workers=0, 
                                    shuffle=False) 





def plot_from_data_loader(dataset):
  image_batch , label_batch = next(iter(dataset))
  plt.figure(figsize=(10,10))
  for i in range(12):
    ax = plt.subplot(3,4,i+1)
    img = image_batch[i]
    img = img.permute(1, 2, 0)
    plt.imshow(img)
    plt.title(label_batch[i])
    plt.axis('off')
    plt.show()

def plot_from_data_custom(dataset):
  plt.figure(figsize=(16,10))
  for i in range(12):
    ax = plt.subplot(3,4,i+1)
    label = dataset.get_file_name(i)[:-4]
    img = dataset[i][0]
    img = img.permute(1, 2, 0)
    plt.imshow(img)
    plt.title(label)
    plt.axis('off')
    plt.show()

plot_from_data_custom(train_data_custom)