import os
import cv2
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader

class BrainMRI(Dataset):

    LABELS = {
        1 : 'yes',
        0 : 'no'
    }
    data = []
    IMG_SIZE = 150
    tumor_count = 0
    
    def __init__(self, data_dir, transform=None):
        #dataset loading and stuff
        
        total = 0
        self.data_dir = data_dir
        for LABEL in self.LABELS:
            label_dir = os.path.join(data_dir,self.LABELS[LABEL])
            for pics in os.listdir(label_dir):
                
                try:
                    path = os.path.join(label_dir,pics)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE,self.IMG_SIZE))
                    self.data.append((torch.from_numpy(np.array(img)),LABEL))
                
                    if LABEL == 1:
                        self.tumor_count += 1
                    total += 1
                
                except Exception as e:
                    pass
        self.n_samples = total
        
    def __getitem__(self,index):
        #dataset indexing
        
        return self.data[index]
    
    def __len__(self):
        #length of the dataset
        
        return self.n_samples
        
        