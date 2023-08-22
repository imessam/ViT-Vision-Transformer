import torch
from utils import *
from torch.utils.data import Dataset


class ImageNet(Dataset):
    
    def __init__(self, data, to_paches_fn, res_patch):
        
        self.data = data
        self.to_paches_fn = to_paches_fn
        self.res_patch = res_patch
        

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, idx):
        
        image = self.data[idx]["image"]
        label = self.data[idx]["label"]
        
        if type(idx) == int:
            image = [image]
        
        image_patches_flatten = flatten_patches(self.to_paches_fn(image, self.res_patch))
        
        return {"image_patches_flatten":image_patches_flatten, "label":label}
    
    
  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        