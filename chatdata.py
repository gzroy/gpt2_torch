import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import random

class ChatDataset(Dataset):
    def __init__(self, dataset_file, block_size, mode="pack", sample_num=10):
        with open(dataset_file, 'rb') as f:
            self.data = pickle.load(f)
        self.block_size = block_size
        self.sample_num = sample_num
        self.mode = mode
        
    def __len__(self):
        return (len(self.data))
    
    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        if self.mode == "pack":
            sample_data = random.sample(self.data, self.sample_num)
            for i in range(self.sample_num):
                if len(x)>=self.block_size:
                    break
                delta_len = self.block_size - len(x)
                sample_x, sample_y = sample_data[i]
                if len(sample_x)>delta_len:
                    continue
                else:
                    x.extend(sample_x)
                    y.extend(sample_y)
            delta_len = self.block_size - len(x)
            x.extend([0 for _ in range(delta_len)])
            y.extend([-1 for _ in range(delta_len)])
        else:
            delta_len = self.block_size - len(x)
            x.extend([0 for _ in range(delta_len)])
            y.extend([-1 for _ in range(delta_len)])
        x = torch.IntTensor(x).long()
        y = torch.IntTensor(y).long()

        return x, y