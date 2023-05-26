from torch.utils.data.dataset import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, data, masks):
        super(Dataset, self).__init__()
        self.data = data
        self.masks = masks

    def __len__(self):
        return self.masks.size

    def __getitem__(self, index):
        sample = torch.from_numpy(self.data[:, index]).float().unsqueeze(0)
        mask = torch.tensor(self.masks[0][index]-1).long()
        return sample,mask
