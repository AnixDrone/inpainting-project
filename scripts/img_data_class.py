from torch.utils.data import Dataset
import os
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self,src_folder,transform = None):
        super(ImageDataset,self).__init__()
        self.transform = transform
        self.src_folder = src_folder
        self.img_list = os.listdir(src_folder)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        if self.transform is not None:
            img_name = self.img_list[index]
            img = Image.open(os.path.join(self.src_folder,img_name))
            return self.transform(img)

