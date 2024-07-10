'''
Converts images to tensor
'''
import torch.utils.data as data
import torch, numpy as np
from processing import remove_invalid_values, open_image
from processing import resample_image, min_max_scaler
from processing import load_files

class ImageLoader(data.Dataset):
    def __init__(self, imgpath, labpath, augmentations=False, num= 0):  
        self.imgpath = imgpath
        self.labpath = labpath
        if num>0:
            self.imgpath = imgpath[:num]
            self.labpath = labpath[:num]
        self.augmentations = augmentations

    def __getitem__(self, index):
        imgs = [open_image(path) for i, path in enumerate(self.imgpath[index]) if i not in (2, 4)]
        lab = remove_invalid_values(open_image(self.labpath[index]))

        imgs = [resample_image(img, (256, 256)) for img in imgs]
        lab = resample_image(lab, (256, 256))

        img = np.concatenate(imgs, axis=2)  # the third dimension
        img = img.transpose((2, 0, 1)) # H W C => C H W

        img, _ = min_max_scaler(img)
        lab, _ = min_max_scaler(lab)

        img = torch.tensor(img.copy(), dtype=torch.float32)
        lab = torch.tensor(lab.copy(), dtype=torch.float32).unsqueeze(0)
        return img, lab

    def __len__(self):
        return len(self.imgpath)