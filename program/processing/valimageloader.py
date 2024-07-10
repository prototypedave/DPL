'''
tiles and loads images for validation
'''
from processing.pre import *
from processing.resampled import *
import torch.utils.data as data
import torch
from processing.fileloader import load_files

class ImageLoader_val(data.Dataset):
    def __init__(self, imgpath, labpath, num= 0):  
        self.imgpath = imgpath
        self.labpath = labpath
        if num>0:
            self.imgpath = imgpath[:num]
            self.labpath = labpath[:num]
        self.images, self.labs, self.files = self.generate_tiles()

    def generate_tiles(self):
        # open image
        images = []
        files = []
        for index in self.imgpath:
            tiles = []
            files.append(index[0])   # vh file
            for i in range(len(index)):
                if i not in (2, 4):
                    img = open_image(index[i])
                    img = resample_image(img, (3000, 3000))
                    tile = tile_image(img, 300)
                    tiles.append(tile)
                   
            
            num_tiles = len(tiles[0])
            for n in range(num_tiles):
                tmp_array = [resample_image(tiles[i][n], (256,256)) for i in range(len(tiles))]
                images.append(tmp_array)
        lab = [remove_invalid_values(open_image(path)) for path in self.labpath]
        lab = [resample_image(img, (3000, 3000)) for path in lab]
        labs = []
        for lb in lab:
            tile = tile_image(lb, 300)
            for p in tile:
                labs.append(resample_image(p, (256,256)))
        assert len(images) == len(labs)
        return images, labs, files

    def __getitem__(self, index):
        img = self.images[index]
        img = np.concatenate(img, axis=2)
        img = img.transpose((2, 0, 1)) # H W C => C H W
        img, _ = min_max_scaler(img)
        lab = self.labs[index]
        lab, dim = min_max_scaler(lab)
        file = self.files[(index // 100)]

        img = torch.tensor(img.copy(), dtype=torch.float32)
        lab = torch.tensor(lab.copy(), dtype=torch.float32).unsqueeze(0)
        return img, lab, dim, file

    def __len__(self):
        return len(self.images)

'''
# Test code
if __name__ == '__main__':
    filepath = "subregions/hamburg"
    imgt, labt, _, _ = load_files(filepath, [1, 0, 0])
    dataset = ImageLoader_val(imgt, labt)
    img, lab, dim, file = dataset[0]
    print(img.shape, lab.shape, dim, file)
        '''
    