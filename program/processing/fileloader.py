'''
Loads and returns dataset file names in arrays
'''
import os
import numpy as np
from os.path import join
#from pre import *
#from resampled import *

def load_files(filepath, split=[0.7, 0.1, 0.2]):
    '''
    :param filepath: the root dir of sen1, sen2 and lab
    :return: img train, lab train, img validate, lab validate
    split data into train/test/val=7:1:2
    '''
    if not os.path.exists(filepath):
        raise ValueError('The path of the dataset does not exist.')
    else:
        sen1_path = [join(filepath, 'sen1', name) for name in os.listdir(join(filepath, 'sen1')) if not name.startswith('.')]
        sen2_path = [join(filepath, 'sen2', name) for name in os.listdir(join(filepath, 'sen2')) if not name.startswith('.')]

    # sen 1
    vh_path = sen1_path[0]
    vv_path = sen1_path[1]
    # sen 2
    baei_path = sen2_path[0]
    ndbi_path = sen2_path[1]
    ndvi_path = sen2_path[2]
    ui_path = sen2_path[3]
    # lab
    lab = [join(filepath, "lab", name) for name in os.listdir(join(filepath, 'lab')) if not name.startswith('.')]

    vh = [join(vh_path, name) for name in os.listdir((vh_path)) if not name.startswith('.')]
    vv = [join(vv_path, name) for name in os.listdir((vv_path)) if not name.startswith('.')]

    baei = [join(baei_path, name) for name in os.listdir((baei_path)) if not name.startswith('.')]
    ndbi = [join(ndbi_path, name) for name in os.listdir((ndbi_path)) if not name.startswith('.')]
    ndvi = [join(ndvi_path, name) for name in os.listdir((ndvi_path)) if not name.startswith('.')]
    ui = [join(ui_path, name) for name in os.listdir((ui_path)) if not name.startswith('.')]

    # assert
    same_size_and_sort([vh, vv, baei, ndbi, ndvi, ui])
    
    num_samples=len(vh)
    vh=np.array(vh)
    vv=np.array(vv)
    lab=np.array(lab)
    baei=np.array(baei)
    ndbi=np.array(ndbi)
    ndvi=np.array(ndvi)
    ui=np.array(ui)

    seqpath = join(filepath, 'seq.txt')
    if os.path.exists(seqpath):
        seq = np.loadtxt(seqpath, delimiter=',')
    else:
        seq = np.random.permutation(num_samples)
        np.savetxt(seqpath, seq, fmt='%d', delimiter=',')
    seq = np.array(seq, dtype='int32')

    num_train = int(num_samples * split[0]) 
    num_val = int(num_samples * split[1])

    train = seq[0:num_train]
    val = seq[num_train:(num_train+num_val)]
    # test = seq[num_train:]

    imgt=np.vstack((vh[train], vv[train], baei[train], ndbi[train], ndvi[train], ui[train])).T
    labt=lab[train]

    imgv=np.vstack((vh[val], vv[val], baei[val], ndbi[val], ndvi[val], ui[val])).T
    labv=lab[val]
    return imgt, labt, imgv, labv

def same_size_and_sort(arr):
    for ar in arr:
        assert len(ar) == len(arr[0])
        ar.sort()


# Test code
if __name__ == '__main__':
    # TILE IMAGES AND SAVE FOR EASIER LOADING
    filepath = "samples"
    imgt, labt, _, _ = load_files(filepath, [1, 0])
    """for i in imgt:
        for j in i:
            parts = j.split('/')
            name = '/' + parts[1] + '/' + parts[2] + '/'
            file = parts[-1].split('.')[0]
            img = open_image(j)
            img = resample_image(img, (3000, 3000))
            tiles = tile_image(img, 300)
            save_tiles(tiles, ("images"+name), file)
    
    for k in labt:
        parts = k.split('/')
        name = '/' + parts[1] + '/'
        file = parts[-1].split('.')[0]
        lab = open_image(k)
        lab = resample_image(lab, (3000, 3000))
        tiles = tile_image(lab, 300)
        save_tiles(tiles, ("images"+name), file)"""