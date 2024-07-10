'''
eval
'''

import torch
from tqdm import tqdm
from processing import inverse_min_max_scaler, untile_image

def evaluate(testdataloader, device, model):
    all_targets = []
    all_predictions = []
    images = []
    labels = []
    with torch.no_grad():
        temp_img = []
        temp_lab = []
        count = 0
        for x, y_true, arr, nameid in tqdm(testdataloader):
            x = x.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True)
            y_pred = model.forward(x)
            count += 1

            all_predictions.append(y_pred.cpu().numpy())
            all_targets.append(y_true.cpu().numpy())
            ypred = inverse_min_max_scaler(y_pred, arr) # scale pixel back to original resolution
            ytrue = inverse_min_max_scaler(y_true, arr)

            temp_img.append(ypred.cpu().numpy())
            temp_lab.append(ypred.cpu().numpy())

            if count >= 100:
                count = 0
                img = untile_image(temp_img, (2560, 2560), (256, 256))
                lab = untile_image(temp_lab, (2560, 2560), (256, 256))
                images.append((img, nameid))
                labels.append((lab, nameid))
                temp_img = []
                temp_lab = []
    return all_targets, all_predictions, images, labels