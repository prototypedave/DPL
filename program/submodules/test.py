'''
code for testing
'''

import torch, numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, root_mean_squared_error

def test_epoch(model, dataloader, device):
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for x, y_true in tqdm(dataloader):
            x = x.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True)
            y_pred = model.forward(x)
            
            all_predictions.append(y_pred.cpu().numpy())
            all_targets.append(y_true.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0).squeeze()  # shape (num_samples, height, width)
    all_predictions = np.concatenate(all_predictions, axis=0).squeeze()

    # Flatten the arrays to calculate the metrics
    all_preds_flat = all_predictions.flatten()
    all_targets_flat = all_targets.flatten()

    mse = mean_squared_error(all_targets_flat, all_preds_flat)
    rmse = root_mean_squared_error(all_targets_flat, all_preds_flat)
    mae = mean_absolute_error(all_targets_flat, all_preds_flat)
    r2 = r2_score(all_targets_flat, all_preds_flat)
    explained_variance = explained_variance_score(all_targets_flat, all_preds_flat)

    return mse, rmse, mae, r2, explained_variance