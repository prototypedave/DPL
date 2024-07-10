"""
 Training
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import logging, torch, random
import numpy as np, torch.nn as nn, torch.optim as optim
from model import DPL
from submodules import checkpoints, train_epoch, save_model, test_epoch
from submodules import plot_rmse, plot_metrics, plot_errors
from processing import load_files, ImageLoader

## Hyper parameters
learning_rate = 0.001
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## setup seeds
torch.manual_seed((1337))
torch.cuda.manual_seed((1337))
np.random.seed((1337))
random.seed((1337))

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    checkpoint_dir = "checkpoints"
    data_path = "images"
    batchsize = 32
    epochs = 300
    n_channels = 16

    trainimg, trainlab, _, _ = load_files(data_path)
    traindataloader = torch.utils.data.DataLoader(
        ImageLoader(trainimg, trainlab, True, num=3),
        batch_size=batchsize, shuffle=True, num_workers=4, pin_memory=True)   
    testdataloader = torch.utils.data.DataLoader(
        ImageLoader(trainimg, trainlab, False, num=3),
        batch_size=batchsize, shuffle=False, num_workers=8, pin_memory=True)

    # model
    model = DPL(num_input_channels=n_channels).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 0
    highest_r2 = float('-inf')
    best_epoch = -1

    model, start_epoch, highest_r2, best_epoch = checkpoints(model, checkpoint_dir, start_epoch, highest_r2, best_epoch, epochs)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    mse_t = []
    mae_t = []
    val_t = []
    r2_t = []
    ev_t = []
    train_loss = []
    for epoch in range(epochs-start_epoch):
        train_rmse = train_epoch(optimizer, epoch, learning_rate, traindataloader, device, criterion, model)
        train_loss.append(train_rmse)
        mse, val_rmse, mae, r2, ev = test_epoch(model, testdataloader, device)
        mse_t.append(mse)
        val_t.append(val_rmse)
        r2_t.append(r2)
        ev_t.append(ev)
        mae_t.append(mae)

        print('Epoch %.0f, Metrics: rmse %.3f, mae %.3f, mse %.3f, R^2 %.3f, Explained Variance %.3f' %
                (epoch, val_rmse, mae, mse, r2, ev))

        # Save models
        highest_r2, best_epoch = save_model(r2, highest_r2, best_epoch, epoch, model, train_loss, val_rmse, checkpoint_dir)

    plot_rmse(train_loss, val_t)
    plot_metrics(r2_t, ev_t)
    plot_errors(mse_t, mae_t)
    

if __name__ == '__main__':
    main()