'''
prediction code
'''
import torch, random
import numpy as np
from processing import load_files, ImageLoader_val
from submodules import checkpoints, plot_histogram, evaluate, plot_overlay
from model import DPL

# seeds
torch.manual_seed((1337))
torch.cuda.manual_seed((1337))
np.random.seed((1337))
random.seed((1337))

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    checkpoint_dir = "checkpoints"
    data_path = "subregions/hamburg"
    batchsize = 1
    n_channels = 16
    epochs = 302

    _, _, valimg, vallab = load_files(data_path, [0.0, 1.0, 0.0])
    testdataloader = torch.utils.data.DataLoader(
        ImageLoader_val(valimg, vallab),
        batch_size=batchsize, shuffle=False, num_workers=2, pin_memory=True)
    model = DPL(num_input_channels=n_channels).to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    start_epoch = 0
    highest_r2 = float('-inf')
    best_epoch = -1
    model, start_epoch, highest_r2, best_epoch = checkpoints(model, checkpoint_dir, start_epoch, highest_r2, best_epoch, epochs)

    model.eval()
    all_targets, all_predictions, images, labels = evaluate(testdataloader, device, model)

    all_targets = np.concatenate(all_targets, axis=0).squeeze()  # shape (num_samples, height, width)
    all_predictions = np.concatenate(all_predictions, axis=0).squeeze()

    all_preds_flat = all_predictions.flatten()
    all_targets_flat = all_targets.flatten()

    # plot histogram
    plot_histogram(all_preds_flat, all_targets_flat, "hamburg")

    # plot overlayed image
    plot_overlay(images, "hamburg", data_path)


if __name__ == '__main__':
    main()
    