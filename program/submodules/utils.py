import matplotlib.pyplot as plt
import os, logging, torch
import numpy as np
from PIL import Image
from processing import resample_image

def adjust_learning_rate(optimizer, epoch, learning_rate):
    if epoch <= 200:
        lr = learning_rate
    elif epoch <= 250:
        lr = learning_rate * 0.1
    elif epoch <= 300:
        lr = learning_rate * 0.01
    else:
        lr = learning_rate * 0.025  # 0.0025 before
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def extract_epoch(filename):
    try:
        return int(filename.split('_')[-1].split('.')[0])
    except (IndexError, ValueError):
        return -1
    

def plot_rmse(train, test):
    train = train[::15]
    test = test[::15]
    plt.plot(train, label="Train rmse")
    plt.plot(test, label="Test rmse")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Training & Test RMSE")
    plt.grid()
    plt.savefig("results/rmse.png")
    plt.close()


def plot_metrics(r2, ev):
    r2 = r2[::15]
    ev = ev[::15]
    plt.plot(r2, label="R^2")
    plt.plot(ev, label="Explained varience")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (out of 1)")
    plt.legend()
    plt.title("Accuracy metrics")
    plt.grid()
    plt.savefig("results/variance.png")
    plt.close()


def plot_errors(mse, mae):
    mse = mse[::15]
    mae = mae[::15]
    plt.plot(mse, label="mean squared error")
    plt.plot(mae, label="mean absolute error")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.title("MSE vs MAE")
    plt.grid()
    plt.savefig("results/meanerrors.png")
    plt.close()


def plot_histogram(all_targets_flat, all_preds_flat, region):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_targets_flat, bins=50, alpha=0.75, color='blue', label='Targets')
    plt.title('Histogram of All Targets')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(all_preds_flat, bins=50, alpha=0.75, color='green', label='Predictions')
    plt.title('Histogram of All Predictions')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    region = "results/" + region + "/height.png"
    plt.tight_layout()
    plt.savefig(region)


def plot_overlay(images, region, data_path):
    for img, nameid in images:
        original_image_path = os.path.join(data_path, nameid[0])
        original_image = Image.open(original_image_path).convert("RGB")
        original_image_np = np.array(original_image)

        resampled_original_image_np = resample_image(original_image_np, (2560, 2560))
        resampled_original_image = Image.fromarray((resampled_original_image_np * 255).astype(np.uint8))

        # Convert predicted height values to an image format
        predicted_image = Image.fromarray((img * 255).astype(np.uint8), 'L')
        predicted_image_np = np.array(predicted_image)
        resampled_predicted_image_np = resample_image(predicted_image_np, (2560, 2560))
        resampled_predicted_image = Image.fromarray((resampled_predicted_image_np * 255).astype(np.uint8))

        grayscale_save_path = os.path.join("predicted_images", f"predicted_{nameid[0]}")
        os.makedirs(os.path.dirname(grayscale_save_path), exist_ok=True)
        resampled_predicted_image.save(grayscale_save_path)

        # Overlay predicted image onto the original image
        overlay_image = Image.blend(resampled_original_image, resampled_predicted_image.convert("RGB"), alpha=0.5)
        overlay_save_path = os.path.join("predictions", f"{region}_{nameid[0]}.png")
        os.makedirs(os.path.dirname(overlay_save_path), exist_ok=True)
        
        # Display the overlayed image
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay_image)
        plt.title(f"Overlay of {nameid[0]}")
        plt.axis('off')
        plt.savefig(overlay_save_path)
        plt.close()
        

def checkpoints(model, checkpoint_dir, start_epoch, highest_r2, best_epoch, epochs):
    if os.path.exists(checkpoint_dir):
        checkpoint_files = os.listdir(checkpoint_dir)
        checkpoint_files = [f for f in checkpoint_files if f.startswith('checkpoint_epoch_')]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=extract_epoch)
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            highest_r2 = checkpoint.get('highest_r2', highest_r2)
            best_epoch = checkpoint.get('best_epoch', best_epoch)
            logging.info(f"Loaded checkpoint from epoch {start_epoch} with highest r2: {highest_r2} from epoch {best_epoch}")
            if start_epoch >= epochs - 1:
                logging.warning("Maximum training epochs reached. Adjust the number of epochs and try again.")
                exit(0)
            return model, start_epoch, highest_r2, best_epoch
        else:
            logging.info("Starting from Scratch")
            return model, start_epoch, highest_r2, best_epoch
    else:
        logging.info("Checkpoint directory does not exist. Starting from Scratch")
        return model, start_epoch, highest_r2, best_epoch


def save_model(r2, highest_r2, best_epoch, epoch, model, train_loss, val_rmse, checkpoint_dir):
    # Save models
    if r2 > highest_r2:
        highest_r2 = r2
        best_epoch = epoch
        savefilename = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': np.mean(train_loss),
            'test_loss': val_rmse,
            'highest_r2': highest_r2,
            'best_epoch': best_epoch,
        }, savefilename)
        #logging.info(f"New best R2: {highest_r2} at epoch {best_epoch}. Saved checkpoint.")
    return highest_r2, best_epoch