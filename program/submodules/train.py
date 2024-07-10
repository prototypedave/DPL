'''
training code
'''
import numpy as np
from tqdm import tqdm
from submodules.utils import adjust_learning_rate

def train_epoch(optimizer, epoch, learning_rate, traindataloader, device, criterion, model):
    adjust_learning_rate(optimizer, epoch, learning_rate)
    model.train() 
    train_mse = 0.
    count = 0
    print_count = 0

    for x, y_true in tqdm(traindataloader):
        x = x.to(device, non_blocking=True)
        y_true = y_true.to(device, non_blocking=True)

        ypred1 = model.forward(x)
        loss = criterion(ypred1, y_true)
        loss_mse = loss.cpu().detach().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #train_loss.append(loss.cpu().detach().numpy())
        train_mse += loss_mse*x.shape[0]
        count += x.shape[0]

        if print_count%20 ==0:
            print('training loss %.3f, rmse %.3f' %
                (loss.item(), np.sqrt(loss_mse)))
        print_count += 1

    return np.sqrt(train_mse/count)
