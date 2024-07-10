import torch.nn as nn

class HeightPredictionLoss(nn.Module):
    ''' Smooth l1 loss '''
    def __init__(self):
        super(HeightPredictionLoss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(self, y_pred, y_true):
        # Calculate Smooth L1 Loss (Huber Loss)
        loss = self.smooth_l1_loss(y_pred, y_true)
        
        return loss
    

class CombinedLoss(nn.Module):
    ''' MSELoss with L1Loss '''
    def __init__(self, mse_weight=1.0, l1_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
    
    def forward(self, outputs, targets):
        mse = self.mse_loss(outputs, targets)
        l1 = self.l1_loss(outputs, targets)
        return self.mse_weight * mse + self.l1_weight * l1