import pytorch_lightning as pl
import torch.nn as nn
import torch

class ASPDNetLightning(pl.LightningModule):

    """
    A PyTorch Lightning wrapper for ASPDNet from Gao et al. (2020).
    See 'Counting from Sky: A Large-scale Dataset for Remote Sensing Object Counting and A Benchmark Method.'
    I've done my best to recreate their training procedure, but in the PyTorch Lightning framework.
    Inputs:
      - model: the ASPDNet PyTorch model to use for training/testing
      - lr: the learning rate for use in the optimizer
    """

    def __init__(self, model, lr):
        super.__init__()

        self.model = model
        self.learning_rate = lr

    def forward(self, X):
        return self.model(X)

    #TODO: we need to downsample GT densities for loss calculations to work correctly!
    def training_step(self, batch, batch_idx):
        X, y = batch

        pred = self.model(X) #the pred densities for the inputs
        loss = nn.functional.mse_loss(pred, y, reduction = 'sum') #this MSE is a different than in the metrics... looks at per-pixel density mismatches between GT and pred

        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch #1 batch == 1 parent image!

        pred = self.model(X) #a bunch of densities
        pred_count = pred.sum() #predicted count over all tiles (so, this is the pred parent image count)
        gt_count = y.sum() #actual parent image count

        return {'pred_count' : pred_count, 'gt_count' : gt_count}

    def test_epoch_end(self, outs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    self.learning_rate,
                                    momentum = 0.95,
                                    weight_decay = 0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size = 30,
                                                       gamma = 0.1)

        return {'optimizer' : optimizer, 'lr_scheduler' : lr_scheduler}

if __name__ == '__main__':
    #TESTING ASPDNetLightning:
    import sys
    sys.path.append('/Users/emiliolr/Desktop/counting-cranes/density_estimation/ASPDNet')
    from ASPDNet.model import ASPDNet
    from torchsummary import summary

    model = ASPDNet()
    # summary(model, (3, 200, 200))
    toy_img = torch.randn(5, 3, 400, 400)
    pred = model(toy_img)
    print(pred.shape)
