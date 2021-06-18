import pytorch_lightning as pl
import torch.nn as nn
import torch
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from evaluation.misc_metrics import mean_percent_error as mpe

class ASPDNetLightning(pl.LightningModule):

    """
    A PyTorch Lightning wrapper for ASPDNet from Gao et al. (2020).
    See 'Counting from Sky: A Large-scale Dataset for Remote Sensing Object Counting and A Benchmark Method.'
    I've done my best to recreate their training procedure using the cleaner PyTorch Lightning framework.
    Inputs:
      - model: the ASPDNet PyTorch model to use for training/testing
      - lr: the learning rate for use in the optimizer
    """

    def __init__(self, model, lr):
        super().__init__()

        self.model = model
        self.learning_rate = lr

    def forward(self, X):
        preds = torch.stack([self.model(tile.unsqueeze(0)).squeeze() for tile in X]) #prediction is one tile at a time, like in the source training script

        return preds

    def training_step(self, batch, batch_idx):
        X, y, _ = batch

        #  here, we train one tile at a time
        preds = torch.stack([self.model(tile.unsqueeze(0)).squeeze() for tile in X]) #the pred densities for the input images
        loss = nn.functional.mse_loss(preds, y, reduction = 'sum') #this MSE is a different than in the metrics... looks at per-pixel density mismatches between GT and pred

        return loss

    def validation_step(self, batch, batch_idx):
        X, y, tile_counts = batch

        preds = torch.stack([self.model(tile.unsqueeze(0)).squeeze() for tile in X])
        pred_count = int(preds.sum())
        gt_count = sum(tile_counts)

        return {'pred_count' : pred_count, 'gt_count' : gt_count}

    def validation_epoch_end(self, outs):
        pred_counts = [out['pred_count'] for out in outs]
        gt_counts = [out['gt_count'] for out in outs]

        count_rmse = mse(gt_counts, pred_counts, squared = False)
        count_mae = mae(gt_counts, pred_counts)
        count_mpe = mpe(gt_counts, pred_counts)

        self.log('Val_RMSE', count_rmse)
        self.log('Val_MAE', count_mae)
        self.log('Val_MPE', count_mpe)

    def test_step(self, batch, batch_idx):
        X, y, tile_counts = batch #1 batch == 1 parent image!

        preds = torch.stack([self.model(tile.unsqueeze(0)).squeeze() for tile in X]) #a bunch of densities
        pred_count = int(preds.sum()) #predicted count over all tiles (so, this is the pred parent image count)
        gt_count = sum(tile_counts) #true count over all tiles

        return {'pred_count' : pred_count, 'gt_count' : gt_count}

    def test_epoch_end(self, outs):
        pred_counts = [out['pred_count'] for out in outs] #one value here for each parent image
        gt_counts = [out['gt_count'] for out in outs]

        count_rmse = mse(gt_counts, pred_counts, squared = False) #these count metrics are at the parent image level
        count_mae = mae(gt_counts, pred_counts)
        count_mpe = mpe(gt_counts, pred_counts)

        self.log('Test_RMSE', count_rmse)
        self.log('Test_MAE', count_mae)
        self.log('Test_MPE', count_mpe)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    self.learning_rate,
                                    momentum = 0.95, #these are hyperparams from the paper's source code...
                                    weight_decay = 0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, #seems roughly equivalent to their LR scheduler
                                                       step_size = 30,
                                                       gamma = 0.1)

        return {'optimizer' : optimizer, 'lr_scheduler' : lr_scheduler}

if __name__ == '__main__':
    #TESTING ASPDNetLightning:
    import json
    from torch.utils.data import DataLoader
    from pytorch_lightning import Trainer

    import sys
    sys.path.append('/Users/emiliolr/Desktop/counting-cranes/density_estimation/ASPDNet')
    sys.path.append('/Users/emiliolr/Desktop/counting-cranes/')
    from ASPDNet.model import ASPDNet
    from bird_dataset import *
    from utils import get_bboxes

    config = json.load(open('/Users/emiliolr/Desktop/counting-cranes/config.json', 'r'))
    DATA_FP = config['data_filepath_local']

    bird_dataset = BirdDataset(root_dir = DATA_FP,
                               transforms = get_transforms(train = False),
                               tiling_method = 'w_o_overlap',
                               annotation_mode = 'points',
                               tile_size = (200, 200))
    bird_subset = torch.utils.data.Subset(bird_dataset, [0, 1])
    dataloader = DataLoader(bird_subset,
                            batch_size = 1,
                            shuffle = True,
                            collate_fn = collate_tiles_density)

    # save_name = '/Users/emiliolr/Desktop/counting-cranes/initial_ASPDNet.pth'
    model = ASPDNet()
    # model.load_state_dict(torch.load(save_name))
    pl_model = ASPDNetLightning(model = model, lr = 1e-5)
    # tiles = torch.randn(5, 3, 200, 200)
    # preds = pl_model(tiles)
    # print(preds.shape, type(preds))
    # for p in preds:
    #     print(p.sum().item())

    trainer = Trainer(max_epochs = 1)
    trainer.fit(pl_model, train_dataloader = dataloader, val_dataloaders = dataloader)
    # trainer.test(pl_model, test_dataloaders = dataloader)
