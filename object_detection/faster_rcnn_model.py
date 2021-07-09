import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pytorch_lightning as pl
from itertools import chain
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from evaluation.pascal_voc_evaluator import get_pascalvoc_metrics
from evaluation.enumerators import MethodAveragePrecision
from evaluation.utils import from_dict_to_BoundingBox
from evaluation.misc_metrics import mean_percent_error as mpe

def get_faster_rcnn(backbone = 'ResNet50', num_classes = 2, pretrained = False, **kwargs):

    """
    A convenience function to get a pre-trained Faster R-CNN model w/a ResNet50 backbone.
    Inputs:
      - backbone: one of ResNet50, ResNet101, or MobileNetV3
      - num_classes: the number of classes to predict
      - pretrained: use Faster R-CNN pretrained on COCO?
      - **kwargs: to be passed on to the Faster R-CNN constructor (mostly hyperparameters)
    Outputs:
      - A PyTorch model
    """

    #Loading a model (pre-training is on COCO)
    if backbone == 'ResNet50':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = pretrained, **kwargs)
    elif backbone == 'ResNet101':
        pass
    elif backbone == 'MobileNetV3':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained = pretrained, **kwargs)

    #Replace the classifier - get input features from the existing model pipeline and then replace!
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

class FasterRCNNLightning(pl.LightningModule):

    """
    A PyTorch Lightning wrapper for Faster R-CNN.
    Inputs:
      - model: the Faster R-CNN PyTorch model to be used
      - lr: the learning rate for use in the optimizer
      - iou_threshold: the threshold to use for validation/test IoU calculations
    """

    def __init__(self, model, lr = 0.001, iou_threshold = 0.5):
        super().__init__()

        self.model = model
        self.learning_rate = lr
        self.iou_threshold = iou_threshold

    def forward(self, X):
        self.model.eval() #adding this in b/c forward pass behavior changes if we're in train mode!

        return self.model(X)

    def predict_counts(self, X):
        self.model.eval()

        preds = self.model(X)
        pred_counts = [len(p['boxes'].tolist()) for p in preds]

        return pred_counts

    def training_step(self, batch, batch_idx):
        X, y, img_fp, annot_fp = batch

        loss_dict = self.model(X, y) #at train time, this is a dictionary of losses
        loss = sum([loss for loss in loss_dict.values()])

        return loss

    def training_epoch_end(self, outs):
        self.log('Total_loss', outs[-1]['loss']) #log the loss from the final batch of the epoch

    def validation_step(self, batch, batch_idx):
        X, y, X_name, y_name = batch

        preds = self.model(X)

        #Getting our predictions into the correct format for AP calculation
        gt_boxes = [from_dict_to_BoundingBox(target, name = name, groundtruth = True) for target, name in zip(y, X_name)]
        gt_boxes = list(chain(*gt_boxes))

        pred_boxes = [from_dict_to_BoundingBox(pred, name = name, groundtruth = False) for pred, name in zip(preds, y_name)]
        pred_boxes = list(chain(*pred_boxes))

        return {'pred_boxes' : pred_boxes, 'gt_boxes' : gt_boxes}

    def validation_epoch_end(self, outs):
        gt_boxes = [out['gt_boxes'] for out in outs] #ground truth
        gt_counts = [len(gt) for gt in gt_boxes] #parent image ground truth counts... remember: a single batch is one (tiled) parent image!
        gt_boxes = list(chain(*gt_boxes))

        pred_boxes = [out['pred_boxes'] for out in outs] #predicted
        pred_counts = [len(pred) for pred in pred_boxes] #parent image pred counts
        pred_boxes = list(chain(*pred_boxes))

        metric = get_pascalvoc_metrics(gt_boxes = gt_boxes,
                                       det_boxes = pred_boxes,
                                       iou_threshold = self.iou_threshold,
                                       method = MethodAveragePrecision.EVERY_POINT_INTERPOLATION)

        per_class = metric['per_class'][1] #we only need metrics from class 1, our only class...
        AP = per_class['AP']
        TP_num = per_class['total TP']
        FP_num = per_class['total FP']

        #Logging key metrics
        count_rmse = mse(gt_counts, pred_counts, squared = False)
        count_mae = mae(gt_counts, pred_counts)
        count_mpe = mpe(gt_counts, pred_counts)

        self.log('Val_AP', AP)
        self.log('Val_TP', TP_num)
        self.log('Val_FP', FP_num)
        self.log('Val_RMSE', count_rmse)
        self.log('Val_MAE', count_mae)
        self.log('Val_MPE', count_mpe)

    def test_step(self, batch, batch_idx):
        X, y, X_name, y_name = batch

        preds = self.model(X)

        gt_boxes = [from_dict_to_BoundingBox(target, name = name, groundtruth = True) for target, name in zip(y, X_name)]
        gt_boxes = list(chain(*gt_boxes))

        pred_boxes = [from_dict_to_BoundingBox(pred, name = name, groundtruth = False) for pred, name in zip(preds, y_name)]
        pred_boxes = list(chain(*pred_boxes))

        return {'pred_boxes' : pred_boxes, 'gt_boxes' : gt_boxes}

    def test_epoch_end(self, outs):
        gt_boxes = [out['gt_boxes'] for out in outs] #ground truth
        gt_counts = [len(gt) for gt in gt_boxes]
        gt_boxes = list(chain(*gt_boxes))

        pred_boxes = [out['pred_boxes'] for out in outs] #predicted
        pred_counts = [len(pred) for pred in pred_boxes]
        pred_boxes = list(chain(*pred_boxes))

        metric = get_pascalvoc_metrics(gt_boxes = gt_boxes,
                                       det_boxes = pred_boxes,
                                       iou_threshold = self.iou_threshold,
                                       method = MethodAveragePrecision.EVERY_POINT_INTERPOLATION)

        per_class = metric['per_class'][1] #we only need metrics from class 1, our only class...
        AP = per_class['AP']
        TP_num = per_class['total TP']
        FP_num = per_class['total FP']

        count_rmse = mse(gt_counts, pred_counts, squared = False) #these count metrics are based on parent image counts - slightly inflated due to tiling (same bird appears in many tiles)...
        count_mae = mae(gt_counts, pred_counts)
        count_mpe = mpe(gt_counts, pred_counts)

        self.log('Test_AP', AP)
        self.log('Test_TP', TP_num)
        self.log('Test_FP', FP_num)
        self.log('Test_RMSE', count_rmse)
        self.log('Test_MAE', count_mae)
        self.log('Test_MPE', count_mpe)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), #using the optimizer setup from the original Faster R-CNN paper
                                    lr = self.learning_rate,
                                    momentum = 0.9,
                                    weight_decay = 0.005)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  mode = 'min',
                                                                  factor = 0.5,
                                                                  patience = 5,
                                                                  min_lr = 0)

        return {'optimizer' : optimizer, 'lr_scheduler' : lr_scheduler, 'monitor' : 'Val_MAE'}

#TESTS:
if __name__ == '__main__':
    #TESTING get_faster_rcnn FUNCTION:
    fake_batch = [torch.randn(3, 200, 200), torch.randn(3, 200, 200)]
    # model = get_faster_rcnn(backbone = 'MobileNetV3', num_classes = 2)

    #  trying a forward pass
    # model.eval()
    # predict = model(fake_batch)
    # print(predict)

    #TRYING OUT PYTORCH LIGHTNING CLASS:
    model = get_faster_rcnn(backbone = 'ResNet50', num_classes = 2, box_detections_per_img = 500)
    # model_fp = '/Users/emiliolr/Desktop/counting-cranes/initial_faster_rcnn.pth'
    # model.load_state_dict(torch.load(model_fp))
    faster_rcnn = FasterRCNNLightning(model, iou_threshold = 0.3)
    # print(faster_rcnn)
    print(faster_rcnn.predict_counts(fake_batch))

    #TRYING OUT MODEL TESTING:
    # from pytorch_lightning import Trainer
    # import json
    # from torch.utils.data import DataLoader
    #
    # config = json.load(open('/Users/emiliolr/Desktop/counting-cranes/config.json', 'r'))
    # DATA_FP = config['data_filepath_local']
    #
    # import sys
    # sys.path.append('/Users/emiliolr/Desktop/counting-cranes')
    # from bird_dataset import *
    #
    # bird_dataset = BirdDataset(root_dir = DATA_FP, transforms = get_transforms(train = False), annotation_mode = 'bboxes', tiling_method = 'random', num_tiles = 5, max_neg_examples = 1)
    # subset = torch.utils.data.Subset(bird_dataset, [1, 5])
    # bird_dataloader = DataLoader(subset, batch_size = 1, shuffle = False, collate_fn = collate_tiles_object_detection)
    #
    # trainer = Trainer(max_epochs = 1)
    # trainer.fit(faster_rcnn, train_dataloader = bird_dataloader)
