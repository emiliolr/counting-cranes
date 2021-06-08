import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pytorch_lightning as pl
from itertools import chain

from evaluation.pascal_voc_evaluator import get_pascalvoc_metrics
from evaluation.enumerators import MethodAveragePrecision
from evaluation.utils import from_dict_to_BoundingBox

#TODO: see medium article if you want to use a different backbone
# - github.com/johschmidt42/PyTorch-Object-Detection-Faster-RCNN-Tutorial/blob/master/faster_RCNN.py
def get_faster_rcnn(num_classes = 2):

    """
    A convenience function to get a pre-trained Faster R-CNN model w/a ResNet50 backbone.
    Inputs:
      - num_classes: the number of classes to predict
    Outputs:
      - A PyTorch model
    """

    #Loading a model pre-trained on COCO - increasing the maximum possible number of detections
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True,
                                                                 box_detections_per_img = 500)

    #Replace the classifier - get input features from the existing model pipeline and then replace!
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

#TODO:
#  - add logging to CSV
#  - do we want to be freezing parameters anywhere here... seems to be learning well, so maybe not!
class FasterRCNNLightning(pl.LightningModule):

    """
    A PyTorch Lightning wrapper for Faster R-CNN.
    Inputs:
      - model: the Faster R-CNN PyTorch model to be used
      - lr: the learning rate for use in the optimizer
      - iou_threshold: the threshold to use for IoU calculations, in the range [0, 1]
    """

    def __init__(self, model, lr = 0.001, iou_threshold = 0.5):
        super().__init__()

        self.model = model
        self.learning_rate = lr
        self.iou_threshold = iou_threshold

        # self.save_hyperparameters()

    def forward(self, X):
        self.model.eval() #adding this in b/c forward pass behavior changes if we're in train mode!
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X, y, img_fp, annot_fp = batch

        loss_dict = self.model(X, y) #at train time, this is a dictionary of losses
        loss = sum([loss for loss in loss_dict.values()])

        return loss

    #TODO: implement validation once testing is working!
    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outs):
        pass

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
        gt_boxes = list(chain(*gt_boxes))
        pred_boxes = [out['pred_boxes'] for out in outs] #predicted
        pred_boxes = list(chain(*pred_boxes))

        metric = get_pascalvoc_metrics(gt_boxes = gt_boxes,
                                       det_boxes = pred_boxes,
                                       iou_threshold = self.iou_threshold,
                                       method = MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                       generate_table = True)

        per_class = metric['per_class'][1] #we only need metrics from class 1, our only class...
        AP = per_class['AP']
        TP_num = per_class['total TP']
        FP_num = per_class['total FP']

        self.log('Test_AP', AP)
        self.log('TP_num', TP_num)
        self.log('FP_num', FP_num)

    def configure_optimizers(self):
        #TODO: add a learning rate scheduler

        #Using the hyperparams from the original Faster R-CNN paper
        optimizer = torch.optim.SGD(self.model.parameters(),
                               lr = self.learning_rate,
                               momentum = 0.9,
                               weight_decay = 0.005)

        return optimizer

#TESTS:
if __name__ == '__main__':
    #TESTING get_faster_rcnn FUNCTION:
    # fake_batch = torch.randn(2, 3, 224, 224)
    # model = get_faster_rcnn(num_classes = 2)

    #  trying a forward pass
    # model.eval()
    # predict = model(fake_batch)
    # print(predict)

    #TRYING OUT PYTORCH LIGHTNING CLASS:
    model = get_faster_rcnn(num_classes = 2)
    model_fp = '/Users/emiliolr/Desktop/counting-cranes/initial_faster_rcnn.pth'
    model.load_state_dict(torch.load(model_fp))
    faster_rcnn = FasterRCNNLightning(model, iou_threshold = 0.1)
    # print(faster_rcnn)

    #TRYING OUT MODEL TESTING:
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning import Trainer
    import json
    from torch.utils.data import DataLoader

    config = json.load(open('/Users/emiliolr/Desktop/counting-cranes/config.json', 'r'))
    DATA_FP = config['data_filepath_local']

    import sys
    sys.path.append('/Users/emiliolr/Desktop/counting-cranes')
    from bird_dataset import *

    bird_dataset = BirdDataset(root_dir = DATA_FP, transforms = get_transforms(), num_tiles = 2, max_neg_examples = 0)
    subset = torch.utils.data.Subset(bird_dataset, [1, 2, 3])
    bird_dataloader = DataLoader(subset, batch_size = 1, shuffle = True, collate_fn = collate_w_tiles)

    # logger = CSVLogger('/Users/emiliolr/Desktop/TEST_logs', name = 'first_experiment')
    trainer = Trainer()
    trainer.test(faster_rcnn, test_dataloaders = bird_dataloader)
