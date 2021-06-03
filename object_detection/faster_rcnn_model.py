import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pytorch_lightning as pl

#TODO: see medium article if you want to use a different backbone
# - github.com/johschmidt42/PyTorch-Object-Detection-Faster-RCNN-Tutorial/blob/master/faster_RCNN.py
def get_faster_rcnn(num_classes = 2):

    """
    A convenience function to get a pre-trained Faster R-CNN model w/a ResNet50 backbone.
    Inputs:
      - num_classes: the nummber of classes to predict
    Outputs:
      - A PyTorch model
    """

    #Loading a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)

    #Replace the classifier - get input features from the existing model pipeline and then replace!
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

#TODO: add logging?
class FasterRCNNLightning(pl.LightningModule):
    def __init__(self, model, lr = 0.001):
        super().__init__()

        self.model = model
        self.learning_rate = lr

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X, y = batch

        loss_dict = self.model(X, y) #at train time, this is a dictionary of losses
        loss = sum([loss for loss in loss_dict.values()])

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    #TODO: stopped here!    
    def test_step(self, batch, batch_idx):
        X, y = batch

        preds = self.model(X)
        true_bboxes = [t['boxes'] for t in y]
        pred_bboxes = [t['boxes'] for t preds]

        return {'pred_bboxes' : pred_bboxes, 'true_bboxes' : true_bboxes}

    def configure_optimizers(self):
        #TODO: add a learning rate scheduler
        optimizer = torch.optim.SGD(self.model.parameters(), #these are th
                               lr = self.learning_rate,
                               momentum = 0.9,
                               weight_decay = 0.005)

        return optimizer

if __name__ == '__main__':
    #TESTING get_faster_rcnn FUNCTION:
    fake_batch = torch.randn(2, 3, 224, 224)
    model = get_faster_rcnn(num_classes = 2)

    #Trying a forward pass
    model.eval()
    predict = model(fake_batch)
    print(predict)

    #TRYING OUT PYTORCH LIGHTNING CLASS:
