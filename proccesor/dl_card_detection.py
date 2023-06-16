import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A

from card_detection import four_point_transform

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


class DetectionModel:
    def __init__(self, weights):
        self.model = torch.load(weights, map_location=torch.device('cpu'))
        self.model.eval()
        preprocessing_fn = smp.encoders.get_preprocessing_fn('mobilenet_v2', 'imagenet')
        self.preprocessing_fn = A.Compose([
                                           A.Lambda(image=preprocessing_fn),
                                           A.Lambda(image=to_tensor)])

    def forward(self, image):
        image = cv2.resize(image, (480, 384))
        image_rgb = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.preprocessing_fn(image=image)
        image = torch.tensor(image['image']).unsqueeze(0)
        with torch.no_grad():
            image = self.model.forward(image)
        image = image.squeeze(0).permute(1, 2, 0).numpy() * 255
        image = image.astype(np.uint8)
        thresh = cv2.threshold(image, 225, 255, cv2.THRESH_BINARY)[1]

        contours,_ = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        image = four_point_transform(image_rgb, box)
        return image
