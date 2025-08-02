# YOLO_Circle
YOLO variant for circular object detection.

## Introduction
We use YOLOv3 as the backbone network for circular object detection.  The YOLOv3 model is trained on the training set and evaluated on the testing set. The trained model is used to detect the circular objects in the testing set. The detection results are compared with the ground truth annotations to evaluate the performance of the model.
## Dataset
The CIRCLE dataset is not available now. We will provide the dataset later. The format of the dataset used for training and testing is the same as the plain YOLO dataset (5 columns: x, y, width, height, class_label). The CIRCLE dataset (self-dataset) contains 618 images of circular objects with same shapes and sizes. The dataset is divided into 80% training set and 20% testing set.
## Model
The YOLOv3 model is used as the backbone network for circular object detection. The model is trained on the CIRCLE dataset and evaluated on the testing set. The model is trained with the default hyperparameters. 
## Results  
The trained model is used to detect the circular objects in the testing set. The detection results are compared with the ground truth annotations to evaluate the performance of the model. The evaluation metrics used are precision, recall, and F1-score. The results are shown in the following table:

| Model backbone | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| YOLOv3 | 0.99      | 0.99   | 0.99     |

The model achieves a high precision, recall, and F1-score on the CIRCLE dataset. The model can be further improved by using more complex models or more training data.
## IoU metric
The Intersection over Union (IoU) metric is used to evaluate the performance of the model. The IoU metric measures the overlap between the predicted bounding circle and the ground truth bounding circle. Here is the visualization of the IoU metric:

![IoU metric](https://github.com/zhenmanzhang/YOLO_Circle/blob/main/IoU_img/image.png)

## Conclusion
The YOLO variant model can be used for circular object detection. The CIRCLE dataset is a self-dataset that contains 618 images of circular objects with same shapes and sizes. The model is trained on the CIRCLE dataset and evaluated on the testing set. The model achieves a high precision, recall, and F1-score on the CIRCLE dataset. The model can be further improved by using more complex models or more training data.!