#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：yolo_circle
@File    ：yolo_circle_main.py
@IDE     ：PyCharm
@Author  ：ZhenmanZhang
@Date    ：2025/8/2 12:18
"""


import os
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

import cv2
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

import albumentations as A
from albumentations.pytorch import ToTensorV2
NO_ALBUMENTATIONS_UPDATE = 1 

from tensorboardX import SummaryWriter



def circle_iou(circle1, circle2, is_pred=True):
    if isinstance(circle1, np.ndarray):
        circle1 = torch.tensor(circle1)
    if isinstance(circle2, np.ndarray):
        circle2 = torch.tensor(circle2)

    if is_pred:
        # circle1 (prediction) and circle2 (label) are both in [x, y, r] format
        # Circle coordinates of prediction
        c1_x = circle1[..., 0:1]
        c1_y = circle1[..., 1:2]
        c1_r = circle1[..., 2:3]

        # Circle coordinates of ground truth
        c2_x = circle2[..., 0:1]
        c2_y = circle2[..., 1:2]
        c2_r = circle2[..., 2:3]

        # Make sure the intersection is at least 0
        d_sq = (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2
        d_c = torch.sqrt(d_sq)  # distance between centers

        # create a empty tensor to store the iou score
        # case1 : circles do not overlap
        iou_score = torch.zeros_like(d_c)

        # case2 : one circle is inside the other
        mask_2 = d_c <= abs(c1_r - c2_r)
        min_r = torch.min(c1_r, c2_r)
        max_r = torch.max(c1_r, c2_r)
        iou_score_2 = min_r ** 2 / max_r ** 2
        iou_score = iou_score + iou_score_2*mask_2

        # case3 : circles overlap
        mask_3 = (d_c < (c1_r + c2_r)) & (d_c > abs(c1_r - c2_r))
        theta1 = 2 * torch.acos((c1_r ** 2 + d_sq - c2_r ** 2) / (2 * c1_r * d_c))
        theta2 = 2 * torch.acos((c2_r ** 2 + d_sq - c1_r ** 2) / (2 * c2_r * d_c))
        area1 = 0.5 * c1_r ** 2 * (theta1 - torch.sin(theta1))
        area2 = 0.5 * c2_r ** 2 * (theta2 - torch.sin(theta2))
        intersection = area1 + area2
        intersection = torch.where(torch.isnan(intersection), torch.tensor(0.0), intersection)
        union = torch.pi * (c1_r ** 2 + c2_r ** 2)-intersection
        iou_score_3 = intersection / union
        iou_score = iou_score + iou_score_3*mask_3
        return iou_score
    else:
        # Calculate intersection area
        intersection_area = (torch.min(circle1[...,0],circle2[..., 0])/2)**2

        # Calculate union area
        union_area = (torch.max(circle1[...,0],circle2[..., 0])/2)**2 - intersection_area

        # Calculate IoU score
        iou_score = intersection_area / union_area

        # Return IoU score
        return iou_score


def nms_ciecles(circles, iou_threshold, threshold):
    # Filter out bounding boxes with confidence below the threshold.
    circles = [c for c in circles if c[1] > threshold]

    # Sort the bounding circles by confidence in descending order.
    sorted_circles  = np.array(sorted(circles, key=lambda x: x[1], reverse=True))

    # Initialize the list of bounding circles after non-maximum suppression.
    circles_nms = []

    while len(sorted_circles) > 0 :
        best_circle = sorted_circles[0]
        circles_nms.append(best_circle)
        ious = circle_iou(best_circle[2:], sorted_circles[1:][...,2:],is_pred=True)
        ious = ious.numpy()
        mask = ious < iou_threshold
        binary_mask = mask.astype(bool).squeeze(1)
        sorted_circles = sorted_circles[1:][binary_mask]

    # Return bounding circles after non-maximum suppression.
    return circles_nms


def convert_cells_to_circles(predictions, anchors, s, is_predictions=True):
    # Batch size used on predictions
    batch_size = predictions.shape[0]
    
    # Number of anchors
    num_anchors = len(anchors)
    
    # List of all the predictions
    circle_predictions = predictions[..., 1:4]

    # If the input is predictions then we will pass the x and y coordinate
    # through sigmoid function and width and height to exponent function and
    # calculate the score and best class.
    if is_predictions:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 1)
        circle_predictions[..., 0:2] = torch.sigmoid(circle_predictions[..., 0:2])
        circle_predictions[..., 2:] = torch.exp(circle_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 4:], dim=-1).unsqueeze(-1)

    # Else we will just calculate scores and best class.
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 4:5]

    # Calculate cell indices
    cell_indices = (torch.arange(s).repeat(predictions.shape[0], 3, s, 1).unsqueeze(-1).to(predictions.device))

    # Calculate x, y, diameter with proper scaling
    x = 1 / s * (circle_predictions[..., 0:1] + cell_indices)
    y = 1 / s * (circle_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    diameter = 1 / s * circle_predictions[..., 2:3]

    # Concatinating the values and reshaping them in (BATCH_SIZE, num_anchors * S * S, 5) shape
    converted_circles = torch.cat((best_class, scores, x, y, diameter), dim=-1).reshape(batch_size, num_anchors * s * s, 5)

    # Returning the reshaped and converted bounding circle list
    return converted_circles.tolist()


def plot_image_circles(image, circles,bi,savefig=False):
    # Getting the color map from matplotlib
    colour_map = plt.get_cmap("rainbow")
    
    # Getting 20 different colors from the color map for 20 different classes
    colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))]

    # Reading the image with OpenCV
    img = np.array(image)
    # Getting the height and width of the image
    h, w, _ = img.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Add image to plot
    ax.imshow(img)


    # Plotting the bounding boxes and labels over the image
    for circle in circles:
        # Get the class from the circle
        class_pred = circle[0]
        # Get the center x and y coordinates
        circle_x = circle[2:3]*w
        circle_y = circle[3:4]*h
        circle_r = circle[4:5]*w/2


        # Create a Circle patch with the bounding circle
        circ = patches.Circle(
            (circle_x, circle_y),
            circle_r,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none")


        # Add the patch to the Axes
        ax.add_patch(circ)

        # Add class name to the patch
        # plt.text(
        #     circle_x-circle_r,
        #     circle_y-circle_r,
        #     s=class_labels[int(class_pred)],
        #     color="white",
        #     verticalalignment="top",
        #     bbox={"color": colors[int(class_pred)], "pad": 0},
        # )

    # Display the plot
    plt.tight_layout()
    if savefig:
        plt.savefig(f"./results_fig/sample_circle_{bi}.png")
    plt.show()

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("==> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("==> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class YoloCircleDataset(Dataset):
    def __init__(
            self, image_dir, label_dir, anchors,
            image_size=512, grid_sizes=[16, 32, 64],
            num_classes=1, transform=None
    ):
        # Read the csv file with image names and labels
        # self.label_list = pd.read_csv(csv_file)
        # Image and label directories
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = os.listdir(image_dir)
        # Image size
        self.image_size = image_size
        # Transformations
        self.transform = transform
        # Grid sizes for each scale
        self.grid_sizes = grid_sizes
        # Anchor boxes
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        # Number of anchor boxes
        self.num_anchors = self.anchors.shape[0]
        # Number of anchor boxes per scale
        self.num_anchors_per_scale = self.num_anchors // 3
        # Number of classes
        self.num_classes = num_classes
        # Ignore IoU threshold
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace(".jpg", ".txt"))

        # Getting the label path
        # label_path = os.path.join(self.label_dir, self.label_list.iloc[idx, 1])
        # We are applying roll to move class label to the last column
        # 5 columns: x, y, width, height, class_label
        bboxes = np.roll(np.loadtxt(fname=label_path,delimiter=" ", ndmin=2), 4, axis=1).tolist()

        # Getting the image path
        # img_path = os.path.join(self.image_dir, self.label_list.iloc[idx, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        # Albumentations augmentations
        if self.transform:
            augs = self.transform(image=image, bboxes=bboxes)
            image = augs["image"]
            bboxes = augs["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        # target : [probabilities, x, y, d, class_label]
        targets = [torch.zeros((self.num_anchors_per_scale, g_s, g_s, 5)) for g_s in self.grid_sizes]

        # Identify anchor box and cell for each bounding circle
        for box in bboxes:

            x, y, width, height, class_label = box
            # d_max = max(width, height) # here is the diameter of the bounding circle
            d_min = min(width, height)
            temp_circle = [x, y, d_min, class_label] # the maximize of width and height is the diameter of the bounding circle

            # Calculate iou of bounding circle with anchor boxes
            iou_anchors = circle_iou(torch.tensor(temp_circle[2:3]), self.anchors,is_pred=False)
            # Selecting the best anchor box
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)

            # At each scale, assigning the bounding circle to the best matching anchor box
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale 
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale 

                # Identifying the grid size for the scale
                s = self.grid_sizes[scale_idx]

                # Identifying the cell to which the bounding circle belongs
                i, j = int(s * y), int(s * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                # Check if the anchor box is already assigned
                if not anchor_taken and not has_anchor[scale_idx]:

                    # Set the probability to 1
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # Calculating the center of the bounding circle relative to the cell
                    x_cell, y_cell = s * x - j, s * y - i

                    # Calculating the diameter of the bounding circle relative to the cell
                    d_cell = d_min*s

                    # Idnetify the circle coordinates
                    circle_coordinates = torch.tensor([x_cell, y_cell, d_cell])

                    # Assigning the circle coordinates to the target
                    targets[scale_idx][anchor_on_scale, i, j, 1:4] = circle_coordinates

                    # Assigning the class label to the target
                    targets[scale_idx][anchor_on_scale, i, j, 4] = int(class_label)

                    # Set the anchor box as assigned for the scale
                    has_anchor[scale_idx] = True

                # If the anchor box is already assigned, check if the IoU is greater than the threshold
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    # Set the probability to -1 to ignore the anchor box
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        # Return the image and the target
        return image, tuple(targets)

def get_transforms(temp="train"):
    if temp == "train":
        # Transform for training
        train_transform = A.Compose(
            [
                # Rescale an image so that maximum side is equal to image_size
                A.LongestMaxSize(max_size=image_size),
                # Pad remaining areas with zeros
                A.PadIfNeeded(
                    min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT
                ),
                # Random color jittering
                A.ColorJitter(
                    brightness=0.5, contrast=0.5,
                    saturation=0.5, hue=0.5, p=0.5
                ),
                # Flip the image horizontally
                A.HorizontalFlip(p=0.5),
                
                # Normalize the image
                A.Normalize(
                    mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255
                ),
                # Convert the image to PyTorch tensor
                ToTensorV2()
            ],
            # Augmentation for bounding boxes
            bbox_params=A.BboxParams(
                format="yolo",
                min_visibility=0.4,
                label_fields=[]
            )
        )
        return train_transform
    elif temp == "test":
        # Transform for testing
        test_transform = A.Compose(
            [
                # Rescale an image so that maximum side is equal to image_size
                A.LongestMaxSize(max_size=image_size),
                # Pad remaining areas with zeros
                A.PadIfNeeded(
                    min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT
                ),
                # Normalize the image
                A.Normalize(
                    mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255
                ),
                # Convert the image to PyTorch tensor
                ToTensorV2()
            ],
            # Augmentation for bounding boxes
            bbox_params=A.BboxParams(
                format="yolo",
                min_visibility=0.4,
                label_fields=[]
            )
        )
        return test_transform
    else:
        raise ValueError(f" temp must be \'train\' or \'test\'")

class YoloCircleDatasetTest(Dataset):
    def __init__(
            self, image_dir, label_dir, anchors,
            image_size=512, grid_sizes=[16, 32, 64],
            num_classes=1, transform=None
    ):
        # Read the csv file with image names and labels
        # self.label_list = pd.read_csv(csv_file)
        # Image and label directories
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = os.listdir(image_dir)
        # Image size
        self.image_size = image_size
        # Transformations
        self.transform = transform
        # Grid sizes for each scale
        self.grid_sizes = grid_sizes
        # Anchor boxes
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        # Number of anchor boxes
        self.num_anchors = self.anchors.shape[0]
        # Number of anchor boxes per scale
        self.num_anchors_per_scale = self.num_anchors // 3
        # Number of classes
        self.num_classes = num_classes
        # Ignore IoU threshold
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        ###
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace(".jpg", ".txt"))
        ###

        # Getting the label path
        # label_path = os.path.join(self.label_dir, self.label_list.iloc[idx, 1])
        # We are applying roll to move class label to the last column
        # 5 columns: x, y, width, height, class_label
        if os.stat(label_path).st_size == 0:
            bboxes = []
        else:
            bboxes = np.roll(np.loadtxt(fname=label_path,delimiter=" ", ndmin=2), 4, axis=1).tolist()


        # Getting the image path
        # img_path = os.path.join(self.image_dir, self.label_list.iloc[idx, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        # Albumentations augmentations
        if self.transform:
            augs = self.transform(image=image, bboxes=bboxes)
            image = augs["image"]
            bboxes = augs["bboxes"]
        circles = [np.squeeze([b[4],b[0],b[1],min(b[2],b[3])]) for b in bboxes]
        return image, circles, label_path



def display_sample(args,bi):
    """
    randomly selects a sample from the dataset and displays it with the bounding circles
    :return:
    """

    # Creating a dataset object
    dataset = YoloCircleDataset(
        image_dir=args.images_path,
        label_dir=args.labels_path,
        grid_sizes=[16, 32, 64],
        anchors=CIRCLE_ANCHORS,
        transform=get_transforms("test")
    )

    # Creating a dataloader object
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
    )

    # Defining the grid size and the scaled anchors
    GRID_SIZE = [16, 32, 64]
    scaled_anchors = torch.tensor(CIRCLE_ANCHORS)/(1/torch.tensor(GRID_SIZE).unsqueeze(1).unsqueeze(1).repeat(1, 3, 1))

    # Getting a batch from the dataloader
    x, y = next(iter(loader))

    # Getting the boxes coordinates from the labels
    # and converting them into bounding boxes without scaling
    circles = []
    for i in range(y[0].shape[1]):
        anchor = scaled_anchors[i]
        circles += convert_cells_to_circles(y[i], is_predictions=False, s=y[i].shape[2], anchors=anchor)[0]

    # Applying non-maximum suppression
    circles = nms_ciecles(circles, iou_threshold=1, threshold=0.7)

    # Plotting the image with the bounding boxes
    plot_image_circles(x[0].permute(1, 2, 0).to("cpu"), circles,bi =bi ,savefig=True)


# Defining CNN Block
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        # Applying convolution
        x = self.conv(x)
        # Applying BatchNorm and activation if needed
        if self.use_batch_norm:
            x = self.bn(x)
            return self.activation(x)
        else:
            return x


# Defining residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()

        # Defining all the layers in a list and adding them based on number of
        # repeats mentioned in the design
        res_layers = []
        for _ in range(num_repeats):
            res_layers += [
                nn.Sequential(
                    nn.Conv2d(channels, channels // 2, kernel_size=1),
                    nn.BatchNorm2d(channels // 2),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.1)
                )
            ]
        self.layers = nn.ModuleList(res_layers)
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    # Defining forward pass
    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer(x)
            if self.use_residual:
                x = x + residual
        return x


# Defining scale prediction class
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Defining the layers in the network
        self.pred = nn.Sequential(
            nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * in_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(2 * in_channels, (num_classes + 4) * 3, kernel_size=1),
        )
        self.num_classes = num_classes

    # Defining the forward pass and reshaping the output to the desired output
    # format: (batch_size, 3, grid_size, grid_size, num_classes + 4)
    # 4 means (cx,cy,r,confidence), where r is the radius of the circle
    def forward(self, x):
        output = self.pred(x)
        output = output.view(x.size(0), 3, self.num_classes +4, x.size(2), x.size(3))
        output = output.permute(0, 1, 3, 4, 2)
        return output


# Class for defining YOLOv3 model
class YOLOCircleNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Layers list for YOLOv3
        self.layers = nn.ModuleList([
            CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1),
            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ResidualBlock(64, num_repeats=1),
            CNNBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128, num_repeats=2),
            CNNBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256, num_repeats=8),
            CNNBlock(256, 512, kernel_size=3, stride=2, padding=1),
            ResidualBlock(512, num_repeats=8),
            CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1),
            ResidualBlock(1024, num_repeats=4),
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            ResidualBlock(1024, use_residual=False, num_repeats=1),
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            ScalePrediction(512, num_classes=num_classes),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2),
            CNNBlock(768, 256, kernel_size=1, stride=1, padding=0),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ResidualBlock(512, use_residual=False, num_repeats=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            ScalePrediction(256, num_classes=num_classes),
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2),
            CNNBlock(384, 128, kernel_size=1, stride=1, padding=0),
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ResidualBlock(256, use_residual=False, num_repeats=1),
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0),
            ScalePrediction(128, num_classes=num_classes)
        ])

    # Forward pass for YOLOv3 with route connections and scale predictions
    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs


# Defining YOLO loss class
class YOLOCircleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target, anchors):
        # Identifying which cells in target have objects and which have no objects
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        # Calculating No object loss
        no_object_loss = self.bce((pred[..., 0:1][no_obj]), (target[..., 0:1][no_obj]),)

        # Reshaping anchors to match predictions
        anchors = anchors.reshape(1, 3, 1, 1, 1)

        # Circle prediction confidence
        circle_preds = torch.cat([self.sigmoid(pred[..., 1:3]),torch.exp(pred[..., 3:4]) * anchors], dim=-1)

        # Calculating intersection over union for prediction and target
        ious = circle_iou(circle_preds[obj], target[..., 1:4][obj]).detach()

        # Calculating Object loss
        object_loss = self.mse(self.sigmoid(pred[..., 0:1][obj]),ious * target[..., 0:1][obj])

        # Predicted circle coordinates
        pred[..., 1:3] = self.sigmoid(pred[..., 1:3])
        # Target circle coordinates
        target[..., 3:4] = torch.log(1e-6 + target[..., 3:4] / anchors)
        # Calculating box coordinate loss
        circle_loss = self.mse(pred[..., 1:4][obj],target[..., 1:4][obj])

        # Claculating class loss
        class_loss = self.cross_entropy((pred[..., 4:][obj]),target[..., 4][obj].long())

        # Total loss
        return (
                circle_loss
                + object_loss
                + no_object_loss
                + class_loss
        )

# Define the train function to train the model
def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    # Creating a progress bar
    progress_bar = tqdm(loader, leave=True)

    # Initializing a list to store the losses
    losses = []

    # Iterating over the training data
    for _, (x, y) in enumerate(progress_bar):
        x = x.to(device)
        y0, y1, y2 = (
            y[0].to(device),
            y[1].to(device),
            y[2].to(device),
        )

        # with torch.cuda.amp.autocast():
        with torch.amp.autocast(device_type=device):
            # Getting the model predictions
            outputs = model(x)
            # Calculating the loss at each scale
            loss = (
                  loss_fn(outputs[0], y0, scaled_anchors[0])
                + loss_fn(outputs[1], y1, scaled_anchors[1])
                + loss_fn(outputs[2], y2, scaled_anchors[2])
            )

        # Add the loss to the list
        losses.append(loss.item())

        # Reset gradients
        optimizer.zero_grad()

        # Backpropagate the loss
        scaler.scale(loss).backward()

        # Optimization step
        scaler.step(optimizer)

        # Update the scaler for next iteration
        scaler.update()

        # update progress bar with loss
        mean_loss = sum(losses) / len(losses)
        progress_bar.set_postfix(loss=mean_loss)

    return mean_loss

def start_train(args):
    # Creating the model from YOLOv3 class
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_name = args.logdir + f"{timestamp}"
    args.logdir = "./runs/" + experiment_name
    writer = SummaryWriter(log_dir=args.logdir)

    args.mode = "train"
    model = YOLOCircleNet(num_classes=args.num_class).to(device)
    writer.add_graph(model, input_to_model=torch.randn((1, 3, args.image_size, args.image_size)).to(device))

    # show model
    # summary(model, (3, args.input_height, args.input_width),device=device)

    # Defining the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.leanring_rate)

    # Defining the loss function
    loss_fn = YOLOCircleLoss()

    # Defining the scaler for mixed precision training
    # scaler = torch.cuda.amp.GradScaler()
    scaler = torch.amp.GradScaler()

    # Defining the train dataset
    train_dataset = YoloCircleDataset(
        image_dir=args.images_path,
        label_dir=args.labels_path,
        anchors=CIRCLE_ANCHORS,
        transform=get_transforms("train"),
        grid_sizes=GRID_SIZES
    )

    # Defining the train data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
    )

    # Scaling the anchors
    scaled_anchors = (torch.tensor(CIRCLE_ANCHORS) * torch.tensor(GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1, 3, 1)).to(device)

    loss=[]
    # Training the model
    for e in range(1, args.num_epochs + 1):
        print("Epoch:", e)
        mean_loss = training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        writer.add_scalar("loss_train", mean_loss, e)

        # Saving the model
        if args.save_model and e % 100==0:
            save_checkpoint(model, optimizer, filename=args.checkpoint_file)
            # save model and check its performance
    writer.close()


def mean_nms_ciecles(circles, iou_threshold, threshold):
    # Filter out bounding boxes with confidence below the threshold.
    circles = [c for c in circles if c[1] > threshold]

    # print(f"number of circles that threshold > {threshold} is {len(circles)}")

    # Sort the bounding circles by confidence in descending order.
    sorted_circles  = np.array(sorted(circles, key=lambda x: x[1], reverse=True))

    # Initialize the list of bounding circles after non-maximum suppression.
    circles_nms = []

    while len(sorted_circles) > 0 :
        best_circle = sorted_circles[0]
        ious = circle_iou(best_circle[2:], sorted_circles[1:][...,2:])
        ious = ious.numpy()
        binary_mask = (ious < iou_threshold).squeeze(1).astype(bool)
        binary_mask_097 = (ious > 0.97).squeeze(1).astype(bool)
        binary_mask_097 = np.insert(binary_mask_097, 0, True)
        temp_best = np.mean(sorted_circles[binary_mask_097], axis=0)
        circles_nms.append(temp_best)

        sorted_circles = sorted_circles[1:][binary_mask]

    # Return bounding circles after non-maximum suppression.
    # print(f"number of circles after NMS under IOU = {iou_threshold} is {len(circles_nms)}")
    return circles_nms

def start_test(args):
    # Taking a sample image and testing the model
    args.mode = "test"

    # Setting the load_model to True
    load_model = True

    # Defining the model, optimizer, loss function and scaler
    model = YOLOCircleNet(num_classes=args.num_class).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.leanring_rate)
    # loss_fn = YOLOCircleLoss()
    # scaler = torch.cuda.amp.GradScaler()
    # scaler = torch.amp.GradScaler()

    # Loading the checkpoint
    if load_model:
        load_checkpoint(args.checkpoint_file, model, optimizer, args.leanring_rate)

    # Defining the test dataset and data loader
    test_dataset = YoloCircleDatasetTest(
        image_dir="/path/to/your/data/images/val",
        label_dir="/path/to/your/data/labels/val",
        anchors=CIRCLE_ANCHORS,
        transform=get_transforms("test")
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
    )

    precisions, recalls, f1_scores, mean_ious = [],[],[],[]
    model.eval()
    # for i in range(len(test_loader)):
    for index,(x,y,label_path) in enumerate(test_loader):

        # Getting a sample image from the test data loader
        # x, y,label_path = next(iter(test_loader))
        print(f"{index:4}: {label_path[0]}")
        x = x.to(device)

        with torch.no_grad():
            # Getting the model predictions
            output = model(x)
            # Getting the bounding boxes from the predictions
            circles = [[] for _ in range(x.shape[0])]
            anchors = (
                    torch.tensor(CIRCLE_ANCHORS)
                    * torch.tensor(GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1, 3, 1)
            ).to(device)

            # Getting bounding boxes for each scale
            for i in range(3):
                batch_size, A, S, _, _ = output[i].shape
                anchor = anchors[i]
                circles_scale_i = convert_cells_to_circles(
                    output[i], anchor, s=S, is_predictions=True
                )
                for idx, (circ) in enumerate(circles_scale_i):
                    circles[idx] += circ

        # Plotting the image with bounding boxes for each image in the batch
        for i in range(batch_size):
            # batch_size == 1
            # Applying non-max suppression to remove overlapping bounding circles
            nms_c = mean_nms_ciecles(circles[i], iou_threshold=0.5, threshold=0.6)
            # Plotting the image with bounding boxes
            plot_image_circles(x[i].permute(1, 2, 0).detach().cpu(), nms_c,bi=i,savefig=False)
            #  measure the model performance, include precision, recall, F1-score, mAP,mAP50
            precision, recall, f1_score, mean_iou = evaluate_prediction(nms_c,y)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)
            mean_ious.append(mean_iou)

        # break # test one image
    print("-----------------------------------------------")
    print(f"Mean Precisions: {np.mean(precisions):.4f}")
    print(f"Mean Recalls: {np.mean(recalls):.4f}")
    print(f"Mean F1_scores: {np.mean(f1_scores):.4f}")
    print(f"Mean mean_ious: {np.mean(mean_ious):.4f}")


def evaluate_prediction(args,predicts,targets):
    """
    :param predict: the length is m
    :param target: the length is n
    :return:
    """
    epsilon = 1e-8

    if len(targets) == 0:
        fp = len(predicts)
        fn = 0
        tp = 0

        recall = (tp + epsilon) / (tp + fn + epsilon)
        precision = (tp + epsilon) / (tp + fp + epsilon)
        f1_score = (2 * (precision * recall)  + epsilon) / (precision + recall + epsilon)
        if len(predicts) == 0:
            iou = 1
        else:
            iou = 0
        return precision, recall, f1_score, iou

    ious = []
    target = np.array([t[0].numpy() for t in targets])[:,[1,2,3]]
    predict = np.array(predicts)[:,[2,3,4]]

    matrix_mn = np.zeros((len(predict), len(target)))

    for idx, pred_circle in enumerate(predict):
        iou_score = circle_iou(pred_circle, target).numpy().squeeze()
        if np.max(iou_score)< 0.5:
            matrix_mn[idx, np.argmax(iou_score)] = 0
        else:
            matrix_mn[idx, np.argmax(iou_score)] = 1
            ious.append(np.max(iou_score))

    fp = len(np.sum(matrix_mn, axis=1))-np.sum(np.sum(matrix_mn, axis=1))
    fn = len(np.sum(matrix_mn, axis=0))-np.sum(np.sum(matrix_mn, axis=0))
    tp = np.sum(np.sum(matrix_mn, axis=1))

    recall = tp / (tp+fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    if len(predicts) != len(targets):
        return 0

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1_score: {f1_score:.4f}")
    print(f"mean_iou: {np.mean(ious):.4f}")

    return precision, recall, f1_score,np.mean(ious)



def test_yolov3_model(args):
    # Setting number of classes and image size
    num_classes = args.num_class

    # Creating model and testing output shapes
    model = YOLOCircleNet(num_classes=num_classes)
    x = torch.randn((1, 3, args.image_size, args.image_size))
    out = model(x)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)

    # Asserting output shapes
    assert model(x)[0].shape == (1, 3, args.image_size//32, args.image_size//32, num_classes + 4)
    assert model(x)[1].shape == (1, 3, args.image_size//16, args.image_size//16, num_classes + 4)
    assert model(x)[2].shape == (1, 3, args.image_size//8, args.image_size//8, num_classes + 4)
    print("Output shapes are correct!")


def get_args():
    parser = argparse.ArgumentParser(description='YOLOv3 Circle.')
    parser.add_argument('--mode', type=str, help='train or test', default='train')
    parser.add_argument('--images_path', type=str, help='path to the images', required=False,
                        default="/path/to/your/data/images/train")
    parser.add_argument('--labels_path', type=str, help='path to the labels', required=False,
                        default="/path/to/your/data/labels/train")
    parser.add_argument("--logdir", default="YOLO_Circle_min", type=str,
                        help="directory to save the tensorboard logs")
    parser.add_argument('--image_size', type=int, help='input width', default=512)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--num_epochs', type=int, help='number of epochs', default=1000)
    parser.add_argument('--leanring_rate', type=float, help='initial learning rate', default=1e-5)
    parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='./')
    parser.add_argument('--load_model', type=bool, help='load model', default=False)
    parser.add_argument('--save_model', type=bool, help='save model', default=True)
    parser.add_argument('--device', type=str, help='cuda or cpu', default='cuda')
    parser.add_argument('--checkpoint_file', type=str, help='checkpoint file name', default='checkpoint_c2.pth.tar')
    parser.add_argument('--num_class', type=int, help='number of class', default=1, required=False)
    args = parser.parse_args()

    # Print the arguments
    print("Training Arguments:")
    print("------------------")
    for k, v in vars(args).items():
        print(f"  {k:20} : {v}")
    print("------------------")

    return args


if __name__ == "__main__":
    # Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # model checkpoint file name
    # checkpoint_file = "checkpoint_temp.pth.tar"

    # Anchor boxes for each feature map scaled between 0 and 1
    # 3 feature maps at 3 different scales based on YOLOv3 paper
    CIRCLE_ANCHORS = [
        [[0.22], [0.48], [0.78]],
        [[0.11], [0.15] , [0.29]],
        [[0.03], [0.06], [0.08]],
    ]

    # Image size
    image_size = 512

    # Grid cell sizes [16, 32, 64]
    GRID_SIZES = [image_size // 32, image_size // 16, image_size // 8]

    # Class labels
    class_labels = ["tube"]

    # get hyperparameter
    args = get_args()

    # Take a sample image and display it with labels
    # for i in range(9):
    #     display_sample(args,i)

    # Check YOLO v3 model
    # test_yolov3_model(args)

    # Training the model
    # start_train(args)

    # Test the trained model
    # start_test(args)