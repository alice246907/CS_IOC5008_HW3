from __future__ import division

from models import Darknet
from utils.utils import load_classes, non_max_suppression, rescale_boxes
from dataloader import ImageFolder

import os
import time
import datetime
import argparse
import json
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import yaml
import numpy as np
import random


def detect(args):

    os.makedirs(args.output_path, exist_ok=True)

    # Set up model
    model = Darknet(args.model_def, img_size=args.img_size).cuda()
    model.load_state_dict(torch.load(f"{args.weight_path}"))
    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(args.image_folder, img_size=args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpu,
    )

    classes = load_classes(args.class_path)  # Extracts class labels from file

    Tensor = (
        torch.cuda.FloatTensor
        if torch.cuda.is_available()
        else torch.FloatTensor
    )

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(
                detections, args.conf_thres, args.nms_thres
            )

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    generated_json(args, imgs, img_detections, classes)
    # plot(imgs, img_detections, classes)


def generated_json(args, imgs, img_detections, classes):
    all_dict = []
    for _, (path, detections) in enumerate(zip(imgs, img_detections)):
        print(path)
        img = np.array(Image.open(path))
        img_dict = {"bbox": [], "score": [], "label": []}
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(
                detections, args.img_size, img.shape[:2]
            )
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                img_dict["bbox"].append(
                    (y1.item(), x1.item(), y2.item(), x2.item())
                )
                img_dict["score"].append(cls_conf.item())
                img_dict["label"].append(int(cls_pred.item()))

        all_dict.append(img_dict)

    with open(args.file_name, "w") as outfile:
        json.dump(all_dict, outfile)


def plot(imgs, img_detections, classes):
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(
                detections, args.img_size, img.shape[:2]
            )
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print(
                    "\t+ Label: %s, Conf: %.5f"
                    % (classes[int(cls_pred)], cls_conf.item())
                )

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[
                    int(np.where(unique_labels == int(cls_pred))[0])
                ]
                # Create a Rectangle patch
                bbox = patches.Rectangle(
                    (x1, y1),
                    box_w,
                    box_h,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(
            f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0
        )
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=str)

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    for key, value in config["detect"].items():
        setattr(args, key, value)
    detect(args)
