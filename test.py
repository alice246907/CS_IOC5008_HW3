from __future__ import division

from models import Darknet
from utils.utils import (
    load_classes,
    ap_per_class,
    get_batch_statistics,
    non_max_suppression,
    xywh2xyxy,
)
import yaml

import argparse
import tqdm
import numpy as np

import torch
from torch.autograd import Variable
from dataloader import ListDataset


def evaluate(
    model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size
):
    model.eval()

    # Get dataloader
    dataset = ListDataset(
        path, img_size=img_size, augment=False, multiscale=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_fn,
    )

    Tensor = (
        torch.cuda.FloatTensor
        if torch.cuda.is_available()
        else torch.FloatTensor
    )

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(
        tqdm.tqdm(dataloader, desc="Detecting objects")
    ):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)  # (b, 2535, 85)
            outputs = non_max_suppression(
                outputs, conf_thres=conf_thres, nms_thres=nms_thres
            )

        sample_metrics += get_batch_statistics(
            outputs, targets, iou_threshold=iou_thres
        )

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))
    ]
    precision, recall, AP, f1, ap_class = ap_per_class(
        true_positives, pred_scores, pred_labels, labels
    )

    return precision, recall, AP, f1, ap_class


def test(args):

    class_names = load_classes(args.class_names)
    print(args.weight_path)
    # Initiate model
    model = Darknet(args.model_def).cuda()
    model.load_state_dict(torch.load(args.weight_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=args.test_data,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        img_size=args.img_size,
        batch_size=8,
    )
    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=str)

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    for key, value in config["test"].items():
        setattr(args, key, value)
    test(args)
