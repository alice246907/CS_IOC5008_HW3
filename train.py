import argparse
import yaml
import torch
from dataloader import ListDataset
import os
from models import Darknet
from utils.logger import Logger
from utils.utils import weights_init_normal, load_classes
import datetime
import time
from test import evaluate
from terminaltables import AsciiTable
import lera
from torch.autograd import Variable


def train(args):

    os.makedirs(args.weight_path, exist_ok=True)

    logger = Logger("logs")
    lera.log_hyperparams(
        {"title": "hw3", "batch size": args.batch_size, "model": "yoloV3-tiny"}
    )
    # data
    dataset = ListDataset(args.train_data, augment=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    class_names = load_classes(args.class_names)

    # model
    model = Darknet(args.model_def).cuda()
    model.apply(weights_init_normal)

    if args.pretrained_epoch != 0:
        model.load_state_dict(
            torch.load(f"{args.weight_path}/{args.pretrained_epoch}.pth")
        )

    optimizer = torch.optim.Adam(model.parameters())
    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(args.pretrained_epoch, args.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i
            imgs = Variable(imgs.cuda())
            targets = Variable(targets.cuda(), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % args.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            if batch_i % args.log_interval == 0:

                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (
                    epoch,
                    args.epochs,
                    batch_i,
                    len(dataloader),
                )

                metric_table = [
                    [
                        "Metrics",
                        *[
                            f"YOLO Layer {i}"
                            for i in range(len(model.yolo_layers))
                        ],
                    ]
                ]

                # Log metrics at each YOLO layer
                for i, metric in enumerate(metrics):
                    formats = {m: "%.6f" for m in metrics}
                    formats["grid_size"] = "%2d"
                    formats["cls_acc"] = "%.2f%%"
                    row_metrics = [
                        formats[metric] % yolo.metrics.get(metric, 0)
                        for yolo in model.yolo_layers
                    ]
                    metric_table += [[metric, *row_metrics]]

                    # Tensorboard logging
                    tensorboard_log = []
                    for j, yolo in enumerate(model.yolo_layers):
                        for name, metric in yolo.metrics.items():
                            if name != "grid_size":
                                tensorboard_log += [(f"{name}_{j+1}", metric)]
                    tensorboard_log += [("loss", loss.item())]
                    logger.list_of_scalars_summary(
                        tensorboard_log, batches_done
                    )

                log_str += AsciiTable(metric_table).table
                log_str += f"\nTotal loss {loss.item()}"

                # Determine approximate time left for epoch
                epoch_batches_left = len(dataloader) - (batch_i + 1)
                time_left = datetime.timedelta(
                    seconds=epoch_batches_left
                    * (time.time() - start_time)
                    / (batch_i + 1)
                )
                log_str += f"\n---- ETA {time_left}"

                print(log_str)
                lera.log({"loss": loss.item()})

            model.seen += imgs.size(0)

        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=args.valid_data,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=args.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
            lera.log({"mAP": AP.mean()})
        if epoch % args.checkpoint_interval == 0:
            torch.save(
                model.state_dict(), f"{args.weight_path}/{epoch + 1}.pth"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=str)

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    for key, value in config["train"].items():
        setattr(args, key, value)
    train(args)
