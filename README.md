# digit detector

### Dataset

    SVHN: http://ufldl.stanford.edu/housenumbers/

### File description

    *   data_preprocess.py : transfer the SVGN data into YOLO format
    *   dataloader.py : load the YOLO format data
    *   model.py : create YOLOv3-tiny module by config/yolov3-tiny.cfg
    *   train.py : training a detector
    *   detect.py : detector the number in testing images
    *   test.py : test mAP
    *   config.yaml : contains all hyberparameter, data path, and weight path
    

