import h5py
import os
import cv2


def get_name(index, hdf5_data):
    name = hdf5_data["/digitStruct/name"]
    return "".join([chr(v[0]) for v in hdf5_data[name[index][0]].value])


def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data["digitStruct"]["bbox"][index].item()
    for key in ["label", "left", "top", "width", "height"]:
        attr = hdf5_data[item][key]
        values = (
            [
                hdf5_data[attr.value[i].item()].value[0][0]
                for i in range(len(attr))
            ]
            if len(attr) > 1
            else [attr.value[0][0]]
        )
        attrs[key] = values
    return attrs


def create_yolo_data(img_root, label_root):
    f = h5py.File(os.path.join(img_root, "digitStruct.mat"), "r")
    for j in range(f["/digitStruct/bbox"].shape[0]):
        img_name = get_name(j, f)
        label_file = open(
            os.path.join(label_root, img_name.replace(".png", ".txt")), "w"
        )
        row_dict = get_bbox(j, f)
        img = cv2.imread(os.path.join(img_root, img_name))
        img_height, img_width, _ = img.shape
        for i in range(len(row_dict["label"])):
            label = int(row_dict["label"][i])
            x = (row_dict["left"][i] + 0.5 * row_dict["width"][i]) / img_width
            y = (row_dict["top"][i] + 0.5 * row_dict["height"][i]) / img_height
            w = row_dict["width"][i] / img_width
            h = row_dict["height"][i] / img_height
            label_file.write(f"{label} {x} {y} {w} {h}\n")
        label_file.close()


if __name__ == "__main__":
    create_yolo_data("./dataset/imgs/train", "./dataset/labels/train")
    # create_yolo_data("./dataset/imgs/test", "./dataset/labels/test")

    f = open("./dataset/train_img.txt", "w")
    for i in range(1, 33402 + 1):
        f.write(f"dataset/imgs/train/{i}.png\n")
    f.close()

    f = open("./dataset/test_img.txt", "w")
    for i in range(1, 13068 + 1):
        f.write(f"dataset/imgs/test/{i}.png\n")
    f.close()
