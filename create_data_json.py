import json
import os

import pandas as pd
from sympy import separatevars
from zipp.glob import separate


def get_image_list(data_path, mode):
    """Taken from https://github.com/KitwareMedical/Medical-SAM-Adapter/blob/e87310ef8f5ab873f1ec9177369eed7034def800/dataset/cxr.py#L17-L38"""
    if mode == 'Training':
        df = pd.read_csv(os.path.join(data_path, 'PreprocessedData-YOLO/train.txt'), header=None)
    elif mode == 'Test':
        df = pd.read_csv(os.path.join(data_path, 'PreprocessedData-YOLO/test.txt'), header=None)

    children_list = df[df[0].str.contains('Children')]
    shenzhen_list = df[df[0].str.contains('Shenzhen')]
    montgomery_list = df[df[0].str.contains('Montgomery')]
    children_size = len(children_list)

    if mode == 'Training':
        # balance the dataset (Children's is the smallest)
        # df = pd.concat([children_list, shenzhen_list.head(children_size), montgomery_list.head(children_size)])
        df = pd.concat([shenzhen_list, montgomery_list])
    elif mode == 'Test':
        # for evaluation, we only care about the performance on Children's dataset
        # df = children_list
        df = pd.concat([shenzhen_list, montgomery_list])

    df = df.sample(frac=1, random_state=1983)  # shuffle the dataframe

    return list(df[0])


def create_json(image_list, separate_lungs=False):
    train_dict = {}
    test_dict = {}
    for image_name in image_list:
        image_path = f"ReorganizedImage/{image_name}.png"
        if separate_lungs:
            label_path_left = f"ReorganizedSegmentation/{image_name}_left_label.png"
            label_path_right = f"ReorganizedSegmentation/{image_name}_right_label.png"
            train_dict[image_path] = [label_path_left, label_path_right]
            test_dict[label_path_left] = image_path
            test_dict[label_path_right] = image_path
        else:
            label_path = f"ReorganizedSegmentation/{image_name}_label.png"
            train_dict[image_path] = [label_path]
            test_dict[label_path] = image_path
    return train_dict, test_dict


if __name__ == "__main__":
    data_path = "M:/Dev/CXR/LungAI/Data/"
    train_set = get_image_list(data_path, 'Training')
    test_set = get_image_list(data_path, 'Test')

    separate_train, _ = create_json(train_set, separate_lungs=True)
    with open("image2label_train_separate.json", 'w') as f:
        json.dump(separate_train, f, indent=4)

    _, separate_test = create_json(test_set, separate_lungs=True)
    with open("label2image_test_separate.json", 'w') as f:
        json.dump(separate_test, f, indent=4)

    combined_train, _ = create_json(train_set, separate_lungs=False)
    with open("image2label_train_combined.json", 'w') as f:
        json.dump(combined_train, f, indent=4)

    _, combined_test = create_json(test_set, separate_lungs=False)
    with open("label2image_test_combined.json", 'w') as f:
        json.dump(combined_test, f, indent=4)
