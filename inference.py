# -*- coding: utf-8 -*-
"""
Created on 21-06-30

@author: SaeheeJeon

Inference using a trained model
(not the output of metric(ArcFace) ! it extracts feature vectors from the last output of backbone Network(e.g. ResNet))
"""
from __future__ import print_function
from absl.flags import FLAGS
from absl import app, flags


import os, yaml
import cv2
from PIL import Image

from models import *
import torch
import numpy as np
import time
from config import Config
from torch.nn import DataParallel

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, "r") as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded


def return_fv(model, device, input_path):

    data = []
    features = []

    transform_train = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # resize image to 256 X 256
            transforms.ToTensor(),  # to tensor & normalize(0~1)
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # normalize(-1~1)
            ),  # (c-m)/s
        ]
    )

    bs = 64

    nloop = math.ceil(len(input_path) / bs)

    for k in range(nloop):

        batch_imgs = []

        for i in range(k * bs, min((k + 1) * bs, len(input_path))):

            img = Image.open(input_path[i])
            batch_imgs.append(transform_train(img))

        batch_imgs = torch.stack(batch_imgs)

        data.append(batch_imgs)

    data = torch.stack(data)

    for data_input in data:

        data_input = data_input.to(device)
        feature = model(data_input)

        features.append(feature)

    features = torch.stack(features)

    return features


def main(_):

    flags.DEFINE_string("cfg_path", "config_train.yaml", "config file path")

    opt = load_yaml(FLAGS.cfg_path)

    device = torch.device("cuda")

    if opt["backbone"] == "resnet18":

        model = ResNet18FineTuning(
            128, dropout=opt["dr_rate"], pretrained=opt["pretrained"]
        )
        # model = resnet_face18(opt.use_se)
    elif opt["backbone"] == "resnet34":
        model = resnet34()
    elif opt["backbone"] == "resnet50":
        model = resnet50()

    model.to(device)
    # model = DataParallel(model)
    # load_model(model, opt.test_model_path)
    model.load_state_dict(torch.load(opt["test_model_path"]))
    model.to(torch.device("cuda"))

    return_fv(model, device, opt["infer_file_path"])
    # identity_list = get_lfw_list(opt.lfw_test_list)
    # img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]
    #
    # model.eval()
    # lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)


if __name__ == "__main__":

    app.run(main)
