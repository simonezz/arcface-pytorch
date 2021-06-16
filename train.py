from __future__ import print_function
import os

import torch

from torch.utils import data
import torch.nn.functional as F
from models import *

import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from utils import Visualizer, view_model
import torch
import numpy as np
import random
import time
from config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from test import *

import wandb


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + "_" + str(iter_cnt) + ".pth")
    torch.save(model.state_dict(), save_name)
    return save_name


if __name__ == "__main__":

    opt = Config()

    torch.manual_seed(opt.torch_seed)

    if opt.display:
        visualizer = Visualizer()
    device = torch.device("cuda")

    # train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)

    transform_train = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # image size 재조정
            transforms.ToTensor(),  # Tensor로 바꾸고 (0~1로 자동으로 normalize)
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # -1 ~ 1 사이로 normalize
            ),  # (c - m)/s 니까...
        ]
    )

    dataset = ImageFolder(root=opt.train_root, transform=transform_train)

    train_len = int(0.7 * len(dataset))
    valid_len = len(dataset) - train_len

    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, valid_len])

    trainloader = data.DataLoader(
        train_set,
        batch_size=opt.train_batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )

    valloader = data.DataLoader(
        val_set,
        batch_size=opt.train_batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )

    # identity_list = get_lfw_list(opt.lfw_test_list)
    # img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    print("{} train iters per epoch:".format(len(trainloader)))

    if opt.loss == "focal_loss":
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == "resnet18":
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == "resnet34":
        model = resnet34()
    elif opt.backbone == "resnet50":
        model = resnet50()

    if opt.metric == "add_margin":
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == "arc_margin":
        metric_fc = ArcMarginProduct(
            512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin
        )
    elif opt.metric == "sphere":
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            [{"params": model.parameters()}, {"params": metric_fc.parameters()}],
            lr=opt.lr,
            weight_decay=opt.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            [{"params": model.parameters()}, {"params": metric_fc.parameters()}],
            lr=opt.lr,
            weight_decay=opt.weight_decay,
        )
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    start = time.time()

    train_accuracies = []
    test_accuracies = []

    for i in range(opt.max_epoch):

        scheduler.step()

        model.train()

        batch_train_acc = []

        for ii, data in enumerate(trainloader):

            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            # print(data)
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            # print(output)
            # print(label)
            acc = np.mean((output == label).astype(int))

            batch_train_acc.append(acc)

            if iters % opt.print_freq == 0:

                time_str = time.asctime(time.localtime(time.time()))
                speed = opt.print_freq / (time.time() - start)
                print(
                    "{} train epoch {} iter {} {} iters/s loss {} acc {}".format(
                        time_str, i, ii, speed, loss.item(), acc
                    )
                )

                start = time.time()

        batch_test_acc = []

        for ii, data in enumerate(valloader):

            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            # print(data)
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)

            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            # print(output)
            # print(label)
            acc = np.mean((output == label).astype(int))

            batch_test_acc.append(acc)

        train_accuracies.append(np.mean(batch_train_acc))

        test_accuracies.append(np.mean(batch_test_acc))

        print("epoch {} train acc {}".format(i + 1, train_accuracies[-1]))
        print("epoch {} test acc {}".format(i + 1, test_accuracies[-1]))

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, i)

        model.eval()
        # acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
        # if opt.display:
        #     visualizer.display_current_results(iters, acc, name='test_acc')
