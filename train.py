from __future__ import print_function

from absl import app, flags, logging
from absl.flags import FLAGS

import yaml

import torch
from torch.utils import data
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR

import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import KFold

from test import *
from models import *

import wandb


def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, "r") as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded


def save_model(model, save_path, name, iter_cnt=""):
    save_name = os.path.join(save_path, name + "_" + str(iter_cnt) + ".pth")
    torch.save(model.module.state_dict(), save_name)
    return save_name


def train_model(
    opt,
    trainloader,
    valloader,
    device,
    save_path,
):
    if opt["loss"] == "focal_loss":
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt["backbone"] == "resnet18":

        # model = resnet_face18(use_se=opt["use_se"], size=opt["size"])

        model = ResNet18FineTuning(
            128, dropout=opt["dr_rate"], pretrained=opt["pretrained"]
        )

    elif opt["backbone"] == "resnet34":
        model = resnet34()
    elif opt["backbone"] == "resnet50":
        model = resnet50()

    if opt["metric"] == "add_margin":
        metric_fc = AddMarginProduct(512, opt["num_classes"], s=30, m=0.35)

    elif opt["metric"] == "arc_margin":

        if opt["size"] == "ori":
            metric_fc = ArcMarginProduct(
                512, opt["num_classes"], s=30, m=0.5, easy_margin=opt["easy_margin"]
            )

        elif opt["size"] == "s":
            metric_fc = ArcMarginProduct(
                256, opt["num_classes"], s=30, m=0.5, easy_margin=opt["easy_margin"]
            )

        else:
            metric_fc = ArcMarginProduct(
                128, opt["num_classes"], s=30, m=0.5, easy_margin=opt["easy_margin"]
            )

    elif opt["metric"] == "sphere":
        metric_fc = SphereProduct(512, opt["num_classes"], m=4)
    else:
        metric_fc = nn.Linear(512, opt["num_classes"])

    print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if opt["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            [{"params": model.parameters()}, {"params": metric_fc.parameters()}],
            lr=opt["lr"],
            weight_decay=opt["weight_decay"],
        )
    else:
        optimizer = torch.optim.Adam(
            [{"params": model.parameters()}, {"params": metric_fc.parameters()}],
            lr=opt["lr"],
            weight_decay=opt["weight_decay"],
        )
    scheduler = StepLR(optimizer, step_size=opt["lr_step"], gamma=0.1)

    start = time.time()

    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []

    best_test_accuracy = 0.0

    for i in range(opt["max_epoch"]):

        scheduler.step()

        model.train()

        batch_train_acc = []
        batch_test_loss = []
        batch_train_loss = []

        for ii, data_t in enumerate(trainloader):

            data_input, label = data_t
            data_input = data_input.to(device)
            label = label.to(device).long()

            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()

            if opt["clipping"]:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opt["clipping_max_norm"]
                )

            optimizer.step()

            iters = i * len(trainloader) + ii

            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()

            acc = np.mean((output == label).astype(int))

            batch_train_acc.append(acc)
            batch_train_loss.append(loss.data.cpu().numpy())

            if iters % opt["print_freq"] == 0:
                time_str = time.asctime(time.localtime(time.time()))
                speed = opt["print_freq"] / (time.time() - start)
                print(
                    "{} train epoch {} iter {} {} iters/s loss {} acc {}".format(
                        time_str, i, ii, speed, loss.item(), acc
                    )
                )

                start = time.time()

        batch_test_acc = []

        for _, data_v in enumerate(valloader):
            data_input_v, label_v = data_v
            data_input = data_input.to(device)
            label_v = label_v.to(device).long()

            feature_v = model(data_input_v)

            output_v = metric_fc(feature_v, label_v)
            loss_v = criterion(output_v, label_v)

            batch_test_loss.append(loss_v.data.cpu().numpy())

            output_v = output_v.data.cpu().numpy()
            output_v = np.argmax(output_v, axis=1)

            label_v = label_v.data.cpu().numpy()

            acc_v = np.mean((output_v == label_v).astype(int))

            batch_test_acc.append(acc_v)

        train_accuracies.append(np.mean(batch_train_acc))

        test_accuracies.append(np.mean(batch_test_acc))

        train_losses.append(np.mean(batch_train_loss))
        test_losses.append(np.mean(batch_test_loss))

        print("epoch {} train loss {}".format(i + 1, train_losses[-1]))
        print("epoch {} test loss {}".format(i + 1, test_losses[-1]))
        print("epoch {} train acc {}".format(i + 1, train_accuracies[-1]))
        print("epoch {} test acc {}".format(i + 1, test_accuracies[-1]))

        if best_test_accuracy < test_accuracies[-1]:

            best_test_accuracy = test_accuracies[-1]
            save_model(model, save_path, opt["backbone"] + " best")

        tags = [
            "loss/train_loss",
            "loss/test_loss",
            "accuracy/train_acc",
            "accuracy/test_acc",
        ]

        for x, tag in zip(
            [
                train_losses[-1],
                test_losses[-1],
                train_accuracies[-1],
                test_accuracies[-1],
            ],
            tags,
        ):

            if wandb:
                wandb.log({tag: x})

        if i % opt["save_interval"] == 0 or i == opt["max_epoch"]:
            save_model(model, save_path, opt["backbone"], i)

        model.eval()

    save_model(model, save_path, "final")
    print("saved at ", save_path)


def main(_):

    flags.DEFINE_string("cfg_path", "config_train.yaml", "config file path")

    opt = load_yaml(FLAGS.cfg_path)

    dir_ind = 0

    while True:
        try:
            os.mkdir(opt["checkpoints_path"] + f"/{dir_ind}")
            save_path = opt["checkpoints_path"] + f"/{dir_ind}"
            break
        except:
            dir_ind += 1

    torch.manual_seed(opt["torch_seed"])

    if wandb:

        wandb_run = wandb.init(
            config=opt,
            resume="allow",
            project=opt["project_name"],
            name=opt["run_name"],
        )

    device = torch.device("cuda")

    transform_train = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # resize image to 256 X 256
            transforms.ToTensor(),  # to tensor & normalize(0~1)
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # normalize(-1~1)
            ),  # (c-m)/s
        ]
    )

    dataset = ImageFolder(root=opt["train_root"], transform=transform_train)

    if opt["cross_validation"]:

        splits = KFold(n_splits=5, shuffle=True, random_state=opt["torch_seed"])

        for fold, (train_idx, val_idx) in enumerate(splits.split(dataset)):

            print(f"---{fold}  Fold")
            train = torch.utils.data.Subset(dataset, train_idx)
            test = torch.utils.data.Subset(dataset, val_idx)

            trainloader = torch.utils.data.DataLoader(
                train,
                batch_size=opt["train_batch_size"],
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )
            valloader = torch.utils.data.DataLoader(
                test,
                batch_size=opt["test_batch_size"],
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )

            print("{} train iters per epoch:".format(len(trainloader)))

            train_model(opt, trainloader, valloader, device, save_path)

    else:

        train_len = int(0.7 * len(dataset))
        valid_len = len(dataset) - train_len

        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_len, valid_len]
        )

        trainloader = data.DataLoader(
            train_set,
            batch_size=opt["train_batch_size"],
            shuffle=True,
            num_workers=opt["num_workers"],
        )

        valloader = data.DataLoader(
            val_set,
            batch_size=opt["test_batch_size"],
            shuffle=True,
            num_workers=opt["num_workers"],
        )

        print("{} train iters per epoch:".format(len(trainloader)))

        train_model(opt, trainloader, valloader, device, save_path)


if __name__ == "__main__":

    app.run(main)
