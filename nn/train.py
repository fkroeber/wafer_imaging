import numpy as np
import pandas as pd
import os
import time
import torch
import torch.optim as optim
import wandb

from .models import NN
from .utils import EarlyStopping
from .data import CSVDataset, RotationTransform
from .losses import FocalLoss, HingeLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import (
    Accuracy,
    AUROC,
    AveragePrecision,
    CalibrationError,
    F1Score,
    Precision,
    Recall,
    MetricCollection,
)


class Trainer:
    """NN training & evaluation on a data set of choice"""

    def __init__(self, **kwargs):
        """
        Define model training & logging parameters
        data_path: data path to csv file
        model_name: model to be trained - see "nn_models" for list of available args
        train_mode: degree of use of pretrained models - see "nn_models" for list of available args
        batch_size: batch size used for training - choose a multiple of 2
        optimiser: optimiser used for training - available optimisers are "SGD", "Adam"
        loss: loss function used for training - options are "ce", "bce", "focal", "hinge", "hinge_squared"
        lr: learning rate used for training
        max_epochs: maximum number of epochs used for training
        patience: patience for early stopping on validation data
        seed: optional seed to reproduce results
        save_path: path to save results for model run
        wandb_project: project name for logging results with wandb
        wandb_runname: name current run logged with wandb
        wandb_tags: tags for current run logged with wandb
        wandb_notes: additional notes/description for current run
        """
        self.data_path = kwargs.get("data_path")
        self.model_name = kwargs.get("model_name")
        self.train_mode = kwargs.get("train_mode")
        self.batch_size = kwargs.get("batch_size")
        self.optimiser = kwargs.get("optimiser")
        self.loss = kwargs.get("loss")
        self.lr = kwargs.get("lr")
        self.max_epochs = kwargs.get("max_epochs")
        self.patience = kwargs.get("patience")
        self.seed = kwargs.get("seed")
        self.save_path = kwargs.get("save_path")
        self.wandb_project = kwargs.get("wandb_project")
        self.wandb_runname = kwargs.get("wandb_runname")
        self.wandb_tags = kwargs.get("wandb_tags", [])
        self.wandb_notes = kwargs.get("wandb_notes")
        # create save folder if necessary
        os.makedirs(self.save_path, exist_ok=True)
        # set seed if specified
        if self.seed:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
        # define wandb session & log model configuration
        wandb.init(
            project=self.wandb_project,
            dir=self.save_path,
            tags=self.wandb_tags,
            notes=self.wandb_notes,
        )
        wandb.run.name = self.wandb_runname
        self._update_logs()

    def _update_logs(self):
        """
        Write config logs to disk & update WandB cloud logs
        """
        configs = {
            k: v for k, v in self.__dict__.items() if type(v) in [str, int, float, bool]
        }
        pd_configs = pd.DataFrame.from_dict(data=configs, orient="index")
        pd_configs.to_csv(
            os.path.join(self.save_path, "model_config.csv"), header=False
        )
        wandb.config.update(configs)

    def prep_model(self):
        """
        Initialize model and dataloaders using CSV file
        """
        # load csv
        df = pd.read_csv(self.data_path)
        # get number of classes
        label_map = {
            label: idx for idx, label in enumerate(sorted(df["label"].unique()))
        }
        # determine number of output neurons
        num_classes = len(label_map)
        if self.loss in ["bce", "hinge", "hinge_squared"] and num_classes == 2:
            num_classes = 1
        # determine metrics
        task = "binary" if num_classes <= 2 else "multiclass"
        self.metric_collection = MetricCollection(
            {
                "acc": Accuracy(task=task),
                "f1": F1Score(task=task),
                "precision": Precision(task=task),
                "recall": Recall(task=task),
                "auroc": AUROC(task=task),
                "auprc": AveragePrecision(task=task),
                "ece": CalibrationError(task=task),
            }
        )
        # init model & transforms
        model_kwargs = {
            "model_name": self.model_name,
            "mode": self.train_mode,
            "num_classes": num_classes,
            "seed": self.seed,
        }
        model_ft, model_transform = NN(**model_kwargs).init_model()
        # define transforms
        data_transforms = {
            "train": transforms.Compose(
                [
                    model_transform,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    RotationTransform(angles=[0, 90, 180, 270]),
                ]
            ),
            "val": model_transform,
            "test": model_transform,
        }
        # split CSV by split column and create datasets
        image_datasets = {
            split: CSVDataset(
                df[df["split"] == split],
                transform=data_transforms[split],
                label_map=label_map,
            )
            for split in ["train", "val", "test"]
        }
        # dataloader parameters
        loader_args = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,
        }
        self.dataloaders = {
            split: DataLoader(image_datasets[split], **loader_args)
            for split in ["train", "val", "test"]
        }
        self.size_train_set = len(image_datasets["train"])
        self.size_val_set = len(image_datasets["val"])
        self.size_test_set = len(image_datasets["test"])
        # set model and count params
        self.model_ft = model_ft
        self.total_params = sum(p.numel() for p in model_ft.parameters())
        self.trainable_params = sum(
            p.numel() for p in model_ft.parameters() if p.requires_grad
        )
        self._update_logs()

    # credits: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    def train_model(self):
        """Train model"""
        # check GPU availability & send the model to GPU
        assert torch.cuda.is_available()
        device = torch.device("cuda:0")
        model_ft = self.model_ft.to(device)
        # gather params to be updated
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        # set up optimiser
        if self.optimiser == "SGD":
            optimizer_ft = optim.SGD(params_to_update, lr=self.lr, momentum=0.9)
        elif self.optimiser == "Adam":
            optimizer_ft = optim.Adam(params_to_update, lr=self.lr)
        # set up loss
        if self.loss == "ce":
            criterion = torch.nn.CrossEntropyLoss()
        elif self.loss == "bce":
            criterion = torch.nn.BCEWithLogitsLoss()
        elif self.loss == "focal":
            criterion = FocalLoss(gamma=2)
        elif self.loss == "hinge":
            criterion = HingeLoss()
        elif self.loss == "hinge_squared":
            criterion = HingeLoss(squared=True)
        else:
            raise ValueError(f"Invalid loss function: {self.loss}")
        # initialise early stopping
        early_stopping = EarlyStopping(
            patience=self.patience, path=os.path.join(self.save_path, "model_params.pt")
        )
        # create logging objects
        self.loss_history = {"train": [], "val": [], "test": []}
        self.metric_history = {"train": [], "val": [], "test": []}
        self.metric_collection.to(device)
        # start training
        since = time.time()
        for epoch in range(self.max_epochs):
            if early_stopping.early_stop:
                break
            print("Epoch {}/{}".format(epoch, self.max_epochs - 1))
            print("-" * 10)
            for phase in ["train", "val", "test"]:
                if phase == "train":
                    model_ft.train()
                else:
                    model_ft.eval()
                self.metric_collection.reset()
                running_loss = 0.0
                # iterate over data
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(device, dtype=torch.float)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer_ft.zero_grad()
                    # forward pass
                    with torch.set_grad_enabled(phase == "train"):
                        if self.loss in ["ce", "focal"]:
                            # excpects outputs (N,C), labels (N)
                            outputs = model_ft(inputs)
                            loss = criterion(outputs, labels)
                            probs = (
                                torch.softmax(outputs, dim=1)[:, 1]
                                if outputs.shape[1] > 1
                                else outputs.squeeze()
                            )
                            preds = torch.argmax(outputs, dim=1)
                        elif self.loss == "bce":
                            # excpects outputs (N,), labels (N)
                            outputs = model_ft(inputs).view(-1)
                            loss = criterion(outputs, labels.float())
                            probs = torch.sigmoid(outputs)
                            preds = torch.round(probs)
                        elif self.loss in ["hinge", "hinge_squared"]:
                            # converts outputs & targets to range [-1, 1] first
                            outputs = model_ft(inputs).view(-1)
                            outputs_ = torch.nn.Tanh()(outputs)
                            labels_ = torch.clone(labels)
                            labels_[labels_ == 0] = -1
                            loss = criterion(outputs_, labels_)
                            probs = torch.sigmoid(outputs)
                            preds = torch.round(probs)
                        else:
                            raise ValueError(f"Invalid loss function: {self.loss}")
                    # backward pass only in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer_ft.step()
                    # add iteration statistics to epoch summary
                    for name, metric in self.metric_collection.items():
                        if name in ["auroc", "auprc", "ece"]:
                            metric.update(probs, labels)
                        else:
                            metric.update(preds, labels)
                    running_loss += loss.item() * inputs.size(0)
                # calculate & log epoch loss & accs
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_metrics = self.metric_collection.compute()
                self.loss_history[phase].append(epoch_loss)
                self.metric_history[phase].append(
                    {k: v.item() for k, v in epoch_metrics.items()}
                )
                # log & print information
                wandb.log(
                    {
                        f"loss_{phase}": epoch_loss,
                        **{f"{k}_{phase}": v for k, v in epoch_metrics.items()},
                    },
                    step=epoch,
                )
                print(
                    f"{phase}\t loss: {epoch_loss:.4f}, "
                    + ", ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()])
                )
                if phase == "test":
                    print()
                # evaluate early stopping
                if phase == "val":
                    early_stopping(epoch_loss, model_ft)

        # load best model
        model_ft.load_state_dict(
            torch.load(os.path.join(self.save_path, "model_params.pt"))
        )
        self.model_ft = model_ft
        # update logs
        self.training_epochs = epoch
        self.training_time_total = round(time.time() - since, 2)
        self.training_time_epoch = round(
            self.training_time_total / self.training_epochs, 2
        )
        pd.DataFrame(self.loss_history).to_csv(
            os.path.join(self.save_path, "model_loss.csv"), index=False
        )
        pd.DataFrame(self.metric_history).to_csv(
            os.path.join(self.save_path, "model_metrics.csv"), index=False
        )
        self._update_logs()
        # set summary vals based on early stopping point
        if self.patience < self.max_epochs:
            val_idx_min = np.argmin(self.loss_history["val"])
            for metric in ["loss", *self.metric_collection.keys()]:
                for phase in ["train", "val", "test"]:
                    if metric == "loss":
                        final_val = self.loss_history[phase][val_idx_min]
                    else:
                        final_val = self.metric_history[phase][val_idx_min][metric]
                    wandb.run.summary[f"best_{metric}_{phase}"] = final_val
        wandb.finish()


if __name__ == "__main__":

    # parse arguments
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(
        description="Train & evaluate neural network for classification",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to input data csv",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="model architecture to be trained, see nn_models.py for avaliable models",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        help="degree of pretraining - see nn_models.py for available args",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="path to save results for model run (model & logs)",
    )
    parser.add_argument("--batch_size", type=int, help="size of the batch")
    parser.add_argument(
        "--optimiser",
        type=str,
        choices=["SGD", "Adam"],
        help="steepest gradient algorithm",
    )
    parser.add_argument("--loss", type=str, default="ce", help="loss function")
    parser.add_argument("--lr", type=float, help="initial learning rate")
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="maximal number of epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="patience for early stopping on validation data",
    )
    parser.add_argument(
        "--seed", nargs="?", type=int, default=None, help="seed to reproduce results"
    )
    parser.add_argument(
        "--wandb_project", type=str, help="project name for logging results with WandB"
    )
    parser.add_argument(
        "--wandb_runname", type=str, help="name current run logged with WandB"
    )
    parser.add_argument(
        "--wandb_tags",
        nargs="*",
        default=[],
        type=str,
        help="tags for current run logged with WandB",
    )
    parser.add_argument(
        "--wandb_notes",
        nargs="?",
        default=None,
        type=str,
        help="additional notes/description for current run",
    )

    # re-parse booleans correctly
    config = vars(parser.parse_args())

    # path creation
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])
    if len(os.listdir(config["save_path"])):
        raise OSError("Save path isn't empty!")

    # train & evaluate model
    trainer = Trainer(**config)
    trainer.prep_model()
    trainer.train_model()
