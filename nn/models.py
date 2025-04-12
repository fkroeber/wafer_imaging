import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class NN:
    """
    Initialising SOTA deep learning models for classification.
    """

    def __init__(self, **kwargs):
        """
        Models can be retrieved with/without pre-trained weights and
        with the last layer being adjusted to the number of required classes.

        model_name: name of the model to be used - available args are...
            "convnext"
            "densenet"
            "resnet"
            "resnext"
            "senet"
            "squeezenet"
            "wideresnet"
            "vit"
            "max_vit"
            "swin_vit"
        mode: degree of use of pretrained models - available args are...
            "scratch"
            "finetune"
            "feat_extract"
        num_classes: number of classes in the final layer of the model as an integer value
        seed: optional seed for reproducibility of model initialisation
        """
        self.model_name = kwargs.get("model_name")
        self.mode = kwargs.get("mode")
        self.num_classes = kwargs.get("num_classes")
        self.seed = kwargs.get("seed")
        # set seed
        if self.seed:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
        # check validity of mode
        allowed_modes = ["scratch", "finetune", "feat_extract"]
        if self.mode not in allowed_modes:
            raise ValueError(
                f"Invalid argument for mode: {self.mode}. Allowed args are: {allowed_modes}"
            )
        # translate mode into information about usage of pretrained weights & need to freeze weights
        self.use_pretrained = self.mode in ["finetune", "feat_extract"]
        self.feature_extract = self.mode == "feat_extract"

    # freeze weights for feat_extrat mode
    def _set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    # define core methods
    def init_model(self):
        """
        Intitialise model specific to the chosen net architecture, train mode and number of classes.
        For each model the model object along with necessary transforms
        to make the use of pretrained weights applicable is returned.
        Note: The data transforms are pretraining specific but
        in most cases for Imagenet pretraining something like
        # transforms.RandomResizedCrop(input_size)
        # transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        """

        # category: CNNs
        if self.model_name == "convnext":
            w = models.ConvNeXt_Tiny_Weights.DEFAULT
            w_apply = w if self.use_pretrained else None
            model = models.convnext_tiny(weights=w_apply)
            self._set_parameter_requires_grad(model, self.feature_extract)
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features, self.num_classes
            )
            transforms = w.transforms()

        elif self.model_name == "densenet":
            w = models.DenseNet121_Weights.DEFAULT
            w_apply = w if self.use_pretrained else None
            model = models.densenet121(weights=w_apply)
            self._set_parameter_requires_grad(model, self.feature_extract)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, self.num_classes)
            transforms = w.transforms()

        elif self.model_name == "resnet":
            w = models.ResNet18_Weights.DEFAULT
            w_apply = w if self.use_pretrained else None
            model = models.resnet18(weights=w_apply)
            self._set_parameter_requires_grad(model, self.feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
            transforms = w.transforms()

        elif self.model_name == "resnext":
            w = models.ResNeXt50_32X4D_Weights.DEFAULT
            w_apply = w if self.use_pretrained else None
            model = models.resnext50_32x4d(weights=w_apply)
            self._set_parameter_requires_grad(model, self.feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
            transforms = w.transforms()

        elif self.model_name == "senet":
            model = torch.hub.load(
                "moskomule/senet.pytorch",
                "se_resnet50",
                pretrained=self.use_pretrained,
                verbose=False,
            )
            self._set_parameter_requires_grad(model, self.feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
            transforms = models.ResNet50_Weights.DEFAULT.transforms()

        elif self.model_name == "squeezenet":
            w = models.SqueezeNet1_0_Weights.DEFAULT
            w_apply = w if self.use_pretrained else None
            model = models.squeezenet1_0(weights=w_apply)
            self._set_parameter_requires_grad(model, self.feature_extract)
            model.classifier[1] = nn.Conv2d(
                512, self.num_classes, kernel_size=(1, 1), stride=(1, 1)
            )
            model.num_classes = self.num_classes
            transforms = w.transforms()

        elif self.model_name == "wideresnet":
            w = models.Wide_ResNet50_2_Weights.DEFAULT
            w_apply = w if self.use_pretrained else None
            model = models.wide_resnet50_2(weights=w_apply)
            self._set_parameter_requires_grad(model, self.feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
            transforms = w.transforms()

        # category: vision transformers
        elif self.model_name == "max_vit":
            w = models.MaxVit_T_Weights.DEFAULT
            w_apply = w if self.use_pretrained else None
            model = models.maxvit_t(weights=w_apply)
            self._set_parameter_requires_grad(model, self.feature_extract)
            num_ftrs = model.classifier[5].in_features
            model.classifier[5] = nn.Linear(num_ftrs, self.num_classes)
            transforms = w.transforms()

        elif self.model_name == "swin_vit":
            w = models.Swin_T_Weights.DEFAULT
            w_apply = w if self.use_pretrained else None
            model = models.swin_t(weights=w_apply)
            self._set_parameter_requires_grad(model, self.feature_extract)
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, self.num_classes)
            transforms = w.transforms()

        elif self.model_name == "vit":
            w = models.ViT_B_16_Weights.DEFAULT
            w_apply = w if self.use_pretrained else None
            model = models.vit_b_16(weights=w_apply)
            self._set_parameter_requires_grad(model, self.feature_extract)
            num_ftrs = model.heads.head.in_features
            model.heads.head = nn.Linear(num_ftrs, self.num_classes)
            transforms = w.transforms()

        else:
            raise ValueError(f"Invalid model name: {self.model_name}")

        return model, transforms
