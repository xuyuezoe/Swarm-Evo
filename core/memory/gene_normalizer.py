"""Discretize extracted code genes into coarse categorical labels."""

from __future__ import annotations

from typing import Dict


class GeneNormalizer:
    """Maps raw gene snippets to deterministic, low-cardinality labels."""

    SECTION_KEYS = [
        "DATA",
        "MODEL",
        "LOSS",
        "OPTIMIZER",
        "REGULARIZATION",
        "INITIALIZATION",
        "TRAINING_TRICKS",
    ]

    def discretize(self, code_genes: Dict[str, str]) -> Dict[str, str]:
        """Return normalized labels for all required sections."""
        normalized: Dict[str, str] = {}
        for section in self.SECTION_KEYS:
            text = (code_genes.get(section) or "").lower()
            if section == "DATA":
                normalized[section] = self._normalize_data(text)
            elif section == "MODEL":
                normalized[section] = self._normalize_model(text)
            elif section == "LOSS":
                normalized[section] = self._normalize_loss(text)
            elif section == "OPTIMIZER":
                normalized[section] = self._normalize_optimizer(text)
            elif section == "REGULARIZATION":
                normalized[section] = self._normalize_regularization(text)
            elif section == "INITIALIZATION":
                normalized[section] = self._normalize_initialization(text)
            elif section == "TRAINING_TRICKS":
                normalized[section] = self._normalize_training_tricks(text)
        return normalized

    def _normalize_data(self, text: str) -> str:
        if "cifar" in text:
            return "CIFAR"
        if "mnist" in text:
            return "MNIST"
        if "imagenet" in text:
            return "ImageNet"
        if any(token in text for token in ("csv", "tabular", "dataframe", "structured")):
            return "Tabular"
        if any(token in text for token in ("timeseries", "temporal", "sequence")):
            return "TimeSeries"
        if any(token in text for token in ("text", "bert", "tokenizer", "nlp")):
            return "TextData"
        if any(token in text for token in ("augment", "cutmix", "mixup", "flip", "crop")):
            return "AugmentedData"
        return "GenericData"

    def _normalize_model(self, text: str) -> str:
        if "resnet" in text:
            return "ResNet"
        if "efficientnet" in text:
            return "EfficientNet"
        if any(token in text for token in ("vit", "vision transformer")):
            return "ViT"
        if "unet" in text:
            return "UNet"
        if "transformer" in text:
            return "Transformer"
        if any(token in text for token in ("lstm", "gru", "rnn")):
            return "RNN"
        if "mlp" in text or "feedforward" in text:
            return "MLP"
        if "cnn" in text or "conv" in text:
            return "SimpleCNN"
        return "UnknownModel"

    def _normalize_loss(self, text: str) -> str:
        if "cross" in text and "entropy" in text:
            return "CrossEntropy"
        if "dice" in text:
            return "DiceLoss"
        if "focal" in text:
            return "FocalLoss"
        if "bce" in text or "binary" in text:
            return "BinaryCE"
        if any(token in text for token in ("mse", "l2")):
            return "MSE"
        if any(token in text for token in ("mae", "l1", "smoothl1", "huber")):
            return "MAE"
        return "GenericLoss"

    def _normalize_optimizer(self, text: str) -> str:
        if "adamw" in text:
            return "AdamW"
        if "adam" in text:
            return "Adam"
        if "sgd" in text or "momentum" in text:
            return "SGD"
        if "rmsprop" in text:
            return "RMSprop"
        if "adagrad" in text:
            return "AdaGrad"
        if "adadelta" in text:
            return "AdaDelta"
        if "lion" in text:
            return "Lion"
        if "lbfgs" in text:
            return "LBFGS"
        return "GenericOpt"

    def _normalize_regularization(self, text: str) -> str:
        if "dropout" in text:
            return "Dropout"
        if "weight decay" in text or "l2" in text:
            return "WeightDecay"
        if "label smoothing" in text:
            return "LabelSmoothing"
        if "batchnorm" in text or "batch norm" in text:
            return "BatchNorm"
        if any(token in text for token in ("mixup", "cutmix")):
            return "MixAug"
        if "gradient penalty" in text:
            return "GradPenalty"
        return "MinimalReg"

    def _normalize_initialization(self, text: str) -> str:
        if "kaiming" in text or "he initialization" in text:
            return "Kaiming"
        if "xavier" in text or "glorot" in text:
            return "Xavier"
        if "lecun" in text:
            return "LeCun"
        if "orthogonal" in text:
            return "Orthogonal"
        if "uniform" in text:
            return "UniformInit"
        if "normal" in text:
            return "NormalInit"
        return "DefaultInit"

    def _normalize_training_tricks(self, text: str) -> str:
        if "cosine" in text:
            return "CosineSchedule"
        if "onecycle" in text:
            return "OneCycle"
        if "warmup" in text:
            return "Warmup"
        if "ema" in text:
            return "EMA"
        if "gradient clip" in text or "grad_clip" in text:
            return "GradClip"
        if "early stop" in text:
            return "EarlyStop"
        return "BaselineTricks"
