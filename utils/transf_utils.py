import torchvision.transforms as transforms

import utils.proj_utils as prju

_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = (0.4914, 0.4822, 0.4465)
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = (0.2023, 0.1994, 0.2010)


def initialize_transform(
    dataset_name=prju.DatasetParameters.CIFAR10_NAME, target_resolution=32
):
    """
    By default, transforms should take in `x` and return `transformed_x`.
    """
    # target_resolution = 32
    # resize_resolution = 32

    transform = transforms.Compose(
        [
            transforms.Resize(target_resolution),
            transforms.CenterCrop((target_resolution, target_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(
                _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN,
                _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD,
            ),
        ]
    )
    return transform


def getBertTokenizer(model):
    from transformers import BertTokenizerFast, DistilBertTokenizerFast

    if model == "bert-base-uncased":
        tokenizer = BertTokenizerFast.from_pretrained(model)
    elif model == "distilbert-base-uncased":
        tokenizer = DistilBertTokenizerFast.from_pretrained(model)
    else:
        raise ValueError(f"Model: {model} not recognized.")

    return tokenizer


def initialize_bert_transform(net, max_token_length=512):
    # assert 'bert' in config.model
    # assert config.max_token_length is not None
    import torch

    tokenizer = getBertTokenizer(net)

    def transform(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_token_length,
            return_tensors="pt",
        )
        if net == "bert-base-uncased":
            x = torch.stack(
                (tokens["input_ids"], tokens["attention_mask"], tokens["token_type_ids"]),
                dim=2,
            )
        elif net == "distilbert-base-uncased":
            x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform
