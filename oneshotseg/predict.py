import argparse
import torch
import logging
import cv2
from PIL import Image
from .model import SyameseUNet
from .utils import plot_img_and_mask
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from .dataset import DavisDataset


def predict_img(
    net,
    full_img,
    reference,
    reference_mask,
    device,
    scale_factor=1,
    out_threshold=0.5,
):
    net.eval()
    img = torch.from_numpy(DavisDataset.preprocess_input(full_img, scale_factor))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    reference = torch.from_numpy(
        DavisDataset.preprocess_reference(reference, reference_mask, scale_factor)
    )
    reference = reference.unsqueeze(0)
    reference = reference.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img, reference)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((full_img.size[1], full_img.size[0])),
                transforms.ToTensor(),
            ]
        )

        full_mask = tf(probs.cpu()).squeeze()

    return full_mask


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        "-d",
        default="DAVIS",
        action="store",
        help="Path to davis dataset",
    )
    parser.add_argument(
        "--input",
        "-i",
        action="store",
        help="Path input image",
        required=True,
    )
    parser.add_argument(
        "--reference",
        "-r",
        action="store",
        help="Path reference image",
        required=True,
    )
    parser.add_argument(
        "--mask",
        "-m",
        action="store",
        help="Path to reference mask",
        required=True,
    )
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="Use bilinear upsampling"
    )
    parser.add_argument(
        "--model", action="store", default=False, help="Path to model", required=True
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Minimum probability value to consider a mask pixel white",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=1.0,
        help="Scale factor for the input images",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    net = SyameseUNet(n_channels=3, n_classes=2, bilinear=args.bilinear)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    logging.info(f"Loading model {args.model}")
    net.load_state_dict(torch.load(args.model, map_location=device), strict=False)
    logging.info("Model loaded")

    if torch.cuda.is_available():
        net.cuda()

    img = Image.open(args.input)
    reference = Image.open(args.reference)
    reference_mask = Image.open(args.mask)

    mask = predict_img(
        net=net,
        full_img=img,
        reference=reference,
        reference_mask=reference_mask,
        scale_factor=args.scale,
        out_threshold=args.mask_threshold,
        device=device,
    )
    plot_img_and_mask(img, mask)
