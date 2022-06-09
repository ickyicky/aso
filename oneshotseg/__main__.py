import argparse
import torch
import logging
from .unet_model import UNet


logging.basicConfig(level=logging.DEBUG)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    "-d",
    default="DAVIS",
    action="store",
    help="Path to davis dataset",
)
parser.add_argument(
    "--example",
    "-e",
    action="store",
    help="Path example image",
)
parser.add_argument(
    "--mask",
    "-m",
    action="store",
    help="Path to example mask",
)
parser.add_argument(
    "--bilinear", action="store_true", default=False, help="Use bilinear upsampling"
)
parser.add_argument("--model", action="store", default=False, help="Path to model")

action = parser.add_mutually_exclusive_group()
action.add_argument("--train", "-t", action="store_true", help="Train model")
action.add_argument("--run", "-r", action="store_true", help="Run model on example")

args = parser.parse_args()

net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device {device}")

if args.model:
    logging.info(f"Loading model {args.model}")
    net.load_state_dict(torch.load(args.model, map_location=device))
