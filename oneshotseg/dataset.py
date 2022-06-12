import logging
import cv2
from os import listdir
from os.path import splitext, join
from pathlib import Path
from random import choice

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from skimage import exposure


class DavisDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        scale: float = 1.0,
        full_shuffle: bool = False,
        ref_shuffle: int = 1,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, "Scale must be between 0 and 1"
        self.scale = scale

        classes = [
            splitext(folder)[0]
            for folder in listdir(images_dir)
            if not folder.startswith(".")
        ]

        self.ids = []

        for cl in classes:
            frames = sorted(
                [splitext(file)[0] for file in listdir(join(images_dir, cl))],
                key=lambda x: int(x),
            )
            refs = (
                (lambda: frames)
                if full_shuffle
                else (lambda: [choice(frames) for _ in range(ref_shuffle)])
            )

            for frame in frames:
                for ref in refs():
                    if ref != frame:
                        self.ids.append((join(cl, frame), join(cl, ref)))

        if not self.ids:
            raise RuntimeError(
                f"No input file found in {images_dir}, make sure you put your images there"
            )
        logging.info(f"Creating dataset with {len(self.ids)} examples")

    def __len__(self):
        return len(self.ids)

    @classmethod
    def _scale(cls, img, scale):
        w, h = img.size
        newW, newH = int(scale * w), int(scale * h)
        assert (
            newW > 0 and newH > 0
        ), "Scale is too small, resized images would have no pixel"
        img = img.resize((newW, newH), resample=Image.NEAREST)
        return img

    @classmethod
    def preprocess_input(cls, img, scale, transpose=True):
        img = cls._scale(img, scale)
        img = np.asarray(img)
        img = exposure.equalize_adapthist(img, clip_limit=0.03)
        if transpose:
            img = img.transpose((2, 0, 1))
        img = img / 255
        return img

    @classmethod
    def preprocess_mask(cls, img, scale):
        img = cls._scale(img, scale)
        img = np.asarray(img)
        if img.ndim > 2:
            img = np.dsplit(img, img.shape[-1])[0].reshape(img.shape[:-1])
        img = img / 255
        return img

    @classmethod
    def preprocess_reference(cls, ref, mask, scale):
        ref = cls.preprocess_input(ref, scale, transpose=False)
        mask = cls.preprocess_mask(mask, scale)
        idx = mask == 0
        ref[idx] = np.zeros_like(ref)[idx]
        ref = ref.transpose((2, 0, 1))
        return ref

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in [".npz", ".npy"]:
            return Image.fromarray(np.load(filename))
        elif ext in [".pt", ".pth"]:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name, ref = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + ".*"))
        img_file = list(self.images_dir.glob(name + ".*"))
        ref_mask_file = list(self.masks_dir.glob(ref + ".*"))
        ref_img_file = list(self.images_dir.glob(ref + ".*"))

        assert (
            len(img_file) == 1
        ), f"Either no image or multiple images found for the ID {name}: {img_file}"
        assert (
            len(ref_img_file) == 1
        ), f"Either no image or multiple images found for the ID {ref}: {ref_img_file}"
        assert (
            len(mask_file) == 1
        ), f"Either no mask or multiple masks found for the ID {name}: {mask_file}"
        assert (
            len(ref_mask_file) == 1
        ), f"Either no mask or multiple masks found for the ID {ref}: {ref_mask_file}"

        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        ref_mask = self.load(ref_mask_file[0])
        ref_img = self.load(ref_img_file[0])

        assert (
            img.size == mask.size == ref_img.size == ref_mask.size
        ), f"Image and mask {name} should be the same size, but are {img.size} and {mask.size}"

        img = self.preprocess_input(img, self.scale)
        mask = self.preprocess_mask(mask, self.scale)
        ref_img = self.preprocess_reference(ref_img, ref_mask, self.scale)

        return {
            "image": torch.as_tensor(img.copy()).float().contiguous(),
            "mask": torch.as_tensor(mask.copy()).long().contiguous(),
            "ref_image": torch.as_tensor(ref_img.copy()).float().contiguous(),
        }
