from PIL import Image, ImageFilter, ImageDraw
import cv2
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import torch
from typing import Tuple


class Subject200KDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 512,
        target_size: int = 512,
        image_size: int = 512,
        padding: int = 0,
        condition_type: str = "subject",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.image_size = image_size
        self.padding = padding
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset) * 2

    def __getitem__(self, idx):

        target = idx % 2
        item = self.base_dataset[idx // 2]


        image = item["image"]
        left_img = image.crop(
            (
                self.padding,
                self.padding,
                self.image_size + self.padding,
                self.image_size + self.padding,
            )
        )
        right_img = image.crop(
            (
                self.image_size + self.padding * 2,
                self.padding,
                self.image_size * 2 + self.padding * 2,
                self.image_size + self.padding,
            )
        )


        target_image, condition_img = (
            (left_img, right_img) if target == 0 else (right_img, left_img)
        )


        condition_img = condition_img.resize(
            (self.condition_size, self.condition_size)
        ).convert("RGB")
        target_image = target_image.resize(
            (self.target_size, self.target_size)
        ).convert("RGB")


        description = item["description"][
            "description_0" if target == 0 else "description_1"
        ]


        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new(
                "RGB", (self.condition_size, self.condition_size), (0, 0, 0)
            )

        return {
            "image": self.to_tensor(target_image),
            "condition": self.to_tensor(condition_img),
            "condition_type": self.condition_type,
            "description": description,

            "position_delta": np.array([0, -self.condition_size // 16]),
            **({"pil_image": image} if self.return_pil_image else {}),
        }

class MOSARGenDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 512,
        target_size: int = None,
        padding: int = 0,
        condition_type: str = "multi_subject",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        drop_condition_prompt_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.padding = padding
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.drop_condition_prompt_prob = drop_condition_prompt_prob
        self.return_pil_image = return_pil_image

        self.to_tensor = T.ToTensor()

    def _image_center_crop(self, image):
        """将图片按照最小边裁剪为正方形"""
        width, height = image.size
        min_size = min(width, height)
        left_pos = (width - min_size) // 2
        top_pos = (height - min_size) // 2
        image = image.crop((left_pos, top_pos, left_pos + min_size, top_pos + min_size))
        return image
    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        target = idx
        item = self.base_dataset[idx]


        target_image = item["tgt_img"]
        condition_img_0 = item["cond_img_0"]
        condition_img_1 = item["cond_img_1"]
        cond_mask_0 = item.get("cond_mask_0", None)
        cond_mask_1 = item.get("cond_mask_1", None)


        if self.target_size is not None:
            target_image = self._image_center_crop(target_image)
            target_image = target_image.resize((self.target_size, self.target_size)).convert("RGB")


        resized_masks = []
        processed_masks = []

        for cond_mask in [cond_mask_0, cond_mask_1]:
            if cond_mask is not None:
                if self.target_size is not None:
                    cond_mask = self._image_center_crop(cond_mask)
                    cond_mask = cond_mask.resize((self.target_size, self.target_size)).convert("L")
                cond_mask_tensor = self.to_tensor(cond_mask)
                cond_mask_tensor = (cond_mask_tensor > 0).to(dtype=torch.bool)
                processed_masks.append(cond_mask_tensor)
            else:
                dummy_mask = torch.ones((1, target_image.height, target_image.width), dtype=torch.bool)
                processed_masks.append(dummy_mask)

        cond_mask_0_tensor, cond_mask_1_tensor = processed_masks


        condition_img_0 = condition_img_0.resize((self.condition_size, self.condition_size)).convert("RGB")
        condition_img_1 = condition_img_1.resize((self.condition_size, self.condition_size)).convert("RGB")


        description = item["prompt"]
        cond_prompt_0 = item["cond_prompt_0"]
        cond_prompt_1 = item["cond_prompt_1"]
        start_prompt = "An undivided, seamless, and harmonious picture with two objects."
        if description.startswith(start_prompt):

            description = description[len(start_prompt):]


        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        drop_condition_prompt_0 = random.random() < self.drop_condition_prompt_prob
        drop_condition_prompt_1 = random.random() < self.drop_condition_prompt_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img_0 = condition_img_1 = Image.new(
                "RGB", (self.condition_size, self.condition_size), (0, 0, 0)
            )
            cond_prompt_0 = cond_prompt_1 = ""
        if drop_condition_prompt_0:
            cond_prompt_0 = ""
        if drop_condition_prompt_1:
            cond_prompt_1 = ""


        return {
            "image": self.to_tensor(target_image),
            "condition_0": self.to_tensor(condition_img_0),
            "condition_1": self.to_tensor(condition_img_1),
            "condition_type": self.condition_type,
            "description": description,
            "cond_prompt_0": cond_prompt_0,
            "cond_prompt_1": cond_prompt_1,
            "cond_mask_0": cond_mask_0_tensor,
            "cond_mask_1": cond_mask_1_tensor,

            "position_delta_0": np.array([0, -self.condition_size // 16]),
            "position_delta_1": np.array([0, -self.condition_size * 2 // 16]),
            **({"pil_image": [target_image, condition_img_0, condition_img_1]} if self.return_pil_image else {}),
        }


class FLIRDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        dataset_root="",
        condition_size: int = 512,
        target_size: int = None,
        padding: int = 0,
        condition_type: str = "infrared",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
        enable_instruct_description: bool = False,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.padding = padding
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.dataset_root = dataset_root
        self.return_pil_image = return_pil_image
        self.enable_instruct_description = enable_instruct_description

        self.to_tensor = T.ToTensor()

    def _image_center_crop(self, image):
        """将图片按照最小边裁剪为正方形"""
        width, height = image.size
        min_size = min(width, height)
        left_pos = (width - min_size) // 2
        top_pos = (height - min_size) // 2
        image = image.crop((left_pos, top_pos, left_pos + min_size, top_pos + min_size))
        return image
    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]


        infrared_path = os.path.join(self.dataset_root, item["infrared_path"])
        vision_path = os.path.join(self.dataset_root, item["vision_path"])
        target_image = Image.open(infrared_path).convert("RGB")
        condition_img_0 = Image.open(vision_path).convert("RGB")


        description_vision_only = item["description_vision_only"]
        description_instruct = item["description_instruct"]
        description = description_vision_only if not self.enable_instruct_description else description_instruct


        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img_0 = Image.new(
                "RGB", condition_img_0.size, (0, 0, 0)
            )

        return {
            "image": self.to_tensor(target_image),
            "condition_0": self.to_tensor(condition_img_0),
            "condition_type": self.condition_type,
            "description": description,
            "vision_path":  vision_path,
            "infrared_path": infrared_path,


            "position_delta_0": np.array([0, 0]),
            **({"pil_image": [target_image, condition_img_0]} if self.return_pil_image else {}),
        }

class InfraredDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        dataset_root="",
        condition_size: int = 512,
        target_size: int = None,
        padding: int = 0,
        condition_type: str = "infrared",
        drop_text_prob: float = 0.0,
        drop_image_prob: float = 0.0,
        return_pil_image: bool = False,
        enable_instruct_description: bool = False,
        resize_resolution: Tuple[int, int] = (640, 512),
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.padding = padding
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.dataset_root = dataset_root
        self.return_pil_image = return_pil_image
        self.enable_instruct_description = enable_instruct_description
        self.resize_resolution = resize_resolution

        self.to_tensor = T.ToTensor()

    def _image_center_crop(self, image):
        """将图片按照最小边裁剪为正方形"""
        width, height = image.size
        min_size = min(width, height)
        left_pos = (width - min_size) // 2
        top_pos = (height - min_size) // 2
        image = image.crop((left_pos, top_pos, left_pos + min_size, top_pos + min_size))
        return image
    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]


        infrared_path = os.path.join(self.dataset_root, item["infrared_path"])
        vision_path = os.path.join(self.dataset_root, item["vision_path"])
        target_image = Image.open(infrared_path).convert("RGB")
        condition_img_0 = Image.open(vision_path).convert("RGB")


        if self.target_size is not None:


            target_image = target_image.resize(self.resize_resolution).convert("RGB")


        if self.condition_size is not None:


            condition_img_0 = condition_img_0.resize(self.resize_resolution).convert("RGB")


        description_vision_only = item["description_vision_only"]
        description_instruct = item["description_instruct"]
        description = description_vision_only if not self.enable_instruct_description else description_instruct


        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img_0 = Image.new(
                "RGB", condition_img_0.size, (0, 0, 0)
            )

        return {
            "image": self.to_tensor(target_image),
            "condition_0": self.to_tensor(condition_img_0),
            "condition_type": self.condition_type,
            "description": description,
            "vision_path":  vision_path,
            "infrared_path": infrared_path,


            "position_delta_0": np.array([0, 0]),
            **({"pil_image": [target_image, condition_img_0]} if self.return_pil_image else {}),
        }

class M3FDDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        dataset_root="",
        condition_size=(512, 384),
        target_size=(512, 384),
        padding: int = 0,
        condition_type: str = "infrared",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
        enable_instruct_description: bool = False,
        scale_ratio: float = 1.02
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.padding = padding
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.dataset_root = dataset_root
        self.return_pil_image = return_pil_image
        self.enable_instruct_description = enable_instruct_description

        self.to_tensor = T.ToTensor()

    def _image_center_crop(self, image, target_height, target_width):
        """将图片按照最小边裁剪为正方形"""
        width, height = image.size

        left_pos = (width - target_width) // 2
        top_pos = (height - target_height) // 2
        image = image.crop((left_pos, top_pos, left_pos + target_width, top_pos + target_height))
        return image

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]


        infrared_path = os.path.join(self.dataset_root, item["infrared_path"])
        vision_path = os.path.join(self.dataset_root, item["vision_path"])
        target_image = Image.open(infrared_path)
        condition_img_0 = Image.open(vision_path)


        condition_img_0 = condition_img_0.resize(self.condition_size).convert("RGB")

        target_image = target_image.resize(self.target_size).convert("RGB")


        description_vision_only = item["description_vision_only"]
        description_instruct = item["description_instruct"]
        description = description_vision_only if not self.enable_instruct_description else description_instruct


        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img_0 = Image.new(
                "RGB", condition_img_0.size, (0, 0, 0)
            )

        return {
            "image": self.to_tensor(target_image),
            "condition_0": self.to_tensor(condition_img_0),
            "condition_type": self.condition_type,
            "description": description,
            "vision_path":  vision_path,
            "infrared_path": infrared_path,


            "position_delta_0": np.array([0, 0]),
            **({"pil_image": [target_image, condition_img_0]} if self.return_pil_image else {}),
        }

class ResizedInfraredDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        dataset_root="",
        condition_size=(512, 384),
        target_size=(512, 384),
        padding: int = 0,
        condition_type: str = "infrared",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
        enable_instruct_description: bool = False,
        scale_ratio: float = 1.02
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.padding = padding
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.dataset_root = dataset_root
        self.return_pil_image = return_pil_image
        self.enable_instruct_description = enable_instruct_description

        self.to_tensor = T.ToTensor()

    def _image_center_crop(self, image, target_height, target_width):
        """将图片按照最小边裁剪为正方形"""
        width, height = image.size

        left_pos = (width - target_width) // 2
        top_pos = (height - target_height) // 2
        image = image.crop((left_pos, top_pos, left_pos + target_width, top_pos + target_height))
        return image

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]


        infrared_path = os.path.join(self.dataset_root, item["infrared_path"])
        vision_path = os.path.join(self.dataset_root, item["vision_path"])
        target_image = Image.open(infrared_path)
        condition_img_0 = Image.open(vision_path)


        condition_img_0 = condition_img_0.resize(self.condition_size).convert("RGB")

        target_image = target_image.resize(self.target_size).convert("RGB")


        description_vision_only = item["description_vision_only"]
        description_instruct = item["description_instruct"]
        description = description_vision_only if not self.enable_instruct_description else description_instruct


        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img_0 = Image.new(
                "RGB", condition_img_0.size, (0, 0, 0)
            )

        return {
            "image": self.to_tensor(target_image),
            "condition_0": self.to_tensor(condition_img_0),
            "condition_type": self.condition_type,
            "description": description,
            "vision_path":  vision_path,
            "infrared_path": infrared_path,


            "position_delta_0": np.array([0, 0]),
            **({"pil_image": [target_image, condition_img_0]} if self.return_pil_image else {}),
        }


class DreamboothDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 512,
        target_size: list = (1024,512),
        padding: int = 0,
        condition_type: str = "multi_subject",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.padding = padding
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image

        self.to_tensor = T.ToTensor()

    def _image_center_crop(self, image):
        """将图片按照最小边裁剪为正方形"""
        width, height = image.size
        min_size = min(width, height)
        left_pos = (width - min_size) // 2
        top_pos = (height - min_size) // 2
        image = image.crop((left_pos, top_pos, left_pos + min_size, top_pos + min_size))
        return image
    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]


        target_image = item["tgt_img"]
        condition_img_0  = item["cond_img_0"]
        condition_img_1 = item["cond_img_1"]

        condition_img_0, condition_img_1, target_image = [
            condition_img_0.resize(
                (self.condition_size, self.condition_size)
                ).convert("RGB"),
            condition_img_1.resize(
                (self.condition_size, self.condition_size)
                ).convert("RGB"),
            target_image.resize(
                (self.target_size[0], self.target_size[1])
                ).convert("RGB"),
        ]


        description = item["description"]


        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img_0 = condition_img_1 = Image.new(
                "RGB", (self.condition_size, self.condition_size), (0, 0, 0)
            )
            cond_prompt_0 = cond_prompt_1 = ""

        return {
            "image": self.to_tensor(target_image),
            "condition_0": self.to_tensor(condition_img_0),
            "condition_1": self.to_tensor(condition_img_1),
            "condition_type": self.condition_type,
            "description": description,

            "position_delta_0": np.array([0, -self.condition_size // 16]),
            "position_delta_1": np.array([0, -self.condition_size * 2 // 16]),
            **({"pil_image": [target_image, condition_img_0, condition_img_1]} if self.return_pil_image else {}),
        }


class ImageConditionDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 512,
        target_size: int = 512,
        condition_type: str = "canny",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
        position_scale=1.0,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.position_scale = position_scale

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    @property
    def depth_pipe(self):
        if not hasattr(self, "_depth_pipe"):
            from transformers import pipeline

            self._depth_pipe = pipeline(
                task="depth-estimation",
                model="LiheYoung/depth-anything-small-hf",
                device="cpu",
            )
        return self._depth_pipe

    def _get_canny_edge(self, img):
        resize_ratio = self.condition_size / max(img.size)
        img = img.resize(
            (int(img.size[0] * resize_ratio), int(img.size[1] * resize_ratio))
        )
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 100, 200)
        return Image.fromarray(edges).convert("RGB")

    def __getitem__(self, idx):
        image = self.base_dataset[idx]["jpg"]
        image = image.resize((self.target_size, self.target_size)).convert("RGB")
        description = self.base_dataset[idx]["json"]["prompt"]

        enable_scale = random.random() < 1
        if not enable_scale:
            condition_size = int(self.condition_size * self.position_scale)
            position_scale = 1.0
        else:
            condition_size = self.condition_size
            position_scale = self.position_scale


        position_delta = np.array([0, 0])
        if self.condition_type == "canny":
            condition_img = self._get_canny_edge(image)
        elif self.condition_type == "coloring":
            condition_img = (
                image.resize((condition_size, condition_size))
                .convert("L")
                .convert("RGB")
            )
        elif self.condition_type == "deblurring":
            blur_radius = random.randint(1, 10)
            condition_img = (
                image.convert("RGB")
                .filter(ImageFilter.GaussianBlur(blur_radius))
                .resize((condition_size, condition_size))
                .convert("RGB")
            )
        elif self.condition_type == "depth":
            condition_img = self.depth_pipe(image)["depth"].convert("RGB")
            condition_img = condition_img.resize((condition_size, condition_size))
        elif self.condition_type == "depth_pred":
            condition_img = image
            image = self.depth_pipe(condition_img)["depth"].convert("RGB")
            description = f"[depth] {description}"
        elif self.condition_type == "fill":
            condition_img = image.resize((condition_size, condition_size)).convert(
                "RGB"
            )
            w, h = image.size
            x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
            y1, y2 = sorted([random.randint(0, h), random.randint(0, h)])
            mask = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, y1, x2, y2], fill=255)
            if random.random() > 0.5:
                mask = Image.eval(mask, lambda a: 255 - a)
            condition_img = Image.composite(
                image, Image.new("RGB", image.size, (0, 0, 0)), mask
            )
        elif self.condition_type == "sr":
            condition_img = image.resize((condition_size, condition_size)).convert(
                "RGB"
            )
            position_delta = np.array([0, -condition_size // 16])

        else:
            raise ValueError(f"Condition type {self.condition_type} not implemented")


        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new(
                "RGB", (condition_size, condition_size), (0, 0, 0)
            )

        return {
            "image": self.to_tensor(image),
            "condition": self.to_tensor(condition_img),
            "condition_type": self.condition_type,
            "description": description,
            "position_delta": position_delta,
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
            **({"position_scale": position_scale} if position_scale != 1.0 else {}),
        }


class CartoonDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 1024,
        target_size: int = 1024,
        image_size: int = 1024,
        padding: int = 0,
        condition_type: str = "cartoon",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.image_size = image_size
        self.padding = padding
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        condition_img = data["condition"]
        target_image = data["target"]


        tag = data["tags"][0]

        target_description = data["target_description"]

        description = {
            "lion": "lion like animal",
            "bear": "bear like animal",
            "gorilla": "gorilla like animal",
            "dog": "dog like animal",
            "elephant": "elephant like animal",
            "eagle": "eagle like bird",
            "tiger": "tiger like animal",
            "owl": "owl like bird",
            "woman": "woman",
            "parrot": "parrot like bird",
            "mouse": "mouse like animal",
            "man": "man",
            "pigeon": "pigeon like bird",
            "girl": "girl",
            "panda": "panda like animal",
            "crocodile": "crocodile like animal",
            "rabbit": "rabbit like animal",
            "boy": "boy",
            "monkey": "monkey like animal",
            "cat": "cat like animal",
        }


        condition_img = condition_img.resize(
            (self.condition_size, self.condition_size)
        ).convert("RGB")
        target_image = target_image.resize(
            (self.target_size, self.target_size)
        ).convert("RGB")


        description = data.get(
            "description",
            f"Photo of a {description[tag]} cartoon character in a white background. Character is facing {target_description['facing_direction']}. Character pose is {target_description['pose']}.",
        )


        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new(
                "RGB", (self.condition_size, self.condition_size), (0, 0, 0)
            )

        return {
            "image": self.to_tensor(target_image),
            "condition": self.to_tensor(condition_img),
            "condition_type": self.condition_type,
            "description": description,

            "position_delta": np.array([0, -16]),
        }
