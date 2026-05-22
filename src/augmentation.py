import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def get_training_transforms(img_size=48):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_validation_transforms(img_size=48):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_test_transforms(img_size=48):
    return get_validation_transforms(img_size)


class FER2013Augmentation:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image):
        if not isinstance(image, torch.Tensor):
            raise TypeError("Expected torch.Tensor")
        if self.prob <= 0:
            return image
        if random.random() < self.prob:
            image = self._horizontal_flip(image)
        if random.random() < self.prob:
            image = self._rotation(image)
        if random.random() < self.prob:
            image = self._random_crop(image)
        if random.random() < self.prob:
            image = self._add_gaussian_noise(image)
        if random.random() < self.prob:
            image = self._adjust_brightness(image)
        return image

    def _horizontal_flip(self, image):
        return torch.flip(image, dims=[-1])

    def _rotation(self, image, max_angle=10):
        angle = random.uniform(-max_angle, max_angle)
        angle_rad = torch.tensor(angle * 3.14159265 / 180.0)
        cos_a = torch.cos(angle_rad).item()
        sin_a = torch.sin(angle_rad).item()
        h, w = image.shape[-2], image.shape[-1]
        cx, cy = w / 2, h / 2
        M = torch.tensor([[cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy],
                          [sin_a, cos_a, -sin_a * cx + (1 - cos_a) * cy]], dtype=torch.float32)
        grid = F.affine_grid(M.unsqueeze(0), (1, 1, h, w), align_corners=False)
        return F.grid_sample(image.unsqueeze(0), grid, align_corners=False).squeeze(0)

    def _random_crop(self, image, scale=0.9):
        _, h, w = image.shape
        crop_h, crop_w = int(h * scale), int(w * scale)
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        cropped = image[:, top:top + crop_h, left:left + crop_w]
        return F.interpolate(cropped.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)

    def _add_gaussian_noise(self, image, std=0.05):
        noise = torch.randn_like(image) * std
        return torch.clamp(image + noise, -1.0, 1.0)

    def _adjust_brightness(self, image, factor_range=(0.8, 1.2)):
        factor = random.uniform(*factor_range)
        return torch.clamp(image * factor, -1.0, 1.0)
