from collections import OrderedDict
from enum import Enum, auto
from hashlib import sha1
from typing import Callable

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


class EnhancementStrategy(Enum):
    DEFAULT = auto()
    HIGH_CONTRAST = auto()
    EDGE_ENHANCE = auto()
    SHARPEN = auto()


_DEFAULT_STRATEGIES: dict[EnhancementStrategy, Callable] = {
    EnhancementStrategy.DEFAULT: lambda img: ImageEnhance.Sharpness(img).enhance(1.5),
    EnhancementStrategy.HIGH_CONTRAST: lambda img: ImageEnhance.Contrast(
        ImageOps.autocontrast(img, cutoff=0.5)
    ).enhance(1.4),
    EnhancementStrategy.EDGE_ENHANCE: lambda img: img.filter(
        ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
    ),
    EnhancementStrategy.SHARPEN: lambda img: img.filter(
        ImageFilter.SHARPEN
    ),
}


class ImageEnhancer:
    def __init__(self, strategy: EnhancementStrategy = EnhancementStrategy.DEFAULT):
        self.strategy = strategy

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.enhance(image)

    def enhance(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        fn = _DEFAULT_STRATEGIES.get(self.strategy, _DEFAULT_STRATEGIES[EnhancementStrategy.DEFAULT])
        return fn(image)


class CompositeEnhancer:
    def __init__(self, steps: list[Callable[[Image.Image], Image.Image]]):
        self.steps = steps

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.enhance(image)

    def enhance(self, image: Image.Image) -> Image.Image:
        for step in self.steps:
            image = step(image)
        return image


def cached_enhance(enhancer: Callable[[Image.Image], Image.Image], max_cache: int = 4):
    cache: OrderedDict[tuple, Image.Image] = OrderedDict()

    def _hash(image: Image.Image) -> str:
        return sha1(image.tobytes()).hexdigest()

    def wrapper(image: Image.Image) -> Image.Image:
        key = _hash(image)
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        result = enhancer(image)
        cache[key] = result
        if len(cache) > max_cache:
            cache.popitem(last=False)
        return result

    return wrapper



