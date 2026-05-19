from functools import lru_cache
from hashlib import sha1

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from vlm_guard.image.enhance import EnhancementStrategy, ImageEnhancer


AUTOCONTRAST_CUTOFF = 0.5
BLOOD_COLOR_BOOST = 1.8
BLOOD_CONTRAST_BOOST = 1.4
BLOOD_UNSHARP = dict(radius=2, percent=150, threshold=3)

TISSUE_CONTRAST_BOOST = 1.2
TISSUE_SHARPNESS_BOOST = 1.3

SKIN_CONTRAST_BOOST = 1.5
SKIN_UNSHARP = dict(radius=2, percent=200, threshold=2)
SKIN_EDGE_AUTOCONTRAST_CUTOFF = 1


def enhance_ntd_image(image: Image.Image, sample_type: str) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")

    enhanced = ImageOps.autocontrast(image, cutoff=AUTOCONTRAST_CUTOFF)
    s = sample_type.lower()

    if "blood" in s:
        g = image.split()[1]
        structure_mask = ImageOps.invert(g)
        enhanced = ImageEnhance.Color(enhanced).enhance(BLOOD_COLOR_BOOST)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(BLOOD_CONTRAST_BOOST)
        sharpened = enhanced.filter(ImageFilter.UnsharpMask(**BLOOD_UNSHARP))
        enhanced = Image.composite(sharpened, enhanced, structure_mask)

    elif "tissue" in s or "biopsy" in s:
        enhanced = ImageOps.equalize(enhanced)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(TISSUE_CONTRAST_BOOST)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(TISSUE_SHARPNESS_BOOST)

    elif "skin" in s:
        base = ImageEnhance.Contrast(enhanced).enhance(SKIN_CONTRAST_BOOST)
        sharpened = base.filter(ImageFilter.UnsharpMask(**SKIN_UNSHARP))
        edge_mask = base.convert("L").filter(ImageFilter.FIND_EDGES)
        edge_mask = ImageOps.autocontrast(edge_mask, cutoff=SKIN_EDGE_AUTOCONTRAST_CUTOFF)
        enhanced = Image.composite(sharpened, base, edge_mask)

    return enhanced
