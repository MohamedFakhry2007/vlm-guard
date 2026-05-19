from PIL import Image, ImageFilter, ImageStat, ImageChops

from vlm_guard.image.quality import check_image_quality


def test_good_image_no_warnings():
    img = Image.new("RGB", (100, 100), color=(200, 150, 100))
    warnings = check_image_quality(img)
    assert len(warnings) == 0


def test_blurry_image_detected():
    uniform = Image.new("RGB", (2, 2), color=(128, 128, 128))
    warnings = check_image_quality(uniform)
    has_blur = any("Blurry" in w for w in warnings)
    assert has_blur


def test_underexposed_detected():
    img = Image.new("RGB", (100, 100), color=(10, 10, 10))
    warnings = check_image_quality(img)
    assert any("Underexposed" in w for w in warnings) or any("Dark" in w for w in warnings)


def test_overexposed_detected():
    img = Image.new("RGB", (100, 100), color=(250, 250, 250))
    warnings = check_image_quality(img)
    assert any("Overexposed" in w for w in warnings)


def test_grayscale_detected():
    gray = Image.new("L", (100, 100), color=(128,))
    img = gray.convert("RGB")
    warnings = check_image_quality(img)
    assert any("Grayscale" in w for w in warnings) or any("Color" in w for w in warnings)
