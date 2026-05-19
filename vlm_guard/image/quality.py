from PIL import Image, ImageFilter, ImageStat


BLUR_EDGE_VAR_MIN = 20
UNDEREXPOSED_MEAN_MAX = 40
OVEREXPOSED_MEAN_MIN = 240
GRAYSCALE_RB_DIFF_MAX = 5


def check_image_quality(image: Image.Image) -> list[str]:
    warnings: list[str] = []

    gray = image.convert("L")
    gray_stat = ImageStat.Stat(gray)

    edges = image.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges.convert("L"))
    if edge_stat.var[0] < BLUR_EDGE_VAR_MIN:
        warnings.append("Blurry / Out of Focus")

    if gray_stat.mean[0] < UNDEREXPOSED_MEAN_MAX:
        warnings.append("Too Dark (Underexposed)")
    if gray_stat.mean[0] > OVEREXPOSED_MEAN_MIN:
        warnings.append("Overexposed (Washed out)")

    r, g, b = image.split()
    mean_r = ImageStat.Stat(r).mean[0]
    mean_b = ImageStat.Stat(b).mean[0]
    if abs(mean_r - mean_b) < GRAYSCALE_RB_DIFF_MAX:
        warnings.append("Low Color Information (Possible Grayscale)")

    return warnings
