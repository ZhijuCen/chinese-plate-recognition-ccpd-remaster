
import albumentations as A


def default_transform(image_size=(224, 64)) -> A.Compose:
    """A default image transformation

    Args:
        image_size (Tuple[int, int]): size of image to be resized.
    Returns:
        albumentations.Compose: requires image.dtype is np.uint8 when called.
    """
    w, h = image_size
    transform = A.Compose([
        A.Resize(h, w),
        A.RandomBrightnessContrast(),
        A.InvertImg(),
        A.ToGray(),
        A.Affine(p=0.1),
    ])
    return transform
