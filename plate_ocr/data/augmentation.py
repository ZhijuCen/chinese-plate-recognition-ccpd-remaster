
import albumentations as A


def default_transform(image_size=(224, 64), is_val=False) -> A.Compose:
    """A default image transformation

    Args:
        image_size (Tuple[int, int]): size of image to be resized.
    Returns:
        albumentations.Compose: requires image.dtype is np.uint8 when called.
    """
    w, h = image_size
    if is_val:
        transform = A.Compose([A.Resize(h, w)])
    else:
        transform = A.Compose([
            A.Resize(h, w),
            A.RandomBrightnessContrast(p=0.2),
            A.InvertImg(p=0.5),
            A.ToGray(p=0.2),
            A.Affine(p=0.1),
        ])
    return transform
