
import albumentations as A


def default_keypoint_transform(
    bbox_format="pascal_voc", kp_format="xy", p=0.5
) -> A.Compose:
    bbox_params = A.BboxParams(bbox_format)
    kp_params = A.KeypointParams(kp_format, remove_invisible=False)
    compose = A.Compose([
        A.Resize(320, 320),
        A.InvertImg(p=p),
        A.ToGray(p=p),
        A.Perspective(p=p),
        A.ShiftScaleRotate(p=p),
        A.Flip(p=p),
    ], bbox_params=bbox_params, keypoint_params=kp_params)
    return compose


def identity_keypoint_transform(
    bbox_format="pascal_voc", kp_format="xy"
) -> A.Compose:
    bbox_params = A.BboxParams(bbox_format)
    kp_params = A.KeypointParams(kp_format, remove_invisible=False)
    compose = A.Compose([
        A.Resize(320, 320),
    ], bbox_params=bbox_params, keypoint_params=kp_params)
    return compose
