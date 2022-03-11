
import albumentations as A


def default_keypoint_transform(
    bbox_format="pascal_voc", kp_format="xy"
) -> A.Compose:
    bbox_params = A.BboxParams(bbox_format)
    kp_params = A.KeypointParams(kp_format)
    compose = A.Compose([
        A.InvertImg(),
        A.ShiftScaleRotate(),
        A.Perspective(),
        A.ToGray(),
    ], bbox_params=bbox_params, keypoint_params=kp_params)
    return compose
