"""Crop of the 1280x1024 camera view out of the 1920x1080 video frames.

Per the EndoVis 2017 dataset info, the camera image is extracted by
cropping the frame at pixel (320, 28). Everything outside is black
letterboxing. Frames and masks share the same geometry, so both must be
cropped identically.
"""

from PIL import Image

CROP_LEFT = 320
CROP_TOP = 28
CROP_WIDTH = 1280
CROP_HEIGHT = 1024
CROP_BOX = (
    CROP_LEFT,
    CROP_TOP,
    CROP_LEFT + CROP_WIDTH,
    CROP_TOP + CROP_HEIGHT,
)


def crop_camera_view(image: Image.Image) -> Image.Image:
    """Crops a 1920x1080 frame or mask down to the 1280x1024 camera view."""
    return image.crop(CROP_BOX)
