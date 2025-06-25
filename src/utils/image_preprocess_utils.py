import io
import os
import base64
from PIL import Image
from typing import Tuple, Union



def encode_image(image_path: str, encode_base64: bool = True) -> Union[str, bytes]:
    """
    Read an image file and return its binary content, optionally encoded in base64.

    Args:
        image_path (str): Path to the image file.
        encode_base64 (bool, optional): If True, the image is returned as a base64-encoded string.
                                        If False, the raw binary content is returned. Defaults to True.

    Returns:
        Union[str, bytes]: The base64-encoded string or raw binary content of the image,
                           depending on the value of `encode_base64`.
    """
    with open(image_path, "rb") as image_file:
        if encode_base64:
            return base64.b64encode(image_file.read()).decode("utf-8")
        return image_file.read()



def encode_resize_image(image_path: str, max_size: Tuple[int], encode_base64: bool = True) -> Union[str, bytes]:
    """
    Resize an image to fit within the specified maximum dimensions and return its content,
    optionally encoded in base64.

    If the original image is smaller than the specified max size, it will be returned without resizing.

    Args:
        image_path (str): Path to the input image file.
        max_size (Tuple[int]): Maximum allowed size as (width, height). The image will be resized
                               proportionally to fit within this box while preserving aspect ratio.
        encode_base64 (bool, optional): If True, the image content is returned as a base64-encoded string.
                                        If False, raw binary data is returned. Defaults to True.

    Returns:
        Union[str, bytes]: The resized image as a base64-encoded string or raw binary data,
                           depending on the value of `encode_base64`.
    """
    with Image.open(image_path) as img:
        original_width, original_height = img.size
        
        # Resize the image
        width_ratio = max_size[0] / original_width
        height_ratio = max_size[1] / original_height
        min_ratio = min(width_ratio, height_ratio)
        
        if min_ratio >= 1:
            return encode_image(image_path, encode_base64)

        new_width = int(original_width * min_ratio)
        new_height = int(original_height * min_ratio)

        img = img.resize((new_width, new_height))

        # Save image to memeory
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG")
        img_buffer.seek(0)

        if encode_base64:
            return base64.b64encode(img_buffer.read()).decode("utf-8")
        return img_buffer.read()
    


def get_image_extension(path: str) -> str:
    """
    Extract and normalize the image file extension from the given file path.

    Args:
        path (str): Path to the image file.

    Raises:
        ValueError: If the file extension is not a supported image format.

    Returns:
        str: Normalized image format string. Returns "png" for PNG files,
             and "jpeg" for JPG or JPEG files. 
    """
    ext = os.path.splitext(path)
    if ext == "png":
        return ext
    elif ext in ["jpeg", "jpg"]:
        return "jpeg"
    else:
        raise ValueError(f"Unsupported image format: {ext}")
