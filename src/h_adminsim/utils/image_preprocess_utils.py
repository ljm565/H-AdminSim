import io
import os
import math
import base64
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
from typing import Callable, Union



def encode_image(image_path: str, encode_base64: bool = True) -> Union[str, bytes]:
    """
    Read an image file and return its binary content, optionally encoded in base64.

    Args:
        image_path (str): Path to the image file.
        encode_base64 (bool, optional): Return a base64 string instead of raw bytes. Defaults to True.

    Returns:
        Union[str, bytes]: Base64 string or raw bytes of the image.
    """
    with open(image_path, "rb") as image_file:
        if encode_base64:
            return base64.b64encode(image_file.read()).decode("utf-8")
        return image_file.read()



def encode_resize_image(image_path: str, max_size: tuple[int, int], encode_base64: bool = True) -> Union[str, bytes]:
    """
    Downscale an image (keeping aspect ratio, saved as JPEG) to fit within ``max_size`` and return its content.
    Images that already fit are returned unchanged.

    Args:
        image_path (str): Path to the input image file.
        max_size (tuple[int, int]): Maximum ``(width, height)``.
        encode_base64 (bool, optional): Return a base64 string instead of raw bytes. Defaults to True.

    Returns:
        Union[str, bytes]: Base64 string or raw bytes of the image.
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
    


def get_image_extension(path: str) -> Union[str, tuple[str, str]]:
    """
    Normalized image format of a path: "png", or "jpeg" for jpg/jpeg.
    Caveat: it compares the whole ``os.path.splitext`` tuple, so as written it always raises.

    Args:
        path (str): Path to the image file.

    Returns:
        Union[str, tuple[str, str]]: The image format.

    Raises:
        ValueError: If the file extension is not a supported image format.
    """
    ext = os.path.splitext(path)
    if ext == "png":
        return ext
    elif ext in ["jpeg", "jpg"]:
        return "jpeg"
    else:
        raise ValueError(f"Unsupported image format: {ext}")



def autopct_format(values: list[int], threshold: float = 5.0) -> Callable[[float], str]:
    """
    Pie-chart label formatter showing percentage and count, e.g. '42.0%\n(21)'.

    Args:
        values (list[int]): Slice values.
        threshold (float, optional): Slices at or below this percentage get no label. Defaults to 5.0.

    Returns:
        Callable[[float], str]: Formatter for matplotlib's ``autopct``.
    """
    def my_autopct(pct: float) -> str:
        total = sum(values)
        count = int(round(pct * total / 100.0))
        return f'{pct:.1f}%\n({count})' if pct > threshold else ''
    return my_autopct



def draw_fail_donut_subplots(fail_data_dict: dict, save_path: str) -> None:
    """
    Draw one donut chart of failure-type distribution per task and save the figure.

    Args:
        fail_data_dict (dict): ``{task name: list of failure types}``.
        save_path (str): Output image path.
    """
    keys = list(fail_data_dict.keys())
    num_plots = len(keys)

    if num_plots == 0:
        return

    # Calculate subplot layout (maximum of 4 columns)
    ncols = min(4, num_plots)
    nrows = math.ceil(num_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 7 * nrows))
    axes = axes.flatten() if num_plots > 1 else [axes]
    cmap = plt.get_cmap('tab10')
    
    for idx, key in enumerate(keys):
        failed_cases = fail_data_dict[key]
        fail_summary = Counter(failed_cases)
        labels = list(fail_summary.keys())
        sizes = list(fail_summary.values())
        sorted_items = sorted(zip(labels, sizes), key=lambda x: x[1], reverse=True)
        labels, sizes = zip(*sorted_items)  
        total = sum(sizes)
        percentages = [s / total * 100 for s in sizes]
        colors = cmap.colors[:len(labels)]

        ax = axes[idx]
        pct_str = autopct_format(sizes, 4.0)
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            autopct=pct_str,
            pctdistance=0.7,
            startangle=90,
            counterclock=False,
            colors=colors,
            wedgeprops=dict(width=0.7)
        )

        legend_labels = [f"{label} ({pct:.1f}%, {size})" if pct < 4.0 else label for label, size, pct in zip(labels, sizes, percentages)]
        ax.legend(
            wedges,
            legend_labels,
            title="Failure Types",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=2,
            fontsize=9
        )
        ax.set_title(f'"{key}"', fontsize=12, pad=10)
        ax.axis('equal')

    # Clear remaining empty subplots
    for idx in range(len(keys), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Failure Type Distribution by Task', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
