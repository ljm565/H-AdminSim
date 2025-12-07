import os
import json
import orjson
from typing import Any
from pathlib import Path

from h_adminsim.utils import log



def txt_load(path: str) -> str:
    """
    Load and return the content of a text file.

    Args:
        path (str): Path to the text file.

    Returns:
        str: The full content of the text file as a string.
    """
    with open(path, 'r') as f:
        content = f.read()
    return content    



def json_load(path: str) -> Any:
    """
    Load and parse a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Any: The parsed Python object (usually a dict or list) from the JSON file.
    """
    with open(path, 'r') as f:
        return json.load(f)



def json_save(path: str, data: dict) -> None:
    """
    Save json file.

    Args:
        path (str): Path to the json file.
        data (dict): Data to save.
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)



def json_save_fast(path: str, data: dict) -> None:
    """
    Save json file more faster.

    Args:
        path (str): Path to the json file.
        data (dict): Data to save.
    """
    with open(path, 'wb') as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))



def yaml_save(file: str='data.yaml', data: Any = None) -> None:
    """
    Save data to an YAML file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (Any, optional): Data to save in YAML format.
    """
    save_path = Path(file)
    log(data.dumps())
    with open(save_path, "w") as f:
        f.write(data.dumps(modified_color=None, quote_str=True))
        log(f"Config is saved at {save_path}")



def get_files(path: str, ext: str = None) -> list[str]:
    """
    Get all files in a directory with a specific extension.

    Args:
        path (str): Folder path to search for files.
        ext (str, optional): Extension that you want to filter. Defaults to None.

    Raises:
        ValueError: If file does not exist.

    Returns:
        list[str]: List of file paths that match the given extension.
    """
    if not os.path.isdir(path):
        raise ValueError(f"Path {path} is not a directory.")
    
    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if ext is None or filename.endswith(ext):
                files.append(os.path.join(root, filename))
    
    return files



def make_project_dir(config) -> Path:
    """
    Make project folder.

    Args:
        config: yaml config.

    Returns:
        (path): project folder path.
    """
    prefix = log('Make project folder')
    project = config.project
    name = config.data_name

    save_dir = os.path.join(project, name)
    if os.path.exists(save_dir):
        log(f'{prefix}: Project {save_dir} already exists. New folder will be created.')
        name = name + str(len(os.listdir(project))+1)
        config.data_name = name
        save_dir = os.path.join(project, name)
    
    os.makedirs(project, exist_ok=True)
    os.makedirs(save_dir)
    
    return Path(save_dir)
