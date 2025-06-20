import os
import json
from typing import Any
from pathlib import Path

from utils import log



def txt_load(path: str):
    with open(path, 'r') as f:
        content = f.read()
    return content    



def json_load(path: str):
    with open(path, 'r') as f:
        return json.load(f)



def yaml_save(file:str='data.yaml', data:Any=None) -> None:
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



def get_files(path:str, ext:str=None) -> list[str]:
    """
    Get all files in a directory with a specific extension.

    Args:
        path (str): Folder path to search for files.
        ext (str, optional): Extension that you want to filter. Defaults to None.

    Raises:
        ValueError: _description_

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
        save_dir = os.path.join(project, name + str(len(os.listdir(project))+1))
    
    os.makedirs(project, exist_ok=True)
    os.makedirs(save_dir)
    
    return Path(save_dir)
