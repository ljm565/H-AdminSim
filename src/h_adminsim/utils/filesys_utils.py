import os
import json
import orjson
from pathlib import Path
from importlib import resources
from typing import TYPE_CHECKING, Any, Optional

from h_adminsim.utils import log

if TYPE_CHECKING:
    from sconf import Config



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



def bundled_prompt(name: str) -> str:
    """
    Path of a prompt file shipped with H-AdminSim, e.g. for a policy's `Decision.negotiate(prompt_path)`.

    Args:
        name (str): File name under ``h_adminsim/assets/prompts`` (e.g. ``staff_negotiation_policy_common_system.txt``).

    Returns:
        str: The prompt's path.
    """
    return str(resources.files('h_adminsim.assets.prompts').joinpath(name))



def json_load(path: str) -> Any:
    """
    Load and parse a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Any: The parsed object (usually a dict or list).
    """
    with open(path, 'r') as f:
        return json.load(f)



def json_save(path: str, data: dict) -> None:
    """
    Save data as an indented JSON file.

    Args:
        path (str): Path to the json file.
        data (dict): Data to save.
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)



def json_save_fast(path: str, data: dict) -> None:
    """
    Save data as an indented JSON file using ``orjson`` (faster; infinities become null).

    Args:
        path (str): Path to the json file.
        data (dict): Data to save.
    """
    with open(path, 'wb') as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))



def yaml_save(file: str='data.yaml', data: Any = None) -> None:
    """
    Save a config object (anything with ``dumps()``, e.g. ``sconf.Config``) to a YAML file.

    Args:
        file (str, optional): Output file path. Defaults to 'data.yaml'.
        data (Any, optional): Config object to save. Defaults to None.
    """
    save_path = Path(file)
    log(data.dumps())
    with open(save_path, "w") as f:
        f.write(data.dumps(modified_color=None, quote_str=True))
        log(f"Config is saved at {save_path}")



def get_files(path: str, ext: Optional[str] = None) -> list[str]:
    """
    Recursively list the files under a directory, optionally filtered by filename suffix.

    Args:
        path (str): Directory to search.
        ext (Optional[str], optional): Filename suffix to keep; None keeps every file. Defaults to None.

    Returns:
        list[str]: Matching file paths.

    Raises:
        ValueError: If ``path`` is not a directory.
    """
    if not os.path.isdir(path):
        raise ValueError(f"Path {path} is not a directory.")
    
    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if ext is None or filename.endswith(ext):
                files.append(os.path.join(root, filename))
    
    return files



def make_project_dir(config: 'Config') -> Path:
    """
    Create ``<project>/<data_name>``; if it exists, a numbered name is used and written back to ``config.data_name``.

    Args:
        config (Config): Config with ``project`` and ``data_name``.

    Returns:
        Path: The created folder.
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
