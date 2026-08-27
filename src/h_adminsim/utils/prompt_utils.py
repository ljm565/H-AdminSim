from importlib import resources



def load_prompt(file_name: str) -> str:
    """
    Load a packaged prompt template from ``h_adminsim.assets.prompts``.

    The trailing newline that editors append to the ``.txt`` files is stripped so a
    packaged template is byte-identical to the same text written as a Python literal.

    Args:
        file_name (str): File name of the prompt inside the packaged prompt directory.

    Returns:
        str: The prompt template text.
    """
    return resources.files("h_adminsim.assets.prompts").joinpath(file_name).read_text().rstrip('\n')
