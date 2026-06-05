from importlib import resources
__version__ = resources.files("h_adminsim").joinpath("version.txt").read_text().strip()