from .admin_staff import *
from .supervisor import SupervisorAgent

from importlib import resources
__version__ = resources.files("h_adminsim").joinpath("version.txt").read_text().strip()