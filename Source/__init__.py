# __init__.py
PACKAGE_NAME = "Analyse batterie"
DEFAULT_PATH = r"C:\Users\tdoucet\\Desktop\\Thèse SSB\\Expériences\\Analyse batterie"

from Source.utiles import extraire_points_contigus, clean_and_enforce_spacing
from Source.pseudo_ocv import detecter_phases_repos