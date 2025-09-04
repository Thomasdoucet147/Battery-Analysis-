import pandas as pd
from scipy.signal import savgol_filter
from .utiles import lire_TEST_EIS   # ta fonction perso pour lire les fichiers


def load_eis_file(path: str) -> pd.DataFrame:
    """Lit un fichier EIS avec lire_TEST_EIS et renvoie un DataFrame brut."""
    freq, real, imag = lire_TEST_EIS(path)
    df = pd.DataFrame({
        "frequency": freq,
        "Re": real,
        "Im": imag,
    })
    return df


def smooth_eis_data(df: pd.DataFrame, window: int = 11, order: int = 2) -> pd.DataFrame:
    """Applique un lissage Savitzky-Golay si assez de points."""
    if len(df) >= window:
        df["Re"] = savgol_filter(df["Re"], window_length=window, polyorder=order)
        df["Im"] = savgol_filter(df["Im"], window_length=window, polyorder=order)
    return df


def sort_eis_data(df: pd.DataFrame) -> pd.DataFrame:
    """Trie les données par fréquence décroissante."""
    return df.sort_values(by="frequency", ascending=False).reset_index(drop=True)


def save_sorted_eis(df: pd.DataFrame, output_path: str) -> None:
    """Sauvegarde les données triées au format tabulé."""
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Fichier sauvegardé : {output_path}")