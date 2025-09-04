import numpy as np # installed with matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.ticker import AutoMinorLocator
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import pandas as pd
import importlib
from numpy import loadtxt

# pyDRTtools' modules
from scipy.signal import hilbert

import seaborn as sns   # for the color palette




# Fonction de lecture d'un fichier
def lire_TEST_BT2000_utf16(fichier):
    Test_time, Current, Voltage, Charge_Capacity, Discharge_Capacity, dVdt = [], [], [], [], [], []
    with open(fichier, "r", encoding="utf-16", errors="ignore") as f:# faire attention à l'encodage, changer en utf-8 si nécessaire
        for i, ligne in enumerate(f):
            if i == 0:
                continue  # Sauter l'en-tête
            colonnes = ligne.strip().split('\t')
            #print(colonnes)
            if len(colonnes) < 13:
                continue  # Sauter les lignes incomplètes
            try:
                Test_time.append(float(colonnes[1].replace(',', '.')))
                Current.append(float(colonnes[6].replace(',', '.')))
                Voltage.append(float(colonnes[7].replace(',', '.')))
                Charge_Capacity.append(float(colonnes[8].replace(',', '.')))
                Discharge_Capacity.append(float(colonnes[9].replace(',', '.')))
                dVdt.append(float(colonnes[12].replace(',', '.')))
            except ValueError:
                continue  # En cas de données non convertible
    df = pd.DataFrame({
        "Test_time": Test_time,
        "Current": Current,
        "Voltage": Voltage,
        "Charge_Capacity": Charge_Capacity,
        "Discharge_Capacity": Discharge_Capacity,
        "dVdt": dVdt
    })       
    return df


# Fonction de lecture d'un fichier
def lire_TEST_BT2000_utf8(fichier):
    Test_time, Current, Voltage, Charge_Capacity, Discharge_Capacity, dVdt = [], [], [], [], [], []
    with open(fichier, "r", encoding="utf-8", errors="ignore") as f:# faire attention à l'encodage, changer en utf-8 si nécessaire
        for i, ligne in enumerate(f):
            if i == 0:
                continue  # Sauter l'en-tête
            colonnes = ligne.strip().split('\t')
            #print(colonnes)
            if len(colonnes) < 13:
                continue  # Sauter les lignes incomplètes
            try:
                Test_time.append(float(colonnes[1].replace(',', '.')))
                Current.append(float(colonnes[6].replace(',', '.')))
                Voltage.append(float(colonnes[7].replace(',', '.')))
                Charge_Capacity.append(float(colonnes[8].replace(',', '.')))
                Discharge_Capacity.append(float(colonnes[9].replace(',', '.')))
                dVdt.append(float(colonnes[12].replace(',', '.')))
            except ValueError:
                continue  # En cas de données non convertible
        df = pd.DataFrame({
        "Test_time": Test_time,
        "Current": Current,
        "Voltage": Voltage,
        "Charge_Capacity": Charge_Capacity,
        "Discharge_Capacity": Discharge_Capacity,
        "dVdt": dVdt
    })       
    return df


# Fonction de lecture d'un fichier
def lire_TEST_BT2020(fichier):
    Test_time, Current, Voltage, Charge_Capacity, Discharge_Capacity, dVdt = [], [], [], [], [], []
    with open(fichier, "r", encoding="utf-16", errors="ignore") as f:# faire attention à l'encodage, changer en utf-8 si nécessaire
        for i, ligne in enumerate(f):
            if i == 0:
                continue  # Sauter l'en-tête
            colonnes = ligne.strip().split('\t')
            #print(colonnes)
            if len(colonnes) < 13:
                continue  # Sauter les lignes incomplètes
            try:
                Test_time.append(float(colonnes[4].replace(',', '.')))
                Current.append(float(colonnes[8].replace(',', '.')))
                Voltage.append(float(colonnes[9].replace(',', '.')))
                Charge_Capacity.append(float(colonnes[10].replace(',', '.')))
                Discharge_Capacity.append(float(colonnes[11].replace(',', '.')))
                dVdt.append(float(colonnes[12].replace(',', '.')))
            except ValueError:
                continue  # En cas de données non convertible
     
    df = pd.DataFrame({
        "Test_time": Test_time,
        "Current": Current,
        "Voltage": Voltage,
        "Charge_Capacity": Charge_Capacity,
        "Discharge_Capacity": Discharge_Capacity,
        "dVdt": dVdt
    })       
    return df

def lire_TEST_BT2020_avec_T(fichier):
    Test_time, Current, Voltage, Charge_Capacity, Discharge_Capacity, dVdt, T1, T2, T3, T4,T5, T6, T7, T8 = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    with open(fichier, "r", encoding="utf-8", errors="ignore") as f:# faire attention à l'encodage, changer en utf-8 si nécessaire
        for i, ligne in enumerate(f):
            if i == 0:
                continue  # Sauter l'en-tête
            colonnes = ligne.strip().split('\t')
            #print(colonnes)
            if len(colonnes) < 13:
                continue  # Sauter les lignes incomplètes
            try:
                Test_time.append(float(colonnes[4].replace(',', '.')))
                Current.append(float(colonnes[8].replace(',', '.')))
                Voltage.append(float(colonnes[9].replace(',', '.')))
                Charge_Capacity.append(float(colonnes[10].replace(',', '.')))
                Discharge_Capacity.append(float(colonnes[11].replace(',', '.')))
                dVdt.append(float(colonnes[12].replace(',', '.')))
                T1.append(float(colonnes[15].replace(',', '.')))
                T2.append(float(colonnes[16].replace(',', '.')))
                T3.append(float(colonnes[17].replace(',', '.')))
                T4.append(float(colonnes[18].replace(',', '.')))
                T5.append(float(colonnes[19].replace(',', '.')))
                T6.append(float(colonnes[20].replace(',', '.')))
                T7.append(float(colonnes[21].replace(',', '.')))
                T8.append(float(colonnes[22].replace(',', '.')))
            except ValueError:
                continue  # En cas de données non convertible
    df = pd.DataFrame({
        "Test_time": Test_time,
        "Current": Current,
        "Voltage": Voltage,
        "Charge_Capacity": Charge_Capacity,
        "Discharge_Capacity": Discharge_Capacity,
        "dVdt": dVdt,
        "T1": T1,
        "T2": T2,
        "T3": T3,
        "T4": T4,
        "T5": T5,
        "T6": T6,
        "T7": T7,
        "T8": T8
    })
    return df

def lire_TEST_EIS(fichier): # à modifier, ce code à été conçu pour la zahner mais cela le fichier arbin est peut-être agencé différemment 
    Freq, Real, Im = [], [], []
    with open(fichier, "r", encoding="utf-8", errors="ignore") as f:# faire attention à l'encodage, changer en utf-8 si nécessaire
        for i, ligne in enumerate(f):
            if i == 0:
                continue  # Sauter l'en-tête
            colonnes = ligne.strip().split('\t')
            #print(colonnes)
            if len(colonnes) < 3:
                continue  # Sauter les lignes incomplètes
            try:
                Freq.append(float(colonnes[0].replace(',', '.')))
                Real.append(float(colonnes[4].replace(',', '.')))
                Im.append(float(colonnes[5].replace(',', '.')))
  

                #print(Test_time)
            except ValueError:
                continue  # En cas de données non convertible
    return Freq, Real, Im



def lire_EIS(path: str) -> pd.DataFrame:
    """
    Lit un fichier EIS au format tabulé et renvoie un DataFrame.
    Re(Z) et -Im(Z) sont en mΩ.
    """
    freqs, re, im = [], [], []
    with open(path, "r") as f:
        for i, ligne in enumerate(f):
            if i == 0:
                continue  # Sauter l'en-tête
            colonnes = ligne.strip().split('\t')
            if len(colonnes) < 3:
                continue
            freqs.append(float(colonnes[0]))
            re.append(1000*float(colonnes[1]))   # Re(Z) en mΩ
            im.append(1000*float(colonnes[2]))  # Im(Z) en mΩ

    df = pd.DataFrame({
        "frequency": freqs,
        "Re": re,
        "Im": im
    })
    return df


    # Pour chaque phase de repos, on garde les séquences suffisamment longues
def extraire_points_contigus(is_rest, times, voltages, capacities, duree_min):
    result_time, result_voltage, result_capacity = [], [], []
    start = None

    for i in range(len(is_rest)):
        if is_rest[i]:
            if start == None:
                start = i
        else:
            if start is not None:
                # Fin d'une séquence de repos
                duration = times[i - 1] - times[start]
                if duration >= duree_min:
                    result_time.extend(times[start:i])
                    result_voltage.extend(voltages[start:i])
                    result_capacity.extend(capacities[start:i])
                start = None
        # Gérer le cas où le repos continue jusqu’à la fin
    if start is not None:
        duration = times[-1] - times[start]
        if duration >= duree_min:
            result_time.extend(times[start:])
            result_voltage.extend(voltages[start:])
            result_capacity.extend(capacities[start:])
    return result_time, result_voltage, result_capacity

def Energie(tension,courant,times):
    """
    Calcule l'énergie en multipliant le courant par la tension et la capacité.
    """
    powers = tension * courant  # en W

# Énergie en Joules (W·s)
    energie_J = np.trapz(powers, times)

# Conversion en Wh
    energie_Wh = energie_J / 3600
    return energie_Wh

    
def clean_and_enforce_spacing(x, y, min_spacing=1e-6):
    x = np.asarray(x)
    y = np.asarray(y)
    # Supprimer NaN et inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # Trier
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Supprimer doublons exacts
    unique_x, unique_indices = np.unique(x, return_index=True)
    x = unique_x
    y = y[unique_indices]

    # Forcer espacement minimal strict
    for i in range(1, len(x)):
        if x[i] - x[i-1] < min_spacing:
            x[i] = x[i-1] + min_spacing

    return x, y



def concatener_donnees(donnees):
    """Concatène les données de plusieurs tests en un dictionnaire unique."""
    keys = ["Test_time", "Current", "Voltage", "Charge_Capacity", "Discharge_Capacity", "dVdt"]
    data_concat = {key: [] for key in keys}
    for test_data in donnees.values():
        for key in keys:
            data_concat[key].extend(test_data[key])
    return data_concat