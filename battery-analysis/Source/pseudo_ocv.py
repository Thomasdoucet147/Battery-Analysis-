
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import savgol_filter

from numpy import loadtxt
from matplotlib import gridspec
from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # ajoute Source au path
from utiles import extraire_points_contigus, clean_and_enforce_spacing



def filtrage_OCV(Voltage, Capacity, time=None, step=2e-3, apply_smoothing=False,
                  window_length=11, polyorder=3):
    """
    Filtrage d'une courbe OCV pour calcul d'IC/dQdV selon un pas de tension minimal.

    Référence de principe : filtrage tous les 2 mV pour IC analysis,
    avec lissage optionnel pour réduire le bruit tout en conservant la position et intensité des pics.

    Paramètres :
    ------------
    Voltage : array-like
        Tension (V)
    Capacity : array-like
        Capacité correspondante (Ah ou mAh)
    time : array-like, optionnel
        Temps correspondant (s)
    step : float, optionnel
        Pas minimal de tension pour conserver les points (V), défaut 0.002 V (2 mV)
    apply_smoothing : bool, optionnel
        Appliquer un lissage Savitzky-Golay sur Capacity avant dérivée
    window_length : int, optionnel
        Longueur de la fenêtre pour le filtre Savitzky-Golay (impair)
    polyorder : int, optionnel
        Ordre du polynôme pour Savitzky-Golay

    Returns :
    ---------
    Voltage_filt : np.array
        Tension filtrée
    Capacity_filt : np.array
        Capacité filtrée et éventuellement lissée
    time_filt : np.array ou None
        Temps filtré si fourni, sinon None
    """

    Voltage = np.array(Voltage)
    Capacity = np.array(Capacity)
    if time is not None:
        time = np.array(time)

    # Création du masque pour filtrage selon pas de tension
    mask = [True]  # garder le premier point
    last_V = Voltage[0]

    for v in Voltage[1:]:
        if abs(v - last_V) >= step:
            mask.append(True)
            last_V = v
        else:
            mask.append(False)

    mask = np.array(mask)

    Voltage_filt = Voltage[mask]
    Capacity_filt = Capacity[mask]
    time_filt = time[mask] if time is not None else None

    # Lissage optionnel de la capacité pour réduire le bruit avant dérivée
    if apply_smoothing:
        if window_length >= len(Capacity_filt):
            # ajustement automatique si fenêtre trop grande
            window_length = len(Capacity_filt) - (len(Capacity_filt) + 1) % 2  # impair
        Capacity_filt = savgol_filter(Capacity_filt, window_length=window_length, polyorder=polyorder)
        
    # Lissage de la tension
        Voltage_filt = savgol_filter(Voltage_filt, window_length=window_length, polyorder=polyorder)
    return Voltage_filt, Capacity_filt, time_filt

def fit_and_plot_dQdV(V, dQdV, label, color, smooth_factor=1e-2, step=0.01, bin_width=0.04):
    """
    This function enables to compute the DVA/ICA
    Dubarry M and Anseán D (2022) Best practices for incremental capacity analysis.
    Front. Energy Res. 10:1023555. doi: 10.3389/fenrg.2022.1023555

    Inputs:
        V: voltage values
        dQdV: incremental capacity values
        label: label for the plot
        color: color for the plot
        smooth_factor: smoothing factor for the spline
        step: step size for the voltage range
        bin_width: width of the bins for averaging
    """

    try:
        # Tracé direct des données
        plt.plot(V, dQdV, '-', label=label, color=color)
        plt.xlabel("Voltage (V)", fontsize=14)
        plt.ylabel("dQdV (Ah/V)", fontsize=14)
        plt.title("ICA Analysis", fontsize=16)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

    except Exception as e:
        print(f"Erreur lors du tracé pour {label}: {e}")

    def fit_and_plot_dVdQ(Q, dVdQ, label, color):
        """
        This function enables to compute the DVA
        Dubarry M and Anseán D (2022) Best practices for incremental capacity analysis.
        Front. Energy Res. 10:1023555. doi: 10.3389/fenrg.2022.1023555
    
        Inputs:
            V: voltage values
            dQdV: incremental capacity values
            label: label for the plot
            color: color for the plot
            smooth_factor: smoothing factor for the spline
            step: step size for the voltage range
            bin_width: width of the bins for averaging
        """
    
        try:
            # Tracé direct des données
            plt.plot(Q, dVdQ, '-', label=label, color=color)
            plt.xlabel("Capacity (Ah)", fontsize=14)
            plt.ylabel("dVdQ (V/Ah)", fontsize=14)
            plt.title("DVA Analysis", fontsize=16)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
    
        except Exception as e:
            print(f"Erreur lors du tracé pour {label}: {e}")


def detecter_phases_repos(data_concat, courant_seuil_pos, courant_seuil_neg, duree_min_repos):
    """Détecte les phases de repos positives et négatives pour le pseudo-OCV."""
    times = np.array(data_concat["Test_time"])
    currents = np.array(data_concat["Current"])
    voltages = np.array(data_concat["Voltage"])
    caps_dis = np.array(data_concat["Discharge_Capacity"])
    caps_cha = np.array(data_concat["Charge_Capacity"])

    # Détection des phases de repos
    is_rest_pos = (courant_seuil_pos - 0.02 < currents) & (currents <= courant_seuil_pos + 0.02)
    is_rest_neg = (-courant_seuil_neg - 0.02 < currents) & (currents <= -courant_seuil_neg + 0.02)

    pseudo_ocv_time_pos, pseudo_ocv_voltage_pos, pseudo_ocv_capacity_cha = extraire_points_contigus(
        is_rest_pos, times, voltages, caps_cha, duree_min_repos
    )

    pseudo_ocv_time_neg, pseudo_ocv_voltage_neg, pseudo_ocv_capacity_dis = extraire_points_contigus(
        is_rest_neg, times, voltages, caps_dis, duree_min_repos
    )

    # Remise à zéro des temps et capacités
    pseudo_ocv_time_pos = [x - pseudo_ocv_time_pos[0] for x in pseudo_ocv_time_pos]
    pseudo_ocv_capacity_cha = [x - pseudo_ocv_capacity_cha[0] for x in pseudo_ocv_capacity_cha]
    pseudo_ocv_time_neg = [x - pseudo_ocv_time_neg[0] for x in pseudo_ocv_time_neg]
    pseudo_ocv_capacity_dis = [x - pseudo_ocv_capacity_dis[0] for x in pseudo_ocv_capacity_dis]

    return (pseudo_ocv_time_pos, pseudo_ocv_voltage_pos, pseudo_ocv_capacity_cha,
            pseudo_ocv_time_neg, pseudo_ocv_voltage_neg, pseudo_ocv_capacity_dis)