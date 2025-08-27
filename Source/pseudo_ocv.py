
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import savgol_filter
import pandas as pd
import importlib
from numpy import loadtxt
from matplotlib import gridspec
from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from Source.utiles import extraire_points_contigus, clean_and_enforce_spacing

def fit_and_plot_dQdV(V, dQdV, label, color, smooth_factor=1e-2, step=0.01, bin_width=0.04):
    if len(V) > 3:
        # Tri des données par ordre croissant de V
        sorted_indices = np.argsort(V)
        V_sorted = V[sorted_indices]
        dQdV_sorted = dQdV[sorted_indices]

        # Moyenne par bins réguliers
        bins = np.arange(V_sorted.min(), V_sorted.max() + bin_width, bin_width)
        V_bin_centers = []
        dQdV_avg = []
        for i in range(len(bins) - 1):
            mask = (V_sorted >= bins[i]) & (V_sorted < bins[i+1])
            if np.any(mask):
                V_bin_centers.append((bins[i] + bins[i+1]) / 2)
                dQdV_avg.append(np.mean(dQdV_sorted[mask]))
        V_bin_centers = np.array(V_bin_centers)
        dQdV_avg = np.array(dQdV_avg)

        if len(V_bin_centers) > 3:
            # Nettoyage stricte et espacement minimal
            V_clean, dQdV_clean = clean_and_enforce_spacing(V_bin_centers, dQdV_avg)

            print(f"V_clean: {V_clean}")
            print(f"diff V_clean: {np.diff(V_clean)}")

            # Plage régulière pour tracer la spline ajustée
            V_range = np.arange(V_clean.min(), V_clean.max(), step)

            try:
                # Ajustement spline régularisée sur données nettoyées
                spline = UnivariateSpline(V_clean, dQdV_clean, s=smooth_factor)
                dQdV_fit = spline(V_range)

                # Tracés
                plt.plot(V_range, dQdV_fit, '-', label=f'{label}', color=color)
                plt.xlabel("Voltage (V)", fontsize=14)
                plt.ylabel("dQdV (Ah/V)", fontsize=14)
                plt.title("ICA Analysis", fontsize=16)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                #plt.tight_layout()
                #plt.show()

            except Exception as e:
                print(f"Erreur lors du fit spline pour {label}: {e}")
        else:
            print(f"Pas assez de points après moyennage pour l'ajustement spline ({label}).")
    else:
        print(f"Pas assez de points pour l'ajustement spline ({label}).")

        
def fit_and_plot_dVdQ(V, dQdV, label, color, smooth_factor=1e-2, step=0.01, bin_width=0.04):
    if len(V) > 3:
        # Tri des données par ordre croissant de V
        sorted_indices = np.argsort(V)
        V_sorted = V[sorted_indices]
        dQdV_sorted = dQdV[sorted_indices]

        # Moyenne par bins réguliers
        bins = np.arange(V_sorted.min(), V_sorted.max() + bin_width, bin_width)
        V_bin_centers = []
        dQdV_avg = []
        for i in range(len(bins) - 1):
            mask = (V_sorted >= bins[i]) & (V_sorted < bins[i+1])
            if np.any(mask):
                V_bin_centers.append((bins[i] + bins[i+1]) / 2)
                dQdV_avg.append(np.mean(dQdV_sorted[mask]))
        V_bin_centers = np.array(V_bin_centers)
        dQdV_avg = np.array(dQdV_avg)

        if len(V_bin_centers) > 3:
            # Nettoyage stricte et espacement minimal
            V_clean, dQdV_clean = clean_and_enforce_spacing(V_bin_centers, dQdV_avg)

            print(f"V_clean: {V_clean}")
            print(f"diff V_clean: {np.diff(V_clean)}")

            # Plage régulière pour tracer la spline ajustée
            V_range = np.arange(V_clean.min(), V_clean.max(), step)

            try:
                # Ajustement spline régularisée sur données nettoyées
                spline = UnivariateSpline(V_clean, dQdV_clean, s=smooth_factor)
                dQdV_fit = spline(V_range)

                # Tracés
                plt.plot(V_range, dQdV_fit, '-', label=f'{label}', color=color)
                plt.xlabel("Capacity (Ah)", fontsize=14)
                plt.ylabel("dQdV (V/Ah)", fontsize=14)
                plt.title("DVA Analysis", fontsize=16)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                #plt.tight_layout()
                #plt.show()

            except Exception as e:
                print(f"Erreur lors du fit spline pour {label}: {e}")
        else:
            print(f"Pas assez de points après moyennage pour l'ajustement spline ({label}).")
    else:
        print(f"Pas assez de points pour l'ajustement spline ({label}).")

        
def detecter_phases_repos(data_concat, courant_seuil_pos, courant_seuil_neg, duree_min_repos):
    """Détecte les phases de repos positives et négatives pour le pseudo-OCV."""
    times = np.array(data_concat["Test_time"])
    currents = np.array(data_concat["Current"])
    voltages = np.array(data_concat["Voltage"])
    caps_dis = np.array(data_concat["Discharge_Capacity"])
    caps_cha = np.array(data_concat["Charge_Capacity"])

    # Détection des phases de repos
    is_rest_pos = (courant_seuil_pos - 0.01 < currents) & (currents <= courant_seuil_pos + 0.01)
    is_rest_neg = (-courant_seuil_neg - 0.01 < currents) & (currents <= -courant_seuil_neg + 0.01)

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