# %% Chargement de données + graphique d'EIS 
import matplotlib.pyplot as plt 
import numpy as np
from Source.preprocessing import load_eis_file, smooth_eis_data, sort_eis_data, save_sorted_eis
from Source.utiles import lire_EIS 
from Source.tracage import plot_eis_nyquist, plot_drt
from Source.eis_drt import gamma

EIS_FILE = "C:\\Users\\tdoucet\\Desktop\\Thèse SSB\\Expériences\\Analyse batterie\\Données\\Essais preliminaires\\Essai_EIS_prerodage\\50mA_EIS_Essais_pré-rodage.txt"
OUTPUT_FILE = "C:\\Users\\tdoucet\\Desktop\\Thèse SSB\\Expériences\\Analyse batterie\\Données\\Essais preliminaires\\Essai_EIS_prerodage\\resultats_filtres\\EIS_post_rodage_sorted.txt"

# Lecture
df_eis = load_eis_file(EIS_FILE)

# Filtrage
#df_eis_smooth = smooth_eis_data(df_eis)

# Tri
df_eis_sorted = sort_eis_data(df_eis)

# Sauvegarde
save_sorted_eis(df_eis_sorted, OUTPUT_FILE)


df_EIS= lire_EIS(OUTPUT_FILE)

print(df_EIS)

# Conversion en dictionnaire compatible avec plot_eis_nyquist
donnees_EIS_filtrees = {
    "EIS": {"Re": df_EIS["Re"], "Im_neg": df_EIS["Im"]},
    # "SOC_50": {"Re": df_SOC_50["Re"], "Im_neg": df_SOC_50["Im"]},
    # "SOC_100": {"Re": df_SOC_100["Re"], "Im_neg": df_SOC_100["Im"]},
}

# Tracé
plot_eis_nyquist(donnees_EIS_filtrees)


# omega en rad/s, tau en s (échelle log espacée)
omega = 2*np.pi*np.array(df_EIS["frequency"])           # shape (Nω,)
              # shape (Nτ,)

# Tes données mesurées:
Z_re_meas = df_EIS["Re"]   # shape (Nω,)
Z_im_meas = -df_EIS["Im"]   # idem (attention au signe: Z = Z' + j Z'')
frequence = df_EIS["frequency"]

# Construire systèmes
gamma, tau, rms_error, rel_error, aire = gamma(Z_im_meas,frequence)

plot_drt(tau, gamma, title="DRT_Im")


#CorrigeDRT, afficher erreurs RMS, mettre le même nombre de points par décade ? 

# %% lecture test OCV 
from Source.utiles import lire_TEST_BT2000_utf16, concatener_donnees
from Source.pseudo_ocv import detecter_phases_repos
from Source.tracage import plot_voltage_and_current, plot_Voltagevstime
import os


# Paramètres

courant_seuil_pos = 1.11 # seuil (en A) pour considérer que le courant est nul
courant_seuil_neg = 1.09
duree_min_repos = 30


# Dossier contenant les fichiers
dossier = r"C:\\Users\tdoucet\\Desktop\\Thèse SSB\\Expériences\\Analyse batterie\\Données\\Essais preliminaires\\Essai_Elec"

# Liste des fichiers à lire
fichiers_OCV = [f"Cyclage_100_DoD_Carac_Initiale_{i}.txt" for i in range(1, 23)]

# Dictionnaire final
donnees_OCV = {}

for i, fichier in enumerate(fichiers_OCV, start=1):
    chemin = os.path.join(dossier, fichier)
    df = lire_TEST_BT2000_utf16(chemin)
    
    # Stockage dans le dictionnaire avec clé OCV_1, OCV_2, ...
    cle = f"OCV_{i}"
    donnees_OCV[cle] = {
        "Test_time": df["Test_time"],
        "Current": df["Current"],
        "Voltage": df["Voltage"],
        "Charge_Capacity": df["Charge_Capacity"],
        "Discharge_Capacity": df["Discharge_Capacity"],
        "dVdt": df["dVdt"]
    }

# Concaténation
data_concat = concatener_donnees(donnees_OCV)

# Tracé
plot_voltage_and_current(data_concat["Test_time"],
                         data_concat["Voltage"],
                         data_concat["Current"],
                         time_slice=None,
                         voltage_label="Voltage (V)",
                         current_label="Current (A)",
                             xlabel="Time (s)",
                             title=None,
                             figsize=(12, 8),
                             voltage_color='red',
                             current_color='blue',
                             voltage_legend="Voltage (V)",
                             current_legend="Current (A)")
    
data_concat = concatener_donnees(donnees_OCV)

# Extraction phases de repos
pseudo_ocv = detecter_phases_repos(data_concat, courant_seuil_pos, courant_seuil_neg, duree_min_repos)
(pseudo_ocv_time_pos, pseudo_ocv_voltage_pos, pseudo_ocv_capacity_cha,
 pseudo_ocv_time_neg, pseudo_ocv_voltage_neg, pseudo_ocv_capacity_dis) = pseudo_ocv


plot_Voltagevstime(pseudo_ocv_time_neg,
                   pseudo_ocv_voltage_neg,
                   time_slice=None,
                   voltage_label="Voltage (V)",
                   xlabel="Time (s)",
                   title=None,
                   figsize=(12, 8),
                   voltage_color='red',
                   voltage_legend="Voltage (V)")


