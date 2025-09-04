# %% Chargement de données + graphique d'EIS 
import matplotlib.pyplot as plt 
import numpy as np
from numpy import log10
from Source.preprocessing import load_eis_file, smooth_eis_data, sort_eis_data, save_sorted_eis
from Source.utiles import lire_EIS 
from Source.tracage import plot_eis_nyquist, plot_drt
from Source.eis_drt import simple_run, EIS_object, peak_analysis
from Source.modelisation import C_diffusion_SOC, tau_moyen_direct
import seaborn as sns 
EIS_FILE = "C:\\Users\\tdoucet\\Desktop\\Thèse SSB\\Expériences\\Analyse batterie\\Données\\Welion22Ah\\Essais preliminaires\\Brut\\50mA_EIS_Essais_pré-rodage.txt"
OUTPUT_FILE = "C:\\Users\\tdoucet\\Desktop\\Thèse SSB\\Expériences\\Analyse batterie\\Données\\Welion22Ah\\Essais preliminaires\\Traité\\EIS_post_rodage_sorted.txt"

# Lecture
df_eis = load_eis_file(EIS_FILE)

# Filtrage
df_eis_smooth = smooth_eis_data(df_eis)

# Tri
df_eis_sorted = sort_eis_data(df_eis_smooth)

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



# Exemple : récupérer les fréquences depuis ton DataFrame
freq = df_EIS["frequency"].to_numpy()       # vecteur des fréquences
Z_re = df_EIS["Re"].to_numpy()             # spectre EIS complexe (ou séparé en Re/Im)
Z_Im= df_EIS["Im"].to_numpy()             # spectre EIS complexe (ou séparé en Re/Im)


entry = EIS_object(freq, Z_re, Z_Im)



# Construire systèmes
# Exécuter simple_run
entry = simple_run(
    entry,
    rbf_type='Gaussian',
    data_used='Combined Re-Im Data',
    induct_used=1,
    der_used='1st order',
    cv_type='GCV',
    reg_param=1E-3,
    shape_control='FWHM Coefficient',
    coeff=0.5
)

# Extraire et tracer la DRT
gamma = entry.gamma
tau_fine = entry.tau_fine

entry2 = peak_analysis(entry, rbf_type='Gaussian', data_used='Combined Re-Im Data', induct_used=1, der_used='1st order', cv_type='GCV', reg_param=1E-3, shape_control='FWHM Coefficient', coeff=0.5, peak_method='separate', N_peaks=7)
Gaussian = entry2.Gaussian
gamma_fit = entry2.out_gamma_fit
fig, ax = plt.subplots(figsize=(7,4))

plot_drt(tau_fine, gamma, title="DRT Im", ax = None)
# for i in range(0,len(gamma_fit)):
#     plot_drt(tau_fine, gamma_fit[i], title="Superposition des pics" + str(i), ax =ax, color = sns.color_palette("deep")[i], label = "Pic " + str(i))
# ax.legend()
plt.show()



R_ohmic = np.trapz(gamma_fit[1], tau_fine)
print(R_ohmic)
R_SEI = np.trapz(gamma_fit[0]+gamma_fit[6], tau_fine)
C_SEI = tau_moyen_direct(tau_fine, gamma_fit[0]+gamma_fit[6])/R_SEI
print(R_SEI)

RT_tc1,RT_tc2 , RT_tc3 =  np.trapz(gamma_fit[3], tau_fine), np.trapz(gamma_fit[4], tau_fine), np.trapz(gamma_fit[2], tau_fine)


C_tc1,C_tc2 , C_tc3 = tau_moyen_direct(tau_fine, gamma_fit[3])/RT_tc1, tau_moyen_direct(tau_fine, gamma_fit[4])/RT_tc2, tau_moyen_direct(tau_fine, gamma_fit[2])/RT_tc3


R_Diffusion = np.trapz(gamma_fit[5], tau_fine)

# Affichage
print("Résistances (Ω) :")
print(f"R_ohmic     = {R_ohmic:.5f}")
print(f"R_SEI       = {R_SEI:.5f}")
print(f"R_tc1       = {RT_tc1:.5f}")
print(f"R_tc2       = {RT_tc2:.5f}")
print(f"R_tc3       = {RT_tc3:.5f}")
print(f"R_Diffusion = {R_Diffusion:.5f}\n")

print("Capacités (F) :")
print(f"C_SEI  = {C_SEI:.5e}")
print(f"C_tc1  = {C_tc1:.5e}")
print(f"C_tc2  = {C_tc2:.5e}")
print(f"C_tc3  = {C_tc3:.5e}")

#plot_drt(tau_fine, Gaussian, title="Gaussian")
#CorrigeDRT, afficher erreurs RMS, mettre le même nombre de points par décade ? 

# %% lecture test OCV 
import matplotlib.pyplot as plt
import numpy as np
from Source.utiles import concatener_donnees, lire_TEST_BT2020_avec_T
from Source.pseudo_ocv import detecter_phases_repos, filtrage_OCV, fit_and_plot_dQdV
from Source.tracage import plot_voltage_and_current, plot_Voltagevstime
import os


# Paramètres

courant_seuil_pos = 1.10 # seuil (en A) pour considérer que le courant est nul
courant_seuil_neg = 1.10
duree_min_repos = 30


# Dossier contenant les fichiers
dossier = r"C:\\Users\\tdoucet\\Desktop\\Thèse SSB\\Expériences\\Analyse batterie\\Données\\Welion22Ah\\Essais preliminaires\\Brut"

# Liste des fichiers à lire
#fichiers_OCV = [f"Carac_OCV_{i}.txt" for i in range(1,20)]
fichiers_OCV = [f"Carac_OCV_1.txt"]
# Dictionnaire final
donnees_OCV = {}

for i, fichier in enumerate(fichiers_OCV, start=1):
    chemin = os.path.join(dossier, fichier)
    df = lire_TEST_BT2020_avec_T(chemin)
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
# plot_voltage_and_current(data_concat["Test_time"],
#                          data_concat["Voltage"],
#                          data_concat["Current"],
#                          time_slice=None,
#                          voltage_label="Voltage (V)",
#                          current_label="Current (A)",
#                              xlabel="Time (s)",
#                              title=None,
#                              figsize=(12, 8),
#                              voltage_color='red',
#                              current_color='blue',
#                              voltage_legend="Voltage (V)",
#                              current_legend="Current (A)")
    
# data_concat = concatener_donnees(donnees_OCV)

# Extraction phases de repos
pseudo_ocv = detecter_phases_repos(data_concat, courant_seuil_pos, courant_seuil_neg, duree_min_repos)
(pseudo_ocv_time_pos, pseudo_ocv_voltage_pos, pseudo_ocv_capacity_cha,
 pseudo_ocv_time_neg, pseudo_ocv_voltage_neg, pseudo_ocv_capacity_dis) = pseudo_ocv



Voltage_filt_pos, Capacity_filt_pos, time_filt_pos = filtrage_OCV(pseudo_ocv_voltage_pos, pseudo_ocv_capacity_cha, pseudo_ocv_time_pos, step=2e-3, apply_smoothing=True)


Soc, C_diff= C_diffusion_SOC(Capacity_filt_pos, Voltage_filt_pos)
C_diff = np.array(C_diff)
print(f"C_DIFF = ", C_diff)




plot_Voltagevstime(Soc,
                   C_diff,
                    time_slice=None,
                    voltage_label="C_diff ",
                    xlabel="SoC (s)",
                    title=None,
                    figsize=(12, 8),
                   voltage_color='blue',
                   voltage_legend="C_diff curve",
                   ax=None)
plt.show()
# fig, ax = plt.subplots(figsize=(12, 8))

# plot_Voltagevstime(pseudo_ocv_time_neg,
#                    pseudo_ocv_voltage_neg,
#                     time_slice=None,
#                     voltage_label="Voltage (V)",
#                     xlabel="Time (s)",
#                     title=None,
#                     figsize=(12, 8),
#                    voltage_color='blue',
#                    voltage_legend="Discharge OCV",
#                    ax=ax)

# plot_Voltagevstime(pseudo_ocv_time_pos,
#                    pseudo_ocv_voltage_pos,
#                         time_slice=None,
#                         voltage_label="Voltage (V)",
#                         xlabel="Time (s)",
#                         title=None,
#                         figsize=(12, 8),
#                         voltage_color='red',
#                         voltage_legend="Charge OCV",
#                         ax=ax)
# ax.legend()
# plt.tight_layout()
# plt.show()

# dQdV_pos = np.gradient(pseudo_ocv_capacity_cha, pseudo_ocv_voltage_pos)
# dQdV_neg = np.gradient(pseudo_ocv_capacity_dis, pseudo_ocv_voltage_neg)

# Voltage_filt_pos, Capacity_filt_pos, time_filt_pos = filtrage_OCV(pseudo_ocv_voltage_pos, pseudo_ocv_capacity_cha, pseudo_ocv_time_pos, step=2e-3, apply_smoothing=True)
# Voltage_filt_neg, Capacity_filt_neg, time_filt_neg = filtrage_OCV(pseudo_ocv_voltage_neg, pseudo_ocv_capacity_dis, pseudo_ocv_time_neg, step=2e-3, apply_smoothing=True)

# dQdV_pos_post_filtre_smooth = np.gradient(Capacity_filt_pos, Voltage_filt_pos)
# dQdV_neg_post_filtre_smooth = np.gradient(Capacity_filt_neg, Voltage_filt_neg)

# dVdQ_pos_post_filtre_smooth = np.gradient(Voltage_filt_pos, Voltage_filt_pos)
# dQdV_neg_post_filtre_smooth = np.gradient(Capacity_filt_neg, Voltage_filt_neg)


# Voltage_filt_pos, Capacity_filt_pos, time_filt_pos = filtrage_OCV(pseudo_ocv_voltage_pos, pseudo_ocv_capacity_cha, pseudo_ocv_time_pos, step=2e-3, apply_smoothing=False)
# Voltage_filt_neg, Capacity_filt_neg, time_filt_neg = filtrage_OCV(pseudo_ocv_voltage_neg, pseudo_ocv_capacity_dis, pseudo_ocv_time_neg, step=2e-3, apply_smoothing=False)

# dQdV_pos_post_filtre = np.gradient(Capacity_filt_pos, Voltage_filt_pos)
# dQdV_neg_post_filtre = np.gradient(Capacity_filt_neg, Voltage_filt_neg)

# # fig, ax = plt.subplots(figsize=(12, 8))

# # fit_and_plot_dQdV(pseudo_ocv_voltage_pos, dQdV_pos, label="Positive Phase", color="red")
# # fit_and_plot_dQdV(pseudo_ocv_voltage_neg, dQdV_neg, label="Negative Phase", color="blue")

# # fit_and_plot_dQdV(Voltage_filt_pos, dQdV_pos_post_filtre, label="Positive Phase filtered", color="black")
# # fit_and_plot_dQdV(Voltage_filt_neg, dQdV_neg_post_filtre, label="Negative Phase filtered", color="black")


# # ax.legend()


# fig, ax = plt.subplots(figsize=(12, 8))


# fit_and_plot_dQdV(Voltage_filt_pos, dQdV_pos_post_filtre_smooth, label="Positive Phase smooth", color="red")
# fit_and_plot_dQdV(Voltage_filt_neg, dQdV_neg_post_filtre_smooth, label="Negative Phase smooth", color="blue")

# ax.legend()
# plt.tight_layout()
# plt.show()

# # %% extraction capacité 

# from Source.utiles import concatener_donnees, lire_TEST_BT2020_avec_T
# from Source.capacite import detecter_phase_cv_decharge
# from Source.tracage import plot_voltage_and_current, plot_Voltagevstime
# import matplotlib.pyplot as plt
# import os


# # Paramètres

# courant_seuil_pos = 1.10 # seuil (en A) pour considérer que le courant est nul
# courant_seuil_neg = 1.10
# duree_min_repos = 30


# # Dossier contenant les fichiers
# dossier = r"C:\\Users\\tdoucet\\Desktop\\Thèse SSB\\Expériences\\Analyse batterie\\Données\\Welion22Ah\\Essais preliminaires\\Brut"

# # Liste des fichiers à lire
# #fichiers_OCV = [f"Carac_OCV_{i}.txt" for i in range(1,20)]
# fichiers_OCV = [f"Carac_OCV_1.txt"]
# # Dictionnaire final
# donnees_OCV = {}

# for i, fichier in enumerate(fichiers_OCV, start=1):
#     chemin = os.path.join(dossier, fichier)
#     df = lire_TEST_BT2020_avec_T(chemin)
#     print(df)
#     # Stockage dans le dictionnaire avec clé OCV_1, OCV_2, ...
#     cle = f"OCV_{i}"
#     donnees_OCV[cle] = {
#         "Test_time": df["Test_time"],
#         "Current": df["Current"],
#         "Voltage": df["Voltage"],
#         "Charge_Capacity": df["Charge_Capacity"],
#         "Discharge_Capacity": df["Discharge_Capacity"],
#         "dVdt": df["dVdt"]
#     }

# # Concaténation
# data_concat = concatener_donnees(donnees_OCV)
# plot_Voltagevstime(df["Test_time"],
#                    df["Discharge_Capacity"],
#                     time_slice=None,
#                     voltage_label="  Discharge Capacity (Ah)",
#                     xlabel="Time (s)",
#                     title=None,
#                     figsize=(12, 8),
#                    voltage_color='blue',
#                    voltage_legend=" Discharge Capacity (Ah)",
#                    ax=None)
# plt.show()
# # Détecter phase CV
# time_cv, courant_cv, tension_cv, discharge_capacity_cv = detecter_phase_cv_decharge(data_concat["Current"], data_concat["Voltage"], data_concat["Test_time"], data_concat["Discharge_Capacity"], seuil_dV=0.00005, seuil_V=2.772, seuil_I=1.05, fenetre=60)

# # Tracé
    
# #fig, ax = plt.subplots(figsize=(12, 8))

# plot_Voltagevstime(time_cv,
#                    tension_cv,
#                     time_slice=None,
#                     voltage_label="Charging current",
#                     xlabel="Time (s)",
#                     title=None,
#                     figsize=(12, 8),
#                    voltage_color='blue',
#                    voltage_legend="Charging current",
#                    ax=None,
#                    marker = 'o')



# plt.show()

# plot_Voltagevstime(time_cv,
#                    discharge_capacity_cv,
#                     time_slice=None,
#                     voltage_label="Discharge capacity (Ah)",
#                     xlabel="Time (s)",
#                     title=None,
#                     figsize=(12, 8),
#                    voltage_color='red',
#                    voltage_legend="Discharge capacity (Ah)",
#                    ax=None,
#                    marker = 'o')

# plt.show()


# # %%
