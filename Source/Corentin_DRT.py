import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
from sklearn.linear_model import Ridge
import matplotlib as mpl
from scipy.optimize import minimize
import itertools

# ============================================================
# === SCRIPT DRT & EIS : Analyse d'un spectre d'impédance ====
# ============================================================
# Ce script lit un fichier Excel contenant des données EIS (Electrochemical Impedance Spectroscopy).
# Il réalise une transformation DRT (Distribution of Relaxation Times).
#
# Principe :
# - L'impédance électrochimique complexe Z est mesurée en fonction de la fréquence.
# - La DRT est une représentation alternative : elle décompose la réponse en une distribution de
#   temps caractéristiques τ, montrant comment chaque processus (charge, transfert, diffusion) contribue.
#
# Résultat :
# - Tracer la courbe Nyquist (Z_re vs -Z_im)
# - Résoudre la DRT par régularisation
# - Reconstruire Z_im à partir de la DRT
# - Comparer l'ajustement et évaluer l'erreur
# - Visualiser la DRT et interpréter l'aire sous la courbe
# ============================================================

# ===================================================
# EXPLICATIONS SUR LA DRT (Distribution des Temps de Relaxation)
# ===================================================
#
# La DRT est une méthode qui permet de décomposer la réponse
# d'impédance complexe d'un système électrochimique (comme une batterie)
# en contributions élémentaires associées à différents mécanismes physiques,
# chacun caractérisé par un temps de relaxation τ (tau).
#
# Le signal mesuré (Z_im) est modélisé comme une somme pondérée d'effets
# à différentes échelles de temps, représentés par la distribution γ(τ).
#
# Mathématiquement, la DRT consiste à résoudre un problème inverse
# où l'on cherche γ(τ) telle que :
#
#    Z_im(ω) ≈ ∫ γ(τ) * Im{1/(1 + jωτ)} d(ln τ)
#
# où ω = 2πf.
#
# Ce problème est mal posé (inversion instable), on utilise donc
# une régularisation (alpha, beta) pour obtenir une solution stable et lisse.
#
# Analyse de la DRT :
# - Les pics dans γ(τ) indiquent des phénomènes distincts avec
#   des temps de relaxation caractéristiques (ex : transfert de charge,
#   diffusion dans l'électrolyte, etc.).
# - La position horizontale des pics (τ) donne l'échelle temporelle
# - L'amplitude des pics est liée à l'intensité du phénomène correspondant.
#
# Résistance série (R0) :
# - Dans un spectre EIS, la résistance en série R0 correspond à la
#   résistance ohmique pure du système (contacts, électrolyte, etc.).
# - Elle se manifeste par le point de départ du diagramme de Nyquist sur l'axe réel.
# - La DRT **ne modélise pas cette résistance R0**, car elle ne représente pas
#   un processus distribué dans le temps mais un simple décalage.
# - Il faut donc toujours considérer R0 séparément lorsque l'on interprète la DRT.
#
# ---------------------------------------------------


# === Réglages d'affichage pour matplotlib ===
mpl.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.bottom'] = True

# === Paramètres fichier ===
# Le fichier doit contenir : fréquence [Hz], Re(Z) [Ohm], Im(Z) [Ohm] en colonnes
file_path = r"C:\Users\tdoucet\Desktop\Thèse SSB\Expériences\Analyse batterie\Données\Essais preliminaires\Essai_EIS_prerodage\resultats_filtres\EIS_post_rodage_sorted.txt"
file_path = file_path.replace("\\", "/")

# === Plage de fréquences et calcul des bornes de tau ===
# tau = 1/(2*pi*f) => tau_min ~ haute fréquence, tau_max ~ basse fréquence
f_min = 0.01  # Hz
f_max = 10000  # Hz

# Définition des bornes de tau :
# On étend artificiellement la plage de temps de relaxation au-delà des fréquences mesurées :
#   tau_max = 10 fois plus grand que 1/(2π f_min)  -> couvre les processus plus lents que la plus basse fréquence.
#   tau_min = 0.1 fois plus petit que 1/(2π f_max) -> couvre les processus plus rapides que la plus haute fréquence.
# Cette marge (~×10) limite les artefacts de bord et stabilise l'inversion DRT (voir Ciucci et al., Electrochimica Acta 2015).
tau_max = 10 / (2 * np.pi * f_min)
tau_min = 0.1 / (2 * np.pi * f_max)

# ------------------------------------------------------------
# === Définition de la matrice noyau K =======================
# K relie la DRT gamma(tau) à Z_im mesuré :
# Z_im(f) = ∫ K(f, tau) * gamma(tau) dln(tau)
# ------------------------------------------------------------
def kernel_matrix(freq, tau):
    omega = 2 * np.pi * freq
    K = 1 / (1 + 1j * np.outer(omega, tau))
    return K
    
# === Paramètres de régularisation ===
alpha = 1e-3  # pénalisation L2 de la magnitude
beta = 1e-3  # pénalisation L2 de la pente (lissage)

# ------------------------------------------------------------
# === Résolution de la DRT ===================================
# On résout un problème inverse régularisé :
#   min ||K.gamma - Z_im||^2 + alpha*||gamma||^2 + beta*||D.gamma||^2
# La contrainte gamma >= 0 garantit la positivité physique.
# ------------------------------------------------------------
def solve_drt(freq, Z_im, tau, alpha=alpha, beta=beta):
    K = np.imag(kernel_matrix(freq, tau))
    n = len(tau)

    # Matrice de différences finies pour régulariser la pente
    D = np.zeros((n - 1, n))
    for i in range(n - 1):
        D[i, i] = -1
        D[i, i + 1] = 1

    # Initialisation : régression Ridge (pas contrainte)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(K, Z_im)
    x0 = np.clip(model.coef_, 0, None)  # impose positifs

    # Fonction objectif : erreur + régularisations
    def objective(gamma):
        res = K @ gamma - Z_im
        smooth = D @ gamma
        return np.sum(res**2) + alpha * np.sum(gamma**2) + beta * np.sum(smooth**2)

    # Gradient analytique pour accélérer L-BFGS-B
    def gradient(gamma):
        res = K @ gamma - Z_im
        smooth = D @ gamma
        return 2 * (K.T @ res) + 2 * alpha * gamma + 2 * D.T @ smooth * beta

    bounds = [(0, None) for _ in range(n)]  # bornes : gamma ≥ 0

    # Minimisation non linéaire
    res = minimize(objective, x0, jac=gradient, bounds=bounds, method='L-BFGS-B',
                   options={'maxfun': 10000, 'maxiter': 10000})

    if not res.success:
        print("Optimisation échouée :", res.message)

    return np.clip(res.x, 0, None)  # toujours positif

# ------------------------------------------------------------
# === Fonction principale d'affichage ========================
# Lit le fichier, résout la DRT, trace EIS, DRT, ajustement
# ------------------------------------------------------------
def plot_eis_and_drt(file_path, sheet_index=1):
    df = pd.read_excel(file_path, sheet_name=sheet_index)
   

    # === Création des figures ===
    fig_eis, ax_eis = plt.subplots(figsize=(6, 5), dpi=600)
    fig_drt, ax_drt = plt.subplots(figsize=(6, 5), dpi=600)
    fig_fit, ax_fit = plt.subplots(figsize=(6, 5), dpi=600)

    # === Etiquettes ===
    custom_labels = ['Cell1_1', 'Cell2_1', 'Cell3_1', 'Cell1_2', 'Cell2_2', 'Cell3_2']
    col_names = df.columns

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(custom_labels))]

    # Echelle log pour tau
    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), 100)

    total_rms_error = 0
    total_rel_error = 0
    num_spectre = 0
    
    all_aires = []  # Liste vide pour stocker les aires de chaque spectre
    
    for i in range(0, len(col_names), 3):
        freq_col = col_names[i]
        re_col = col_names[i + 1]
        im_col = col_names[i + 2]

        freq = df[freq_col].to_numpy()
        Z_re = df[re_col].to_numpy() # conversion Ohm → mOhm
        Z_im_eis = df[im_col].to_numpy() * -1

        # Nettoyage : fréquence > 0, pas de NaN
        mask = (freq > 0) & (~np.isnan(freq)) & (~np.isnan(Z_im_eis))
        freq = freq[mask]
        Z_re = Z_re[mask]
        Z_im_eis = Z_im_eis[mask]

        label = custom_labels[num_spectre]

        # === Courbe Nyquist ===
        ax_eis.plot(Z_re, Z_im_eis, marker='o', linestyle='-', mfc='none',
                    label=label, linewidth=1.1, color=colors[num_spectre])

        # === Résolution DRT ===
        Z_im_for_drt = -Z_im_eis  # signe standard
        gamma = solve_drt(freq, Z_im_for_drt, tau, alpha=alpha)

        # === Courbe DRT ===
        mask_tau = (tau >= tau_min) & (tau <= tau_max)
        ax_drt.plot(tau[mask_tau], gamma[mask_tau], label=label, color=colors[num_spectre])

        # === Reconstruction Z_im ===
        K_full = np.imag(kernel_matrix(freq, tau))
        Z_im_reconstructed = K_full @ gamma

        # === Comparaison mesuré/reconstruit ===
        ax_fit.plot(freq, Z_im_eis, 'o', mfc='none', color=colors[num_spectre], label=f"{label} (measured)")
        ax_fit.plot(freq, -Z_im_reconstructed, '--', color=colors[num_spectre], label=f"{label} (reconstructed)")

        # === Résiduel ===
        error = Z_im_for_drt - Z_im_reconstructed
        ax_fit.plot(freq, error, ':', label='Residual')

        # === Erreurs quantitatives ===
        rms_error = np.sqrt(np.mean(error**2))
        rel_error = rms_error / np.max(np.abs(Z_im_for_drt))
        total_rms_error += rms_error
        total_rel_error += rel_error
        num_spectre += 1
        
        # === Intégrale de la DRT : équivalent résistance ===
        dln_tau = np.diff(np.log(tau))
        gamma_moy = 0.5 * (gamma[1:] + gamma[:-1])
        aire = np.sum(gamma_moy * dln_tau)
        all_aires.append(aire)

    # === Liste des Z complexes ===
    Z_complex_list = []
    Z_max_list = []
    
    for i in range(0, len(col_names), 3):
        re_col = col_names[i + 1]
        im_col = col_names[i + 2]
        Z_re = df[re_col].to_numpy() * 1000
        Z_im = df[im_col].to_numpy() * -1000
        Z_complex = Z_re + 1j * Z_im
        Z_complex_list.append(Z_complex)
        Z_max_list.append(np.nanmax(np.abs(Z_complex)))
    
    # === Toutes les paires ===
    rms_pairs = []
    rel_rms_pairs = []
    
    print("\n==== Dispersion inter-spectres ====")
    for i, j in itertools.combinations(range(len(Z_complex_list)), 2):
        Z1 = Z_complex_list[i]
        Z2 = Z_complex_list[j]
        diff = Z1 - Z2
    
        # RMS absolue
        rms_diff = np.sqrt(np.nanmean(np.abs(diff)**2))
        rms_pairs.append(rms_diff)
    
        # RMS relative (%)
        Z_ref = max(Z_max_list[i], Z_max_list[j])
        rel_rms = 100 * rms_diff / Z_ref
        rel_rms_pairs.append(rel_rms)
    
        label_i = custom_labels[i]
        label_j = custom_labels[j]
    
        print(f"Paire {label_i} vs {label_j} : RMS = {rms_diff:.3f} mOhm, RMS relative = {rel_rms:.2f} %")
    
    # === Moyennes ===
    rms_mean = np.mean(rms_pairs)
    rel_rms_mean = np.mean(rel_rms_pairs)
    
    print(f"\n RMS moyenne toutes paires = {rms_mean:.3f} mOhm")
    print(f" RMS relative moyenne = {rel_rms_mean:.2f} %")

    # === Statistiques globales ===
    mean_rms_error = total_rms_error / num_spectre
    mean_rel_error = total_rel_error / num_spectre

    aire_moyenne = np.mean(all_aires)
        
    print(f"\n==== Résumé ====")
    print(f"Erreur RMS moyenne = {mean_rms_error:.4f} mΩ")
    print(f"Erreur relative RMS moyenne = {mean_rel_error:.2%}")
    print(f"Aire moyenne ≈ {aire_moyenne:.6f} mΩ")

    # === Mise en forme Nyquist ===
    ax_eis.set_xlabel(r"$\mathbf{Z_{re}}$ / $\mathbf{mΩ}$", fontsize=12)
    ax_eis.set_ylabel(r"$\mathbf{-Z_{im}}$ / $\mathbf{mΩ}$", fontsize=12)
    ax_eis.set_aspect('equal', adjustable='box')
    ax_eis.tick_params(axis='both', labelsize=12, top=True, right=True, direction="in")
    ax_eis.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_eis.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax_eis.tick_params(axis='x', which='minor', top=True, direction="in")
    ax_eis.tick_params(axis='y', which='minor', right=True, direction="in")
    ax_eis.grid(which='major', color='lightgray', linestyle='--', linewidth=0.8)
    ax_eis.legend(ncol=2)
    ax_eis.set_ylim(0, 60)
    ax_eis.set_xlim(10, 70)

    # === Mise en forme DRT ===
    ax_drt.set_xscale('log')
    ax_drt.set_xlim(tau_min, 10*tau_max)
    ax_drt.set_ylim(-0.2)
    ax_drt.set_xlabel(r'$\mathbf{Time\ constant\ } \boldsymbol{\tau} \mathbf{\ (s)}$', fontsize=12) 
    ax_drt.set_ylabel(r'$\mathbf{DRT\ } \boldsymbol{\gamma}(\boldsymbol{\tau}) \mathbf{\ /\ \Omega}$', fontsize=12)
    ax_drt.grid(which='major', color='lightgray', linestyle='--', linewidth=1)
    ax_drt.grid(which='minor', color='lightgray', linestyle=':', linewidth=0.9)
    ax_drt.legend(ncol=2)
    ax_drt.tick_params(axis='both', labelsize=12, top=True, right=True, direction="in")
    ax_drt.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax_drt.tick_params(axis='x', which='minor', top=True, direction="in")
    ax_drt.tick_params(axis='y', which='minor', right=True, direction="in")
    ax_drt.xaxis.set_major_locator(LogLocator(base=10.0))
    ax_drt.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))
    ax_drt.xaxis.set_minor_formatter(NullFormatter())

    # === Mise en forme reconstruction ===
    ax_fit.set_xscale('log')
    ax_fit.set_xlabel("Frequency / Hz")
    ax_fit.set_ylabel(r"$Z_{im}$ / mΩ")
    ax_fit.grid(True, which='both', linestyle='--')
    ax_fit.legend(fontsize=8, ncol=1, bbox_to_anchor=(1.4, 1), loc='upper right', frameon=True)

    fig_eis.tight_layout()
    fig_drt.tight_layout()
    plt.show()

# === Lancement ===
plot_eis_and_drt(file_path, sheet_index=1)

