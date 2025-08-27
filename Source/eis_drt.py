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
alpha = 4e-2  # pénalisation L2 de la magnitude
beta = 4e-2  # pénalisation L2 de la pente (lissage)

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
                   options={'maxfun': 15000, 'maxiter': 15000}) #10000

    if not res.success:
        print("Optimisation échouée :", res.message)

    return np.clip(res.x, 0, None)  # toujours positif

# ------------------------------------------------------------
# === Fonction principale d'affichage ========================
# Lit le fichier, résout la DRT, trace EIS, DRT, ajustement
# ------------------------------------------------------------

def gamma(Z_im, freq):
    # Echelle log pour tau
    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), 100)

    # # Nettoyage : fréquence > 0, pas de NaN
    mask = (freq > 0) & (~np.isnan(freq)) & (~np.isnan(Z_im))
    freq = freq[mask]
    Z_im_raw = Z_im[mask]      # valeurs du fichier

    # === Résolution DRT ===
    gamma = solve_drt(freq, Z_im, tau, alpha=alpha, beta=beta)
    print(gamma)

    # === Reconstruction Z_im ===
    K_full = np.imag(kernel_matrix(freq, tau))
    Z_im_reconstructed = K_full @ gamma

    # === Résiduel ===
    error = Z_im- Z_im_reconstructed

    # === Erreurs quantitatives ===
    rms_error = np.sqrt(np.mean(error**2))
    rel_error = rms_error / np.max(np.abs(Z_im))

    # === Intégrale de la DRT : équivalent résistance ===
    dln_tau = np.diff(np.log(tau))
    gamma_moy = 0.5 * (gamma[1:] + gamma[:-1])
    aire = np.sum(gamma_moy * dln_tau)

    return gamma, tau, rms_error, rel_error, aire