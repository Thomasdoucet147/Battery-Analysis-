# ----------------

# Le but de ce code est de calculer la capacité durant tout le cycle de vieillissement. 
# Ce fichier comporte l:
# - Fonction pour détecter la phase CV
#

# ----------------

import numpy as np

def detecter_phase_cv_decharge(Courant, tension, time, discharge_capacity, seuil_dV=1e-4, seuil_V=4.19, seuil_I=0.05, fenetre=60):
    """
    Détecte la phase CV d'une décharge CC-CV avec lissage par moyennage temporel.
    
    Arguments:
        Courant : array-like, courant mesuré (A, négatif en décharge)
        tension : array-like, tension mesurée (V)
        time    : array-like, temps correspondant (s)
        seuil_dV : float, seuil max |dV/dt| (V/s) pour considérer V comme constant
        seuil_I : float, valeur absolue max du courant (A) pour considérer qu'il tend vers zéro
        fenetre : float, durée de la fenêtre de moyennage en secondes
    
    Retourne:
        time_cv, courant_cv, tension_cv : arrays correspondant à la phase CV de décharge (lissés)
    """
    Courant = np.array(Courant)
    tension = np.array(tension)
    time = np.array(time)
    discharge_capacity = np.array(discharge_capacity)
    
    # taille de la fenêtre (en nombre de points)
    dt = np.mean(np.diff(time))
    n_fenetre = max(1, int(fenetre / dt))

    # lissage par moyenne glissante (convolution)
    def smooth(x, n):
        return np.convolve(x, np.ones(n)/n, mode="same")

    I_smooth = smooth(Courant, n_fenetre)
    V_smooth = smooth(tension, n_fenetre)

    # dérivée lissée
    dVdt = np.gradient(V_smooth, time)

    # masque CV : tension quasi constante ET courant positif proche de zéro
    masque = (np.abs(dVdt) < seuil_dV) & (I_smooth < 0) & (I_smooth  < -seuil_I) & (V_smooth < seuil_V)

    return time[masque], I_smooth[masque], V_smooth[masque], discharge_capacity[masque]
