import numpy as np



def C_diffusion_SOC(OCV_voltage, OCV_current):
    """
    Calcule la capacité de diffusion normalisée (0-1) en fonction de la SOC.
    
    Paramètres
    ----------
    OCV_voltage : array-like
        Tension OCV (V)
    OCV_current : array-like
        Capacité cumulée ou courant intégré (Ah ou mAh)
    
    Retour
    -------
    SOC : array-like
        État de charge normalisé (0-1)
    C_norm : array-like
        Capacité différentielle normalisée dQ/dV
    """
    OCV_voltage = np.asarray(OCV_voltage)
    OCV_current = np.asarray(OCV_current)

    # Capacité différentielle
    dQdV = np.gradient(OCV_current) / np.gradient(OCV_voltage)

    # Capacité totale pour normalisation
    Q_tot = OCV_current[-1] - OCV_current[0]

    # Capacité normalisée
    C_norm = dQdV / Q_tot

    # SOC normalisée : 0 à 1
    SOC = (OCV_current - OCV_current[0]) / Q_tot

    return SOC, C_norm

def RC(R,C,freq):
    omega = 2 * np.pi * freq
    K = 1 / (1 + 1j * np.outer(omega, R*C))
    return K


def tau_moyen_direct(tau, gamma):
    """
    Calcule le temps caractéristique moyen d'un pic DRT pondéré par gamma.

    Paramètres
    ----------
    tau : array-like
        temps caractéristiques (s)
    gamma : array-like
        intensité de la DRT correspondante (Ω ou Ω/unit tau)

    Retour
    -------
    tau_mean : float
        tau moyen (s)
    """
    tau = np.asarray(tau)
    gamma = np.asarray(gamma)

    if np.sum(gamma) == 0:
        return np.nan  # éviter division par zéro

    tau_mean = np.sum(tau * gamma) / np.sum(gamma)
    return tau_mean

def voltage_from_current(I, dt, R_ohmic, R_SEI, C_SEI, 
                         R_tc_list, C_tc_list,C_diff, R_diff, V0=0.0, Nb_RC_diff= 5):
    """
    Simule la tension d'un circuit équivalent à partir du courant I(t)
    
    Paramètres
    ----------
    I : array-like
        Courant d'entrée (A), positif en charge
    dt : float
        Pas de temps (s)
    R_ohmic : float
        Résistance ohmique
    R_SEI : float
        Résistance SEI
    C_SEI : float
        Capacité SEI
    R_tc_list : list of float
        Liste des résistances de transfert de charge [R_tc1, R_tc2, R_tc3,...]
    C_tc_list : list of float
        Liste des capacités correspondantes [C_tc1, C_tc2, C_tc3,...]
    R_diff : float, optionnel
        Résistance diffusion/Warburg
    V0 : float, optionnel
        Tension initiale
    
    Retour
    ------
    V : ndarray
        Tension simulée (V)
    """
    I = np.asarray(I)
    N = len(I)
    V = np.zeros(N)
    V_RC_SEI = 0.0
    V_RC_tc = np.zeros(len(R_tc_list))
    
    for n in range(N):
        # tension instantanée
        V_inst = R_ohmic * I[n]
        # SEI RC
        dV_SEI = (I[n] - V_RC_SEI / R_SEI) * dt / C_SEI
        V_RC_SEI += dV_SEI
        V_inst += V_RC_SEI

        # RC transfert de charge
        for i, (R_tc, C_tc) in enumerate(zip(R_tc_list, C_tc_list)):
            dV_tc = (I[n] - V_RC_tc[i] / R_tc) * dt / C_tc
            V_RC_tc[i] += dV_tc
            V_inst += V_RC_tc[i]

        # R_diffusion simple (ohmique)
        for i in range(Nb_RC_diff):
            V_inst += 6*R_diff * I[n] / (Nb_RC_diff*np.pi)**2
            dV_tc = (I[n] - V_RC_tc[i] / R_tc) * dt / C_tc
            V_RC_tc[i] += dV_tc
        # enregistrer tension
        V[n] = V0 + V_inst

    return V

