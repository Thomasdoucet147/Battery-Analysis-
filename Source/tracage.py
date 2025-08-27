import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec # for the contour plots
from numpy import loadtxt
import matplotlib as mpl

# === Réglages d'affichage pour matplotlib ===
mpl.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.bottom'] = True
def Voltage_Current_vs_C_rate(c_rates,
                         values,
                         ylabel=r"$\mathbf{Maximal \ discharge \ capacity}$/ $\mathbf{Ah}$",
                         xlabel=r"$\mathbf{Discharge \ rate (D-rate)}$",
                         title="Maximal Discharge Capacity vs C-rate",
                         legend_label="Max discharge capacity (Ah)",
                         color='blue',
                         marker='o',
                         linestyle='-',
                         linewidth=1.1,
                         figsize=(8, 6)):
    """
    Trace une courbe générique en fonction du C-rate, avec personnalisation complète.

    Paramètres :
    ------------
    c_rates : array-like
        Valeurs du taux de décharge (C-rates).
    values : array-like
        Valeurs à tracer (capacités, résistances, tensions, etc.).
    ylabel : str, optionnel
        Étiquette de l'axe des ordonnées (avec LaTeX possible).
    xlabel : str, optionnel
        Étiquette de l'axe des abscisses.
    title : str, optionnel
        Titre du graphique.
    legend_label : str, optionnel
        Texte de la légende.
    color : str, optionnel
        Couleur de la courbe.
    marker : str, optionnel
        Type de marqueur.
    linestyle : str, optionnel
        Style de ligne.
    linewidth : float, optionnel
        Épaisseur de ligne.
    figsize : tuple, optionnel
        Taille de la figure.
    """
    values = np.array(values)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(c_rates, values, marker=marker, linestyle=linestyle, color=color,
            mfc='none', label=legend_label, linewidth=linewidth)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    ax.legend(loc='best', frameon=False)

    ax.set_aspect('auto', adjustable='box')
    ax.tick_params(axis='both', labelsize=12, top=True, right=True, direction="in")

    plt.tight_layout()
    ax.grid(which='major', color='lightgray', linestyle='--', linewidth=0.8)

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis='x', which='minor', top=True, direction="in")
    ax.tick_params(axis='y', which='minor', right=True, direction="in")

    plt.show()



    
def plot_voltage_and_current(times,
                             voltages,
                             currents,
                             time_slice=None,
                             voltage_label="Voltage (V)",
                             current_label="Current (A)",
                             xlabel="Time (s)",
                             title=None,
                             figsize=(12, 8),
                             voltage_color='red',
                             current_color='blue',
                             voltage_legend="Voltage (V)",
                             current_legend="Current (A)"):
    """
    Trace deux sous-graphes partagés pour la tension et le courant en fonction du temps.

    Paramètres :
    ------------
    times : array-like
        Tableau des temps (s).
    voltages : array-like
        Tableau des tensions (V).
    currents : array-like
        Tableau des courants (A).
    time_slice : tuple (start, end), optionnel
        Permet de tronquer les signaux sur un intervalle donné.
    voltage_label : str, optionnel
        Label pour l’axe Y du graphe tension.
    current_label : str, optionnel
        Label pour l’axe Y du graphe courant.
    xlabel : str, optionnel
        Label de l’axe X (temps).
    title : str, optionnel
        Titre global de la figure.
    figsize : tuple, optionnel
        Taille de la figure.
    voltage_color : str, optionnel
        Couleur pour la courbe de tension.
    current_color : str, optionnel
        Couleur pour la courbe de courant.
    voltage_legend : str, optionnel
        Texte de la légende tension.
    current_legend : str, optionnel
        Texte de la légende courant.
    """
    # Appliquer le slicing si demandé
    if time_slice is not None:
        start, end = time_slice
        times = times[start:end]
        voltages = voltages[start:end]
        currents = currents[start:end]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Tension
    ax1.plot(times, voltages, color=voltage_color, label=voltage_legend, linewidth=1.5)
    ax1.set_ylabel(rf"$\mathbf{{{voltage_label}}}$", fontsize=12, color=voltage_color)
    ax1.tick_params(axis='y', labelcolor=voltage_color, right=True)
    ax1.tick_params(axis='both', labelsize=12, top=True, direction="in")
    ax1.legend(loc='upper right', frameon=False)
    ax1.grid(which='major', color='lightgray', linestyle='--', linewidth=0.8)
    ax1.yaxis.set_minor_locator(AutoMinorLocator(5))

    # Courant
    ax2.plot(times, currents, color=current_color, label=current_legend, linewidth=1.5)
    ax2.set_ylabel(rf"$\mathbf{{{current_label}}}$", fontsize=12, color=current_color)
    ax2.set_xlabel(rf"$\mathbf{{{xlabel}}}$", fontsize=12)
    ax2.tick_params(axis='y', labelcolor=current_color, right=True)
    ax2.tick_params(axis='both', labelsize=12, top=True, direction="in")
    ax2.legend(loc='upper right', frameon=False)
    ax2.grid(which='major', color='lightgray', linestyle='--', linewidth=0.8)
    ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(5))

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()



    
def plot_Voltagevstime(times,
                        voltages,
                        time_slice=None,
                        voltage_label="Voltage (V)",
                        xlabel="Time (s)",
                        title=None,
                        figsize=(12, 8),
                        voltage_color='red',
                        voltage_legend="Voltage (V)"):
    """
    Trace la tension en fonction du temps.

    Paramètres :
    ------------
    times : array-like
        Tableau des temps (s).
    voltages : array-like
        Tableau des tensions (V).
    time_slice : tuple (start, end), optionnel
        Permet de tronquer les signaux sur un intervalle donné.
    voltage_label : str, optionnel
        Label pour l’axe Y du graphe tension.
    xlabel : str, optionnel
        Label de l’axe X (temps).
    title : str, optionnel
        Titre global de la figure.
    figsize : tuple, optionnel
        Taille de la figure.
    voltage_color : str, optionnel
        Couleur pour la courbe de tension.
    voltage_legend : str, optionnel
        Texte de la légende tension.
    """
    # Appliquer le slicing si demandé
    if time_slice is not None:
        start, end = time_slice
        times = times[start:end]
        voltages = voltages[start:end]

    fig, ax = plt.subplots(figsize=figsize)

    # Tension
    ax.plot(times, voltages, color=voltage_color, label=voltage_legend, linewidth=1.5)
    ax.set_ylabel(rf"$\mathbf{{{voltage_label}}}$", fontsize=12, color=voltage_color)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.tick_params(axis='y', labelcolor=voltage_color, right=True)
    ax.tick_params(axis='both', labelsize=12, top=True, direction="in")
    ax.legend(loc='upper right', frameon=False)
    ax.grid(which='major', color='lightgray', linestyle='--', linewidth=0.8)
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()

    
def plot_eis_nyquist(donnees_EIS_filtrees, xlim=(-1, 13), ylim=(-2.4, 5)):
    """
    Affiche un tracé de type Nyquist à partir des données EIS filtrées.

    Paramètres
    ----------
    donnees_EIS_filtrees : dict
        Dictionnaire contenant les données EIS pour différents courants.
        Chaque entrée doit être un sous-dictionnaire avec les clés "Re" et "Im_neg".
    
    xlim : tuple, optionnel
        Limites de l'axe x (par défaut : (-1, 13)).
    
    ylim : tuple, optionnel
        Limites de l'axe y (par défaut : (-2.4, 5)).
    """
    sns.set_palette("Set1")
    fig, ax = plt.subplots(figsize=(8, 6))

    for courant, data in donnees_EIS_filtrees.items():
        ax.plot(
            data["Re"], data["Im_neg"],
            marker='o', linestyle='-', mfc='none',
            label=f"{courant} Filtré", linewidth=1.1
        )

    # Mise en forme des axes
    ax.set_xlabel(r"$\mathbf{Z}_{\mathbf{re}}$ / $\mathbf{m\Omega}$", fontsize=12)
    ax.set_ylabel(r"$-\mathbf{Z}_{\mathbf{im}}$ / $\mathbf{m\Omega}$", fontsize=12)
    ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.78, 1), frameon=False)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', labelsize=12, top=True, right=True, direction="in")

    # Ajustement de la grille et des ticks
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis='x', which='minor', top=True, direction="in")
    ax.tick_params(axis='y', which='minor', right=True, direction="in")
    ax.grid(which='major', color='lightgray', linestyle='--', linewidth=0.8)

    plt.tight_layout()
    plt.show()

    
def plot_bode(df, title="Bode"):
    sns.set_palette("Set1")
    fig1, ax1 = plt.subplots(figsize=(7,4))
    fig2, ax2 = plt.subplots(figsize=(7,4))
    f = df["frequency"].values
    Zmag = np.hypot(df["Re"].values, -df["Im"].values)
    phase = np.degrees(np.arctan2(-df["Im"].values, df["Re"].values))
    ax1.semilogx(f, Zmag, marker='o', ls='-'); ax1.set_xlabel("f (Hz)"); ax1.set_ylabel("|Z| (mΩ)"); ax1.grid(ls='--', alpha=0.5)
    ax2.semilogx(f, phase, marker='o', ls='-'); ax2.set_xlabel("f (Hz)"); ax2.set_ylabel("Phase (deg)"); ax2.grid(ls='--', alpha=0.5)
    fig1.suptitle(title+" — |Z|"); fig2.suptitle(title+" — phase")
    plt.tight_layout(); plt.show()

def plot_drt(tau, gamma, title="DRT"):
#☻    sns.set_palette("Set2")
    fig, ax = plt.subplots(figsize=(7,4))
    ax.semilogx(tau, gamma, '-',color=sns.color_palette("deep")[0], linewidth=1.2)
    ax.set_xlabel(r"$\tau$ (s)")
    ax.set_ylabel(r"$\gamma(\tau)$ (m$\Omega$)")
    ax.grid(which='major', color='lightgray', linestyle='--', linewidth=1)
    ax.grid(which='minor', color='lightgray', linestyle=':', linewidth=0.9)
    ax.set_title(title)
    plt.tight_layout(); plt.show()