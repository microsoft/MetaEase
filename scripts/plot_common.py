import json
import os
from typing import Dict, Any, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import colorsys
import re

METHOD_COLORS = {
    "MetaEase": "#4DB84D",      # Deep blue
    # "MetaOpt": "#A23B72",       # Deep magenta
    # "HillClimbing": "#F18F01",  # Orange
    # "SimulatedAnnealing": "#C73E1D",  # Red
    # "Random": "#6B5B95",        # Purple
    # "SampleBasedGradient": "#008000" # Deep green
    "HillClimbing": "#F6B44D",        # Lighter orange
    "SimulatedAnnealing": "#C97B67",  # Lighter red
    "Random": "#9385AC",              # Lighter purple
    "SampleBasedGradient": "#2E86AB"  # Lighter green
}

METHOD_LABELS = {
    "MetaEase": "MetaEase",
    "MetaOpt": "MetaOpt",
    "HillClimbing": "Hill Climbing",
    "Random": "Random",
    "SimulatedAnnealing": "Simulated Annealing",
    "SampleBasedGradient": "Sample-Based Gradient"
}

METHOD_ORDER = ["MetaOpt",
                "MetaEase",
                "SimulatedAnnealing",
                "HillClimbing",
                "SampleBasedGradient",
                "Random"
                ]

METHOD_HATCHES = {"MetaOpt": "/",
                  "MetaEase": "",
                  "SimulatedAnnealing": "..",
                  "Random": "-",
                  "HillClimbing": "\\",
                  "SampleBasedGradient": "o"}

def setup_plot_style():
    """Set up publication-quality plot style."""
    sns.set_style("whitegrid")
    # plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "text.usetex": False,  # Disable LaTeX rendering
            "legend.fontsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.monospace": ["Courier New", "Courier", "DejaVu Sans Mono"],
            "mathtext.fontset": "dejavuserif",  # math styled similar to Times
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "hatch.color": 'black',
            "hatch.linewidth": 0.5,
            # make a and y labels bold
            # "axes.labelweight": "bold",
            # "axes.titleweight": "bold",
            # make x tick labels bold
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.major.size": 14,
            "ytick.major.size": 14,
            # distance between xtick and the plot vertical line
            "xtick.major.pad": 5, 
            "ytick.major.pad": 5,
            # make tick colors black
            "xtick.color": "black",
            "ytick.color": "black"
        }
        
    )
    plt.rc("axes", labelpad=15)
