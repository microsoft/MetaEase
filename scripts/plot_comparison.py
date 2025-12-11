from plot_common import setup_plot_style
setup_plot_style()
# (name, gap, time (s))
MetaEase = [('abilene', 1118.8800, 5490.0), ('b4', 1430.0000, 4643.0), ('diamond',  20.0, 406.0), ('swan', 400.0000, 13590.0)]
MetaOpt = [('abilene', 1140, 1912.143),('b4', 1710, 2275.782), ('diamond', 20, 12.875), ('swan', 400, 1894.827)]
import matplotlib.pyplot as plt
import numpy as np

# Extract names, gaps, and times
names = [x[0] for x in MetaEase]
gap_ease = np.array([x[1] for x in MetaEase])
time_ease = np.array([x[2] for x in MetaEase])
gap_opt = np.array([x[1] for x in MetaOpt])
time_opt = np.array([x[2] for x in MetaOpt])

# Compute percentages

gap_percent = (gap_ease) / gap_opt * 100
time_percent = (time_ease - time_opt) / 3600

# 1. Plot: Gap % vs Time %
plt.figure(figsize=(7, 5))
plt.scatter(gap_percent, time_percent, color='blue')
for i, name in enumerate(names):
    plt.text(gap_percent[i], time_percent[i], name, fontsize=10, ha='right', va='bottom')
plt.xlim(0, 102)
plt.xlabel('$\Delta$ Gap (%)')
plt.ylabel('$\Delta$ Time (h)')
plt.grid(True)
plt.tight_layout()
plt.savefig('main_gap_vs_time.png', dpi=300, bbox_inches='tight')

# 2. Plot: Gap % only
plt.figure(figsize=(7, 5))
plt.bar(names, gap_percent, color='orange')
plt.ylim(0, 100)
plt.ylabel('$\Delta$ Gap (%)')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('main_gap_only.png', dpi=300, bbox_inches='tight')