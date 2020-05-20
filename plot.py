"""
Usage:
    plot.py [options] MODEL_NAME

Options:
    -h --help                        Show this screen.
    --upscale UP                     Test on larger data. Gets appended to model name. Remember to add underscore (e.g. _2x) [default: ]
    --use-BFS-for-termination        Use BFS for deciding if more augmenting paths exist. [default: False]
"""

from docopt import docopt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import subprocess

def line_count(filename):
    return int(subprocess.check_output('wc -l {}'.format(filename), shell=True).split()[0])

sns.set()
args = docopt(__doc__)
upscale = args["--upscale"]
print("ARGS", args)

FONT_SZ = 48
LABELPAD = 20
LINEWIDTH = 3
plt.rcParams["figure.figsize"] = [25, 13]
fig, ax1 = plt.subplots()

ax1.set_xlabel("Epoch", fontsize=FONT_SZ)
ax1.set_ylabel("Maximum flow error", fontsize=FONT_SZ, labelpad=LABELPAD)
plt.xticks(fontsize=FONT_SZ)
plt.yticks(fontsize=FONT_SZ)
name = args["MODEL_NAME"]

plt.rcParams.update({'lines.markeredgewidth': LINEWIDTH})
plt.rcParams.update({'lines.linewidth': LINEWIDTH})
plt.rcParams.update({'errorbar.capsize': LINEWIDTH+3})
thresholds = ['BFS-based'] if args["--use-BFS-for-termination"] else [1, 3, 5]
LIMIT = line_count('./results/results_'+name+upscale+'_' + thresholds[0]+'.txt')
x = list(range(1, LIMIT+1))

for threshold in thresholds:
    y_values = []
    stds = []
    with open('./results/results_'+name+upscale+'_{}.txt'.format(threshold)) as f:
        for line in f:
            y = [float(x) for x in line.split()]
            y_values.append(np.average(y))
            stds.append(np.std(y))
    print(len(x), len(y_values), len(stds))
    line = ax1.errorbar(x, y_values[:LIMIT], stds[:LIMIT], ls='--')
    line[-1][0].set_linestyle('--')


ax2 = ax1.twinx()
for threshold in thresholds:
    accuracy_values = []
    stds_acc = []
    with open('./results/results_where_the_same_'+name+upscale+'_{}.txt'.format(threshold)) as f:
        for line in f:
            y = [float(x) for x in line.split()]
            accuracy_values.append(np.average(y))
            stds_acc.append(np.std(y))
    label = "${}$" if args["--use-BFS-for-termination"] else "$t={}$"
    line = ax2.errorbar(x, accuracy_values[:LIMIT], stds_acc[:LIMIT], label=label.format(threshold))

ax2.legend(prop={"size": FONT_SZ}, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.18))

ax2.set_ylabel("Maximum flow accuracy", fontsize=FONT_SZ, labelpad=LABELPAD)
ax2.tick_params(axis='y', labelsize=FONT_SZ)

plt.savefig('./figures/experiments_'+name+upscale+'.png')
