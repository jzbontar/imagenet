import re
import sys

import pylab as plt

train_stats = {}
val_stats = {}
for fname in sys.argv[1:]:
    file = open(fname)
    try:
        args = next(file)
        if not args.strip():
            args = next(file)
    except StopIteration:
        break
    title = fname.split('.')[-1] + args.strip()
    train_stats_file = []
    val_stats_file = []
    for line in file:
        if line.startswith('train'):
            nll = float(re.search('train_nll=(\S+)', line).group(1))
            train_stats_file.append(nll)
        if line.startswith('val'):
            top1 = float(re.search('val_top1=(\S+)', line).group(1))
            val_stats_file.append(top1)
    train_stats[title] = train_stats_file
    val_stats[title] = val_stats_file

plt.figure(figsize=(12, 6))
if 1:
    for k, v in train_stats.items():
        if v[-1] < 6.8:
            plt.plot(v, label=k)
else:
    for k, v in val_stats.items():
        if v and v[-1] < 0.99:
            plt.plot(v, 'o-', label=k)
plt.grid()
plt.legend(loc=3 ,prop={'size': 8})
plt.show()
