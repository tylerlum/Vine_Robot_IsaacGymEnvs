# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import seaborn as sns
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
filename = '/home/tylerlum/Downloads/wandb_export_2022-12-10T19_11_04.755-08_00.csv'

# %%
df = pd.read_csv(filename)

# %%
df.head()

# %%
num_data_columns = len(df.keys())
ncols = int(math.sqrt(num_data_columns))
nrows = math.ceil(num_data_columns / ncols)

# %%
sns.violinplot(data=df, x="joint_pos_0")

# %%
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
axes = axes.reshape(-1)
for i, (columnName, columnData) in enumerate(df.iteritems()):
    sns.violinplot(data=df, x=columnName, ax=axes[i])
    # axes[i].hist(columnData.values)
    axes[i].set_xlabel(columnName)
    axes[i].set_ylabel('frequency')
fig.tight_layout()

# %%
plt.hist(df["joint_pos_0"])

# %%
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
axes = axes.reshape(-1)
for i, (columnName, columnData) in enumerate(df.iteritems()):
    axes[i].hist(columnData.values)
    axes[i].set_xlabel(columnName)
    axes[i].set_ylabel('frequency')
fig.tight_layout()

# %%
np.random.seed(19680801)
number_of_bins = 100

# An example of three data sets to compare
number_of_data_points = 387
# labels = ["A", "B", "C"]
# data_sets = [np.random.normal(0, 1, number_of_data_points),
#              np.random.normal(6, 1, number_of_data_points),
#              np.random.normal(-3, 1, number_of_data_points)]
labels, data_sets = [], []
for columnName, columnData in df.iteritems():
    labels.append(columnName)
    data_sets.append(columnData.values)

# Computed quantities to aid plotting
hist_range = (np.min(data_sets), np.max(data_sets))
binned_data_sets = [
    np.histogram(d, range=hist_range, bins=number_of_bins)[0]
    for d in data_sets
]
binned_maximums = np.max(binned_data_sets, axis=1)
# x_locations = np.arange(0, sum(binned_maximums), np.max(binned_maximums))
x_locations = np.arange(0, np.max(binned_maximums) * binned_maximums.size, np.max(binned_maximums))

# The bin_edges are the same for all of the histograms
bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
heights = np.diff(bin_edges)

# Cycle through and plot each histogram
fig, ax = plt.subplots(figsize=(10, 3))
for x_loc, binned_data in zip(x_locations, binned_data_sets):
    lefts = x_loc - 0.5 * binned_data
    ax.barh(centers, binned_data, height=heights, left=lefts)

ax.set_xticks(x_locations)
ax.set_xticklabels(labels, rotation=90)

ax.set_ylabel("Data values")
ax.set_xlabel("Data sets")

plt.show()

# %%
