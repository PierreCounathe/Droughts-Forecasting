import pickle
import random
import re
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook

tqdm.pandas()

import statsmodels.api as sm
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler

import preprocessing


def plot_confusion_matrix(matrix, labels):
    plt.figure(1, figsize=(8.7, 7))
    ax = sns.heatmap(matrix, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticklabels(labels)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_yticklabels(labels, rotation=90, va="center")
    ax.set_xlabel("Predicted Label", labelpad=18, fontdict=dict(weight="bold"))
    ax.set_ylabel("True Label", labelpad=15, fontdict=dict(weight="bold"))
    colorax = plt.gcf().axes[-1]
    colorax.tick_params(length=0)
    plt.show()


def plot_confusion_matrices(matrices, labels, macro_f1s, maes):
    fig, axes = plt.subplots(1, 6, sharex=True, sharey=True, figsize=(20, 5.6))
    for i, (matrix, macro_f1, mae) in enumerate(zip(matrices, macro_f1s, maes)):
        sns.heatmap(
            matrix, ax=axes[i], annot=True, cmap="YlGnBu", vmin=0, vmax=1, cbar=False
        )
        axes[i].set_xticklabels(labels)
        axes[i].xaxis.tick_top()
        axes[i].xaxis.set_label_position("top")
        axes[i].tick_params(axis="both", which="both", length=0)
        axes[i].set_yticklabels(labels, rotation=90, va="center")
        axes[i].text(
            3, 7, f"MAE: {mae}\nMacro F1: {macro_f1}", fontsize=14, ha="center"
        )
        axes[i].set_xlabel(
            f"Predicted Label\nWeek {i+1}",
            labelpad=15,
            fontdict=dict(weight="bold"),
            fontsize=14,
        )
        axes[0].set_ylabel(
            "True Label", labelpad=12, fontdict=dict(weight="bold"), fontsize=14
        )
    fig.suptitle(
        f"avg. MAE: {np.mean([float(x) for x in maes]):.3f}\n"
        f"avg. Macro F1: {np.mean([float(x) for x in macro_f1s]):.3f}",
        fontsize=14,
        fontweight="bold",
        y=0.02,
    )
    fig.tight_layout()
    plt.show()


def summarize(y_true, y_pred):
    weeks_true = np.split(y_true, 6, 1)
    weeks_pred = np.split(y_pred, 6, 1)
    matrices = []
    macro_f1s = []
    maes = []
    for y_true, y_pred in zip(weeks_true, weeks_pred):
        y_true = preprocessing.round_and_intify(y_true.flatten())
        y_pred = preprocessing.round_and_intify(y_pred.flatten())
        matrix = metrics.confusion_matrix(
            y_true, y_pred, normalize="true", labels=[0, 1, 2, 3, 4, 5]
        )
        matrices += [matrix]
        report = metrics.classification_report(y_true, y_pred, digits=3)
        r = re.compile("(?<=macro avg\s{6}\d.\d{3}\s{5}\d.\d{3}\s{5})\d.\d{3}")
        macro_f1s += [r.search(report).group(0)]
        maes += [f"{np.mean(abs(y_true - y_pred)):.3f}"]
    warnings.filterwarnings("ignore")
    labels = ["None", "D0", "D1", "D2", "D3", "D4"]
    plot_confusion_matrices(matrices, labels, macro_f1s, maes)


def scores_at_week(y_target_week, y_pred_week):
    y_true = preprocessing.round_and_intify(y_target_week)
    y_pred = preprocessing.round_and_intify(y_pred_week)
    report = metrics.classification_report(y_true, y_pred, digits=3)
    r = re.compile("(?<=macro avg\s{6}\d.\d{3}\s{5}\d.\d{3}\s{5})\d.\d{3}")
    f1 = float(r.search(report).group(0))
    mae = np.mean(abs(y_true - y_pred))
    warnings.filterwarnings("ignore")
    return (f1, mae)


def multiple_weeks_score(y_target, y_pred):
    weeks_true = np.split(y_target, 6, 1)
    weeks_pred = np.split(y_pred, 6, 1)
    macro_f1s = []
    maes = []
    for y_true, y_pred in zip(weeks_true, weeks_pred):
        y_true = preprocessing.round_and_intify(y_true.flatten())
        y_pred = preprocessing.round_and_intify(y_pred.flatten())
        report = metrics.classification_report(y_true, y_pred, digits=3)
        r = re.compile("(?<=macro avg\s{6}\d.\d{3}\s{5}\d.\d{3}\s{5})\d.\d{3}")
        macro_f1s += [float(r.search(report).group(0))]
        maes += [np.mean(abs(y_true - y_pred))]
    warnings.filterwarnings("ignore")
    return np.mean(macro_f1s), np.mean(maes)
