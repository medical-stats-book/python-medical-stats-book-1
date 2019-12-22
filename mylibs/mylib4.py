import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import gc
import math
import csv
import pickle


def alpha(dict_bp, w, sex, x):
    out_classes = dict_bp[sex]['out']['class'][x].astype('f8')
    return w * out_classes.sum()


def beta(dict_bp, w, sex, x):
    exp_classes = dict_bp[sex]['exp']['class'][x].astype('f8')
    return (np.exp(w * exp_classes)).sum()


def get_baseline(df_summary, sex, age):
    is_sex = (df_summary['sex'] == sex)
    is_age = (df_summary['age'] == age)
    return df_summary.loc[is_sex & is_age, 'baseline'].values[0]


def load_bases():
    # Load dict_bp
    f = open('./pseudo_medical/processed/incl_bp/dict_bp.binaryfile', 'rb')  # rb: Read Binary
    dict_bp = pickle.load(f)
    f.close()
    # Load opt_result
    f = open('./pseudo_medical/processed/incl_bp/opt_result.binaryfile', 'rb')  # rb: Read Binary
    opt_result = pickle.load(f)
    f.close()
    # Load df_summary
    df_summary = pd.read_csv('./pseudo_medical/processed/incl_bp/df_summary_with_CI.csv')
    return (dict_bp, opt_result, df_summary)


def predict_rate(df_summary, dict_bp, w, exp_or_out, sex, x):
    baseline = get_baseline(df_summary, sex, x)
    bp_classes = dict_bp[sex][exp_or_out]['class'][x].astype('f8')
    return baseline * np.exp(w * bp_classes)


def count_rates_exp_dif_out(rate_exp, count_exp, rate_out, count_out):
    for r in rate_out:
        count_exp[rate_exp == r] = (count_exp[rate_exp == r] - count_out[rate_out == r])
    return (rate_exp, count_exp)


def make_actual_and_score(df_summary, dict_bp, w, sex, age):
    rates_exp = predict_rate(df_summary, dict_bp, w, 'exp', sex, age)
    rates_out = predict_rate(df_summary, dict_bp, w, 'out', sex, age)
    rate_exp, count_exp = np.unique(rates_exp, return_counts=True)
    rate_out, count_out = np.unique(rates_out, return_counts=True)
    # edo: Exposure Diff Outcome
    (rate_edo, count_edo) = count_rates_exp_dif_out(rate_exp, count_exp, rate_out, count_out)
    y_actual = np.append(np.repeat(0, count_edo.sum()),
                         np.repeat(1, count_out.sum()))
    y_score  = np.append(np.repeat(rate_edo, count_edo),
               np.repeat(rate_out, count_out))
    return (y_actual, y_score)


def make_whole_actual_and_score(df_summary, dict_bp, w):
    Y_actual, Y_score = [], []
    for sex in ['M', 'F']:
        for age in np.arange(65):
            i = (sex == 'F') * 65 + age
            (y_actual, y_score) = make_actual_and_score(df_summary, dict_bp, w, sex, age)
            try:
                Y_actual = np.append(Y_actual, y_actual)
                Y_score = np.append(Y_score, y_score)
            except:
                Y_actual = y_actual
                Y_score = y_score
    return (Y_actual, Y_score)
