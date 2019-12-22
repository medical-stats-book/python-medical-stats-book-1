import mylibs.mylib1 as mylib1  # 本書 1 章の関数群を含む自作ライブラリ

import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import gc
import math
import csv

#############
##   3-1   ##
#############

def calc_bp_class(dbp, sbp):
    return ((dbp >= 80) | (sbp >= 120)) * 1 \
         + ((dbp >= 85) | (sbp >= 130)) * 1 \
         + ((dbp >= 90) | (sbp >= 140)) * 1 \
         + ((dbp >= 100) | (sbp >= 160)) * 1 \
         + ((dbp >= 110) | (sbp >= 180)) * 1

#############
##   3-4   ##
#############

# df_info, df_commons を健診レコードを持つ者のみに絞り込む
def filter_iid_with_hc(df_hc, df_info, df_commons):
    df_info = df_info[df_info['iid'].isin(df_hc['iid'])]
    df_commons = df_commons[df_commons['iid'].isin(df_hc['iid'])]
    return df_hc, df_info, df_commons


# ALB table, Exposure table を作成
def create_obs_ym_and_alb_table_and_exposure_table(
        start_study_t, end_study_t, df_info):
    npa_obs_ym = mylib1.create_npa_obs_ym(start_study_t, end_study_t)
    df_info = mylib1.convert_ym_to_t_in_info(df_info)
    df_alb_table = mylib1.create_alb_table_from_info(npa_obs_ym, df_info)
    df_exposure_table = mylib1.create_exposure_table_from_alb_table(
                            npa_obs_ym, df_alb_table)
    return npa_obs_ym, df_alb_table, df_exposure_table


# Event table の作成
def create_admission_table(npa_obs_ym, df_alb_table, df_commons):
    df_event_table = df_alb_table.copy()
    df_event_table[npa_obs_ym] = 0
    # 入院レコードのみに限定
    df_hosps = df_commons[df_commons['receipt_type'] == 'inpatient']
    dfg = df_hosps.groupby(['iid', 'admission_ym'])
    df_admissions = dfg['days'].sum().reset_index()
    for ym in npa_obs_ym:
        is_ym = (df_admissions['admission_ym'] == ym)
        iid_in_ym = df_admissions.loc[is_ym, 'iid'].values
        in_iid_in_ym = df_event_table['iid'].isin(iid_in_ym)
        df_event_table.loc[in_iid_in_ym, ym] = 1
    return df_event_table


# DBP table の作成
def create_bp_tables(npa_obs_ym, df_alb_table, df_hc, dbp_or_sbp):
    df_bp_table = df_alb_table.copy()
    df_bp_table[npa_obs_ym] = 0
    df_hc.index = df_hc.iid
    idx = df_bp_table.index
    df_bp_table.index = df_bp_table.iid
    for ym in npa_obs_ym:
        ss_bp = df_hc.loc[(df_hc.hc_ym == ym), dbp_or_sbp]
        df_bp_table[ym] = ss_bp
    df_bp_table[npa_obs_ym] = df_bp_table[npa_obs_ym].fillna(method='ffill', axis=1)
    df_bp_table.index = idx
    return df_bp_table
