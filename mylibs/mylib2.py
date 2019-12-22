import mylibs.mylib1 as mylib1  # 本書 1 章の関数群を含む自作ライブラリ

import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import gc
import math
import csv

#############
##   2-2   ##
#############

# 月次入院／外来累積発生率
def make_admission_cumrate(df_admission):
    less_age_75 = (df_admission.alb_max < 75)
    df_admission = df_admission.loc[less_age_75]
    cols = df_admission.columns[3:]
    df_admission_cumrate = df_admission.copy()
    df_admission_cumrate[cols] = df_admission[cols].cumsum(axis=1)
    return df_admission_cumrate


# レセプトの擬似生成
# ==================

# df_info の i 番目の (iid, sex, birth_t, start_obs_t, end_obs_t) を取得
def get_info_for_i(df_info, i):
    cols = ['iid', 'sex', 'birth_ym', 'start_obs_ym', 'end_obs_ym']
    (iid, sex, birth_ym, start_obs_ym, end_obs_ym) = df_info.iloc[i][cols]
    birth_t = mylib1.ym_to_t(birth_ym)
    start_obs_t = mylib1.ym_to_t(start_obs_ym)
    end_obs_t = mylib1.ym_to_t(end_obs_ym)
    return (iid, sex, birth_t, start_obs_t, end_obs_t)


# 入院発生率 adm(sex, alb) を取得
def get_ss_admission_cumrate(df_admission_cumrate, sex, alb):
    cols = df_admission_cumrate.columns[3:]
    is_sex = (df_admission_cumrate.sex == sex)
    more_alb_min = (df_admission_cumrate.alb_min <= alb)
    less_alb_max = (alb <= df_admission_cumrate.alb_max)
    df = df_admission_cumrate.loc[(is_sex & more_alb_min & less_alb_max), cols]
    ss_adm_cumrate = pd.Series(df.columns, index=df.values[0])
    return ss_adm_cumrate


# 入院用に一様乱数 u(rs) (0 <= r <= 1)を生成 & ランダムシード更新
def get_random_num_for_adm(rs):
    np.random.seed(rs)
    u = np.random.rand()  # 一様乱数
    rs = rs + 1           # 乱数シード更新
    return (u, rs)


# (入院発生？, 入院原因傷病 dis) = 入院乱数シミュレーション(u)
def get_admission_disease(ss_adm_cumrate, u):
    ss = ss_adm_cumrate[u < ss_adm_cumrate.index]
    if len(ss) == 0:
        return (False, None)
    else:
        return (True, ss.iloc[0])


# mu = 平均在院日数(sex, alb, dis)
def get_avg_hospdays(df_days, sex, alb, dis):
    is_sex = (df_days.sex == sex)
    more_alb_min = (df_days.alb_min <= alb)
    less_alb_max = (alb <= df_days.alb_max)
    return df_days.loc[(is_sex & more_alb_min & less_alb_max), dis].values[0]


# 在院日数 days を指数分布(mu) で決定
def get_random_days(mu, rs):
    np.random.seed(rs)
    # 指数分布乱数を整数値に切り上げ
    days = math.ceil(np.random.exponential(mu))
    rs = rs + 1  # 乱数シード更新
    return (days, rs)


# 新しい rid を発行
def make_new_rid(rid):
    i = int(rid[1:]) + 1
    return 'r' + str(i).zfill(8)


# 共通レセプト(入院) df_common(iid, rid, ym, admission_ym, days, m) の作成
def create_df_common(iid, rid, ym, receipt_type, admission_ym, days_in_month):
    cols = ['iid', 'rid', 'ym', 'receipt_type', 'admission_ym', 'days']
    values = [iid, rid, ym, receipt_type, admission_ym, days_in_month]
    return pd.DataFrame(values, cols).T


# 月をまたぐ入院における m+1 ヶ月目の入院レセプトを発行
def create_df_common_inpatient_after_m_months(
        iid, rid, ym, admission_ym, days, m):
    ym = mylib1.t_to_ym(mylib1.ym_to_t(admission_ym) + m / 12)
    if m == 0:
        days_in_month = min(15, days)
    else:
        days_in_month = min(30, days - 15 - 30 * (m - 1))
    return create_df_common(iid, rid, ym, 'inpatient',
                           admission_ym, days_in_month)


# csv に df_common を追加
def add_df_xxx_on_csv(open_csv_object, df_xxx):
    if len(df_xxx) == 0:
        pass
    else:
        open_csv_object.writerow(df_xxx.values[0])


# 傷病レセプト df_disease(iid, rid, first_ym, icd10) の作成
def create_df_disease(iid, rid, first_ym, dis):
    cols = ['iid', 'rid', 'first_ym', 'icd10_code']
    values = [iid, rid, first_ym, dis]
    return pd.DataFrame(values, cols).T


# 診療行為レセプト(入院) df_treatment(iid, rid, treatment_code) の作成
def create_df_treatment(iid, rid, treatment_code):
    cols = ['iid', 'rid', 'treatment_code']
    values = [iid, rid, treatment_code]
    return pd.DataFrame(values, cols).T


# 外来発生率 out(sex, alb) を取得
def get_outpatient_rate(df_outpatient, sex, alb):
    cols = df_outpatient.columns[3:]
    is_sex = (df_outpatient.sex == sex)
    more_alb_min = (df_outpatient.alb_min <= alb)
    less_alb_max = (alb <= df_outpatient.alb_max)
    df = df_outpatient.loc[(is_sex & more_alb_min & less_alb_max), cols]
    ss_out_rate = pd.Series(df.columns, index=df.values[0])
    return ss_out_rate


# 外来用に一様乱数 u(rs) (0 <= r <= 1)を生成 & ランダムシード更新
def get_random_num_for_out(rs):
    np.random.seed(rs)
    u = np.random.rand(19)  # 一様乱数
    rs = rs + 1             # 乱数シード更新
    return (u, rs)


# 外来発生の有無と外来原因傷病を乱数シミュレーション
def get_outpatient_diseases(ss_out_rate, us):
    diss = ss_out_rate[us < ss_out_rate.index.values].values
    if len(diss) == 0:
        return (False, None)
    else:
        return (True, diss)


dict_csv_paths = {'common': './pseudo_medical/records/excl_bp/commons.csv',
                  'disease': './pseudo_medical/records/excl_bp/diseases.csv',
                  'treatment': './pseudo_medical/records/excl_bp/treatments.csv'}

def create_receipts(dict_csv_paths):
    commons_csv = open(dict_csv_paths['common'], 'w')
    open_common_object = csv.writer(commons_csv)
    cols = ['iid', 'rid', 'ym', 'receipt_type', 'admission_ym', 'days']
    open_common_object.writerow(cols)

    diseases_csv = open(dict_csv_paths['disease'], 'w')
    open_disease_object = csv.writer(diseases_csv)
    cols = ['iid', 'rid', 'first_ym', 'icd10_code']
    open_disease_object.writerow(cols)

    treatments_csv = open(dict_csv_paths['treatment'], 'w')
    open_treatment_object = csv.writer(treatments_csv)
    cols = ['iid', 'rid', 'treatment_code']
    open_treatment_object.writerow(cols)

    rid = 'r00000000'
    rs = 0

    for i in np.arange(len(df_info)):  # df_info を上から順に参照
        gc.collect()
        # df_info の i 番目の (iid, sex, birth_t, start_obs_t, end_obs_t) を取得
        (iid, sex, birth_t, start_obs_t, end_obs_t) = get_info_for_i(df_info, i)
        t = start_obs_t
        while start_obs_t <= t <= end_obs_t:
            alb = int(t - birth_t)
            # 入院発生率 adm(sex, alb) を取得
            ss_adm_cumrate = get_ss_admission_cumrate(
                df_admission_cumrate, sex, alb)
            # 入院用に一様乱数 u(rs) (0 <= r <= 1)を生成 & ランダムシード更新
            (u, rs) = get_random_num_for_adm(rs)
            # (入院発生？, 入院原因傷病 dis) = 入院乱数シミュレーション(u)
            (does_adm_occur, dis) = get_admission_disease(ss_adm_cumrate, u)
            if does_adm_occur:
                mu = get_avg_hospdays(df_days, sex, alb, dis)
                # 在院日数 days を指数分布(mu) で決定
                (days, rs) = get_random_days(mu, rs)
                # 入院発生年月、入院は全て月央に発生すると仮定
                admission_ym = mylib1.t_to_ym(t)
                # 月をまたぐ継続入院でなくても次のコードブロックを実行
                does_hosp_continue = True
                m = 0  # 初月入院を 0、次月から月をまたぐごとに +1
                while does_hosp_continue:
                    ym = mylib1.t_to_ym(t)
                    rid = make_new_rid(rid)
                    # 基本レセプト(入院)の発行
                    # 基本レセプト(入院)の作成
                    df_common = create_df_common_inpatient_after_m_months(
                                iid, rid, ym, admission_ym, days, m)
                    add_df_xxx_on_csv(open_common_object, df_common)
                    # 傷病レセプトの発行
                    df_disease = create_df_disease(iid, rid, admission_ym, dis)
                    add_df_xxx_on_csv(open_disease_object, df_disease)
                    # 診療行為レセプト(入院)の発行
                    df_treatment = create_df_treatment(iid, rid, 'A100')
                    add_df_xxx_on_csv(open_treatment_object, df_treatment)
                    # 診療行為レセプト(手術)を入院の 10% に発行
                    (u, rs) = get_random_num_for_adm(rs)
                    if u <= 0.1:
                        df_treatment = create_df_treatment(iid, rid, 'K000')
                        add_df_xxx_on_csv(open_treatment_object, df_treatment)

                    t = t + 1 / 12  # t を１ヶ月ずらす
                    does_hosp_continue = (days - 15 - 30 * m > 0)
                    m = m + 1

            else:  # 入院発生なし
                # 外来発生率 out(sex, alb) を取得
                ss_out_rare = get_outpatient_rate(df_outpatient, sex, alb)
                # 外来用に一様乱数 u(rs) (0 <= r <= 1)を生成 & ランダムシード更新
                (us, rs) = get_random_num_for_out(rs)
                # (外来発生？, 複数外来原因傷病 diss) = 外来乱数シミュレーション(u)
                (does_occur_outpatient, diss) \
                    = get_outpatient_diseases(ss_out_rate, us)
                if does_occur_outpatient:
                    # 外来発生年月、外来は全て月央に発生すると仮定
                    ym = mylib1.t_to_ym(t)
                    for dis in diss:
                        rid = make_new_rid(rid)
                        # 基本レセプト(外来)の発行
                        # 基本レセプト(外来) df_common(iid, rid, ym) の作成
                        df_common = create_df_common(iid, rid, ym,
                                                   'outpatient', '-', 1)
                        add_df_xxx_on_csv(open_common_object, df_common)
                        # 傷病レセプトの発行
                        df_disease = create_df_disease(iid, rid, ym, dis)
                        add_df_xxx_on_csv(open_disease_object, df_disease)
                        # 診療行為レセプト(外来)の発行
                        df_treatment = create_df_treatment(iid, rid, 'A000')
                        add_df_xxx_on_csv(open_treatment_object, df_treatment)
                        df_treatment = create_df_treatment(iid, rid, 'F000')
                        add_df_xxx_on_csv(open_treatment_object, df_treatment)
                        # 診療行為レセプト(手術)を外来の 1% に発行
                        (u, rs) = get_random_num_for_adm(rs)
                        if u <= 0.01:
                            df_treatment = create_df_treatment(iid, rid, 'K000')
                            add_df_xxx_on_csv(open_treatment_object, df_treatment)

                t = t + 1 / 12  # t を１ヶ月ずらす

    commons_csv.close()
    diseases_csv.close()
    treatments_csv.close()
