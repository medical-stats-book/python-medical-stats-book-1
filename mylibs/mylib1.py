import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc

#############
##   1-2   ##
#############

def ym_to_t(ym):
    """
    "yyyy/mm" 形式の文字列を、時刻 t (スカラー) に変換。
    yyyy 年 mm 月の月央を t に変換。
    """
    y = int(ym[:4])
    m = (int(ym[5:]) - 0.5) / 12
    return y + m


def t_to_ym(t):
    """
    時刻 t (スカラー) を "yyyy/mm" 形式の文字列に変換。
    時刻 t が yyyy 年 mm 月の1日～月末にあれば、 "yyyy/mm" を返す。
    """
    y = int(t)
    m = int((t - y) * 12) + 1
    m = max(min(m, 12), 1)
    # 月部分を2桁表示になるよう、つまり
    # "2010/1" でなく "2010/01" となるよう
    # zfill(2) でパディングする。
    return str(y) + '/' + str(m).zfill(2)


# 擬似生成のためのパラメタと pd.DataFrame は引数として与える
# ==========================================================
# 人口推計の累積比率 CSV を読み込み
# df_birth = pd.read_csv('./sources/ipss_birth.csv')
# start_study_t = 2010    # 分析開始時点 (2010年1月)
# end_study_t = 2019.999  # 分析終了時点 (2019年12月)
# mu: 指数分布のパラメタ
# 観察開始期間、観察終了期間を決定する。
# mu = 10
# N = 5000            # N 人の加入者を生成する。
# family_ratio = 0.3  # 全体の内、扶養家族の占める割合
# IPSS の月次死亡率 CSV を読み込み
# df_mortality = pd.read_csv('./sources/ipss_mortality.csv')

##########################
# 死亡情報以外の擬似生成 #
##########################

def create_birth_and_obs(df_birth, start_study_t, end_study_t,
                         mu, N, family_ratio):
    """
    誕生年は df_birth に従うこととし
    観察開始期間と観察終了期間は指数分布に従うという仮定のもと
    df_info を作成。
    """
    # 変数初期化
    i = 0
    rs = 0              # rs: Random Seed
    np.random.seed(rs)  # 乱数シードをセット
    df_info = pd.DataFrame()

    while len(df_info) < N:
        r = np.random.rand()  # 一様乱数
        rs = rs + 1           # 乱数シード更新
        np.random.seed(rs)
        ss = df_birth[df_birth['cum_ratio'] >= r].iloc[0]  # ss: pd.SerieS
        sex = ss.sex
        by = ss.year  # bs: Birth Year
        # 誕生月を一様に分布させる。
        # Birth Time -> Birth Year Month
        bt = by + np.random.rand()
        rs = rs + 1
        np.random.seed(rs)
        bym = t_to_ym(bt)
        # 疑似データにおいては
        # 観察開始期間、観察終了期間は指数分布で決定する。
        start_t = start_study_t - mu + np.random.exponential(mu)
        rs = rs + 1
        np.random.seed(rs)
        end_t = start_t + np.random.exponential(mu)
        rs = rs + 1
        np.random.seed(rs)
        # | は論理和 (or)
        if (end_t < start_study_t) | (end_study_t < start_t):
            # 分析期間に在籍しない場合 -> 何もしない
            pass
        else:
            # iid: 加入者ID
            cols = ['iid', 'sex', 'family', 'birth_ym',
                    'start_obs_ym', 'end_obs_ym']
            df_exposure = pd.DataFrame(np.zeros(len(cols)).reshape(1, len(cols)),
                                       columns=cols)
            # 本人: 1, 家族: 2
            family = 2 - (np.random.rand() > family_ratio)
            rs = rs + 1
            np.random.seed(rs)
            df_exposure.loc[:, cols] = ['i' + str(i).zfill(6), sex, family, bym,
                                 t_to_ym(start_t), t_to_ym(end_t)]
            # df_info の下に
            # 新しく作成した１列データフレーム df_exposure を追加する。
            df_info = pd.concat([df_info, df_exposure], axis=0)
        i = i + 1

    # index の振り直し
    df_info = df_info.reset_index()
    # reset_index() により作成されてしまった "index" という列を消す。
    del df_info['index']
    return df_info


def truncate_obs_period(df_info, start_study_t, end_study_t):
    """
    df_info を start_study_t と end_study_t で挟まれる分析期間に絞り込む。
    この手続きはデータの擬似生成のために必要だが、
    実データではデータの最終更新日で打ち切られるため、実務においては不要。
    """
    # 観察開始年月の分析開始時点での制限
    df_info['start_obs_t'] = df_info['start_obs_ym'].apply(ym_to_t)
    #                                               ---------------
    #        時刻が文字列のままでも大小比較は可能ですが、
    #        数値に変換しておくと次の年齢の計算で便利なので数値にすることとします。
    more_start_study = (df_info['start_obs_t'] > start_study_t)
    df_info['start_obs_t'] = more_start_study * df_info['start_obs_t'] \
                           + ~more_start_study * start_study_t
    #                        -----------------
    #                        ~ は論理否定演算子、False→True, True→False, 0→1、1→0
    df_info['start_obs_ym'] = df_info['start_obs_t'].apply(t_to_ym)

    # 観察終了年月の分析終了時点での制限
    df_info['end_obs_t'] = df_info['end_obs_ym'].apply(ym_to_t)
    less_end_study = (df_info['end_obs_t'] < end_study_t)
    df_info['end_obs_t'] = less_end_study * df_info['end_obs_t'] \
                         + ~less_end_study * end_study_t
    df_info['end_obs_ym'] = df_info['end_obs_t'].apply(t_to_ym)
    df_info['birth_t'] = df_info['birth_ym'].apply(ym_to_t)
    return df_info


def change_dtypes(df_info):
    """
    データ型を変換してメモリ効率アップ
    """
    types = {'iid': 'str',
             'sex': 'str',
             'family': 'int8',
             'birth_ym': 'str',
             'start_obs_ym': 'str',
             'end_obs_ym': 'str',
             'start_obs_t': 'float32',
             'end_obs_t': 'float32',
             'birth_t': 'float32'}
    df_info = df_info.astype(types)
    return df_info

######################
# 死亡情報の擬似生成 #
######################

# 月次満年齢テーブルの作成
# ========================

def create_npa_obs_ym(start_study_t, end_study_t):
    """
    np.array 形式で、一連の観察期間 "yyyy/mm" を作成
    npa_pbs_ym: NumPy Array of OBServation Year and Month
    """
    t = start_study_t
    obs_ym = []
    while t < end_study_t:
        ym = t_to_ym(t)
        obs_ym.append(ym)
        t = ym_to_t(ym) + 1/12
    return np.array(obs_ym)


def convert_ym_to_t_in_npa(npa_obs_ym):
    """
    一連の観察期間を t (スカラー) に変換
    """
    # np.array に apply がないため
    # pd.Series に一度変換して apply して np.array に戻す。
    return np.array(pd.Series(npa_obs_ym).apply(ym_to_t))


def create_alb_table(npa_obs_ym, df_info):
    """
    iid と観察年月の組み合わせの各セルごとに満年齢を持つ df_alb_table を作成
    """
    npa_obs_t = convert_ym_to_t_in_npa(npa_obs_ym)
    npa_birth_t = np.array(df_info['birth_t'])
    df_alb_table = df_info.copy()
    for i in np.arange(len(npa_obs_ym)):
        # "//" で商(整数)をとる。ちなみに剰余演算子は "%"
        # df_alb_table に新しい列を追加しながら、計算した ALB 列を代入している。
        df_alb_table[npa_obs_ym[i]] = (npa_obs_t[i] - npa_birth_t) // 1
        df_alb_table[npa_obs_ym[i]] = df_alb_table[npa_obs_ym[i]].astype('int8')
    return df_alb_table

# 月次死亡率テーブルの作成
# ========================

def create_mortality_table_frame(df_mortality, npa_obs_ym, df_alb_table):
    """
    df_alb_table をコピーして df_mortality_table の枠を作成。
    各セルの満年齢に対応する月次死亡率はこのコード以後に代入される。
    """
    df_mortality_table = df_alb_table.copy()
    # int8 から float64 に変換することで、死亡率がより精緻に取り扱える
    for i in np.arange(len(npa_obs_ym)):
        df_mortality_table[npa_obs_ym[i]] = \
                df_mortality_table[npa_obs_ym[i]].astype('float64')
    return df_mortality_table


def get_mortality(df_mortality, sex, alb):
    """
    (sex, alb) に対応する月次死亡率を取得
    """
    if alb < 0 or 100 <= alb:
        return 0
    else:
        return df_mortality.loc[alb, sex]


def calc_monthly_mortality_col(df_mortality, df_alb_table, ym):
    """
    df_mortality_table の ym 列に関する月次死亡率を np.array 形式で計算
    """
    sexes = df_alb_table['sex']  # 加入者の性別一覧
    albs = df_alb_table[ym]      # 加入者の "yyyy/mm" における満年齢一覧
    mortalities = [get_mortality(df_mortality, sex, alb) for (sex, alb)
                   in zip(sexes, albs)]  # リスト内包表記
    # np.isnan の使用と、月次死亡率計算でブロードキャスティングを使うため
    # リスト形式を np.array 形式に変換。
    # ブロードキャスティングについては以下を参照。
    # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    mortalities = np.array(mortalities)
    mortalities[np.isnan(mortalities)] = 0
    return mortalities


def create_mortality_table(df_mortality, npa_obs_ym,
                           df_alb_table, df_mortality_table):
    """
    月次死亡率 を値として持つ df_mortality_table を作成
    """
    for ym in npa_obs_ym:
        mortalities = calc_monthly_mortality_col(df_mortality, df_alb_table, ym)
        df_mortality_table[ym] = mortalities
    return df_mortality_table

# 月次エクスポージャテーブルの作成
# ================================

def create_exposure_table(npa_obs_ym, df_alb_table):
    """
    エクスポージャ単位を [人月] とし、df_exposure_table を作成
    """
    df_exposure_table = df_alb_table.copy()
    for ym in npa_obs_ym:
        # Boolean values for Start_obs_t
        bs = (df_exposure_table['start_obs_t'] <= ym_to_t(ym))
        # Boolean values for End_obs_t
        be = (ym_to_t(ym) <= df_exposure_table['end_obs_t'])
        # Boolean values for Birth_t
        bb = (df_exposure_table['birth_t'] <= ym_to_t(ym))
        # Boolean values for ALB
        # 健保組合は65歳以上を保障しない
        ba = (df_alb_table[ym] < 65)
        df_exposure_table[ym] = (bs & be & bb & ba) * 1
    return df_exposure_table

# 月次死亡テーブルの作成
# ======================

def create_death_table(npa_obs_ym, df_alb_table, df_mortality_table,
                       df_exposure_table):
    """
    df_alb_table と同じ形状を持つ乱数列を作成し、
    df_mortality_table の月次死亡率との比較によって、死亡状態を乱数生成して
    df_death_table を作成。
    """
    df_death_table = df_alb_table.copy()
    rs = 0              # rs: Random Seed
    np.random.seed(rs)  # 乱数シードをセット
    df_random = np.random.random(df_death_table[npa_obs_ym].shape)
    df_death_table[npa_obs_ym] = (df_random < df_mortality_table[npa_obs_ym]) \
                                  * df_exposure_table[npa_obs_ym] * 1
    return df_death_table

# 死亡を最初の１つに確定すると同時に、死亡後のエクスポージャを０に変更
# ====================================================================

def adjust_death_table(npa_obs_ym, df_death_table):
    """
    cumsum を2回適用して、1となるセルのみを1とし、それ以外を0とする。
    """
    df_death_table[npa_obs_ym] \
        = (df_death_table[npa_obs_ym].cumsum(axis=1).cumsum(axis=1) == 1) * 1
    # 整数型に変更してメモリ効率を上げる。
    for i in np.arange(len(npa_obs_ym)):
        df_death_table[npa_obs_ym[i]] \
            = df_death_table[npa_obs_ym[i]].astype('int8')
    return df_death_table


def adjust_exposure_table(npa_obs_ym, df_exposure_table, df_death_table):
    """
    死亡後のエクスポージャを0に変更。
    """
    before_death_filter = (df_death_table[npa_obs_ym].cumsum(axis=1).cumsum(axis=1) <= 1) * 1
    df_exposure_table[npa_obs_ym] = df_exposure_table[npa_obs_ym] * before_death_filter
    return df_exposure_table

# 死亡情報の加入者情報への付与
# ============================

def add_death_flag_and_adjust_exposure(npa_obs_ym, df_info, df_death_table):
    """
    加入者情報データ df_info に死亡情報を追加
    """
    df_info['death'] = 0
    df_info['death'] = df_info['death'].astype('int8')
    df_info.loc[(df_death_table[npa_obs_ym].sum(axis=1) == 1), 'death'] = 1
    #            --------------------------------------
    #                     死亡レコードがある者
    return df_info


def adjust_end_obs_ym(npa_obs_ym, df_info, df_exposure_table, df_death_table):
    """
    df_info, df_exposure_table, df_death_table の間で
    観察終了年月が整合的となるよう、観察終了年月を調整する。
    """
    # 当月のエクスポージャが 1、翌月のエクスポージャが 0 の時
    # 当月が観察終了年月となる。
    # エクスポージャを 1 ヶ月ずらした差分が 1 となる年月が観察最終年月。
    df_exp_dif = df_exposure_table[npa_obs_ym] \
               - df_exposure_table[npa_obs_ym].shift(-1, axis=1)
    # ただし分析終了年月の翌月は存在しないため None となる。
    # 分析終了年月 (2019/12) にエクスポージャが 1 の場合
    # 分析終了年月年月が観察終了年月となる。
    df_exp_dif[npa_obs_ym[-1]] = df_exposure_table[npa_obs_ym[-1]]
    # 差分が -1 の箇所を 0 に変更。
    df_exp_dif[df_exp_dif < 0] = 0
    # エクスポージャが全て 0 となった加入者を除外
    exposure_exists = (df_exposure_table[npa_obs_ym].sum(axis=1) > 0)
    df_info = df_info[exposure_exists].copy()
    df_exposure_table = df_exposure_table[exposure_exists]
    # 擬似生成で誕生年月が観察開始以後にある場合が生じてしまった場合、
    # 現実にはあり得ないため、誕生年月を観察開始年月に変更。
    b = (df_info['start_obs_ym'] < df_info['birth_ym'])
    df_info.loc[b, 'birth_ym'] = df_info.loc[b, 'start_obs_ym'].copy()
    df_info.loc[b, 'birth_t'] = df_info.loc[b, 'start_obs_t'].copy()
    # エクスポージャの最終年月を観察終了年月とし、
    # 本来の加入者情報データにあるべきでない、
    # 時刻 t に関する情報を削除し、 カラムを並べ替える。
    # エクスポージャの最終年月を観察終了年月とする。
    for i in df_exposure_table.index:
        end_of_exp = (df_exp_dif.loc[i, npa_obs_ym] == 1)
        df_info.loc[i, 'end_obs_ym'] = npa_obs_ym[end_of_exp][0]
    # 時刻 t に関する情報を削除し、カラムを並べ替える。
    info_cols = ['iid', 'sex', 'family', 'birth_ym',
                 'start_obs_ym', 'end_obs_ym', 'death']
    return df_info[info_cols]


def create_df_info(df_birth, df_mortality, start_study_t, end_study_t,
                   mu, N, family_ratio):
    """
    df_info を作成する
    """
    df_info = create_birth_and_obs(df_birth, start_study_t, end_study_t,
                                   mu, N, family_ratio)
    df_info = truncate_obs_period(df_info, start_study_t, end_study_t)
    df_info = change_dtypes(df_info)
    npa_obs_ym = create_npa_obs_ym(start_study_t, end_study_t)
    df_alb_table = create_alb_table(npa_obs_ym, df_info)
    df_mortality_table = create_mortality_table_frame(df_mortality,
                                                      npa_obs_ym, df_alb_table)
    df_mortality_table = create_mortality_table(df_mortality, npa_obs_ym,
                                                df_alb_table, df_mortality_table)
    df_exposure_table = create_exposure_table(npa_obs_ym, df_alb_table)
    df_death_table = create_death_table(npa_obs_ym, df_alb_table, df_mortality_table, df_exposure_table)
    df_death_table = adjust_death_table(npa_obs_ym, df_death_table)
    df_exposure_table = adjust_exposure_table(npa_obs_ym, df_exposure_table, df_death_table)
    df_info = add_death_flag_and_adjust_exposure(npa_obs_ym, df_info, df_death_table)
    df_info = adjust_end_obs_ym(npa_obs_ym, df_info, df_exposure_table, df_death_table)
    # メモリ節約
    del df_birth
    del df_mortality
    del df_alb_table
    del df_mortality_table
    del df_exposure_table
    del df_death_table
    gc.collect()
    return df_info


#############
##   1-3   ##
#############


def calc_LCL(l, d, e):
    """
    精緻な信頼区間、下限 (Lower Confidence Limit) を計算
    """
    from scipy.stats import norm
    # 正規分布の累積分布関数の逆関数
    z = norm.ppf(1 - e / 2)
    delta = z**2 * (d + z**2 / 4 - d**2 / l)
    return (d + z**2 / 2 - delta**(0.5)) / (l + z**2)


def calc_UCL(l, d, e):
    """
    精緻な信頼区間、上限 (Upper Confidence Limit) を計算
    """
    from scipy.stats import norm
    z = norm.ppf(1 - e / 2)
    delta = z**2 * (d + z**2 / 4 - d**2 / l)
    return (d + z**2 / 2 + delta**(0.5)) / (l + z**2)

#############
##   1-4   ##
#############

# 観察死亡率の推定
# ================


def convert_ym_to_t_in_info(df_info):
    """
    df_info の文字列型の年月を数値型に変換
    """
    col_ts  = ['start_obs_t',  'end_obs_t',  'birth_t']
    col_yms = ['start_obs_ym', 'end_obs_ym', 'birth_ym']
    for (col_t, col_ym) in zip(col_ts, col_yms):
        df_info[col_t] = df_info[col_ym].apply(ym_to_t)
    return df_info


def create_alb_table_from_info(npa_obs_ym, df_info):
    """
    ALB テーブルを作成
    """
    df_alb_table = df_info.copy()
    npa_birth_t = np.array(df_info['birth_t'])
    for ym in npa_obs_ym:
        t = ym_to_t(ym)
        df_alb_table[ym] = (t - npa_birth_t) // 1
        df_alb_table[ym] = df_alb_table[ym].astype('int8')
    return df_alb_table


def create_exposure_table_from_alb_table(npa_obs_ym, df_alb_table):
    """
    月次エクスポージャテーブルを作成
    """
    df_exposure_table = df_alb_table.copy()
    for ym in npa_obs_ym:
        # Boolean values for Start_obs_t
        bs = (df_exposure_table['start_obs_t'] <= ym_to_t(ym))
        # Boolean values for End_obs_t
        be = (ym_to_t(ym) <= df_exposure_table['end_obs_t'])
        df_exposure_table[ym] = (bs & be) * 1
    return df_exposure_table


def create_event_table_from_info(npa_obs_ym, df_alb_table, event):
    """
    月次イベントテーブルを作成。
    event = 'death' で月次死亡テーブルを作成。
    """
    N = len(df_alb_table)
    df_event_table = df_alb_table.copy()
    df_event_table[npa_obs_ym] = np.zeros((N, len(npa_obs_ym)))
    i_of_events = df_event_table[df_event_table[event] == 1].index
    for i in i_of_events:
        event_ym = df_event_table.loc[i, 'end_obs_ym']
        df_event_table.loc[i, event_ym] = 1
    return df_event_table


def count_exposure_and_event(npa_obs_ym, df_alb_table,
                             df_exposure_table, df_event_table):
    """
    ALB ごとにエクスポージャと死亡数を集計し df_summary を作成
    """
    N = len(df_alb_table)
    cols = ['sex', 'alb', 'exposure', 'event']
    df_summary = pd.DataFrame(np.zeros((200, 4)), columns=cols)
    df_summary['sex'] = np.concatenate([np.repeat('M', 100),
                                        np.repeat('F', 100)])
    df_summary['alb'] = np.concatenate([np.arange(100), np.arange(100)])
    types = {'alb': 'int8', 'exposure': 'int32', 'event': 'int8'}
    df_summary = df_summary.astype(types)
    for sex in ['M', 'F']:
        for alb in np.arange(100):
            # print(sex, alb)
            sex_filter = (df_alb_table['sex'] == sex).values.reshape(N, 1) * 1
            alb_filter = (df_alb_table[npa_obs_ym] == alb).values * 1
            i_smry = (sex == 'F') * 100 + alb
            df_summary.loc[i_smry, 'exposure'] = (
                    df_exposure_table[npa_obs_ym].values
                    * sex_filter * alb_filter
                    ).sum().sum()
            df_summary.loc[i_smry, 'event'] = (df_event_table[npa_obs_ym].values
                                               * sex_filter
                                               * alb_filter).sum().sum()
    return df_summary


def estimate_rate(df_summary, e=0.05):
    """
    df_summary に月次死亡率の期待値と 95%信頼区間を追加
    """
    df_summary['obs_rate'] = df_summary['event'] / df_summary['exposure']
    df_summary['LCL'] = calc_LCL(df_summary['exposure'],
                                        df_summary['event'], e)
    df_summary['UCL'] = calc_UCL(df_summary['exposure'],
                                        df_summary['event'], e)
    return df_summary


def add_true_mortality(df_mortality, df_summary):
    """
    df_summary に真の死亡率を追加
    """
    df_summary['true_rate'] = np.zeros(len(df_summary))
    for sex in ['M', 'F']:
        for alb in np.arange(100):
            i_smry = (sex == 'F') * 100 + alb
            df_summary.loc[i_smry, 'true_rate'] \
                = get_mortality(df_mortality, sex, alb)
    return df_summary
