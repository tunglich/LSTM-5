import numpy as np
import imp
import talib
import talib.abstract as abstract
import pandas as pd
import copy
from . import data

data = data.Data()

def fundamental_features():
    
    T3900繼續營業部門稅前純益 = data.get('繼續營業單位稅前淨利（淨損）')
    T3970經常稅後淨利 = data.get('本期淨利（淨損）').fillna(data.get('本期稅後淨利（淨損）'))
    T2000權益總計 = data.get('權益總計').fillna(data.get('權益總額'))
    T3295營業毛利 = data.get('營業毛利（毛損）').fillna(data.get('營業毛利（毛損）淨額'))
    T3100營業收入淨額 = data.get("營業收入合計")
    T3700營業外收入及支出 = data.get("營業外收入及支出合計").fillna(data.get("營業外損益合計"))
    T3910所得稅費用 = (data.get("所得稅費用（利益）合計").fillna(data.get("本期所得稅費用（利益）"))
        .fillna(data.get("當期所得稅費用（利益）"))
        .fillna(data.get("所得稅費用（利益）淨額"))
        .fillna(data.get("所得稅費用（利益）"))
        .fillna(-data.get("所得稅（費用）利益").fillna(data.get("所得稅利益（費用）")).dropna(how='all', axis=1))
    )
    T3300營業費用 = data.get("營業費用合計")
    T7211折舊 = data.get('折舊費用', table='cash_flows')
    T7212攤提 = data.get('攤銷費用', table='cash_flows')

    T3920繼續營業部門純益 = T3900繼續營業部門稅前純益 - T3910所得稅費用
    T0170存貨 = data.get("存貨合計").fillna(data.get("存貨"))
    股本 = data.get('股本合計')
    T0400不動產廠房設備合計 = (data.get("不動產、廠房及設備合計")
                      .fillna(data.get("不動產、廠房及設備"))
                      .fillna(data.get("不動產及設備合計"))
                      .fillna(data.get('不動產、廠房及設備淨額'))
                      .fillna(data.get("不動產及設備－淨額"))
                     )
    T0960非流動資產 = data.get("非流動資產合計").fillna(data.get("非流動資產淨額"))
    T1800非流動負債 = data.get("非流動負債合計")
    T1100流動負債 = data.get("流動負債合計").fillna(data.get("流動負債總額"))
    T0100流動資產 = data.get("流動資產合計")
    T3501財物成本 = data.get('財務成本').fillna(data.get("財務成本淨額"))
    T0010資產總額 = T0100流動資產 + T0960非流動資產
    T3971本期綜合損益總額 = data.get("本期綜合損益總額").fillna(data.get("綜合損益總額"))
    T3395營業利益 = data.get('營業利益（損失）').fillna(data.get('營業利益'))
    T2403稅前息前折舊前淨利 = T3900繼續營業部門稅前純益 + T3501財物成本 + T7211折舊 + T7212攤提
    T2402稅前息前淨利 = T3900繼續營業部門稅前純益 + T3501財物成本
    R202用人費用率 = data.get("管理費用合計").fillna(data.get("管理費用"))
    T3356研究發展費 = data.get("研究發展費用").fillna(data.get("研究發展費用合計")).fillna(data.get("委託研究費"))
    T7210營運現金流 = data.get("營運產生之現金流入（流出）")
    T3950歸屬母公司淨利 = data.get("歸屬於母公司業主之權益合計")
    T1000負債總額 = data.get("負債總額").fillna(data.get("負債總計"))
    T3200營業成本 = data.get("營業成本合計")
    T7324取得不動產廠房及設備 = (data.get("取得不動產、廠房及設備")
                      .fillna(data.get("取得不動產及設備")))
    #2018年(含)以後所得稅率為20% 
    #2010年(含)至2017年(含)期間所得稅率為17% 
    #2009年(含)以前所得稅率為25% 

    import pandas as pd
    所得稅率 = pd.DataFrame(0.2, index=T3501財物成本.index, columns=T3501財物成本.columns)
    所得稅率.loc[:'2017'] = 17
    所得稅率.loc[:'2009'] = 25

    # 獲利能力指標

    R101_ROA稅後息前 = T3920繼續營業部門純益 + T3501財物成本 * (1 - 所得稅率) / (T0010資產總額 + T0010資產總額.shift()) * 2 * 100
    R11V_ROA綜合損益 = T3971本期綜合損益總額 / (T0010資產總額 + T0010資產總額.shift()) * 2 * 100
    R103_ROE稅後 =  T3920繼續營業部門純益 /(T2000權益總計 + T2000權益總計.shift()).abs() * 2 * 100
    R11U_ROE綜合損益 =  (T3971本期綜合損益總額/((T2000權益總計+T2000權益總計.shift())/2)) * 100

    R145_稅前息前折舊前淨利率 = T2403稅前息前折舊前淨利/T3100營業收入淨額*100
    R105_營業毛利率 = T3295營業毛利 / T3100營業收入淨額 *100
    R106_營業利益率 = T3395營業利益 / T3100營業收入淨額 *100
    R107_稅前淨利率 = T3900繼續營業部門稅前純益 / T3100營業收入淨額 *100
    R108_稅後淨利率 = T3970經常稅後淨利 / T3100營業收入淨額 *100
    R112_業外收支營收率 = T3700營業外收入及支出 / T3100營業收入淨額 * 100
    R179_貝里比率 = T3295營業毛利 / T3300營業費用 * 100

    # 成本費用率指標

    R203_研究發展費用率 = T3356研究發展費 / T3300營業費用 * 100
    R205_現金流量比率 = T7210營運現金流 / T1100流動負債 *100
    R207_稅率 = (T3910所得稅費用 / T3900繼續營業部門稅前純益 *100)
    R207_稅率[(T3900繼續營業部門稅前純益 <= 0) | (T3910所得稅費用 <= 0)] = 0

    # 償債能力指標
    R304_每股營業額 = T3100營業收入淨額/股本 * 10
    R305_每股營業利益 = T3395營業利益 / 股本 * 10
    R303_每股現金流量 = T7210營運現金流 / 股本 * 10
    R306_每股稅前淨利 = T3900繼續營業部門稅前純益 / 股本 * 10
    R314_每股綜合損益 = T3971本期綜合損益總額 / 股本 * 10
    R315_每股稅前淨利 = T3900繼續營業部門稅前純益 / 股本 * 10
    R316_每股稅後淨利 = T3950歸屬母公司淨利 / 股本 * 10
    R504_總負債除總淨值 = T1000負債總額 / T2000權益總計 * 100
    R505_負債比率 = T1000負債總額 / T0010資產總額 * 100
    R506_淨值除資產 = T2000權益總計 / T0010資產總額 * 100


    # 成長率指標
    def 成長率(s):
        return (s / s.shift(4) - 1) * 100

    R401_營收成長率 = 成長率(T3100營業收入淨額)
    R402_營業毛利成長率 = 成長率(T3295營業毛利)
    R403_營業利益成長率 = 成長率(T3395營業利益)
    R404_稅前淨利成長率 = 成長率(T3900繼續營業部門稅前純益)
    R405_稅後淨利成長率 = 成長率(T3970經常稅後淨利)
    R406_經常利益成長率 = 成長率(T3920繼續營業部門純益)
    R408_資產總額成長率 = 成長率(T0010資產總額)
    R409_淨值成長率  = 成長率(T2000權益總計)

    # 償債能力指標
    R501_流動比率 = T0100流動資產 / T1100流動負債 *100
    R502_速動比率 = (T0100流動資產 - T0170存貨) / T1100流動負債 *100
    R503_利息支出率 = T3501財物成本 /(T3920繼續營業部門純益 + T3501財物成本 *(1-所得稅率)) *100


    R678_營運資金 = T0100流動資產 - T1100流動負債
    R607_總資產週轉次數 = T3100營業收入淨額 / T0010資產總額.rolling(4).sum()
    R610_存貨週轉率 = T3200營業成本 / T0170存貨.rolling(4).sum()
    R612_固定資產週轉次數 = T3100營業收入淨額 - T0400不動產廠房設備合計
    R613_淨值週轉率次 = T3100營業收入淨額 / T2000權益總計.rolling(4).sum()
    R69B_自由現金流量 = T3970經常稅後淨利 + T7211折舊 + T7212攤提 + T7324取得不動產廠房及設備 +((
        T0100流動資產 - T1100流動負債) - (T0100流動資產.shift() - T1100流動負債.shift()))
    fundamental_data = {
        
        # 獲利能力指標
        'R101_ROA稅後息前': R101_ROA稅後息前,
        'R11V_ROA綜合損益': R11V_ROA綜合損益,
        'R103_ROE稅後': R103_ROE稅後,
        'R11U_ROE綜合損益': R11U_ROE綜合損益,
        'R145_稅前息前折舊前淨利率': R145_稅前息前折舊前淨利率,
        'R105_營業毛利率': R105_營業毛利率,
        'R106_營業利益率': R106_營業利益率,
        'R107_稅前淨利率': R107_稅前淨利率,
        'R108_稅後淨利率': R108_稅後淨利率,
        'R112_業外收支營收率': R112_業外收支營收率,
        'R179_貝里比率': R179_貝里比率,
        
        # 成本費用率指標
        'R203_研究發展費用率': R203_研究發展費用率,
        'R205_現金流量比率': R205_現金流量比率,
        'R207_稅率': R207_稅率,
        
        # 償債能力指標
        'R304_每股營業額': R304_每股營業額,
        'R305_每股營業利益': R305_每股營業利益,
        'R303_每股現金流量': R303_每股現金流量,
        'R306_每股稅前淨利': R306_每股稅前淨利,
        'R314_每股綜合損益': R314_每股綜合損益,
        'R315_每股稅前淨利': R315_每股稅前淨利,
        'R316_每股稅後淨利': R316_每股稅後淨利,
        'R504_總負債除總淨值': R504_總負債除總淨值,
        'R505_負債比率': R505_負債比率,
        'R506_淨值除資產': R506_淨值除資產,
        
        # 成長率指標
        'R401_營收成長率': R401_營收成長率,
        'R402_營業毛利成長率': R402_營業毛利成長率,
        'R403_營業利益成長率': R403_營業利益成長率,
        'R404_稅前淨利成長率': R404_稅前淨利成長率,
        'R405_稅後淨利成長率': R405_稅後淨利成長率,
        'R406_經常利益成長率': R406_經常利益成長率,
        'R408_資產總額成長率': R408_資產總額成長率,
        'R409_淨值成長率': R409_淨值成長率,
        
        # 償債能力指標
        'R501_流動比率': R501_流動比率,
        'R502_速動比率': R502_速動比率,
        'R503_利息支出率': R503_利息支出率,
        'R678_營運資金': R678_營運資金,
        'R607_總資產週轉次數': R607_總資產週轉次數,
        'R610_存貨週轉率': R610_存貨週轉率,
        'R612_固定資產週轉次數': R612_固定資產週轉次數,
        'R613_淨值週轉率次': R613_淨值週轉率次,
        'R69B_自由現金流量': R69B_自由現金流量,
    }
    return pd.DataFrame({name: df.unstack() for name, df in fundamental_data.items()})    


def talib_features(df:pd.DataFrame, indicators:list, multipliers:list, preprocessing:dict=None):

    features = {}

    for name in indicators:

        f = getattr(abstract, name)
        org_params = dict(f.parameters)
        outputs = dict

        for m in multipliers:

            params = copy.copy(org_params)
            params = {k:int(v*m) for k,v in params.items()}

            values = f(df, **params)
            
            if preprocessing and name in preprocessing:
                values = preprocessing[name](values)

            if isinstance(values, pd.Series):
                features[name + '_' + str(params)] = values
            elif isinstance(values, pd.DataFrame):
                for output_name, series in values.items():
                    features[name+'_'+output_name+'_'+str(params)] = series
            else:
                raise Exception("features of %s should be pd.Series or pd.DataFrame. Get %s instead" % 
                               (name, type(values)))

        f.parameters = org_params

    return pd.DataFrame(features)

def add_profit_prediction(dataset):
    
    dates = sorted(list(set(dataset.reset_index()['date'])))
    
    adj_open = data.get_adj('開盤價')
    
    tomorrow_adj_open = adj_open.shift(-1)
    tomorrow_adj_open = tomorrow_adj_open.reindex(dates, method='bfill')
    
    
    p = (tomorrow_adj_open.shift(-1) / tomorrow_adj_open)
    dataset['return'] = p.unstack()
    
def add_rank_prediction(dataset):
    
    dates = sorted(list(set(dataset.reset_index()['date'])))
    
    adj_open = data.get_adj('開盤價')
    
    tomorrow_adj_open = adj_open.shift(-1)
    tomorrow_adj_open = tomorrow_adj_open.reindex(dates, method='bfill')
    
    
    p = (tomorrow_adj_open.shift(-1) / tomorrow_adj_open).rank(axis=1, pct=True)
    dataset['rank'] = p.unstack()
    
def add_feature(dataset, feature_name, feature):
    feature_series = feature.reindex(dataset.index.levels[1], method='ffill').unstack().reindex(dataset.index)
    dataset[feature_name] = feature_series
    
    
def drop_extreme_case(dataset, feature_names, thresh=0.01):
    
    extreme_cases = pd.Series(False, index=dataset.index)
    for f in feature_names:
        tf = dataset[f]
        extreme_cases = extreme_cases | (tf < tf.quantile(thresh)) | (tf > tf.quantile(1-thresh))
    dataset = dataset[~extreme_cases]
    return dataset


    