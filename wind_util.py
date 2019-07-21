# -*- coding: utf-8 -*-

import pandas as pd
import QuantLib as ql
import datetime

def wset_to_df(wset):
    df = pd.DataFrame()
    for i in range(len(wset.Fields)):
        df[wset.Fields[i]] = wset.Data[i]
    df['DATE'] = pd.to_datetime(wset.Times[0])
    return df

def wss_to_df(wss):
    df = pd.DataFrame()
    for i in range(len(wss.Fields)):
        df[wss.Fields[i]] = wss.Data[i]
    df['CODE'] = wss.Codes
    return df

def topy_date(date):
    if type(date) == ql.Date:
        return datetime.date(date.year, date.month, date.dayOfMonth)
    else:
        return date

def toql_date(date):
    if type(date) == datetime.date:
        return ql.Date(date.day, date.month, date.year)
    else:
        return date