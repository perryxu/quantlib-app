# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 21:21:27 2019

@author: Perry
"""

# Black Scholes Basics

import math
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
import QuantLib as ql
from scipy.stats import norm
from WindPy import w
from wind_util import *
  
def bs_npv_analytical_std(s,vol,k,t,quote,r,q=0,cat='c'):
    cat = cat.lower()[:1]
    t = t/365
    vol = vol/100
    r = r/100
    q = q/100
    d1 = (math.log(s/k) + (r-q+0.5*vol*vol)*t) / (vol*math.sqrt(t))
    d2 = d1 - vol*math.sqrt(t)
    if cat == 'c':
        return s*math.exp(-q*t)*norm.cdf(d1) - k*math.exp(-r*t)*norm.cdf(d2)
    elif cat == 'p':
        return k*math.exp(-r*t)*norm.cdf(-d2) - s*math.exp(-q*t)*norm.cdf(-d1) 
    else:
        raise ValueError('undefined option type')

def bs_delta_analytical_std(s,vol,k,t,quote,r,q=0,cat='c'):
    cat = cat.lower()[:1]
    t = t/365
    vol = vol/100
    r = r/100
    q = q/100
    d1 = (math.log(s/k) + (r-q+0.5*vol*vol)*t) / (vol*math.sqrt(t))
    if cat == 'c':
        return math.exp(-q*t)*norm.cdf(d1)
    elif cat == 'p':
        return -math.exp(-q*t)*norm.cdf(-d1) 
    else:
        raise ValueError('undefined option type')

def bs_gamma_analytical_std(s,vol,k,t,quote,r,q=0,cat='c'):
    cat = cat.lower()[:1]
    t = t/365
    vol = vol/100
    r = r/100
    q = q/100
    d1 = (math.log(s/k) + (r-q+0.5*vol*vol)*t) / (vol*math.sqrt(t))
    if cat == 'c' or cat == 'p':
        return math.exp(-q*t)*norm.pdf(d1)/(s*vol*math.sqrt(t))
    else:
        raise ValueError('undefined option type')

def bs_vega_analytical_std(s,vol,k,t,quote,r,q=0,cat='c'):
    cat = cat.lower()[:1]
    t = t/365
    vol = vol/100
    r = r/100
    q = q/100
    d1 = (math.log(s/k) + (r-q+0.5*vol*vol)*t) / (vol*math.sqrt(t))
    if cat == 'c' or cat == 'p':
        return s*math.exp(-q*t)*norm.pdf(d1)*math.sqrt(t)
    else:
        raise ValueError('undefined option type')

def bs_impvol_analytical_std(it = 100, **kwarg):
    kwarg['vol'] = 25
    for i in range(it):
        kwarg['vol'] -= 100*(bs_npv_analytical_std(**kwarg) - kwarg['quote'])/bs_vega_analytical_std(**kwarg)
    return kwarg['vol']

op1 = {'s':2.924,'vol':21.92,'k':2.9,'t':100.0,'r':3.5,'cat':'c','quote':0.16001699}
value_op1 = []
spot_op1 = np.arange(2.2,3.6,0.05)
for i in spot_op1:
    op1['s'] = i
    value_op1.append(bs_vega_analytical_std(**op1))
#plt.subplot(2,2,1)
#plt.plot(spot_op1, value_op1)


if __name__ == '__main__':
    if w.isconnected():
        pass
    else:
        w.start(waitTime = 60)

    spot = 2.924
    rundate = datetime.date(2019,7,18)
    optstatic = wset_to_df(w.wset("optionchain","date="+rundate.strftime("%Y%m%d")+";us_code=510050.SH;option_var=全部;call_put=全部"))
    optnpv = wss_to_df(w.wss(','.join(optstatic['option_code'].tolist()), "close,us_impliedvol","tradeDate="+rundate.strftime("%Y%m%d")+";priceAdj=U;cycle=D"))
#    optnpv.set_index('CODE',inplace=True)
    optshot = optstatic.merge(optnpv, how = 'left', left_on = 'option_code', right_on = 'CODE')
    optsave = optshot[(((optshot['call_put'] == '认购') * (optshot['strike_price'] >= (spot - 0.0)))
                    + ((optshot['call_put'] == '认沽') * (optshot['strike_price'] <= (spot + 0.0))))]
    optsurface = pd.pivot_table(optsave, values = 'US_IMPLIEDVOL', columns = ['strike_price'], index = ['expiredate'],
                             aggfunc = np.average, margins = False)
    optsurface.dropna(axis=1, inplace=True)    

    ts_rd_flat = ql.YieldTermStructureHandle(ql.FlatForward(toql_date(rundate), op1['r']/100, ql.Actual365Fixed()))
    ts_rq_flat = ql.YieldTermStructureHandle(ql.FlatForward(toql_date(rundate), 0, ql.Actual365Fixed()))
    
    optsurfaceT = ql.Matrix(len(optsurface.columns), len(optsurface.index))
    for i in range(optsurfaceT.rows()):
        for j in range(optsurfaceT.columns()):
            optsurfaceT[i][j] = optsurface.values[j][i]

    opt_var_surface = ql.BlackVarianceSurface(toql_date(rundate), ql.China(),
                                              [toql_date(rundate + datetime.timedelta(days=int(i))) for i in optsurface.index],
                                              optsurface.columns.tolist(),optsurfaceT,ql.Actual365Fixed())           

    opt_var_surface.setInterpolation('bicubic')
    opt_var_surface.enableExtrapolation()

    opt_locvol_surface = ql.LocalVolSurface(ql.BlackVolTermStructureHandle(opt_var_surface), ts_rd_flat, ts_rq_flat, spot)    
    opt_locvol_surface.enableExtrapolation()
    print("black vol: {0}\nloc vol: {1}".format(opt_var_surface.blackVol(0.9,2.83), opt_locvol_surface.localVol(0.9,2.83)))

    # to price a double no touch option value
    # 所有的upout downin的期权都是美式观察欧式交割的，行权方式选择European
    dnt = ql.DoubleBarrierOption(ql.DoubleBarrier.KnockOut,2.7,2.93,0.0,
                                 ql.CashOrNothingPayoff(ql.Option.Call,0.0,1.0),
                                 ql.EuropeanExercise(ql.Date(18,10,2019)))
    upnt = ql.BarrierOption(ql.Barrier.UpOut,2.93,0.0,
                                 ql.CashOrNothingPayoff(ql.Option.Call,0.0,1.0),
                                 ql.EuropeanExercise(ql.Date(18,10,2019)))
    downnt = ql.BarrierOption(ql.Barrier.DownOut,2.7,0.0,
                                 ql.CashOrNothingPayoff(ql.Option.Call,0.0,1.0),
                                 ql.EuropeanExercise(ql.Date(18,10,2019)))
    european_option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, 2.8),
                                       ql.EuropeanExercise(ql.Date(18,10,2019)))
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    bsm_process = ql.BlackScholesMertonProcess(spot_handle, ts_rq_flat, ts_rd_flat, 
                                               ql.BlackVolTermStructureHandle(opt_var_surface))
#    dnt.setPricingEngine(ql.BinomialBarrierEngine(bsm_process,'crr',400))
#    upnt.setPricingEngine(ql.BinomialBarrierEngine(bsm_process,'crr',400))
#    downnt.setPricingEngine(ql.BinomialBarrierEngine(bsm_process,'crr',400))

    dnt.setPricingEngine(ql.BinomialDoubleBarrierEngine(bsm_process,'crr',400))
    upnt.setPricingEngine(ql.BinomialBarrierEngine(bsm_process,'crr',400))
    downnt.setPricingEngine(ql.BinomialBarrierEngine(bsm_process,'crr',400))
    print(dnt.NPV())
    print(upnt.NPV())
    print(downnt.NPV())
#    bs_price = dnt.delta()
#npv_op1 = bs_npv_analytical_std(**op1)
#impvol = bs_impvol_analytical_std(**op1)
#delta_op1 = bs_delta_analytical_std(**op1)
#gamma_op1 = bs_gamma_analytical_std(**op1)
#vega_op1 = bs_vega_analytical_std(**op1)/100