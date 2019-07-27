# -*- coding: utf-8 -*-

import math
import datetime
import matplotlib.pyplot as plt
import wind_util as wu
from QuantLib import *

today = Date(25,7,2019)
Settings.instance().evaluationDate = today
shibor = USDLibor(Period(3,Months))
china = China()

# SHIBOR DISCOUNT CURVE

# addHoliday applys effect to all the object of the same category within the console session
china.addHoliday(Date(24,1,2019))
china.addHoliday(Date(25,1,2019))
china.addHoliday(Date(26,1,2019))
china.addHoliday(Date(27,1,2019))
china.addHoliday(Date(28,1,2019))
china.addHoliday(Date(29,1,2019))

helpers = [DepositRateHelper(QuoteHandle(SimpleQuote(rate)),Period(*tenor),
                             offset,China(),Following,False,Actual360())
            for rate, offset, tenor in [(0.02481,0,(1,Days)),
                                        (0.02561,1,(1,Weeks)),
                                        (0.02631,1,(2,Weeks)),
                                        (0.02590,1,(1,Months)),
                                        (0.02625,1,(3,Months))]]

helpers += [SwapRateHelper(QuoteHandle(SimpleQuote(rate/100)),Period(*tenor),china,
                           Annual,Unadjusted,Actual360(),shibor, QuoteHandle(), Period(1, Days))
             for rate, tenor in [(2.7562, (6, Months)),(2.8350, (9, Months)),
                                 (2.8875, (1, Years)), (3.0200, (2, Years)), 
                                 (3.1400, (3, Years)), (3.2575, (4, Years)), (3.3662, (5,Years))] ]

shibor_curve = PiecewiseLogCubicDiscount(0, China(), helpers, Actual365Fixed())
shibor_curve.enableExtrapolation()

print('\n'.join(['{0}\t{1:.4f}'.format(d,shibor_curve.zeroRate(d,Actual365Fixed(),Continuous).rate()*100) 
                for d in shibor_curve.dates()]))

today = shibor_curve.referenceDate()
end = today + Period(2,Years)
# better way to generate a series of dates than using schedule
dates = [Date(serial) for serial in range(today.serialNumber(),end.serialNumber()+1)]
shibor_fwd_1d = [shibor_curve.forwardRate(d,china.advance(d,30,Days),Actual360(),Simple).rate() for d in dates]
dates_ax = [wu.topy_date(i) for i in dates]
plt.figure()
plt.plot(dates_ax,shibor_fwd_1d)
plt.show()