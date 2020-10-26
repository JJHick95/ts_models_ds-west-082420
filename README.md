
# Time Series Models

## TOC

- [Random Walk Model](#random_walk)
- [Improvements on the FSM](#arma_models)
    - [Autoregressive Model](#ar_model)
    - [Moving Average Model](#ma_model)
    - [ACF and PACF](#acf_pacf)
- [auto_arima](#auto_arima)



```python
from src.student_caller import one_random_student, three_random_students
from src.student_list import student_first_names

```


```python
"What is the formula for the coeffient of determination, aka R-Squared?"
one_random_student(student_first_names)
```

If we think back to our lecture on the bias-variance tradeoff, a perfect model is not possible.  There will always be noise (inexplicable error).

If we were to remove all of the patterns from our time series, we would be left with white noise, which is written mathematically as:

$$\Large Y_t =  \epsilon_t$$

The error term is randomly distributed around the mean, has constant variance, and no autocorrelation.


```python
# Let's make some white noise!
from random import gauss as gs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

rands = []

# Create a 1000 cycle for-loop
for _ in range(1000):
    # Append 1000 random numbers from the standard normal distribution
    rands.append(gs(0, 1))
    
series = pd.Series(rands)
```


```python
fig, ax = plt.subplots(figsize=(10, 7))

# Create an array with the same shape as our random gaussian series.
X = np.linspace(-10, 10, 1000)

ax.plot(X, series)
ax.set_title('White Noise Time Series');
```

We know this data has no true pattern governing its fluctuations (because we coded it with a random function).

Any attempt at a model would be fruitless.  The next point in the series could be any value, completely independent of the previous value.

We will assume that the timeseries data that we are working with is more than just white noise.

# Train Test Split

Let's reimport our chicago gun crime data, and prepare it in the same manner as the last notebook.



```python
# The process below is the same preprocessing we performed in the TS data notebook.

ts = pd.read_csv('data/Gun_Crimes_Heat_Map.csv', index_col='Date', parse_dates=True)

# ts_minute = ts.groupby('Date').count()['ID']
daily_count = ts.resample('D').count()['ID']

# Remove outliers
daily_count = daily_count[daily_count < 90]

# Create date_range object with all dates between beginning and end dates
ts_dr = pd.date_range(daily_count.index[0], daily_count.index[-1])

# Create empty series the length of the date range
ts_daily = np.empty(shape=len(ts_dr))
ts_daily = pd.Series(ts_daily)

# Add datetime index to series and fill with daily count
ts_daily = ts_daily.reindex(ts_dr)
ts_daily = ts_daily.fillna(daily_count)

# There are missing values which we fill via linear interpolation
ts_daily = ts_daily.interpolate()

# Downsample to the week level
ts_weekly = ts_daily.resample('W').mean()

```


```python
fig, ax = plt.subplots()
ax.plot(ts_weekly)
ax.set_title("Weekly Reports of Gun Offenses in Chicago")
```

Train test split for a time series is a little different than what we are used to.  Because **chronological order matters**, we cannot randomly sample points in our data.  Instead, we cut off a portion of our data at the end, and reserve it as our test set.


```python
# find the index which allows us to split off 20% of the data
# Calculate the 80% mark by multiplying the row index of the shape attribute by .8
# Use the built in round function to find the nearest integer
end_of_train_index = None
```


```python
# Define train and test sets according to the index found above
train = None
test = None
```


```python
fig, ax = plt.subplots()
ax.plot(train)
ax.plot(test)
ax.set_title('Train test Split');
```

We will now set aside our test set, and build our model on the train.

<a id='random_walk'></a>

# Random Walk

A good first attempt at a model for a time series would be to simply predict the next data point with the point previous to it.  

We call this type of time series a random walk, and it is written mathematically like so.

$$\Large Y_t = Y_{t-1} + \epsilon_t$$

$\epsilon$ represents white noise error.  The formula indicates that the difference between a point and a point before it is white noise.

$$\Large Y_t - Y_{t-1}=  \epsilon_t$$

This makes sense, given one way we described making our series stationary was by applying a difference of a lag of 1.

Let's make a simple random walk model for our Gun Crime dataset.

WE can perform this with the shift operator, which shifts our time series according to periods argument.


```python
# The prediction for the next day is the original series shifted to the future by one day.
# pass period= 1 argument to the shift method called at the end of train.
random_walk = None

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

train[0:30].plot(ax=ax, c='r', label='original')
random_walk[0:30].plot(ax=ax, c='b', label='shifted')
ax.set_title('Random Walk')
ax.legend();
```

We will use a random walk as our **FSM**.  

That being the case, let's use a familiar metric, RMSE, to assess its strength.


## Individual Exercise (3 min): Calculate RMSE


```python
# Either proceed by hand or using the mean_squared_error function
# from sklearn.metrics module

```

<a id='arma_models'></a>

# Improvement on FSM: Autoregressive and Moving Average Models

Lets plot the residuals from the random walk model.


```python
residuals = None

plt.plot(residuals.index, residuals)
plt.plot(residuals.index, residuals.rolling(30).std())
```

If we look at the rolling standard deviation of our errors, we can see that the performance of our model varies at different points in time.


```python
plt.plot(residuals.index, residuals.rolling(30).var())
```

That is a result of the trends in our data.

In the previous notebook, we ended by indicating most Time Series models expect to be fed **stationary** data.  Were able to make our series stationary by differencing our data.

Let's repeat that process here. 

In order to make our life easier, we will use statsmodels to difference our data via the **ARIMA** class. 

We will break down what ARIMA is shortly, but for now, we will focus on the I, which stands for **integrated**.  A time series which has been be differenced to become stationary is said to have been integrated [1](https://people.duke.edu/~rnau/411arim.htm). 

There is an order parameter in ARIMA with three slots: (p, d, q).  d represents our order of differencing, so putting a one there in our model will apply a first order difference.





```python
from statsmodels.tsa.arima_model import ARIMA
```


```python
# create an arima_model object, and pass the training set and order (0,1,0) as arguments
# then call the .fit() method
rw = ARIMA()

# Just like our other models, we can now use the predict method. 
# Add typ='levels' argument to predict on original scale

```

We can see that the differenced predictions (d=1) are just a random walk


```python
random_walk
```


```python
# We put a typ='levels' to convert our predictions to remove the differencing performed.
y_hat = rw.predict(typ='levels')

# RMSE is equivalent to Random Walk RMSE
np.sqrt(mean_squared_error(train[1:], y_hat))
```

By removing the trend from our data, we assume that our data passes a significance test that the mean and variance are constant throughout.  But it is not just white noise.  If it were, our models could do no better than random predictions around the mean.  

Our task now is to find **more patterns** in the series.  

We will focus on the data points near to the point in question.  We can attempt to find patterns to how much influence previous points in the sequence have. 

If that made you think of regression, great! What we will be doing is assigning weights, like our betas, to previous points.

<a id='ar_model'></a>

# The Autoregressive Model (AR)

Our next attempt at a model is the autoregressive model, which is a timeseries regressed on its previous values

### $y_{t} = \phi_{0} + \phi_{1}y_{t-1} + \varepsilon_{t}$

The above formula is a first order autoregressive model (AR1), which finds the best fit weight $\phi$ which, multiplied by the point previous to a point in question, yields the best fit model. 

In our ARIMA model, the **p** variable of the order (p,d,q) represents the AR term.  For a first order AR model, we put a 1 there.


```python
# fit an 1st order differenced AR 1 model with the ARIMA class, 
# Pass train and order (1,1,0)
ar_1 = ARIMA(train, (1,1,0)).fit()

# predict while passing typ='levels' as an argument
ar_1.predict(typ='levels')
```

The ARIMA class comes with a nice summary table.  


```python
ar_1.summary()
```

But, as you may notice, the output does not include RMSE.

It does include AIC. We briefly touched on AIC with linear regression.  It is a metric with a strict penalty applied to we used models with too many features.  A better model has a lower AIC.

Let's compare the first order autoregressive model to our Random Walk.


```python
rw_model = ARIMA(train, (0,1,0)).fit()
rw_model.summary()
```


```python
print(f'Random Walk AIC: {rw.aic}')
print(f'AR(1,1,0) AIC: {ar_1.aic}' )

```

Our AIC for the AR(1) model is lower than the random walk, indicating improvement.  

Let's stick with the RMSE, so we can compare to the hold out data at the end.


```python
y_hat_ar1 = ar_1.predict(typ='levels')
rmse_ar1 = np.sqrt(mean_squared_error(train[1:], y_hat_ar1))
rmse_ar1
```


```python
y_hat_rw = rw.predict(typ='levels')
rmse_rw = np.sqrt(mean_squared_error(train[1:], y_hat_rw))
```


```python
print(rmse_rw)
print(rmse_ar1)
```

Checks out. RMSE is lower as well.

Autoregression, as we said before, is a regression of a time series on lagged values of itself.  

From the summary, we see the coefficient of the 1st lag:


```python
# print the arparams attribute of the fit ar_1 model
```

We come close to reproducing this coefficients with linear regression, with slight differences due to how statsmodels performs the regression. 


```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(np.array(train.diff().shift(1).dropna()).reshape(-1,1), train[1:].diff().dropna())
print(lr.coef_)
```

We can also factor in more than just the most recent point.
$$\large y_{t} = \phi_{0} + \phi_{1}y_{t-1} + \phi_{2}y_{t-2}+ \varepsilon_{t}$$

We refer to the order of our AR model by the number of lags back we go.  The above formula refers to an **AR(2)** model.  We put a 2 in the p position of the ARIMA class order


```python
# Fit a 1st order difference 2nd order ARIMA model 
ar_2 = None

y_hat_ar_2 = ar_2.predict(typ='levels')
```


```python
ar_2.summary()
```


```python
rmse_ar2 = np.sqrt(mean_squared_error(train[1:], y_hat_ar_2))
print(rmse_ar2)
```


```python
print(rmse_rw)
print(rmse_ar1)
print(rmse_ar2)
```

<a id='ma_model'></a>

# Moving Average Model (MA)

The next type of model is based on error.  The idea behind the moving average model is to make a prediciton based on how far off we were the day before.

$$\large Y_t = \mu +\epsilon_t + \theta * \epsilon_{t-1}$$

The moving average model is a pretty cool idea. We make a prediction, see how far off we were, then adjust our next prediction by a factor of how far off our pervious prediction was.

In our ARIMA model, the q term of our order (p,d,q) refers to the MA component. To use one lagged error, we put 1 in the q position.



```python
ma_1 = ARIMA(train, (0,0,1)).fit()
y_hat = ma_1.predict(typ='levels')
y_hat[20:30]
```


```python
ma_1.summary()
```


```python
# Reproduce the prediction for 2014-06-01

# Set prior train to the train value on 2014-05-25
prior_train = None

# Set prior prediction to the y_hat on 2014-05-25
prior_y_hat = None

# calculate the next days prediction by multiplying the ma.L1.y coefficient 
# by the error: i.e. prior_train - prior_hat
# and adding the mean of the training set, or the ma_1.params['const']


# Value should match below
print(y_hat['2014-06-01'])
```

We can replacate all of the y_hats with the code below:


```python
y_hat_manual = ((train - y_hat)*ma_1.maparams[0] +ma_1.params['const']).shift()[20:30]
y_hat_manual
```

Let's look at the 1st order MA model with a 1st order difference


```python
ma_1 = ARIMA(train, <replace_with_correct_order>).fit()
rmse_ma1 = None

```


```python

print(rmse_rw)
print(rmse_ar1)
print(rmse_ar2)
print(rmse_ma1)
```

It performs better than a 1st order AR, but worse than a 2nd order

Just like our AR models, we can lag back as far as we want. Our MA(2) model would use the past two lagged terms:

$$\large Y_t = \mu +\epsilon_t + \theta_{t-1} * \epsilon_{t-1} + \theta_2 * \epsilon_{t-2}$$

and our MA term would be two.


```python
ma_2 = ARIMA(train, (0,1,2)).fit()
y_hat = ma_2.predict(typ='levels')
rmse_ma2 = np.sqrt(mean_squared_error(train[1:], y_hat))
```


```python
print(rmse_rw)
print(rmse_ar1)
print(rmse_ar2)
print(rmse_ma1)
print(rmse_ma2)
```

# ARMA

We don't have to limit ourselves to just AR or MA.  We can use both AR terms and MA terms.

for example, an ARMA(2,1) model is given by:

 $$\large Y_t = \mu + \phi_1 Y_{t-1}+\phi_2 Y_{t-2}+ \theta \epsilon_{t-1}+\epsilon_t$$


# Pair (5 minutes)


```python
# With a partner, find the best performing combination 
# of autoregressive, moving average, and differencing terms
# by comparing the rmse score of different orders fit on the training set.

# Compare to our previous fits
print(rmse_rw)
print(rmse_ar1)
print(rmse_ar2)
print(rmse_ma1)
print(rmse_ma2)

```

<a id='acf_pacf'></a>

# ACF and PACF

We have been able to reduce our AIC by chance, adding fairly random p,d,q terms.

We have two tools to help guide us in these decisions: the **autocorrelation** and **partial autocorrelation** functions.

## PACF

In general, a partial correlation is a **conditional correlation**. It is the  amount of correlation between a variable and a lag of itself that is not explained by correlations at all lower-order-lags.  

If $Y_t$ is correlated with $Y_{t-1}$, and $Y_{t-1}$ is equally correlated with $Y_{t-2}$, then we should also expect to find correlation between $Y_t$ and $Y_{t-2}$.   

Thus, the correlation at lag 1 "propagates" to lag 2 and presumably to higher-order lags. The partial autocorrelation at lag 2 is therefore the difference between the actual correlation at lag 2 and the expected correlation due to the propagation of correlation at lag 1.



For an AR process, we run a linear regression on lags according to the order of the AR process. The coefficients calculated factor in the influence of the other variables.   

Since the PACF shows the direct effect of previous lags, it helps us choose AR terms.  If there is a significant positive value at a lag, consider adding an AR term according to the number that you see.

Some rules of thumb: 

    - A sharp drop after lag "k" suggests an AR-K model.
    - A gradual decline suggests an MA.

![ar1_pacf](img/ar1_pacf.png)


```python
# source: https://online.stat.psu.edu/stat510/book/export/html/665
```

## ACF

The autocorrelation plot of our time series is simply a version of the correlation plots we used in linear regression.  In place of the independent features we include the lags. 




```python
df = pd.DataFrame(train)
df.columns = ['lag_0']
df['lag_1'] = train.shift()
df['lag_2'] = train.shift(2)
df['lag_3'] = train.shift(3)
df['lag_4'] = train.shift(4)
df['lag_5'] = train.shift(5)
df.corr()
```


```python
list(df.corr()['lag_0'].index)
plt.bar(list(df.corr()['lag_0'].index), list(df.corr()['lag_0']))
```

The error terms in a Moving Average process are built progressively by adjusting the error of the previous moment in time.  Each error term therein includes the indirect effect of the error term before it. Because of this, we can choose the MA term based on how many significant lags appear in the ACF.

![acf_ma1](img/ma1_acf.png)

Let's bring in the pacf and acf from statsmodels.


```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
```


```python
# Original data

plot_acf(train);
```

The above autocorrelation shows that there is correlation between lags up to about 12 weeks back.  

When Looking at the ACF graph for the original data, we see a strong persistent correlation with higher order lags. This is evidence that we should take a **first difference** of the data to remove this autocorrelation.

This makes sense, since we are trying to capture the effect of recent lags in our ARMA models, and with high correlation between distant lags, our models will not come close to the true process.

The shaded area of the graph is the convidence interval.  When the correlation drops into the shaded area, that means there is no longer statistically significant correlation between lags.


```python
plot_pacf(train.diff().dropna());
```


```python
plot_acf(train.diff().dropna());
```

This autocorrelation plot can now be used to get an idea of a potential MA term.  Our differenced series shows negative significant correlation at lag of 1 suggests adding 1 MA term.  There is also a statistically significant 2nd, term, so adding another MA is another possibility.


> If the ACF of the differenced series displays a sharp cutoff and/or the lag-1 autocorrelation is negative--i.e., if the series appears slightly "overdifferenced"--then consider adding an MA term to the model. The lag at which the ACF cuts off is the indicated number of MA terms. [Duke](https://people.duke.edu/~rnau/411arim3.htm#signatures)

The plots above suggest that we should try a 1st order differenced MA(1) or MA(2) model on our weekly gun offense data.


The ACF can be used to identify the possible structure of time series data. That can be tricky going forward as there often isn’t a single clear-cut interpretation of a sample autocorrelation function.

Luckily, we have auto_arima

<a id='auto_arima'></a>

# auto_arima

Luckily for us, we have a Python package that will help us determine optimal terms.


```python
from pmdarima import auto_arima

auto_arima(train, start_p=0, start_q=0, max_p=6, max_q=3, seasonal=False, trace=True)
```

According to auto_arima, our optimal model is a first order differenced, AR(1)MA(2) model.

Let's plot our training predictions.


```python
aa_model = ARIMA(train, (1,1,2)).fit()
y_hat_train = aa_model.predict(typ='levels')

fig, ax = plt.subplots()
ax.plot(y_hat_train)
ax.plot(train)
```


```python
# Let's zoom in:

fig, ax = plt.subplots()
ax.plot(y_hat_train[50:70])
ax.plot(train[50:70])
```


```python
aa_model.summary()
```


```python
rmse = np.sqrt(mean_squared_error(train[1:], y_hat_train))
rmse
```

# Test

Now that we have chosen our parameters, let's try our model on the test set.


```python
test
```


```python
y_hat_test = aa_model.predict(start=test.index[0], end=test.index[-1],typ='levels')

fig, ax = plt.subplots()
ax.plot(y_hat_test)

```


```python
fig, ax = plt.subplots()
ax.plot(y_hat_test)
ax.plot(test)
```


```python
np.sqrt(mean_squared_error(test, y_hat_test))
```

Our predictions on the test set certainly leave something to be desired.  

Let's take another look at our autocorrelation function of the original series.


```python
plot_acf(ts_weekly);
```

Let's increase the lags


```python
plot_acf(ts_weekly, lags=75);
```

There seems to be a wave of correlation at around 50 lags.
What is going on?

![verkempt](https://media.giphy.com/media/l3vRhBz4wCpJ9aEuY/giphy.gif)

# SARIMA

Looks like we may have some other forms of seasonality.  Luckily, we have SARIMA, which stands for Seasonal Auto Regressive Integrated Moving Average.  That is a lot.  The statsmodels package is actually called SARIMAX.  The X stands for exogenous, and we are only dealing with endogenous variables, but we can use SARIMAX as a SARIMA.



```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
```

A seasonal ARIMA model is classified as an **ARIMA(p,d,q)x(P,D,Q)** model, 

    **p** = number of autoregressive (AR) terms 
    **d** = number of differences 
    **q** = number of moving average (MA) terms
     
    **P** = number of seasonal autoregressive (SAR) terms 
    **D** = number of seasonal differences 
    **Q** = number of seasonal moving average (SMA) terms


```python
import itertools
p = q = range(0, 2)
pdq = list(itertools.product(p, [1], q))
seasonal_pdq = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, [1], q))]
print('Examples of parameter for SARIMA...')
for i in pdq:
    for s in seasonal_pdq:
        print('SARIMAX: {} x {}'.format(i, s))

```


```python
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod =SARIMAX(train,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{} - AIC:{}'.format(param,param_seasonal,results.aic))
        except: 
            print('hello')
            continue
```

Let's try the third from the bottom, ARIMA(1, 1, 1)x(0, 1, 1, 52)12 - AIC:973.5518935855749


```python
sari_mod =SARIMAX(train,order=(1,1,1),
                  seasonal_order=(0,1,1,52),
                  enforce_stationarity=False,
                  enforce_invertibility=False).fit()

```


```python
y_hat_train = sari_mod.predict(typ='levels')
y_hat_test = sari_mod.predict(start=test.index[0], end=test.index[-1],typ='levels')

fig, ax = plt.subplots()
ax.plot(train)
ax.plot(test)
ax.plot(y_hat_train)
ax.plot(y_hat_test)
```


```python
# Let's zoom in on test
fig, ax = plt.subplots()

ax.plot(test)
ax.plot(y_hat_test)

```


```python
np.sqrt(mean_squared_error(test, y_hat_test))
```

# Forecast

Lastly, let's predict into the future.

To do so, we refit to our entire training set.


```python
sari_mod =SARIMAX(ts_weekly,order=(1,1,1),seasonal_order=(0,1,1,52),enforce_stationarity=False,enforce_invertibility=False).fit()
```


```python
forecast = sari_mod.forecast(steps = 52)
```


```python
fig, ax = plt.subplots()

ax.plot(ts_weekly)
ax.plot(forecast)
ax.set_title('Chicago Gun Crime Predictions\n One Year out')
```
