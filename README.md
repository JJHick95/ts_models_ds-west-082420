
# Time Series Models

## TOC

- [Random Walk Model](#random_walk)
- [Improvements on the FSM](#arma_models)
    - [Autoregressive Model](#ar_model)
    - [Moving Average Model](#ma_model)
    - [ACF and PACF](#acf_pacf)
- [auto_arima](#auto_arima)


If we think back to our lecture on the bias-variance tradeoff, a perfect model is not possible.  There will always be noise (inexplicable error).

If we were to remove all of the patterns from our time series, we would be left with white noise, which is written mathematically as:

$$\Large Y_t =  \epsilon_t$$

The error term is randomly distributed around the mean, has constant variance, and no autocorrelation.

We know this data has no true pattern governing its fluctuations (because we coded it with a random function).

Any attempt at a model would be fruitless.  The next point in the series could be any value, completely independent of the previous value.

We will assume that the timeseries data that we are working with is more than just white noise.

# Train Test Split

Let's reimport our chicago gun crime data, and prepare it in the same manner as the last notebook.


Train test split for a time series is a little different than what we are used to.  Because **chronological order matters**, we cannot randomly sample points in our data.  Instead, we cut off a portion of our data at the end, and reserve it as our test set.


```python
end_of_train_index = round(ts_weekly.shape[0]*.8)
```


```python
# Define train and test sets according to the index found above
train = ts_weekly[:end_of_train_index]
test = ts_weekly[end_of_train_index:]
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
# pass period=1 argument to the shift method called at the end of train.
random_walk = train.shift(-1)

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
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(train[1:], random_walk.dropna())
rmse = np.sqrt(mse)
print(rmse)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-6-ea3fc3c5ed2b> in <module>
          1 #__SOLUTION__
          2 from sklearn.metrics import mean_squared_error
    ----> 3 mse = mean_squared_error(train[1:], random_walk.dropna())
          4 rmse = np.sqrt(mse)
          5 print(rmse)


    NameError: name 'train' is not defined



```python
# By hand
residuals = random_walk - train
mse_rw = (residuals.dropna()**2).sum()/len(residuals-1)
np.sqrt(mse_rw.sum())
```

<a id='arma_models'></a>

# Improvement on FSM: Autoregressive and Moving Average Models

Lets plot the residuals from the random walk model.


```python
residuals = random_walk - train

plt.plot(residuals.index, residuals)
plt.plot(residuals.index, residuals.rolling(30).std())
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~/anaconda3/lib/python3.7/site-packages/pandas/core/ops/array_ops.py in na_arithmetic_op(left, right, op, str_rep)
        148     try:
    --> 149         result = expressions.evaluate(op, str_rep, left, right)
        150     except TypeError:


    ~/anaconda3/lib/python3.7/site-packages/pandas/core/computation/expressions.py in evaluate(op, op_str, a, b, use_numexpr)
        207     if use_numexpr:
    --> 208         return _evaluate(op, op_str, a, b)
        209     return _evaluate_standard(op, op_str, a, b)


    ~/anaconda3/lib/python3.7/site-packages/pandas/core/computation/expressions.py in _evaluate_numexpr(op, op_str, a, b)
        120     if result is None:
    --> 121         result = _evaluate_standard(op, op_str, a, b)
        122 


    ~/anaconda3/lib/python3.7/site-packages/pandas/core/computation/expressions.py in _evaluate_standard(op, op_str, a, b)
         69     with np.errstate(all="ignore"):
    ---> 70         return op(a, b)
         71 


    ~/anaconda3/lib/python3.7/site-packages/pandas/core/ops/roperator.py in rsub(left, right)
         12 def rsub(left, right):
    ---> 13     return right - left
         14 


    TypeError: unsupported operand type(s) for -: 'NoneType' and 'float'

    
    During handling of the above exception, another exception occurred:


    TypeError                                 Traceback (most recent call last)

    <ipython-input-105-7cd9e2d2a6be> in <module>
          1 #__SOLUTION__
    ----> 2 residuals = random_walk - train
          3 
          4 plt.plot(residuals.index, residuals)
          5 plt.plot(residuals.index, residuals.rolling(30).std())


    ~/anaconda3/lib/python3.7/site-packages/pandas/core/ops/common.py in new_method(self, other)
         62         other = item_from_zerodim(other)
         63 
    ---> 64         return method(self, other)
         65 
         66     return new_method


    ~/anaconda3/lib/python3.7/site-packages/pandas/core/ops/__init__.py in wrapper(left, right)
        501         lvalues = extract_array(left, extract_numpy=True)
        502         rvalues = extract_array(right, extract_numpy=True)
    --> 503         result = arithmetic_op(lvalues, rvalues, op, str_rep)
        504 
        505         return _construct_result(left, result, index=left.index, name=res_name)


    ~/anaconda3/lib/python3.7/site-packages/pandas/core/ops/array_ops.py in arithmetic_op(left, right, op, str_rep)
        195     else:
        196         with np.errstate(all="ignore"):
    --> 197             res_values = na_arithmetic_op(lvalues, rvalues, op, str_rep)
        198 
        199     return res_values


    ~/anaconda3/lib/python3.7/site-packages/pandas/core/ops/array_ops.py in na_arithmetic_op(left, right, op, str_rep)
        149         result = expressions.evaluate(op, str_rep, left, right)
        150     except TypeError:
    --> 151         result = masked_arith_op(left, right, op)
        152 
        153     return missing.dispatch_fill_zeros(op, left, right, result)


    ~/anaconda3/lib/python3.7/site-packages/pandas/core/ops/array_ops.py in masked_arith_op(x, y, op)
        110         if mask.any():
        111             with np.errstate(all="ignore"):
    --> 112                 result[mask] = op(xrav[mask], y)
        113 
        114     result, _ = maybe_upcast_putmask(result, ~mask, np.nan)


    ~/anaconda3/lib/python3.7/site-packages/pandas/core/ops/roperator.py in rsub(left, right)
         11 
         12 def rsub(left, right):
    ---> 13     return right - left
         14 
         15 


    TypeError: unsupported operand type(s) for -: 'NoneType' and 'float'


If we look at the rolling standard deviation of our errors, we can see that the performance of our model varies at different points in time.

That is a result of the trends in our data.

In the previous notebook, we ended by indicating most Time Series models expect to be fed **stationary** data.  Were able to make our series stationary by differencing our data.

Let's repeat that process here. 

In order to make our life easier, we will use statsmodels to difference our data via the **ARIMA** class. 

We will break down what ARIMA is shortly, but for now, we will focus on the I, which stands for **integrated**.  A time series which has been be differenced to become stationary is said to have been integrated[1](https://people.duke.edu/~rnau/411arim.htm). 

There is an order parameter in ARIMA with three slots: (p, d, q).  d represents our order of differencing, so putting a one there in our model will apply a first order difference.





```python
# create an ARIMA object, and pass the training set and order (0,1,0) as arguments
rw = ARIMA(train, (0,1,0)).fit()
# Add typ='levels' argument to predict on original scale
rw.predict(typ='levels')
```




    2014-01-12    31.187619
    2014-01-19    18.987619
    2014-01-26    24.559048
    2014-02-02    24.559048
    2014-02-09    22.273333
                    ...    
    2019-02-10    24.559048
    2019-02-17    27.844762
    2019-02-24    30.701905
    2019-03-03    30.844762
    2019-03-10    28.701905
    Freq: W-SUN, Length: 270, dtype: float64



We can see that the differenced predictions (d=1) are just a random walk

By removing the trend from our data, we assume that our data passes a significance test that the mean and variance are constant throughout.  But it is not just white noise.  If it were, our models could do no better than random predictions around the mean.  

Our task now is to find **more patterns** in the series.  

We will focus on the data points near to the point in question.  We can attempt to find patterns to how much influence previous points in the sequence have. 

If that made you think of regression, great! What we will be doing is assigning weights, like our betas, to previous points.

<a id='ar_model'></a>

# The Autoregressive Model (AR)

Our next attempt at a model is the autoregressive model, which is a timeseries regressed on its previous values

### $y_{t} = \phi_{0} + \phi_{1}y_{t-1} + \varepsilon_{t}$

The above formula is a first order autoregressive model (AR1), which finds the best fit weight $\phi$ which, multiplied by the point previous to a point in question, yields the best fit model. 


```python

# A linear regression model with fed with a shifted timeseries 
# yields coefficients which approximate ARIMA

from sklearn.linear_model import LinearRegression

lr_ar_1 = LinearRegression()
lr_ar_1.fit(pd.DataFrame(train.diff().dropna().shift(1).dropna()), np.array(train[1:].diff().dropna()))
lr_ar_1.coef_
lr_ar_1.intercept_
lr_ar_1.predict(pd.DataFrame(train[1:].diff().dropna())) + train[2:]



```




    2014-01-19    23.011317
    2014-01-26    24.601717
    2014-02-02    22.968474
    2014-02-09    18.641207
    2014-02-16    16.458602
                    ...    
    2019-02-10    26.949503
    2019-02-17    29.928984
    2019-02-24    30.846652
    2019-03-03    29.356266
    2019-03-10    28.132108
    Freq: W-SUN, Length: 269, dtype: float64



In our ARIMA model, the **p** variable of the order (p,d,q) represents the AR term.  For a first order AR model, we put a 1 there.


```python
# fit an 1st order differenced AR 1 model with the ARIMA class, 
# Pass train and order (1,1,0)
ar_1 = ARIMA(train, (1,1,0)).fit()

# We put a typ='levels' to convert our predictions to remove the differencing performed.
ar_1.predict(typ='levels')
```




    2014-01-12    31.198536
    2014-01-19    22.556496
    2014-01-26    22.944514
    2014-02-02    24.569538
    2014-02-09    22.950500
                    ...    
    2019-02-10    26.027893
    2019-02-17    26.896905
    2019-02-24    29.879050
    2019-03-03    30.813585
    2019-03-10    29.337404
    Freq: W-SUN, Length: 270, dtype: float64



The ARIMA class comes with a nice summary table.  

But, as you may notice, the output does not include RMSE.

It does include AIC. We briefly touched on AIC with linear regression.  It is a metric with a strict penalty applied to we used models with too many features.  A better model has a lower AIC.

Let's compare the first order autoregressive model to our Random Walk.

Our AIC for the AR(1) model is lower than the random walk, indicating improvement.  

Let's stick with the RMSE, so we can compare to the hold out data at the end.

Checks out. RMSE is lower as well.

Autoregression, as we said before, is a regression of a time series on lagged values of itself.  

From the summary, we see the coefficient of the 1st lag:


```python
ar_1.arparams
```




    array([-0.29167099])



We come close to reproducing this coefficients with linear regression, with slight differences due to how statsmodels performs the regression. 

We can also factor in more than just the most recent point.
$$\large y_{t} = \phi_{0} + \phi_{1}y_{t-1} + \phi_{2}y_{t-2}+ \varepsilon_{t}$$

We refer to the order of our AR model by the number of lags back we go.  The above formula refers to an **AR(2)** model.  We put a 2 in the p position of the ARIMA class order


```python
# Fit a 1st order difference 2nd order ARIMA model 
ar_2 = ARIMA(train, (2,1,0)).fit()

y_hat_ar_2 = ar_2.predict(typ='levels')
```

<a id='ma_model'></a>

# Moving Average Model (MA)

The next type of model is based on error.  The idea behind the moving average model is to make a prediciton based on how far off we were the day before.

$$\large Y_t = \mu +\epsilon_t + \theta * \epsilon_{t-1}$$

The moving average model is a pretty cool idea. We make a prediction, see how far off we were, then adjust our next prediction by a factor of how far off our pervious prediction was.

In our ARIMA model, the q term of our order (p,d,q) refers to the MA component. To use one lagged error, we put 1 in the q position.



```python
# Reproduce the prediction for 2014-06-01

#prior true value
print(train['2014-05-25'])
prior_train = train['2014-05-25']
# prior prediction
print(y_hat['2014-05-25'])
prior_y_hat = y_hat['2014-05-25']

(prior_train - prior_y_hat) * ma_1.params['ma.L1.y'] + ma_1.params['const']

```

    31.142857142857142
    33.38774771043386





    33.56600793267698



We can replacate all of the y_hats with the code below:

Let's look at the 1st order MA model with a 1st order difference


```python
ma_1 = ARIMA(train, (0,1,1)).fit()
rmse_ma1 = np.sqrt(mean_squared_error(train[1:], ma_1.predict(typ='levels')))
print(rmse_rw)
print(rmse_ar1)
print(rmse_ar2)
print(rmse_ma1)
```

    4.697542272439977
    4.502200686486479
    4.2914159019782545
    4.3253589327542565


It performs better than a 1st order AR, but worse than a 2nd order

Just like our AR models, we can lag back as far as we want. Our MA(2) model would use the past two lagged terms:

$$\large Y_t = \mu +\epsilon_t + \theta_{t-1} * \epsilon_{t-1} + \theta_2 * \epsilon_{t-2}$$

and our MA term would be two.

# ARMA

We don't have to limit ourselves to just AR or MA.  We can use both AR terms and MA terms.

for example, an ARMA(2,1) model is given by:

 $$\large Y_t = \mu + \phi_1 Y_{t-1}+\phi_2 Y_{t-2}+ \theta \epsilon_{t-1}+\epsilon_t$$


# Pair (5 minutes)


```python
arma_22 = ARIMA(train, (2,1,2)).fit()
rmse_22 = np.sqrt(mean_squared_error(train[1:], arma_22.predict(typ='levels')))
```


```python
print(rmse_rw)
print(rmse_ar1)
print(rmse_ar2)
print(rmse_ma1)
print(rmse_ma2)
print(rmse_22)
```

    4.697542272439977
    4.502200686486479
    4.2914159019782545
    4.3253589327542565
    4.261197978485248
    4.225492682757378


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

## ACF

The autocorrelation plot of our time series is simply a version of the correlation plots we used in linear regression.  In place of the independent features we include the lags. 



Unlike the PACF, shows both the direct and indirect correlation between lags. In other words, in the above plot, there is significant correlation between the day of interest 12 lags back.  But this assumes that lag 1 is correlated to lag 2, lag 2 to 3, and so forth.  

The error terms in a Moving Average process are built progressively by adjusting the error of the previous moment in time.  Each error term therein includes the indirect effect of the error term before it. Because of this, we can choose the MA term based on how many significant lags appear in the ACF.

![acf_ma1](img/ma1_acf.png)

Let's bring in the pacf and acf from statsmodels.

The above autocorrelation shows that there is correlation between lags up to about 12 weeks back.  

When Looking at the ACF graph for the original data, we see a strong persistent correlation with higher order lags. This is evidence that we should take a **first difference** of the data to remove this autocorrelation.

This makes sense, since we are trying to capture the effect of recent lags in our ARMA models, and with high correlation between distant lags, our models will not come close to the true process.

The shaded area of the graph is the convidence interval.  When the correlation drops into the shaded area, that means there is no longer statistically significant correlation between lags.

This autocorrelation plot can now be used to get an idea of a potential MA term.  Our differenced series shows negative significant correlation at lag of 1 suggests adding 1 MA term.  There is also a statistically significant 2nd, term, so adding another MA is another possibility.


> If the ACF of the differenced series displays a sharp cutoff and/or the lag-1 autocorrelation is negative--i.e., if the series appears slightly "overdifferenced"--then consider adding an MA term to the model. The lag at which the ACF cuts off is the indicated number of MA terms. [Duke](https://people.duke.edu/~rnau/411arim3.htm#signatures)

The plots above suggest that we should try a 1st order differenced MA(1) or MA(2) model on our weekly gun offense data.


The ACF can be used to identify the possible structure of time series data. That can be tricky going forward as there often isn’t a single clear-cut interpretation of a sample autocorrelation function.

Luckily, we have auto_arima

<a id='auto_arima'></a>

# auto_arima

Luckily for us, we have a Python package that will help us determine optimal terms.

According to auto_arima, our optimal model is a first order differenced, AR(1)MA(2) model.

Let's plot our training predictions.

# Test

Now that we have chosen our parameters, let's try our model on the test set.

Our predictions on the test set certainly leave something to be desired.  

Let's take another look at our autocorrelation function of the original series.

Let's increase the lags

There seems to be a wave of correlation at around 50 lags.
What is going on?

![verkempt](https://media.giphy.com/media/l3vRhBz4wCpJ9aEuY/giphy.gif)

# SARIMA

Looks like we may have some other forms of seasonality.  Luckily, we have SARIMA, which stands for Seasonal Auto Regressive Integrated Moving Average.  That is a lot.  The statsmodels package is actually called SARIMAX.  The X stands for exogenous, and we are only dealing with endogenous variables, but we can use SARIMAX as a SARIMA.


A seasonal ARIMA model is classified as an **ARIMA(p,d,q)x(P,D,Q)** model, 

    **p** = number of autoregressive (AR) terms 
    **d** = number of differences 
    **q** = number of moving average (MA) terms
     
    **P** = number of seasonal autoregressive (SAR) terms 
    **D** = number of seasonal differences 
    **Q** = number of seasonal moving average (SMA) terms

Let's try the third from the bottom, ARIMA(1, 1, 1)x(0, 1, 1, 52)12 - AIC:973.5518935855749

# Forecast

Lastly, let's predict into the future.

To do so, we refit to our entire training set.
