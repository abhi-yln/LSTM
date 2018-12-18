# LSTM
# Time Series Analysis for Stock Prediction using LSTM
Stock buyer wants to decide when to buy stocks and when to sell them to gain profit. This is where time series modelling comes in. machine learning models that can look at the history of a sequence of data and correctly predict the future elements of the sequence are going to be.
Stock market prices are highly unpredictable and volatile. This means that there are no consistent patterns in the data that allows us to model stock prices over time near-perfectly.


## Introduction to LSTM: Making Stock Movement Predictions Far into the Future

Long Short-Term Memory (LSTM) models are extremely powerful time-series models. A LSTM can predict an arbitrary number of steps into the future. A LSTM module (or a cell) has 5 essential components which allows them to model both long-term and short-term data. 
* Cell state ($c_t$) - This represents the internal memory of the cell which stores both short term memory and long-term memories
* Hidden state ($h_t$) - This is output state information calculated w.r.t. current input, previous hidden state and current cell input which you eventually use to predict the future stock market prices. Additionally, the hidden state can decide to only retrive the short or long-term or both types of memory stored in the cell state to make the next prediction.
* Input gate ($i_t$) - Decides how much information from current input flows to the cell state
* Forget gate ($f_t$) - Decides how much information from the current input and the previous cell state flows into the current cell state
5. Output gate ($o_t$) - Decides how much information from the current cell state flows into the hidden state, so that if needed LSTM can only pick the long-term memories or short-term memories and long-term memories

And the equations for calculating each of these entities are as follows.

* $i_t = \sigma(W_{ix}x_t + W_{ih}h_{t-1}+b_i)$
* $\tilde{c}_t = \sigma(W_{cx}x_t + W_{ch}h_{t-1} + b_c)$
* $f_t = \sigma(W_{fx}x_t + W_{fh}h_{t-1}+b_f)$
* $c_t = f_t c_{t-1} + i_t \tilde{c}_t$
* $o_t = \sigma(W_{ox}x_t + W_{oh}h_{t-1}+b_o)$
* $h_t = o_t tanh(c_t)$


