from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
ddos_predictions = list()
history = new_count_df
for ddos in range(len(ddos_data)):
  model = ARIMA(history, order=(10,1,0))
  fitted_model = model.fit(disp=0)
  output =fitted_model.forecast()
         
   
pred = output[0]
ddos_predictions.append(pred)
error = mean_squared_error(ddos_data,ddos_predictions)
