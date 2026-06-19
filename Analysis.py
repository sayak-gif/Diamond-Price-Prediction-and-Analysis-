

# import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

data = pd.read_csv('/kaggle/input/dmddata/diamonds.csv')

data

data.info()

data.describe()

data.isnull().sum()

x = data[['carat','cut','color','clarity','depth','table','x','y','z']]
y = data['price']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state = 42)

#defining order from lowest to highest quality for cut and clarity
ordr_cut = ['Fair','Good','Very Good','Premium','Ideal']
ordr_clarity = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

ord_encd_cut = OrdinalEncoder(categories=[ordr_cut])
ord_encd_clarity = OrdinalEncoder(categories= [ordr_clarity])

x_train['cut'] = ord_encd_cut.fit_transform(x_train[['cut']])
x_test['cut'] = ord_encd_cut.transform(x_test[['cut']])

x_train['clarity'] = ord_encd_clarity.fit_transform(x_train[['clarity']])
x_test['clarity'] = ord_encd_clarity.transform(x_test[['clarity']])

ord_encd = OneHotEncoder(sparse_output=False)

color_data_xtr = pd.DataFrame(data = ord_encd.fit_transform(x_train[['color']]), columns = ord_encd.get_feature_names_out(), index = x_train.index)

color_data_xtr

color_data_xts = pd.DataFrame(data = ord_encd.transform(x_test[['color']]), columns = ord_encd.get_feature_names_out(), index = x_test.index)

color_data_xts

x_train_en = pd.concat([x_train.drop('color',axis=1),color_data_xtr],axis=1)

x_train_en

x_test_en = pd.concat([x_test.drop('color',axis=1),color_data_xts],axis=1)

x_test_en

mns  = StandardScaler()

x_tr_scl = mns.fit_transform(x_train_en)

x_ts_scl = mns.transform(x_test_en)

x_train_scl = pd.DataFrame(data= x_tr_scl,index =x_train.index, columns = mns.get_feature_names_out())

x_train_scl

x_test_scl = pd.DataFrame(data= x_ts_scl,index =x_test.index, columns = mns.get_feature_names_out())

x_test_scl

dt = pd.concat([x_train_scl,y_train],axis=1)

xtr_correlat = dt.corr()

dt.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(data= dt.corr(),cmap = 'coolwarm', annot = True,fmt=".2f",annot_kws={"size": 10})
plt.title('Correlation Matrix')
plt.show()

lr = LinearRegression()

lr.fit(x_train_scl,y_train)

pred = lr.predict(x_test_scl)

r2_lr = r2_score(y_test,pred)
print('r2 score: ',r2_lr)

mae_lr = mean_absolute_error(y_test,pred)
print('mean absolute error: ',mae_lr)

mse_lr = mean_squared_error(y_test,pred)
print('mean squared error: ',mse_lr)

ridg = Ridge()

ridg.fit(x_train_scl,y_train)

rg_pred = ridg.predict(x_test_scl)

r2_ridg = r2_score(y_test,rg_pred)
print('r2_score: ',r2_ridg)
mea_ridg = mean_absolute_error(y_test,rg_pred)
print('Mean Absolute Error: ',mea_ridg)
mse_ridg = mean_squared_error(y_test,rg_pred)
print('Mean Squared error: ',mse_ridg)

ls = Lasso()

ls.fit(x_train_scl,y_train)

ls_pred = ls.predict(x_test_scl)

r2_ls = r2_score(y_test,ls_pred)
print('r2_score: ',r2_ls)
mae_ls = mean_absolute_error(y_test,ls_pred)
print('Mean absolute error: ',mae_ls)
mse_ls = mean_squared_error(y_test,ls_pred)
print('Mean squared error: ',mse_ls)

ENr = ElasticNet()

ENr.fit(x_train_scl,y_train)

enr_pred = ENr.predict(x_test_scl)

r2_enr = r2_score(y_test,enr_pred)
print('r2_score: ',r2_enr)
mae_enr = mean_absolute_error(y_test,enr_pred)
print('Mean absolute error: ',mae_enr)
mse_enr = mean_squared_error(y_test,enr_pred)
print('Mean squared error: ',mse_enr)

knr = KNeighborsRegressor()

knr.fit(x_train_scl,y_train)

knr_pred = ENr.predict(x_test_scl)

r2_knr = r2_score(y_test,enr_pred)
print('r2_score: ',r2_knr)
mae_knr = mean_absolute_error(y_test,enr_pred)
print('Mean absolute error: ',mae_knr)
mse_knr = mean_squared_error(y_test,enr_pred)
print('Mean squared error: ',mse_knr)

dtr = DecisionTreeRegressor()

dtr.fit(x_train_scl,y_train)

dtr_pred = dtr.predict(x_test_scl)

r2_dtr = r2_score(y_test,enr_pred)
print('r2_score',r2_dtr)
mae_dtr = mean_absolute_error(y_test,enr_pred)
print('Mean absolute error',mae_dtr)
mse_dtr = mean_squared_error(y_test,enr_pred)
print('Mean squared error',mse_dtr)

rfr = RandomForestRegressor()

rfr.fit(x_train_scl,y_train)

p = rfr.predict(x_test_scl)

r2_rfr = r2_score(y_test,p)
print('r2_score: ',r2_rfr)
mae_rfr = mean_absolute_error(y_test,p)
print('Mean absolute error: ',mae_rfr)
mse_rfr = mean_squared_error(y_test,p)
print('Mean squared error: ',mse_rfr)

fr_imp = rfr.feature_importances_
frtr = x_train_scl.columns

# Create a DataFrame
importance_df = pd.DataFrame({'Feature': frtr, 'Importance': fr_imp})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from Random Forest')
plt.gca().invert_yaxis()
plt.show()

adr = AdaBoostRegressor()

adr.fit(x_train_scl,y_train)

adr_pred = adr.predict(x_test_scl)

r2_adr = r2_score(y_test,adr_pred) 
print('r2_score: ',r2_adr)
mae_adr = mean_absolute_error(y_test,adr_pred) 
print('Mean absolute error: ',mae_adr)
mse_adr = mean_squared_error(y_test,adr_pred) 
print('Mean squared error: ',mse_adr)

svr = SVR()

svr.fit(x_train_scl,y_train)

pred_svr = svr.predict(x_test_scl)

r2_svr = r2_score(y_test,pred_svr)
print('r2_score: ',r2_svr)
mae_svr = mean_absolute_error(y_test,pred_svr)
print('Mean absolute error: ',mae_svr)
mse_svr = mean_squared_error(y_test,pred_svr)
print('Mean squared error: ',mse_svr)

import optuna

def objective(trial):

    n_estimators = trial.suggest_int('n_estimators',10,200)
    #criterion = trial.suggest_categorical('criterion',["squared_error", "absolute_error", "friedman_mse", "poisson"])
    max_depth = trial.suggest_int('max_depth',2,20)
    min_samples_split = trial.suggest_int('min_samples_split',2,20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf',1,20)

    rndf = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split)
    rndf.fit(x_train_scl,y_train)
    rf_p = rndf.predict(x_test_scl)
    accuracy = r2_score(y_test,rf_p)
    
    return accuracy

study_rf = optuna.create_study(direction='maximize')

study_rf.optimize(objective,n_trials = 150)

mse = np.mean((y_test - p)**2)
rmse = np.sqrt(mse_rfr)
std_dev = np.std(y_test)

print(f"RMSE: {rmse}")
print(f"Target Standard Deviation: {std_dev}")

rn = RandomForestRegressor(n_estimators =178,max_depth = 19,min_samples_split= 9, min_samples_leaf =  1)

rn.fit(x_train_scl,y_train)

pp = rn.predict(x_test_scl)

r2_score(y_test,pp)

mean_absolute_error(y_test,pp)

mean_squared_error(y_test,pp)

new_xtr = pd.DataFrame(data = [[4.43,2.0,4.0,50.8,39.0,33.29,21.89,10.98,0.0,0.0,0.0,0.0,1.0,0.0,0.0]],
                       columns = ['carat','cut','clarity','depth','table','x','y','z','color_D','color_E','color_F','color_G','color_H','color_I','color_J'])

new_xtr

new_xts_scl = pd.DataFrame(mns.transform(new_xtr),columns = mns.get_feature_names_out())

prd = rn.predict(new_xts_scl)

prd



# Data to populate the table
# Replace these values with the actual metrics you've calculated for each algorithm
results_data = {
    "Algorithm": ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression",
                  "KNN Regression", "Decision Tree Regression", "Random Forest Regression", "AdaBoost Regression", "Support Vector regression"],
    "R2_Score": [0.909969, 0.909969, 0.90993, 0.839108, 0.839108, 0.839108,  0.981029,  0.9106437, 0.42078],  # Replace with actual R2 scores
    "Mean Absolute Error": [782.07808, 782.19929, 783.336519, 1060.20752, 1060.20752, 1060.20752, 269.923276, 888.093195, 1499.58817],  # Replace with actual MAEs
    "Mean Squared Error": [ 1404089.347886, 1404118.604529, 1404701.608846, 2509260.74634, 2509260.74634, 2509260.74634, 295859.71611, 1393596.28211, 9033385.12825]  # Replace with actual MSEs
}

# Creating DataFrame
results_df = pd.DataFrame(results_data)

# Set the 'Algorithm' column as the index
results_df.set_index("Algorithm", inplace=True)



# Display the table


pd.options.display.float_format = True
results_df

results_df
