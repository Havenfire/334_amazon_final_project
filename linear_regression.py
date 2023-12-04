import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

df = pd.read_csv('fullDF.csv')
df

'''
model 1:
linear regression model using discount, list price, stars, reviews, and revenue
'''



df_1=df[['stars', 'reviews', 'listPrice', 'discount','revenue']].copy()
# data preprocessing: drop observations with NA
df_1.dropna(inplace=True)

'''
selected variables: 'stars', 'reviews', 'listPrice', 'discount','revenue'
variables: 'category_id' and 'category_cluster' are not selected because there are too many categories. There is neither any order associated with ids. Thus they are not the most ideal choices for linear regression model
create correlation heatmap between all selected variables
We found very low level of correlation between between all variables
We do not need to worry about mulcollinearity.
However, all correlation are very low, which indicates that it might be unrealistic for us to build a linear model to predict stars and price.
'''

subset_df = df_1[['stars', 'reviews', 'listPrice', 'discount', 'revenue']]
correlation_matrix = subset_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

#model 1: use discount, list price, stars, and reviews to predict revenue
x_1 = df_1[['stars', 'reviews', 'listPrice', 'discount']]
y_1 = df_1['revenue']
x1_train, x1_test, y1_train, y1_test = train_test_split(x_1, y_1, test_size=0.2, random_state=104)
scaler = StandardScaler()
x1_train_scaled = scaler.fit_transform(x1_train)
x1_test_scaled = scaler.transform(x1_test)

linear_model = LinearRegression()
linear_model.fit(x1_train_scaled, y1_train)

print("Linear regression model details:")
print("Coefficients:", linear_model.coef_)
print(pd.DataFrame({"Variables": ['stars', 'reviews', 'listPrice', 'discount'], "Coefficients": linear_model.coef_}))
print("Intercept:", linear_model.intercept_)

linear_predictions = linear_model.predict(x1_test_scaled)
linear_rmse = np.sqrt(mean_squared_error(y1_test, linear_predictions))
linear_r2 = r2_score(y1_test, linear_predictions)
n = len(y1_test)
standard_error = linear_rmse / np.sqrt(n - 2)

print("RMSE:", linear_rmse)
print("Standard Error of the Regression:", standard_error)
print(f'Linear Regression R-squared: {linear_r2:.2f}')

# We also used Lasso regression model with the best parameter found through k-fold cross validation and grid search 
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(x1_train_scaled, y1_train)
lasso_predictions = lasso_model.predict(x1_test_scaled)
lasso_rmse = np.sqrt(mean_squared_error(y1_test, lasso_predictions))
print('Lasso Regression RMSE', lasso_rmse)
lasso_r2 = r2_score(y1_test, lasso_predictions)
print('Lasso Regression R-squared', lasso_r2)

# use k-fold to find best alpha for lasso regression
kf = KFold(n_splits=5, shuffle=True, random_state=104)
param_grid = {'alpha': np.logspace(-4, 2, 100)}
lasso_model = Lasso()
grid_search = GridSearchCV(lasso_model, param_grid, scoring='neg_mean_squared_error', cv=kf)
grid_search.fit(x1_train_scaled, y1_train)
best_alpha = grid_search.best_params_['alpha']
best_lasso_model = Lasso(alpha=best_alpha)
best_lasso_model.fit(x1_train_scaled, y1_train)
best_lasso_predictions = best_lasso_model.predict(x1_test_scaled)
best_lasso_rmse = np.sqrt(mean_squared_error(y1_test, best_lasso_predictions))
best_lasso_r2 = r2_score(y1_test, best_lasso_predictions)
print(f'Best Alpha: {best_alpha:.4f}')
print(f'Best Lasso Regression RMSE: {best_lasso_rmse:.2f}')
print(f'Best Lasso Regression R-squared: {best_lasso_r2:.2f}')

'''
conclusion: Results from linear regression models support the correlation matrix. It means that we need more information other than
stars, review, list price, and discount to predict revenue.

To improve our linear regression models, we can include information from 'title', 'category_id', and 'category_cluster'

To use 'category_id', and 'category_cluster', we can use one hot encoding to create a dummy variable for each category and each cluster.
However, simply adding those extra variables to linear regression model does not make sense. Having a different category should not be a simple addition or subtraction to predicted revenue.
Instead, a more realistic logic is that variables 'stars', 'reviews', 'listPrice', 'discount' have changes revenue at different scales given different goods category.
We thus want to create interactive terms between every category/cluster and the variables 'stars', 'reviews', 'listPrice', 'discount'.
However, due to the size of data and number of categories and clusters, this approach would generate gigabytes of data and thus failed.

In conclusion, linear regression models do not perform well as the linear relationship between the continuous variables are very weak.
Linear regression model is neither ideal when handeling this amount of different categories, or extracting information from text. 
'''

# #model 2 (failed): use category, discount, list price, stars, and reviews to predict revenue
# df_2=df[['category_id', 'stars', 'reviews', 'listPrice', 'discount','revenue']].copy()
# df_2.dropna(inplace=True)

# #one hot encoding to create category dummies.
# df_2_encoded = pd.get_dummies(df_2, columns=['category_id'], prefix='category')
# # create interactive terms
# interaction_terms = ['stars', 'reviews', 'listPrice', 'discount']
# for term in interaction_terms:
#     for category_col in df_2_encoded.columns[df_2_encoded.columns.str.startswith('category_')]:
#         interaction_name = f'{term}_{category_col}'
#         df_2_encoded[interaction_name] = df_2_encoded[term] * df_2_encoded[category_col]
     
# x_2 = df_2_encoded.drop(['revenue'], axis=1)
# y_2 = df_2_encoded['revenue']
# x2_train, x2_test, y2_train, y2_test = train_test_split(x_2, y_2, test_size=0.2, random_state=42)
# scaler = StandardScaler()
# x2_train_scaled = scaler.fit_transform(x2_train)
# x2_test_scaled = scaler.transform(x2_test)
# linear_model = LinearRegression()
# linear_model.fit(x2_train_scaled, y2_train)
# predictions = linear_model.predict(x2_test_scaled)
# rmse = np.sqrt(mean_squared_error(y2_test, predictions))
# print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')


