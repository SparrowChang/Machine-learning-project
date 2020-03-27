# Machine-learning-project


### Problem Definition
Use current on-hand wafer data (etest, probe, pre burn-in, post burn-in) to analyze the better method to find and reduce uncessary test cost and keep the must burn-in test items to reveal the real reliability window and perform efficiently, especially for burn-in tests.

### Get Data & Exploratory Data Analysis
We use Python 

### Data Clean/Preprocessing & Feature Engineering
There are a total of 223 test items (probe test+ post burin-in test) here, assume the worst burn-in test item as my target variable.
that is, the variable that needs to be predicted, and the remaining 222 test items are used as feature variables. The characteristic variable here is not the characteristic variable of a wafer, but the characteristic variable of the whole lot where the lot contains 23 wafers (different X and Y). I need to predict the worst burn-in test result in the lot based on the characteristic variables of the 23 wafers. Already wafer-center for both porbe test and post burn-in test.

### Model Training
In this project, use Python pandas, numpy, sklearn and matplotlib.
The method we use XGBoost regression (also with weak classification concept) 

Preliminary predictions
Before doing feature extraction, I used the original feature variables to train the model by XGBoos method (both with regression and classification concept)
Used XGBoost's parameter settings to see the performance of the model.
First I create training and test sets, then make predictions: 
random_state = 0 to ensure the reproducibility

### The key point is not to do feature extraction, directly set the default parameter of XGBoost

xgb_model1 = XGBRegressor()
xgb_model1.fit(X_train, y_train, verbose=False)

y_train_pred1 = xgb_model1.predict(X_train)
y_test_pred1 = xgb_model1.predict(X_test)

print('Train r2 score: ', r2_score(y_train, y_train_pred1))
print('Test r2 score: ', r2_score(y_test, y_test_pred1))
train_mse1 = mean_squared_error(y_train, y_train_pred1)
test_mse1 = mean_squared_error(y_test, y_test_pred1)
train_rmse1 = np.sqrt(train_mse1)
test_rmse1 = np.sqrt(test_mse1)
print('Train RMSE: %.4f' % train_rmse1)
print('Test RMSE: %.4f' % test_rmse1)

### Visualization

plt.figure(figsize=(7, 7))
plt.ylabel("y_test")
plt.xlabel("y_pred")
plt.scatter(y_test_pred1, y_test)


### Model Predict & Testing
###### 
# Feature extraction
# Need to do a correct study of all the feature variables when doing feature extraction. 
# Some feature variables may need to be combined, and some feature variables need to be decomposed. I have to expand more than current feature variables. 
# First need to check the data distribution of these characteristic variables by histogram.
# look at the two characteristics of wafer coordinates (X and Y). They are the geographic coordinates of the wafer.
# Using these two features can provide excellent visualization by the worst post burn-in result.

plt.figure(figsize=(13, 8))

ax = plt.subplot(1, 2, 1)
ax.set_title("Training Data")
ax.set_autoscaley_on(False)
ax.set_ylim([9, 98])
ax.set_autoscalex_on(False)
ax.set_xlim([5, 38])
plt.grid()
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(X_train["X"],
            X_train["Y"],
            cmap="coolwarm",
            c=y_train["Target"] / y_train["Target"].max())

ax = plt.subplot(1,2,2)
ax.set_title("Test Data")
ax.set_autoscaley_on(False)
ax.set_ylim([9, 98])
ax.set_autoscalex_on(False)
ax.set_xlim([5, 38])
plt.grid()
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(X_test["X"],
            X_test["Y"],
            cmap="coolwarm",
            c=y_test["Target"] / y_test["Target"].max())
_ = plt.plot()

# Create a heat map of the correlation coefficient matrix to see the correlation between the feature variables:

import seaborn as sns
pd.set_option('precision',2)
plt.figure(figsize=(12, 12))
sns.heatmap(df.drop(['Target'],axis=1).corr(), square=True)
plt.suptitle("Pearson Correlation Heatmap")
plt.savefig('Pearson Correlation Heatmap.png')
plt.show();

# It can be seen from the heat map of the above correlation coefficient matrix that post burn-in test items have a strong positive correlation with each other than probe test items.
# look at the correlation between the feature variable and the target variable:

corr_with_Target_value = df.corr()["Target"].sort_values(ascending=False)
plt.figure(figsize=(50,10))
plt.ylabel('Correlation with post burn-in_value')
corr_with_Target_value.drop("Target").plot.bar()
plt.savefig('Correlation value.png')
plt.show(); 

### Model Optimization
###### 
# Synthetic features, by sum top 3 correlation post burn-in test items (BI_TEST_132,BI_TEST_36, BI_TEST_33)
# Explore the relationship between neighborhood population density and median house value

df["Synthetic feature"] =(df['BI_TEST_132'] + df['BI_TEST_36'] + df['BI_TEST_133'])
df["Synthetic feature"]=df["Synthetic feature"]/3
print(df["Synthetic feature"])
print(df)

#Next use the new feature set with synthetic features (by group top correlation test items)to train our XGBoost model again

X = df.loc[:, df.columns !=  'Target'] 
y = df.loc[:, df.columns == 'Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# From the above prediction results, the RMSE on the test set didn't been reduced from 0.0655 to 0.0675
# Guess the test items sum up already is lower the correlation

# Check wafer Y corridinate effect, from burn-in purpose pick up the die from each wafer top or bottom
# Let's first look at the data distribution of the dimensions and target variables:

# From the perspective of the distribution of dimensions, assume wafer coordinate Y is slightly sensitive to the worst bin failure. 
# Then decompose wafer coordinate Y into two intervals, Y(5 to 15) and Y (55 to 85) will be stored as one-hot

# Use BI_TEST_43 (with good and bad result) to seperate
print(df.BI_TEST_43.describe())
df.BI_TEST_43.hist()

# Sum up
# Feature synthesis and feature bucket, only explain that feature bucketing is not ideal for our dataset or for XGBoost.
# The effect of using feature bucketing is still a good method, but it is slightly worse than XGBoost.
#bucket_0 (< 0.175)
#bucket_1 (0.175 ~ 0.22)
#bucket_2 (> 0.22)
