# Machine-learning-project

## Problem Definition
Use current on-hand wafer data (etest, probe, pre burn-in, post burn-in) to find a suitable method to analyze them. In order to reduce uncessary wafer burn-in test cost and only keep the essential test items to reveal the real reliability window and perform more efficient.

## Get Data & Exploratory Data Analysis
Python file: 1_etests.ipynb, 2_probe.ipynb, 3_prebi.ipynb, 4_postbi.ipynb

Use excel, statistics software JMP and Python functions (pandas, numpy, sklearn and matplotlib) to check the data missing or outlier (over upper/lower limit). Refer 1_etests.ipynb to 4_postbi.ipynb, there are without missing daya only outier data. From these steps, I would know the data when etests each wafer only with 5 measurement dies, but probe each wafer more than 2300 dies as a whole wafer test. Pre burn-in and post burn-in, their test limits are the same, it shows that the dies survives through etest (~100% pass rate), probe (at least Yield 98.9%) to final burn-in (pre burn-in test pass rate close to 98%), however after the tourture test (post burn-in test pass rate some down to 0%, large variation), they indicates that some burn-in test are more strict, reversible, easily fail at the early reliability gating. And the burn-in test area are major from wafer top or wafer bottom near wafer edge area.
So I check then combine probe/post burn-in test items as the same wafer recenter by coorindates (X and Y). for further training/test sets. For data analysis, I use XGBoost regression method (also with weak classification concept) to perform the prediction by r2 score and RMSE. 

## Data Clean/Preprocessing & Feature Engineering
CSV file (in Datasets): XGBoost_probe_postbi_combine.csv  
Python file: XGBoost_probe_postbi_combine.ipynb

There are a total of 223 test items (probe test+ post burin-in test) and 23 wafers here, assume the worst post burn-in test item (in the code, I pick-up BI_TEST_35, pass rate 0%) as my target variable. That is, the variable that needs to be predicted, and the remaining 222 test items are used as feature variables. The feature variable here is not the feature variable of a wafer, but the feature variable of the whole lot, which contains 23 wafers. The major is to predict the worst burn-in test result in the lot based on the feature variables of the 23 wafers and find out other high correlation prerequisite test items.

## Model Training, Predict & Testing
Preliminary predictions
1. Before doing feature extraction, I used the original feature variables to train the model by XGBoos default parameter to see the performance of the model. Hints: training and test sets, to ensure the reproducibility predictions (random_state = 0) 

```python
X = df.loc[:, df.columns !=  'Target'] 
y = df.loc[:, df.columns == 'Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

```python
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
```

1.Train/Test | score/RMSE
--- | ---
Train r2 score | 0.9999948598635676
Test r2 score | 0.8696620493512542
Train RMSE | 0.0004            
Test RMSE | 0.0655 

#### Visualization
```python
plt.figure(figsize=(7, 7))
plt.ylabel("y_test")
plt.xlabel("y_pred")
plt.scatter(y_test_pred1, y_test)
```

Feature extraction. Need to do a correct study of all the feature variables when doing feature extraction. Some feature variables may need to be combined, and some feature variables need to be decomposed. I have to expand more than current feature variables. First need to check the data distribution of these feature variables by histogram. Look at the two features of wafer coordinates (X and Y). They are the geographic coordinates of the wafer. Using these two features can provide excellent visualization by the worst post burn-in result.

```python
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
```

#### Create a heat map of the correlation coefficient matrix to see the correlation between the feature variables
```python
import seaborn as sns
pd.set_option('precision',2)
plt.figure(figsize=(12, 12))
sns.heatmap(df.drop(['Target'],axis=1).corr(), square=True)
plt.suptitle("Pearson Correlation Heatmap")
plt.savefig('Pearson Correlation Heatmap.png')
plt.show();
```

It can be seen from the heat map of the above correlation coefficient matrix that post burn-in test items have a strong positive correlation with each other than probe test items. Look at the correlation between the feature variable and the target variable:

```python
corr_with_Target_value = df.corr()["Target"].sort_values(ascending=False)
plt.figure(figsize=(50,10))
plt.ylabel('Correlation with post burn-in_value')
corr_with_Target_value.drop("Target").plot.bar()
plt.savefig('Correlation value.png')
plt.show(); 
```

![image](https://github.com/SparrowChang/Machine-learning-project/blob/master/Images/Correlation%20value.png)

## Model Optimization

2. Synthetic features, by sum top 3 correlation post burn-in test items (BI_TEST_132,BI_TEST_36, BI_TEST_33). Explore the relationship between Synthetic features and Target value

```python
df["Synthetic feature"] =(df['BI_TEST_132'] + df['BI_TEST_36'] + df['BI_TEST_133'])
df["Synthetic feature"]=df["Synthetic feature"]/3
```
Next use the new feature set with synthetic features (by group top correlation test items) to train our XGBoost model again
From the above prediction results, the RMSE on the test set didn't been reduced from 0.0655 to 0.0675
Guess the test items sum up induces lower the correlation

2.Train/Test | score/RMSE
--- | ---
Train r2 score | 0.9999943800854812
Test r2 score | 0.861673815963637
Train RMSE | 0.0004            
Test RMSE | 0.0674 

3. Check wafer Y corridinate effect, from burn-in purpose pick up the die from each wafer top or bottom (by Y coordinates)
Let's first look at the data distribution of the dimensions and target variables:

From the perspective of the distribution of dimensions, assume wafer coordinate Y is slightly sensitive to the worst bin failure. 
Then decompose wafer coordinate Y into two intervals, Y(5 to 15) and Y (55 to 85) will be stored as one-hot

```python
Y_RANGES = zip(range(5, 15), range(55, 85))
for r in Y_RANGES:
    X_train["lat_%d_to_%d" % r] = X_train["Y"].apply(lambda l:1.0 if l>=r[0] and l<r[1] else 0.0)
    
Y_RANGES = zip(range(5, 15), range(55, 85))
for r in Y_RANGES:
    X_test["lat_%d_to_%d" % r] = X_test["Y"].apply(lambda l:1.0 if l>=r[0] and l<r[1] else 0.0)
```

3.Train/Test | score/RMSE
--- | ---
Train r2 score | 0.9999949389112046
Test r2 score | 0.8566688507098852
Train RMSE | 0.0004            
Test RMSE | 0.0686 

4. Use BI_TEST_43 (with good and bad result, pass rate 30.6%) to seperate

```python
print(df.BI_TEST_43.describe())
df.BI_TEST_43.hist()
#bucket_0 (< 0.175)
#bucket_1 (0.175 ~ 0.22)
#bucket_2 (> 0.22)
```
```python
for r in ((-0.8, 0.175),(0.175, 0.22),(0.22, 0.3)):
    X_train["pop_%d_to_%d" % r] = X_train["BI_TEST_43"].apply(lambda l:0.01 if l>=r[0] and l<r[1] else 0.0)
    
for r in ((-0.8, 0.175),(0.175, 0.22),(0.22, 0.3)):
    X_test["pop_%d_to_%d" % r] = X_test["BI_TEST_43"].apply(lambda l:0.1 if l>=r[0] and l<r[1] else 0.0)
```

4.Train/Test | score/RMSE
--- | ---
Train r2 score: | 0.9999934040677946
Test r2 score:  | 0.8259931383031068
Train RMSE: | 0.0004            
Test RMSE: | 0.0675 

## Sum up
Feature synthesis (Test r2 score:  0.8617) and feature bucket(Test r2 score:  0.8567/ 0.8260), only explain that feature bucketing is not ideal for our dataset or for XGBoost. The effect of using feature bucketing is still a good method, but it is slightly worse than XGBoost. (Test r2 score:  0.8697)
