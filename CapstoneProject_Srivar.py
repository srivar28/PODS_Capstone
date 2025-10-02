#%% Initialisation
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
import random
random.seed(17271624)

#random seed = 17271624
alpha = 0.005

#%% Loader
data = np.genfromtxt('rmpCapstoneNum.csv', delimiter = ',')
data_txt = pd.read_csv('rmpCapstoneQual.csv',header = None)

check = ~np.isnan(data[:, 2])

data = data[check]
data_txt = data_txt[check]

pre_filteration_median_rating = np.median(data[:,2])
pre_filteration_75th_percentile =  np.percentile(data[:,2],75)


#%% Handling Missing Data

#Plotting data loss per threshold
threshold_counts = {k: (data[:,2] > k).sum() for k in range(1, 11)}

plt.figure(figsize=(8, 5))
plt.plot(list(threshold_counts.keys()), list(threshold_counts.values()), marker='o',color = 'blue', alpha = 0.5)
plt.title("Number of Professors Retained vs. Rating Threshold ")
plt.xlabel("Rating Threshold")
plt.ylabel("Number of Professors Retained")
plt.grid(True)
plt.tight_layout()
plt.show()


#Plottion standard dev based on threshold
volatility_data = []

for i in range(1,11):
    temp = data[data[:,2] == i]
    std_dev = np.std(temp[:,0],ddof= 1)
    volatility_data.append((i, std_dev))

ks, stds = zip(*volatility_data)

plt.figure(figsize=(8, 5))
plt.plot(ks, stds, marker='o',color = 'blue', alpha = 0.5)
plt.title("Volatility of Average Ratings vs. Number of Ratings")
plt.xlabel("Number of Ratings")
plt.ylabel("Std Dev of Avg Rating")
plt.xticks(range(1, 11))
plt.grid(True)
plt.tight_layout()
plt.show()

check = data[:,2] >= 4

data = data[check]
data_txt = data_txt[check]



#%% Question 1

check = []
for i in range(len(data)):
    if ((data[i,6] == 0 and data[i,7] == 0) or (data[i,6] == 1 and data[i,7] == 1)):
        check.append(False)
    else:
        check.append(True)

q1_data = data[check]

#No female column(becasue we already have that information in male column)
#Too many nan in would take again

corr_matrix = np.corrcoef(q1_data[:,[1,2,3,4,5,6]], rowvar=False) 
#Neglible correlaton between predictors 

X =  q1_data[:,[1,2,3,5,6]]
y = q1_data[:,0]


fullModel = LinearRegression().fit(X,y)
rSqrFull = fullModel.score(X,y) 

b0, b1 = fullModel.intercept_, fullModel.coef_

X_sm = sm.add_constant(X)  # Adds intercept term
model = sm.OLS(y, X_sm).fit()

print(model.summary())

alpha = 0.005
conf_int = model.conf_int(alpha=alpha)

print(model.pvalues)


#%% Question 1
#Mann Whitney U - test
# Control for experience - Maybe use number of ratings as a proxy for experience

check = []
for i in range(len(data)):
    if ((data[i,6] == 0 and data[i,7] == 0) or (data[i,6] == 1 and data[i,7] == 1)):
        check.append(False)
    else:
        check.append(True)

data = data[check]

data_male = data[data[:,6] == 1]
data_female = data[data[:,7]==1]


u1,p1 = stats.mannwhitneyu(data_female[:,0], data_male[:,0]) 

plt.figure(figsize=(8, 6))
plt.boxplot([data_male[:,0], data_female[:,0]], labels=['Male Professors', 'Female Professors'])
plt.title('Comparison of Ratings by Gender')
plt.ylabel('Rating')
plt.grid(True)
plt.show()

n1 = len(data_male)
n2 = len(data_female)

rank_biserial = 1 - (2 * u1) / (n1 * n2)

#It is significant however effect size is small


#%% Question 2 & 6

q1_data = data

#No gender column -> small effect size (check this)

corr_matrix = np.corrcoef(q1_data[:,[1,2,3,5]], rowvar=False) 
#Neglible correlaton between predictors 

X =  q1_data[:,[1,2,3,5]]
y = q1_data[:,0]

fullModel = LinearRegression().fit(X,y)
rSqrFull = fullModel.score(X,y) 

b0, b1 = fullModel.intercept_, fullModel.coef_

X_sm = sm.add_constant(X)  # Adds intercept term
model = sm.OLS(y, X_sm).fit()

print(model.summary())

alpha = 0.005
conf_int = model.conf_int(alpha=alpha)

print(model.pvalues)


#%% Question 2
#Mann Whitney U -test


#Not signifcant at 50
#A bit of experience 
exp_cap = pre_filteration_75th_percentile

check = data[:, 2] >= exp_cap

data_exp = data[check]
data_inexp = data[~check]


u2,p2 = stats.mannwhitneyu(data_inexp[:,0], data_exp[:,0]) 
if p2 >= alpha:
    print("Not ")
print("Significant")

n1 = len(data_exp)
n2 = len(data_inexp)

rank_biserial = 1 - (2 * u2) / (n1 * n2)


plt.figure(figsize=(8, 6))
plt.boxplot([data_exp[:,0], data_inexp[:,0]], labels=['Experienced Professors', 'Inexperienced Professors'])
plt.title('Comparison of Ratings by Experience')
plt.ylabel('Rating')
plt.grid(True)
plt.show()

#%% Question 3
plt.figure(figsize=(8, 6))
plt.scatter(data[:,0], data[:,1],color = 'blue', alpha = 0.5)
plt.title("Average Rating vs. Average Difficulty")
plt.xlabel("Average Difficulty")
plt.ylabel("Average Rating")
plt.show()

q1_pearson_corr = np.corrcoef(data[:,0], data[:,1])[0,1]

q1_spearman_corr, _ = spearmanr(data[:,0], data[:,1])


#%% Question 4
#Linear Regression

q4_data = data

q4_data[:,5] = q4_data[:,5]/q4_data[:,2]


corr_matrix = np.corrcoef(q4_data[:,[1,2,3,5]], rowvar=False) 
#Neglible correlaton between predictors 

X =  q4_data[:,[1,2,3,5]]
y = q4_data[:,0]


fullModel = LinearRegression().fit(X,y)
rSqrFull = fullModel.score(X,y) 

b0, b1 = fullModel.intercept_, fullModel.coef_

X_sm = sm.add_constant(X)  # Adds intercept term
model = sm.OLS(y, X_sm).fit()

print(model.summary())

alpha = 0.005
conf_int = model.conf_int(alpha=alpha)

print(model.pvalues)


#%% Question 4

q4_data = data
q4_data[:, 5] = q4_data[:, 5] / q4_data[:, 2] 

thresholds = np.arange(0, 1, 0.1)

threshold_counts = {k: (q4_data[:, 5] > k).sum() for k in thresholds}

plt.figure(figsize=(8, 5))
plt.plot(list(threshold_counts.keys()), list(threshold_counts.values()), marker='o', color='blue', alpha=0.5)
plt.title("Number of Professors Retained vs. Ratio Threshold")
plt.xlabel("Ratio Threshold")
plt.ylabel("Number of Professors Retained")
plt.grid(True)
plt.tight_layout()
plt.show()


threshold = 0.4
check = q4_data[:,5] >= threshold

online_data = q4_data[check]
offline_data = q4_data[~check]

u4,p4 = stats.mannwhitneyu(online_data[:,0], offline_data[:,0]) 
if p4 >= alpha:
    print("Not ")
print("Significant")

n1 = len(online_data)
n2 = len(offline_data)

rank_biserial = 1 - (2 * u4) / (n1 * n2)


plt.figure(figsize=(8, 6))
plt.boxplot([online_data[:,0], offline_data[:,0]], labels=['Online Professors', 'Offline Professors'])
plt.title('Online Professors vs Offline Professors')
plt.ylabel('Rating')
plt.grid(True)
plt.show()


#%% Question 5
q5_data = data
check = ~np.isnan(q5_data[:,4])
q5_data = q5_data[check]


plt.figure(figsize=(8, 6))
plt.scatter(q5_data[:,0], q5_data[:,4],color = 'blue', alpha = 0.5)
plt.title("Would Take Again Percentage vs. Average Rating")
plt.xlabel("Average Rating")
plt.ylabel("Would Take Again Percentage")
plt.show()


q5_pearson_corr = np.corrcoef(q5_data[:,0], q5_data[:,4])[0,1]

q5_spearman_corr, _ = spearmanr(q5_data[:,0], q5_data[:,4])

#%% Question 6
check = data[:,3] == 1
pepper = data[check][:,0]
non_pepper = data[~check][:,0]



u6,p6 = stats.mannwhitneyu(non_pepper, pepper) 
if p6 >= alpha:
    print("Not ")
print("Significant")

n1 = len(pepper)
n2 = len(non_pepper)

rank_biserial = 1 - (2 * u6) / (n1 * n2)


plt.figure(figsize=(8, 6))
plt.boxplot([pepper, non_pepper], labels=['Received a Pepper', "Didn't Receive a Pepper"])
plt.title("Professors Who Received a Pepper vs Professors Who Didn't Receive a Pepper")
plt.ylabel('Rating')
plt.grid(True)
plt.show()


#%% Question 7

check = []
for i in range(len(data)):
    if ((data[i,6] == 0 and data[i,7] == 0) or (data[i,6] == 1 and data[i,7] == 1)):
        check.append(False)
    else:
        check.append(True)

data = data[check]


X7 = data[:,1]
y7 = data[:,0]


X7 = X7.reshape(-1,1)


X7_train, X7_test, y7_train, y7_test = train_test_split(X7, y7, test_size=0.2, random_state=17271624)

model = LinearRegression().fit(X7_train, y7_train)
y7_pred = model.predict(X7_test)

r2_7 = r2_score(y7_test, y7_pred)
rmse_7 = root_mean_squared_error(y7_test, y7_pred)

q7coeffs = model.coef_

plt.figure(figsize=(8, 5))
plt.scatter(X7_test, y7_test, color='blue', alpha=0.5, label='Actual Ratings')

plt.plot(X7_test, y7_pred, color='red', linewidth=2, label='Regression Line')

plt.xlabel("Average Difficulty")
plt.ylabel("Average Rating")
plt.title(f"Linear Regression: Avg Rating vs. Difficulty\n$R^2$ = {r2_7:.3f}, RMSE = {rmse_7:.3f}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



#%% Question 8

check = []
for i in range(len(data)):
    if ((data[i,6] == 0 and data[i,7] == 0) or (data[i,6] == 1 and data[i,7] == 1) or np.isnan(data[i,4])):
        check.append(False)
    else:
        check.append(True)

data = data[check]


X8 = data[:,[1,2,3,4,5,6]]
y8 = data[:,0]



corr_matrix = np.corrcoef(X8, rowvar=False) 

X8_train, X8_test, y8_train, y8_test = train_test_split(X8, y8, test_size=0.2, random_state=17271624)

alph = 1.0 

ridge = Ridge(alpha=alph) 
ridge.fit(X8_train, y8_train)

y8_pred_ridge = ridge.predict(X8_test)

ridge_rmse = root_mean_squared_error(y8_test, y8_pred_ridge)
ridge_r2 = r2_score(y8_test, y8_pred_ridge)


print("\nRidge Regression Results (Q8):")
print("R-squared:", round(ridge_r2, 4))
print("RMSE:", round(ridge_rmse, 4))
print("Intercept:", ridge.intercept_)
print("Coefficients:", ridge.coef_)

feature_names = ['Difficulty', 'NumRatings', 'Pepper', 'WouldTakeAgain', 'OnlineCount', 'Male']

intercept = ridge.intercept_
coeffs = ridge.coef_


plt.figure(figsize=(8, 5))
plt.scatter(y8_test, y8_pred_ridge, alpha=0.5, color='blue', label='Predicted vs Actual')
min_val = min(min(y8_test), min(y8_pred_ridge))
max_val = max(max(y8_test), max(y8_pred_ridge))
plt.plot([min_val, max_val], [min_val, max_val], color = 'red', linewidth=2, label='Ideal Fit')

plt.xlabel("Actual Average Rating")
plt.ylabel("Predicted Average Rating")
plt.title(f"Ridge Regression: Predicted vs Actual Ratings\n$R^2$ = {ridge_r2:.3f}, RMSE = {ridge_rmse:.3f}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



#%% Question 9

check = []
for i in range(len(data)):
    if ((data[i,6] == 0 and data[i,7] == 0) or (data[i,6] == 1 and data[i,7] == 1) or np.isnan(data[i,4])):
        check.append(False)
    else:
        check.append(True)

data = data[check]

X9 = data[:,0].reshape(len(data),1) 
y9 = data[:,3]


X9_train, X9_test, y9_train, y9_test = train_test_split(X9, y9, test_size=0.2, random_state=17271624)


# Fit model:
model9 = LogisticRegression().fit(X9_train,y9_train)

y9_pred = model9.predict_proba(X9_test)[:,1]

    
auc = roc_auc_score(y9_test, y9_pred)
fpr, tpr, _ = roc_curve(y9_test, y9_pred)

plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for   Prediction (Q9)')
plt.legend()
plt.show()

# Logistic curve plot
x1 = np.linspace(0, 5, 500).reshape(-1,1)
y1 = model9.predict_proba(x1)[:,1]

plt.plot(x1, y1, color='red', linewidth=3)
plt.scatter(X9, y9, color='black', s=10, alpha=0.5)
plt.hlines(0.5, 0, 5, colors='gray', linestyles='dotted')
plt.xlabel('Average Rating')
plt.xlim([0, 5])
plt.ylabel('Pepper (Yes/No)')
plt.yticks([0,1])
plt.title('Logistic Regression: Pepper vs Average Rating')
plt.show()


y9_pred_class = (y9_pred >= 0.5).astype(int)

cm9 = confusion_matrix(y9_test, y9_pred_class)

tn, fp, fn, tp = confusion_matrix(y9_test, y9_pred_class).ravel()

specificity9 = tn / (tn + fp)
accuracy9 = accuracy_score(y9_test, y9_pred_class)
precision9 = precision_score(y9_test, y9_pred_class)
recall9 = recall_score(y9_test, y9_pred_class) 




#%% Question 10
check = []
for i in range(len(data)):
    if ((data[i,6] == 0 and data[i,7] == 0) or (data[i,6] == 1 and data[i,7] == 1) or np.isnan(data[i,4])):
        check.append(False)
    else:
        check.append(True)

data10 = data[check]

X10 = data10[:,[0,1,2,4,5,6]]
y10 = data10[:,3]

X10_train, X10_test, y10_train, y10_test = train_test_split(X10, y10, test_size=0.2, random_state=17271624)


model10 = LogisticRegression(class_weight='balanced').fit(X10_train, y10_train)

y10_pred = model10.predict_proba(X10_test)[:,1]


auc = roc_auc_score(y10_test, y10_pred)
fpr, tpr, _ = roc_curve(y10_test, y10_pred)

plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Pepper Prediction (Q10)')
plt.legend()
plt.show()

y10_pred_class = (y10_pred >= 0.5).astype(int)

cm10 = confusion_matrix(y10_test, y10_pred_class)

tn, fp, fn, tp = confusion_matrix(y10_test, y10_pred_class).ravel()

specificity10 = tn / (tn + fp)
accuracy10 = accuracy_score(y10_test, y10_pred_class)
precision10 = precision_score(y10_test, y10_pred_class)
recall10 = recall_score(y10_test, y10_pred_class) 


#%% Extra Credit

difficulty = data[:, 1]
fields = data_txt.iloc[:, 0].astype(str)

math_check = fields.str.lower().str.contains("math", na=False)
cs_check = fields.str.lower().str.contains("computer", na=False)

math_difficulty = difficulty[math_check.values]
cs_difficulty = difficulty[cs_check.values]

math_difficulty_median = np.median(math_difficulty)
cs_difficulty_median = np.median(cs_difficulty)

u11,p11 = stats.mannwhitneyu(cs_difficulty,math_difficulty) 
if p11 >= alpha:
    print("Not ")
print("Significant")

n1 = len(math_difficulty)
n2 = len(cs_difficulty)

rank_biserial = 1 - (2 * u11) / (n1 * n2)


plt.figure(figsize=(8, 6))
plt.boxplot([cs_difficulty, math_difficulty], labels=['CS Professor Average Difficulty', 'Math Professor Average Difficulty'])
plt.title('Comparison of Average Difficulty of CS and Math Professors')
plt.ylabel('Average Professor Difficulty')
plt.grid(True)
plt.show()














