import numpy as np 
import matplotlib.pyplot as plt
import random
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import add_dummy_feature
from sklearn.linear_model import LogisticRegression

random.seed(42)

r = Path("/home/vinish/Dropbox/Machine Learning/book/lectures")

Path.PosixPath()

# number of observations 
n = 100000

# ------------------------------------
# ------------------------------------

# Part A. Simulate data

# ------------------------------------
# ------------------------------------

# The pseudo DGP takes the following form:
# 1. College education boosts health by 10 percent.
# 2. Income boosts health by 20 percent.
# 3. 40 percent more people from higher income households have college education.
# 4. Having insurance boosts health by 5 percent.

# ---------------------

# A. Generating income

# ---------------------

# lognormal dist
income_log = np.random.lognormal(mean=1, sigma=0.5, size=n)

# plot income
plt.figure(figsize=(10, 6))
plt.hist(income_log, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
plt.xlabel('Log income', fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Histogram of Log income", fontsize=14, fontweight="bold")
# Add grid for easier reading
plt.grid(True, alpha=0.3, linestyle='--')
# Improve layout
plt.tight_layout()
plt.savefig(Path(r, "python_output/log_income_hist.pdf"))

# income (dollar measure)
income = income_log * 20000

plt.figure(figsize=(10, 6))
plt.hist(income, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
plt.xtitle("Income", fontsize=12)
plt.ytitle("Frequency", fontsize=12)
plt.title("Histogram of Income", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(Path(r, "python_output/income_hist.pdf"))

# high income based on median 
median_inc = np.median(income)
high_mask = np.zeros(n).astype('bool')
high_mask[income>=median_inc] = True
high_mask

high_income=np.where(high_mask==True,1,0)
print(f"mean of high income: {high_income.mean()}")

# ----------------------

# B. college level education

# ----------------------
def gen_college(prob, N=1):
    
    # @Arg prob: probability used for a binomial outcome
    # @Arg N: size fixed at 1
    a=np.random.binomial(1, prob, N)
    return a

college = []

# get n number of college values with given probability from a binomial dist
for i in range(0, n):

    coll_i = gen_college(prob = 0.3 + 0.4*high_income[i])
    college.append(coll_i)

# arrange values in the list as an array
college = np.array(college).ravel()

# print fraction with/without college by high/low income
print(f"fraction of people with college within high income: {sum(college[high_mask==True]/sum(high_mask))} \n \n") 
print(f"fraction of people without college within high income: {1 - sum(college[high_mask==True]/sum(high_mask))} \n \n") 

print(f"fraction of people with college within low income: {sum(college[high_mask==False]/(n-sum(high_mask)))} \n \n") 
print(f"fraction of people without college within high income: {1 - sum(college[high_mask==False]/(n-sum(high_mask)))} \n \n") 


# ------------------------------

# C. insurance 

# ------------------------------
# note that insurance is exogeneous in the model
ins = np.random.binomial(1, 0.5, n)
print(f"fraction of insured individuals: {ins.mean()}")

def gen_health(prob, N=1):

    h = np.random.binomial(1, prob, N)
    return(h)

health = [] # list for health

for i in range(0, n):

    prob_i = 0.6 + 0.1*college[i] + 0.2*high_income[i] + 0.05*ins[i]
    health_i = gen_health(prob_i)
    health.append(health_i)

# arrange list to an array and flatten
health = np.array(health).ravel()

# --------------------------------------------
# --------------------------------------------

# Part B. Build a linear regression model

# --------------------------------------------
# --------------------------------------------

# ---------------------------------

# Specify X and y

# ---------------------------------
X = pd.DataFrame({"college": college,
                  "income": income,
                  "high_income": high_income,
                  "insurance": ins
}
)
print(f"columns on X: {X.columns}")

y = pd.DataFrame({"y": health})

# call the linear regression function from sklearn library
lin_reg = LinearRegression()
print(lin_reg)

# fit using actual X and y vals and build several models

# model 1: include college only
lin_reg = LinearRegression()
mod1 = lin_reg.fit(pd.DataFrame(X.loc[:,"college"]), y)
mod1.intercept_
mod1.coef_

# model 2: college + income (continuous)
lin_reg = LinearRegression()
mod2 = lin_reg.fit(X.loc[:, ["college", "income"]], y) 
mod2.intercept_
mod2.coef_

# model 3: college + high_income instead of income 
lin_reg = LinearRegression()
mod3 = lin_reg.fit(X.loc[:, ["college", "high_income"]], y)
mod3.intercept_
mod3.coef_

# model 4: college + high_income + income (NOTE: income is redundant here)
lin_reg = LinearRegression()
mod4 = lin_reg.fit(X.loc[:, ["college", "high_income", "income"]], y)
mod4.intercept_
mod4.coef_

# model 5: college + high_income + ins (Specification thats used to simulate the DGP)
lin_reg = LinearRegression()
mod5 = lin_reg.fit(X.loc[:, ["college", "high_income", "insurance"]], y)
mod5.intercept_
mod5.coef_

# Build results table

res = {"model1": {"intercept": mod1.intercept_.ravel()[0],
                "college": mod1.coef_.ravel()[0]
                },
       'model2': {"intercept": mod2.intercept_.ravel()[0],
                'college': mod2.coef_.ravel()[0],
                'income': mod2.coef_.ravel()[1]
                },
        'model3': {'intercept': mod3.intercept_.ravel()[0],
                   'college': mod3.coef_.ravel()[0],
                   'high income': mod3.coef_.ravel()[1]},
        'model4': {'intercept': mod4.intercept_.ravel()[0],
                   'college': mod4.coef_.ravel()[0],
                   'high income': mod4.coef_.ravel()[1],
                   'income': mod4.coef_.ravel()[2]},
        'model5': {'intercept': mod5.intercept_.ravel()[0],
                   'college': mod5.coef_.ravel()[0],
                   'high income': mod5.coef_.ravel()[1],
                   'insurance': mod5.coef_.ravel()[2]}

}

results_table = pd.DataFrame(res)

print(results_table.round(4))

results_table.attrs

dir(results_table)

results_table


# -------------------------------

# Logistic model

# -------------------------------

# simulate new y's: they come from probabilities off of the logistic function
# true thetas
theta_true = np.array([0.6, 0.1, 0.2 ,0.05]).reshape((4, 1))
X_new = add_dummy_feature(X.loc[:, ["college","high_income", "insurance"]])
# get probabilites from the logistic function
prob = 1 / (1 + np.exp(-X_new @ theta_true))
health_new = [] 

for i in range(n):
    health_new_i= gen_health(prob[i]) # binomial with given probabilities 
    health_new.append(health_new_i)

health_new = np.array(health_new).ravel()
np.mean(health_new)

# initialize thetas
theta = np.zeros(4).reshape((4, 1))
y_new = np.array(health_new).reshape(n, 1)
eta = 0.00001 # learning rate
loss = []     # loss
epsilon = 1e-15 # to avoid blowing up 

for i in range(5000):
    
    z = np.clip(X_new @ theta, -500, 500) # avoids blowing up
    sigma = 1 / (1 + np.exp(-z))
    gradient = X_new.T @ (sigma - y_new) 
    theta = theta - eta*gradient
    loss_val = np.mean(y_new*(-np.log(sigma+epsilon)) + (1-y_new)*(-np.log(1-sigma+epsilon)))
    loss.append(loss_val)

loss = np.array(loss).ravel()

# plot the loss as a function of interation
plt.figure(figsize=(10, 8))
plt.scatter(np.linspace(1, 5000, 5000), loss)
plt.xlabel('epoch')
plt.ylabel('logit loss')
plt.title('Number of interation and loss')
plt.savefig(Path(r, 'python_output/logit_loss.pdf'))


# compare it with the results from sklearn 
mod_logit = LogisticRegression(max_iter=5000, fit_intercept=False)
mod_logit.fit(X_new, y_new)

print(f"coefficients from logistic regression: {mod_logit.coef_}")
print(f"coefficients from gradient descent: {theta}")

# compare it with linear regression model (incorrect functional form)
mod_lr = LinearRegression(fit_intercept=False)
mod_lr = mod_lr.fit(X_new, y_new)
print(f"coefficients from LR: {mod_lr.coef_}")

comparison = pd.DataFrame({
                            'true_theta': theta_true.ravel(),
                            'sklearn': mod_logit.coef_.ravel(),
                            'Gradient Descent': theta.ravel(),
                            'Linear Regression': mod_lr.coef_.ravel()
})


print(comparison)













