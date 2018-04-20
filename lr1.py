import matplotlib.pyplot as plt
from sklearn import linear_model
y = [1, 2, 3, 4, 5]
X = [[1], [2.5], [2], [3.5], [5]]
lr = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
lr.fit(X, y)
lrR2 = lr.score(X, y)
Y0 = [(lr.coef_[0] * 0 + lr.intercept_) * lrR2, 6]
Y6 = [(lr.coef_[0] * 0 + lr.intercept_) * lrR2, (lr.coef_[0] * 6 + lr.intercept_) * lrR2]
print("coef_ value = ", lr.coef_)
print("intercept_ value = ", lr.intercept_)
print("Coeficient of Determination R² = ", round(lrR2, 2))
# Plot outputs
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(y, X,  color='red', linewidth=3)
ax.scatter(6, (lr.coef_[0] * 6 + lr.intercept_) * lrR2,  color='#FF00FF', linewidth=3)
ax.plot(Y0, Y6, color='green', linewidth=3)
ax.set_ylim(-1, 8)
ax.set_xlim(-1, 8)
plt.show()
