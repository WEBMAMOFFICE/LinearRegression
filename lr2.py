import matplotlib.pyplot as plt
from sklearn import linear_model

currency_names = [1, 3, 4, 7, 10, 12, 14]
X_prices1 = [[1.0014], [3.0114], [4.9999], [7.0077], [10.1400], [12.7777], [14.1414]]
lr = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
lrf = lr.fit(X_prices1, currency_names)
y_price1 = [[1.7714], [3.1470], [4.0000], [7.1477], [10.5914], [12.5914], [14.5914]]
y_price2 = [[1.3414], [3.7114], [4.5555], [7.3477], [10.3477], [12.2577], [14.7777]]
y_price3 = [[1.9514], [3.3414], [4.1455], [7.7777], [10.7777], [12.1477], [14.9514]]
print("coef_ value = %s" % (lrf.coef_,))
print("intercept_ value = %s" % (lrf.intercept_,))
print("Coeficient of Determination R² = %s" % (round(lrf.score(X_prices1, currency_names), 4),))
lrp1 = lr.predict(y_price1)
print("\n--- Prediction # 1")
for pr, curr in zip(currency_names, lrp1):
    print("%10s%s%-10s" % (pr, "--->".center(10), round(curr, 4)))
lrf1 = lr.fit(y_price1, currency_names)
print("coef_ value = %s" % (lrf1.coef_,))
print("intercept_ value = %s" % (lrf1.intercept_,))
print("Coeficient of Determination R² = %s" % (round(lrf1.score(y_price1, currency_names), 4),))
print("\n--- Prediction # 2")
lrp2 = lr.predict(y_price2)
for pr, curr in zip(currency_names, lrp2):
    print("%10s%s%-10s" % (pr, "--->".center(10), round(curr, 4)))
lrf2 = lr.fit(y_price2, currency_names)
print("coef_ value = %s" % (lrf2.coef_,))
print("intercept_ value = %s" % (lrf2.intercept_,))
print("Coeficient of Determination R² = %s" % (round(lrf2.score(y_price1, currency_names), 4),))
print("\n--- Prediction # 3")
lrp3 = lr.predict(y_price3)
for pr, curr in zip(currency_names, lrp3):
    print("%10s%s%-10s" % (pr, "--->".center(10), round(curr, 4)))
lrf3 = lr.fit(y_price3, currency_names)
print("coef_ value = %s" % (lrf3.coef_,))
print("intercept_ value = %s" % (lrf3.intercept_,))
print("Coeficient of Determination R² = %s" % (round(lrf3.score(y_price3, currency_names), 4),))
# Plot outputs
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter([1, 3, 7, 10, 14], [currency_names[0], X_prices1[0][0], lrp1[0], lrp2[0], lrp3[0]], s=7, color='#770000')
ax.plot([1, 3, 7, 10, 14], [currency_names[0], X_prices1[0][0], lrp1[0], lrp2[0], lrp3[0]], ls='dotted', lw=1.4, aa=True, color="#770000")
ax.plot([1, 14], [X_prices1[0][0], lrp3[0]], color='green', linewidth=1)
ax.scatter([1, 3, 7, 10, 14], [currency_names[1], X_prices1[1][0], lrp1[1], lrp2[1], lrp3[1]], s=7, color='#AA0000')
ax.plot([1, 3, 7, 10, 14], [currency_names[1], X_prices1[1][0], lrp1[1], lrp2[1], lrp3[1]], ls='dotted', lw=1.4, aa=True, color="#AA0000")
ax.plot([1, 14], [X_prices1[1][0], lrp3[1]], color='green', linewidth=1)
ax.scatter([1, 3, 7, 10, 14], [currency_names[2], X_prices1[2][0], lrp1[2], lrp2[2], lrp3[2]], s=7, color='#BB0000')
ax.plot([1, 3, 7, 10, 14], [currency_names[2], X_prices1[2][0], lrp1[2], lrp2[2], lrp3[2]], ls='dotted', lw=1.4, aa=True, color="#BB0000")
ax.plot([1, 14], [X_prices1[2][0], lrp3[2]], color='green', linewidth=1)
ax.scatter([1, 3, 7, 10, 14], [currency_names[3], X_prices1[3][0], lrp1[3], lrp2[3], lrp3[3]], s=7, color='#CC0000')
ax.plot([1, 3, 7, 10, 14], [currency_names[3], X_prices1[3][0], lrp1[3], lrp2[3], lrp3[3]], ls='dotted', lw=1.4, aa=True, color="#CC0000")
ax.plot([1, 14], [X_prices1[3][0], lrp3[3]], color='green', linewidth=1)
ax.scatter([1, 3, 7, 10, 14], [currency_names[4], X_prices1[4][0], lrp1[4], lrp2[4], lrp3[4]], s=7, color='#DD0000')
ax.plot([1, 3, 7, 10, 14], [currency_names[4], X_prices1[4][0], lrp1[4], lrp2[4], lrp3[4]], ls='dotted', lw=1.4, aa=True, color="#DD0000")
ax.plot([1, 14], [X_prices1[4][0], lrp3[4]], color='green', linewidth=1)
ax.scatter([1, 3, 7, 10, 14], [currency_names[5], X_prices1[5][0], lrp1[5], lrp2[5], lrp3[5]], s=7, color='#EE0000')
ax.plot([1, 3, 7, 10, 14], [currency_names[5], X_prices1[5][0], lrp1[5], lrp2[5], lrp3[5]], ls='dotted', lw=1.4, aa=True, color="#EE0000")
ax.plot([1, 14], [X_prices1[5][0], lrp3[5]], color='green', linewidth=1)
ax.scatter([1, 3, 7, 10, 14], [currency_names[6], X_prices1[6][0], lrp1[6], lrp2[6], lrp3[6]], s=7, color='#FF0000')
ax.plot([1, 3, 7, 10, 14], [currency_names[6], X_prices1[6][0], lrp1[6], lrp2[6], lrp3[6]], ls='dotted', lw=1.4, aa=True, color="#FF0000")
ax.plot([1, 14], [X_prices1[6][0], lrp3[6]], color='green', linewidth=1)
ax.set_ylim(-1, 15)
ax.set_xlim(-1, 15)
plt.show()