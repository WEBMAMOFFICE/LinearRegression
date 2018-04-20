import matplotlib.pyplot as plt

X1 = [1, 2, 3, 4, 5]
X2 = [1, 2.5, 2, 3.5, 5]
Y1 = [-0.1, 6]
Y2 = [-0.1, 5.3]

# Plot outputs
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X1, X2,  color='red', linewidth=3)
ax.scatter(6, 5.3,  color='#FF00FF', linewidth=3)
ax.plot(Y1, Y2, color='green', linewidth=3)
ax.set_ylim(-1, 8)
ax.set_xlim(-1, 8)
plt.show()