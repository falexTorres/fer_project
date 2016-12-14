import numpy as np
import matplotlib.pyplot as plt

n = 5687

X = np.load('../fer_X_train_final.npy')
y = np.load('../fer_y_train_final.npy')
plt.xlabel(y[n][0])
plt.imshow(X[n], cmap='gray')
plt.show()
