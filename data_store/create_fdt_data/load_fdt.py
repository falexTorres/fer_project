import numpy as np
import matplotlib.pyplot as plt

n = 1234

X_fdt = np.load('fer_X_test_fdt.npy')
y = np.load('fer_y_train.npy')
X = np.load('./create_fdt_data/fer_X_test.npy')
plt.xlabel("without")
plt.imshow(X[n], cmap='gray')
plt.show()
plt.xlabel("with")
plt.imshow(X_fdt[n], cmap='gray')
plt.show()
