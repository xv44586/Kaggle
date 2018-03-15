from sklearn.preprocessing import StandardScaler
import numpy as np

data = [[0, 0], [0, 0], [1, 1], [1, 1]]
data = np.array(data)
scaler = StandardScaler()
print(data[:, -1])
scaler.fit(data[:-1])

print(scaler.mean_)
print(scaler.var_)
