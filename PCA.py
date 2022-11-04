import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('ASD meta abundance.csv')

del df["Taxonomy"]

df_min_max_scaled = df.copy()

# apply normalization techniques
for column in df_min_max_scaled.columns:
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (
                df_min_max_scaled[column].max() - df_min_max_scaled[column].min())

u, s, v_T = np.linalg.svd(df_min_max_scaled, full_matrices=True)

fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
ax1 = plt.semilogy(s, '-o')

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')

data_array = pd.DataFrame.to_numpy(df_min_max_scaled)

for j in range(data_array.shape[0]):
    pc1 = v_T[0, :] @ data_array[j, :].T
    pc2 = v_T[1, :] @ data_array[j, :].T
    pc3 = v_T[2, :] @ data_array[j, :].T
    ax.scatter(pc1, pc2, pc3, marker='o')

plt.show()

