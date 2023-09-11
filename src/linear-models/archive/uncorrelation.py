import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

a = np.array([1, 12, 30, 4, 15])
b = np.array([1, 6, 7, 8, -13])
y = (np.sqrt(a) + np.sin(b)).round(3)

df = pd.DataFrame({"A": a, "B": b, "y": y})

print(df)

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(6, 6))
ax1.scatter(df.A, df.y)
ax1.set_xlabel("A")
ax1.set_ylabel("y")
ax2.scatter(df.B, df.y)
ax2.set_xlabel("B")
fig.suptitle("Uncorrelated data")
plt.show()
