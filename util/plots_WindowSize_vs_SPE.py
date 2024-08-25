import pandas as pd
import matplotlib.pyplot as plt
# data = pd.read_csv('../results/WindowSize_vs_SPE.csv')
data = pd.read_csv('/home/sarthak/my_projects/argset/results/WindowSize_vs_SPE.csv')
plt.figure()
plt.errorbar(data['Window Size'], data['spe'], yerr=data['std'], fmt='.-')
plt.grid()
plt.title('Window Size vs SPE charge')
plt.show()