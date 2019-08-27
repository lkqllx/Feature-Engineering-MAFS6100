import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def heatmap(df, name):
    plt.figure(figsize=(20, 12))
    sns.heatmap(df, vmin=-1, vmax=1, xticklabels=1, cmap="YlGnBu")
    plt.title(name, fontsize=24)
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('mean.csv', index_col=0)
    df.sort_values(by=df.index[-1], inplace=True, axis=1)
    heatmap(df, 'Linear Regression Model')

    df = pd.read_csv('ridge_mean.csv', index_col=0)
    df.sort_values(by=df.index[-1], inplace=True, axis=1)
    heatmap(df, 'Ridge Regression Model')
