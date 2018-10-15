import seaborn as sns
df = pd.DataFrame(d)
sns.set(style='darkgrid')
sns.lineplot(x='num', y='sqr', data=df)