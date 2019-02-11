import seaborn as sns
df = pd.DataFrame(d)
sns.set(style='darkgrid')
sns.lineplot(x='num', y='sqr', data=df)
sns.boxplot(data=df,x='overall_rating',y='name')
sns.kdeplot(x, shade=True, color="r") #kernel Density plot 
sns.distplot(x, shade=True, color="r") #Histogram and Density Plot
sns.barplot(x, shade=True, color="r")
sns.countplot(x="class", data=titanic) #Count Plot
