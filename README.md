# data-science-project

import pandas as pd
data = {

    "Watch Time": [1.5, 1.0, 1.5, 1.5, 2.0, 2.5, 3.0, 4.0, 2.0, 6.0],
    "Language": ["English", "English", "Hindi", "Hindi", "English", "Hindi", "Hindi", "Marathi", "Telugu", "Tamil"],
    "Age": [12, 13, 11, 15, 12, 15, 10, 6, 45, 60]
}

df = pd.DataFrame(data)
df
**Univariate Analysis **
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.histplot(df['Watch Time'], kde=True)
plt.title('Distribution of Watch Time')
plt.xlabel('Watch Time')
plt.ylabel('Density')
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.boxplot(x='Watch Time', data=df)
plt.title('Distribution of Watch Time using BoxPlot')
plt.xlabel('Watch Time')
plt.show()

plt.figure(figsize=(8,6))
sns.violinplot(x='Watch Time', data=df)
plt.title('Distribution of Watch Time using ViolinPlot')
plt.xlabel('Watch Time')
plt.show()
***Measures of Central Tendency***
print("Mean:", df['Watch Time'].mean())
print("Median:", df['Watch Time'].median())
print("Mode:", df['Watch Time'].mode().iloc[0])
***Standard Deviation***
print(df[["Watch Time", "Age"]].std())
print(df[["Watch Time"]].std())
print(df['Watch Time'].std(ddof=0))
print(df['Age'].std(ddof=0))
Source : Perplexity


The difference between .std() and .std(ddof=0) in Python's NumPy and Pandas libraries arises from the way the standard deviation is calculated, specifically in the denominator used in the formula124.

.std(): This function calculates the sample standard deviation, which is used when you're working with a sample of a larger population and want to estimate the population standard deviation. By default, it uses N-1 in the denominator, where N is the number of elements in the dataset14. The ddof argument stands for "Delta Degrees of Freedom", and its default value is 1, leading to the use of N-1 in the denominator1. This is also known as Bessel's correction2.

.std(ddof=0): This calculates the population standard deviation, which is used when you have data for the entire population. It uses N in the denominator14. Setting ddof=0 makes the calculation match the formula for the population standard deviation
correlation= df.select_dtypes(exclude='object_').corr()
print(correlation)
plt.figure(figsize=(8,6))
sns.heatmap(correlation, annot=True, cmap='coolwarm',fmt=".2f")
plt.title('Corr')
plt.show()
***Bivariate Analysis*** 
sns.barplot(x="Language",y="Age",data=df,palette="inferno")
sns.barplot(x="Language",y="Watch Time",data=df,palette="crest")
sns.boxplot(x="Language",y="Watch Time",data=df,palette="crest")
sns.violinplot(x="Watch Time",y="Age",data=df,palette="mako")
import pandas as pd
data02 = {

    "Watch Time": [6.0, 5.5, 5.0, 4.5, 4.0, 3.0, 6.0, 7.5, 8.0 ,9.0],
    "Occupation": ["KinderGarten Toddler", "Grade 3 Student", "Grade 4 Student", "Grade 7 Student", "Grade 7 Student", "Grade 10 Student",
                   "Summer Break Employee", "Home Maker", "Home Maker", "Retired Officer"],
    "Language": ["English", "English", "Hindi", "Hindi", "English", "Hindi", "Hindi", "Marathi", "Telugu", "Tamil"],
    "Age": [5, 9, 10, 12, 12, 15, 28, 32, 45, 62]
}

df02 = pd.DataFrame(data02)
df02
***Univariate Analysis ***
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.histplot(df02['Watch Time'], kde=True)
plt.title('Distribution of Watch Time')
plt.xlabel('Watch Time')
plt.ylabel('Density')
plt.show()
print("Mean:", df02['Watch Time'].mean())
print("Median:", df02['Watch Time'].median())
print("Mode:", df02['Watch Time'].mode().iloc[0])
***Using ggplot histogram from Plotnine - Watch Time***
from plotnine import ggplot, aes, geom_histogram, labs

histogram = (ggplot(df02, aes(x='Watch Time')) +
             geom_histogram(binwidth=1, fill='cornflowerblue', color='black') +
             labs(title='Histogram of Points',
                  x='Watch Time',
                  y='Frequency'))

# Save histogram to file
histogram.save("histogram.png")

print(histogram)
***Bivariate Analysis : Positive and Negative Correlation using Scatter Plot***
from plotnine import ggplot, aes, geom_histogram, labs, geom_point, geom_smooth
(ggplot(df02, aes(x='Watch Time', y='Age'))
 + geom_point(color='blue', size=3)
 + labs(title='Age vs. Watch Time',
        x='Watch Time',
        y='Age'))
Positive Correlation : As 'Age' increases from 28 onwards, we rightly observe that the 'Watch Time' also increases.

Therefore, positive increase of one variable causing simultanous increase in the other as well is known as Positive Correlation.
Negative Correlation : As 'Age' increases from 5 until 15, we see a decline in the 'Watch Time'.

Therefore, negative decrease of one variable causing simultanous increase in the otheri s known as Negative Correlation.
***Mutivariate Analysis using 3D Plot***
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

data03 = {

    "Watch Time": [6.0, 5.5, 5.0, 4.5, 4.0, 3.0, 6.0, 7.5, 8.0 ,9.0],
    "Occupation": ["KinderGarten Toddler", "Grade 3 Student", "Grade 4 Student", "Grade 7 Student", "Grade 7 Student", "Grade 10 Student",
                   "Summer Break Employee", "Home Maker", "Home Maker", "Retired Officer"],
    "Language": ["English", "English", "Hindi", "Hindi", "English", "Hindi", "Hindi", "Marathi", "Telugu", "Tamil"],
    "Behaviour_Score": [2,4,6,8,10,12,14,16,18,20],
    "Age": [5, 9, 10, 12, 12, 15, 28, 32, 45, 62],
}

df03 = pd.DataFrame(data03)

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Use different colors for different categories
colors = {'English': 'red', 'Hindi': 'green', 'Marathi': 'blue', 'Telugu':'pink', 'Tamil':'purple'}

# Scatter plot
for lang in df03['Language'].unique():
    subset = df03[df03['Language'] == lang]
    ax.scatter(subset['Watch Time'], subset['Age'], subset['Behaviour_Score'],
               c=colors[lang], label=lang, marker='o', s=50)

# Set labels and title
ax.set_xlabel('Watch Time')
ax.set_ylabel('Age')
ax.set_zlabel('Behaviour_Score')
ax.set_title('3D Scatter Plot')

# Add legend
ax.legend()

# Show the plot
plt.show()
