import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
import pandas as pd
df=pd.read_excel('titanic.xls')
original_df=pd.DataFrame.copy(df)
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
def handle_non_numerical_data(df):
    columns=df.columns.values
    for column in columns:
        text_digit_vals={}
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[column].dtype!=np.int64 and df[column].dtype!=np.float64:
            column_contents=df[column].values.tolist()
            unique_elements=set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique]=x
                    x+=1
            df[column]=list(map(convert_to_int, df[column]))
    return df
df=handle_non_numerical_data(df)
df.drop(['ticket','home.dest'], 1, inplace=True)
X=np.array(df.drop(['survived'], 1).astype(float))
X=preprocessing.scale(X)
y=np.array(df['survived'])

#centers=[[1,1,1],[5,5,5],[3,10,10]]
#X, _= make_blobs(n_samples=100, centers=centers, cluster_std=1)
ms=MeanShift()
ms.fit(X)
labels=ms.labels_
cluster_centers=ms.cluster_centers_
original_df['cluster_group']=np.nan
for i in range(len(X)):
    original_df['cluster_group'].iloc[i]=labels[i]
n_clusters_=len(np.unique(labels))
survival_rates={}
for i in range(n_clusters_):
    temp_df=original_df[(original_df['cluster_group']==float(i))]
    #print("temp_df: ", temp_df)
    survival_cluster=temp_df[(temp_df['survived']==1)]
    survival_rate=len(survival_cluster)/len(temp_df)
    survival_rates[i]=survival_rate
print(survival_rates)
'''
print(cluster_centers)
n_clusters_=len(np.unique(labels))
print("No. of estimated clusters: ", n_clusters_)
colors=10*['r','g','b','c','k','y','m']
fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')
for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')
ax.scatter(cluster_centers[:,0], cluster_centers[:,1], cluster_centers[:,2], marker='x', color='k', s=150, linewidths=5, zorder=10)
plt.show()
'''