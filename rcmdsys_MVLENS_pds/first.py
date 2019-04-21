import pandas as pd
import numpy as np


def getSimilarity():
    pass

df = pd.read_csv('/Users/vill/Desktop/推荐系统导论/ml-100k/u2.base.csv', sep='\t', header=None)
df.columns = ['userId', 'movieId', 'rate', 'time']
# df.set_index("userId")
# df.set_index("movieId")
# df.set_index("rate")
user1 = df.loc[df['userId'] == 1]
user2 = df.loc[df['userId'] == 2]
print(user2)
m = 0
n = 0
# print(len(user2))

# print(user2.iloc[0].loc['rate'])
print(user2.loc[df['rate'] > 3, df['movieId'] > 200])
