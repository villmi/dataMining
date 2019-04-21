import DecisionTree.Tools.DecisionTreeToolPlus as dt

import pandas as pd
import json

watermelon = pd.read_csv("/Users/vill/Desktop/🍉➕.csv", encoding="utf-8", index_col="编号")

j = dt.buildDecisionTreeByGain(watermelon, "好瓜")
j = json.dumps(j, ensure_ascii=False, indent=8)
print(j)
