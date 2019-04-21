import pandas as pd
import math
import json


# 通过属性名称进行分类,dataFrame是数据集，name是属性名称，例如通过属性名"纹理"，可以分为"清晰"，"模糊"，"稍糊"三类
# 如果把useAttribute设置为True 则会同时还会返回具体的属性名称
def getTypes(dataFrame, name, useAttribute=False):
    if dataFrame[name].dtypes != float:
        types = dataFrame.groupby([name]).count().reset_index(name)
        ls = []
        # 根据属性的名称将数据分为n类
        for i in range(len(types)):
            a = dataFrame.loc[dataFrame[name] == str(types.iloc[i].iloc[0])]
            if len(a) != 0:
                ls.append(a)
        if not useAttribute:
            return ls
        else:
            attributes = []
            for i in range(len(types)):
                attributes.append(types.iloc[:, 0][i])
            return ls, attributes
    elif dataFrame[name].dtypes == float:
        dataFrame = dataFrame.sort_values(by=[name])
        candidates = []
        for i in range(len(dataFrame) - 1):
            candidate = dataFrame.iloc[i].loc[name] + dataFrame.iloc[i+1].loc[name]
            candidate = candidate / 2
            candidates.append(candidate)
        return candidates, dataFrame


# 计算信息熵，dataFrame是数据集，name是属性名称，root是最后的判断结果，比如西瓜数据集就是"好瓜"
def getEntropy(dataFrame, name, root):  # 此函数用于求根结点的信息熵
    # 根据输入属性名称获该索引的数据
    if dataFrame[name].dtypes != float:
        ls = getTypes(dataFrame=dataFrame, name=name)
        if name != root:
            entroy = []
            count = []
            # 计算该属性的信息熵
            for i in ls:
                la = getTypes(dataFrame=i, name=root)
                e = 0
                for j in la:
                    b = len(j) / len(i)
                    e -= b * math.log(b, 2)
                entroy.append(e)
                count.append(len(i))
            return entroy, count
        else:
            e = 0
            for i in ls:
                b = len(i) / len(dataFrame)
                e -= b * math.log(b, 2)
            return e
    elif dataFrame[name].dtypes == float:
        ts, dataFrame = getTypes(dataFrame, name)
        ess = []
        for t in ts:
            es = []
            a = dataFrame.loc[dataFrame[name] <= t]
            b = dataFrame.loc[dataFrame[name] > t]
            e = 0
            if len(a) + len(b) == len(dataFrame):
                la = getTypes(dataFrame=a, name=root)
                lb = getTypes(dataFrame=b, name=root)
                for j in la:
                    v = len(j) / len(a)
                    e -= v * math.log(v, 2)
                es.append(e)
                e = 0
                for j in lb:
                    v = len(j) / len(b)
                    e -= v * math.log(v, 2)
                es.append(e)
            ess.append(es)
        return ess, ts, dataFrame


# 获取信息增益
def getGains(dataFrame, root):
    d = {}
    rag = {}
    rootEntropy = getEntropy(dataFrame=dataFrame, name=root, root=root)
    for i in range(dataFrame.shape[1]):
        name = str(dataFrame.iloc[:, i].name)
        if name != root:
            en = 0
            if dataFrame[name].dtypes != float:
                pe, count = getEntropy(dataFrame=dataFrame, name=name, root=root)
                for j in range(len(pe)):
                    en += pe[j] * count[j] / len(dataFrame)
                gain = rootEntropy - en
            elif dataFrame[name].dtypes == float:
                ess, ts, dataFrame = getEntropy(dataFrame=dataFrame, name=name, root=root)
                count = 0
                mmax = 0
                for t in ts:
                    es = ess[count]
                    en = len(dataFrame.loc[dataFrame[name] <= t]) / len(dataFrame) * es[0] + len(dataFrame.loc[dataFrame[name] > t]) / len(dataFrame) * es[1]
                    count += 1
                    en = rootEntropy - en
                    # print("{}:{}".format(t, en))
                    if mmax < en:
                        mmax = en
                        rag[name] = t
                gain = mmax
            d[name] = gain
    d = pd.DataFrame.from_dict(d, orient='index')
    d = d.sort_values(by=[0], ascending=False)
    # if rag[d.iloc[0].name] is not None:
    #     return d, rag[d.iloc[0].name]
    # else:
    #     return d, 0
    try:
        vv = rag[d.iloc[0].name]
        return d, vv
    except KeyError:
        return d, 0


# 通过ID3建立决策树
def buildDecisionTreeByGain(dataFrame, root, d={}, attr=""):
    gains, rag = getGains(dataFrame=dataFrame, root=root)
    if gains.iloc[0, 0] != 0:
        frame = dataFrame
        name = str(gains.iloc[0].name)
        tps = []
        atrbs = []
        if rag == 0:
            d[name] = {}
            tps, atrbs = getTypes(dataFrame=frame, name=name, useAttribute=True)
        else:
            d["{}?{}".format(name, rag)] = {}
            dataFrame = dataFrame.sort_values(by=[name])
            tps.append(dataFrame.loc[dataFrame[name] <= rag])
            tps.append(dataFrame.loc[dataFrame[name] > rag])
            atrbs.append("<=")
            atrbs.append(">")
            name = "{}?{}".format(name, rag)
        for atrb in atrbs:
            d[name][atrb] = {}
        count = 0
        for tp in tps:
            tmp = buildDecisionTreeByGain(dataFrame=tp, root=root, attr=atrbs[count], d=d[name][atrbs[count]])
            d[name][atrbs[count]] = tmp
            count += 1
    else:
        return str(dataFrame[root].iloc[0])
    return d


# 获取信息增益率
def getGain_ratio(dataFrame, root):
    d = {}
    rootEntropy = getEntropy(dataFrame=dataFrame, name=root, root=root)
    for i in range(dataFrame.shape[1]):
        name = str(dataFrame.iloc[:, i].name)
        if name != root:
            pe, count = getEntropy(dataFrame=dataFrame, name=name, root=root)
            en = 0
            IV = 0
            for j in range(len(pe)):
                en += pe[j] * count[j] / len(dataFrame)
            for c in range(len(count)):
                v = count[c] / len(dataFrame)
                v = v * math.log2(v)
                IV -= v
            print("%s: %f" % (name, IV))
            d[name] = (rootEntropy - en) / IV
    d = pd.DataFrame.from_dict(d, orient='index')
    d = d.sort_values(by=[0], ascending=False)
    return d


# 通过C4.5建立决策树，dataFrame就是数据集,root是最后的判断结果，比如西瓜数据集就是"好瓜"
def buildDecisionTreeByGainRatio(dataFrame, root, d={}):
    gains, rag = getGain_ratio(dataFrame=dataFrame, root=root)
    if gains.iloc[0, 0] != 0:
        frame = dataFrame
        name = str(gains.iloc[0].name)
        d[name] = {}
        tps, atrbs = getTypes(dataFrame=frame, name=name, useAttribute=True)
        for atrb in atrbs:
            d[name][atrb] = {}
        count = 0
        for tp in tps:
            tmp = buildDecisionTreeByGain(dataFrame=tp, root=root, attr=atrbs[count], d=d[name][atrbs[count]])
            d[name][atrbs[count]] = tmp
            count += 1
    else:
        return str(dataFrame[root].iloc[0])
    return d


def getGini(dataFrame, name):
    types = getTypes(dataFrame=dataFrame, name=name)
    gini = 0
    for t in types:
        gini += (len(t)/len(dataFrame)) ** 2
    gini = 1 - gini
    return gini


def getGini_index(dataFrame, root):
    l = {}
    for i in range(dataFrame.shape[1]):
        name = str(dataFrame.iloc[:, i].name)
        if name != root:
            l[name] = {}
            types = dataFrame.groupby([name]).count().reset_index(name)
            for i in range(len(types)):
                attr = str(types.iloc[i].iloc[0])
                a = dataFrame.loc[dataFrame[name] == attr]
                b = dataFrame.loc[dataFrame[name] != attr]
                ga = getGini(a, root)
                gb = getGini(b, root)
                l[name][attr] = len(a)/len(dataFrame) * ga + len(b)/len(dataFrame) * gb
    return l


def buildDecisionTreeByCART(dataFrame, root, d={}):
    if len(dataFrame.groupby([root]).count().reset_index(root)) == 1:
        return str(dataFrame[root].iloc[0])
    else:
        all_gini_index = getGini_index(dataFrame, root)
        m = {}
        mmin = 1
        for t in all_gini_index:
            a = min(all_gini_index[t], key=all_gini_index[t].get)
            if all_gini_index[t][a] <= mmin:
                mmin = all_gini_index[t][a]
                m = {t: a}
        if len(m) == 1:
            l = []
            for t in m:
                d[t] = {}
                l.append(dataFrame.loc[dataFrame[t] == m[t]])
                l.append(dataFrame.loc[dataFrame[t] != m[t]])
                count = 0
                for i in l:
                    if count == 0:
                        d[t][m[t]] = {}
                        tmp = buildDecisionTreeByCART(i, root, d=d[t][m[t]])
                        d[t][m[t]] = tmp
                        count += 1
                    else:
                        d[t]["不{}".format(m[t])] = {}
                        tmp = buildDecisionTreeByCART(i, root, d=d[t]["不{}".format(m[t])])
                        d[t]["不{}".format(m[t])] = tmp
    return d


def main():
    watermelon = pd.read_csv("/Users/vill/Desktop/🍉➕.csv", encoding="utf-8", index_col="编号")
    # j = buildDecisionTreeByCART(watermelon, "好瓜")
    # j = json.dumps(j, ensure_ascii=False, indent=8)
    # print(j)
    j = buildDecisionTreeByGain(watermelon, "好瓜")
    j = json.dumps(j, ensure_ascii=False, indent=8)
    print(j)


if __name__ == '__main__':
    main()
