import operator
import math

class DecisionTree:
    def __init__(self):
        pass

    # 初始化数据
    def loadData(self):
        """
        天气：[晴，阴，雨] -> [2, 1, 0]
        温度：[炎热，适中，寒冷] -> [2, 1, 0]
        湿度：[高，低] -> [1, 0]
        风速：[强，弱] -> [1, 0]
        举办活动：[是，否] -> [yes, no]
        :return:
        """
        data = [
            [2,2,1,0, "yes"],
            [2,2,1,1, "no"],
            [1,2,1,0, "yes"],
            [0,0,0,0, "yes"],
            [0,0,0,1, "no"],
            [1,0,0,1, "yes"],
            [2,1,1,0, "no"],
            [2,0,0,0, "yes"],
            [0,1,0,0, "yes"],
            [2,1,0,1, "yes"],
            [1,2,0,0, "no"],
            [0,1,1,1, "no"]
        ]
        # 属性
        features = ["天气","温度","湿度","风速"]
        return data, features

    # 计算信息熵:E(D)
    def ShannoEnt(self, data):
        numData = len(data)
        labelCounts = dict()    # 用于统计label

        for feature in data:
            label = feature[-1] # label是这行数据的最后一个元素
            labelCounts.setdefault(label,0)
            labelCounts[label] += 1

        shannoEnt = 0.0
        for key in labelCounts:
            # 计算每个label的概率：出现次数/总次数
            prob = float(labelCounts[key])/numData
            # 计算信息熵
            shannoEnt -= prob * math.log2(prob)
        return shannoEnt

    def splitData(self, data, axis, value):
        """
        :param data: 数据集
        :param axis: 属性下标，表示天气、温度、湿度、风速
        :param value: 指定值
        :return:
        """
        retData = []
        for feature in data:
            if feature[axis] == value:
                # 将拥有相同特征的数据集抽取出来
                reducedFeature = feature[:axis]
                print(reducedFeature)
                reducedFeature.extend(feature[axis+1:]) # 追加元素
                print(reducedFeature)
                retData.append(reducedFeature)  # 添加一个对象
        return retData

    # 计算信息增益，得出最好的数据集划分方式
    def chooseBestFeatureToSplit(self,data):
        numFeature = len(data[0])-1 # 获取属性数量，最后一个不是属性
        baseEntropy = self.ShannoEnt(data)  # 获取信息熵E(D)
        bestInfoGain = 0.0
        bestFeature = -1
        for i in range(numFeature):
            # 获取第i个特征的所有取值
            featureList = [result[i] for result in data]
            # 去重
            uniqueFeatureList = set(featureList)

            newEntropy = 0.0
            for value in uniqueFeatureList:
                # 得到特征为i，值为value的数据集，以便计算E(D|A)
                splitDataSet = self.splitData(data, i, value)
                # 计算p(D|A=value) = num(A=value)/D总次数
                prob = len(splitDataSet) / float(len(data))
                # 计算E(D|A)
                newEntropy += prob * self.ShannoEnt(splitDataSet)
            # 计算增益
            infoGain = baseEntropy - newEntropy

            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    # 递归构建决策树
    def majorityCnt(self, yesOrNoList):
        labelCount = dict()
        for vote in yesOrNoList:
            labelCount.setdefault(vote,0)
            labelCount[vote] += 1
        sortedLabelCount = sorted(labelCount.items(),key=lambda x:x[1],reverse=True)
        print(sortedLabelCount)
        return sortedLabelCount[0][0]

    # 创建决策树
    def createTree(self, data, features):
        """
        :param data: 原始数据
        :param features: 特征属性：天气，温度，湿度，风速
        :return:
        """
        # 使用“=”产生新变量，实际上两者是一样的，避免后面del()函数对原变量产生影响
        features = list(features)
        # 标签集合
        yesOrNoList = [line[-1] for line in data]
        print("yesOrNoList:\nbegin\n{}\nend".format(yesOrNoList))
        if yesOrNoList.count(yesOrNoList[0]) == len(yesOrNoList):
            return yesOrNoList[0]

        if len(data[0]) == 1:
            return self.majorityCnt(yesOrNoList)

        # 根据信息熵的增益，选取最优属性
        bestFeature = self.chooseBestFeatureToSplit(data)
        # 获取标签对应的值
        bestFeaLabel = features[bestFeature]
        # 以最优属性创建一个节点
        myTree = {bestFeaLabel:{}}
        featureValues = [example[bestFeature] for example in data]
        uniqueFeatureValues = set(featureValues)
        #   清空features[bestFeature]，在下一次使用时清零.即在features中删除当前最优属性值
        del (features[bestFeature])
        # 遍历当前最优属性的value
        for value in uniqueFeatureValues:
            subFeatures = features[:]
            # 递归创建决策树
            myTree[bestFeaLabel][value] = self.createTree(
                self.splitData(data,bestFeature,value), subFeatures
            )
        return myTree

    def predict(self, tree, features, x):
        print("tree:\n",tree)
        for key1 in tree.keys():
            secondDict = tree[key1]
            featIndex = features.index(key1)
            for key2 in secondDict.keys():
                if x[featIndex] == key2:
                    if type(secondDict[key2]).__name__ == "dict":
                        classLabel = self.predict(secondDict[key2],features,x)
                    else:
                        classLabel = secondDict[key2]
        return classLabel

if __name__ == '__main__':
    dtree = DecisionTree()
    data, features = dtree.loadData()
    myTree = dtree.createTree(data,features)
    print("myTree:\n",myTree)
    label = dtree.predict(myTree, features, [1,1,1,0])
    print("新数据[1,1,1,0]对应的是否举办活动为：{}".format(label))