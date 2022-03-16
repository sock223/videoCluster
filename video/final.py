"""
目标： 通过找用户共同点评过的电影，将用户分组。从而推荐用户喜欢的电影。

优化空间： 1：当前未对电影进行分类，可对电影进行分类细化后，分别分析用户对各类电影的喜好，从而分组。

"""

""" ************** import ************** """
from pyspark import SparkContext, SparkConf
import pandas as pd
import pymysql

pymysql.install_as_MySQLdb()
from pyspark.sql import SparkSession, SQLContext
import numpy as np
from matplotlib import pyplot as plt
from pyspark.sql import functions as F

""" import end """

""" ************** variables **************"""
threshold = 3  # 最少看过多少部重复的电影。
var_threshold = 0.3
movie_least_num = 150
person_a = []
person_b = []
value = []
corr = pd.DataFrame()

""" variables end """

""" ************** function **************"""


def cal_loss(person1, person2):
    """
    :param person1: 用户a [用户id 电影1打分 电影2打分 ... ]
    :param person2: 用户b [用户id 电影1打分 电影2打分 ... ]
    :return: None

    根据ab用户重叠的电影打分，来计算打分差的方差
    """
    # p1 p2 为 用户对*相同*电影打分的集合
    p1 = []
    p2 = []

    for i in range(1, len(person1)):
        if not pd.isna(person1[i]) and not pd.isna(person2[i]) and person1[i] > 0 and person2[i] > 0:
            p1.append(person1[i])
            p2.append(person2[i])

    """
        核函数：低维不可分 -> 高维可分  得有标签ß
        用来建模预测新用户的分类。

        本次聚类，没有标签。

        n维空间上的某两个点 之间的夹角      
    """
    if len(p1) >= threshold and len(p2) >= threshold:
        # cos_value = sum( a * b for a, b in zip(p1,p2)) / (sum( a * a for a in p1) * sum( b*b for b in p2))
        var_value = np.var(np.array([a - b for a, b in zip(p1, p2)]))

        person_a.append(person1[0])
        person_b.append(person2[0])

        # value.append(cos_value)
        value.append(var_value)

        return value
    else:
        return None


def add_cluster(old, p=None, c=None):
    """
    :param old: 原有的cluster  [ [用户a, 用户b] , 当前分数和 : dataframe ]
    :param p: 新用户的打分情况 : dataframe
    :param c: 新的cluster
    :return: 是否可以加入该cluster , 新的分数和
    """
    score1 = old[1]
    score2 = p.apply(lambda x: x * (-1)) if p is not None else c[1].apply(lambda x: x * (-1))

    # 删除 列中任有一个是na的
    df = pd.concat([score1, score2], axis=0)
    df = df.dropna(axis=1, how='any')
    # print(df)
    var_value = df.sum(axis=0).var()
    # print(var_value)
    if var_value <= var_threshold:
        score3 = old[1]
        for t in range(len(old[0]) - 1):
            #  插入N行平均值
            score3.loc[score3.shape[0]] = old[1].iloc[0, :]

        # print(len(old[0]))
        # print(score3)
        score_mean = pd.concat([score3, p if p is not None else c[1]], axis=0).mean(axis=0).to_frame().T

        return True, score_mean
    else:
        return False, None


def my_cluster(d):
    """
    :param d:  用户打分方差数据集     col = [用户a id , 用户b id, 打分差值的方差]
    :return:
    1: 从大到小排列 用户相关性的表
    2: 如果两个用户都没在已分配列表中，则新建一个簇，并添加至已分配列表
    3：如果其中一个用户在已分配列表中，则另一个也加入相应簇，并添加至已分配列表
    4：如果两个都在已分配列表中，则两个簇合并。
        PS: 如果不合并， [A,B]  [C,D]  [A,C] 则 D 会以C的喜好推荐，而不会以AB的喜好推荐。

    5：问题： 如此处理后，所有用户均分入同一个簇，所以当value到该聚类重心的距离大于某一个阈值后，认为两个用户不属于同一个簇
    """
    n = 1
    assigned = []
    cluster = []  # [ [ [用户a, 用户b] , 当前分数均值 : dataframe ] , [ [用户c, 用户d] , 当前分数均值 : dataframe ] ]

    for a, b, v in d.values.tolist():

        if n % 100 == 0:
            print(n, ' / ', len(d))

        filter_user = df_pivot.filter((df_pivot['userId'] == a) | (df_pivot['userId'] == b)).toPandas()
        if a not in assigned and b not in assigned:
            cluster.append([[a, b], filter_user.iloc[:, 1:].mean(axis=0).to_frame().T])
            assigned.append(a)
            assigned.append(b)
            # print(cluster)

        elif (a in assigned and b not in assigned) or (a not in assigned and b in assigned):
            for c in cluster:
                if a in c[0] or b in c[0]:
                    able, score = add_cluster(c,
                                              p=filter_user[filter_user['userId'] == (b if a in assigned else a)].iloc[
                                                :, 1:])
                    if able:
                        c[0].append(b if a in assigned else a)
                        c[1] = score
                        assigned.append(b if a in assigned else a)
                    else:
                        pass

        else:  # [a,b,c,d] a,b    [a,c] [b,d] a,b
            # 【a,b 】 [c,d]   b,d => [a,b,c,d]   ,   a,c = >
            clusterA = None
            clusterB = None
            for c in cluster:
                if a in c[0]:
                    clusterA = c
                if b in c[0]:
                    clusterB = c

            if clusterA[0] == clusterB[0]:
                pass  # a,b 由于 [c,a] [d,b] 根据 [c,d] 已经合并成 [a,b,c,d] 无需处理
            else:
                # 移出[a,c] [b,d] 添加 [a,b,c,d]
                able, score = add_cluster(clusterA, c=clusterB)
                if able:
                    cluster.remove(clusterA)
                    cluster.remove(clusterB)
                    clusterA[0].extend(clusterB[0])
                    new_cluster = [[i for i in set(clusterA[0])], score]
                    cluster.append(new_cluster)
                    print(cluster)
                else:
                    pass
        n += 1
    return cluster


"""  function end """

""" ************** 数据获取 ************** """
spark = SparkSession. \
    Builder(). \
    appName('Rating'). \
    master('local'). \
    getOrCreate()

rating = spark.read.format('csv').option('header', True).option('multiline', True).load(
    '/Users/yibo/PycharmProjects/xingshulin/spark测试/video/dataset/ratings.csv')
rating = rating.select('userId', 'movieId', 'rating')
rating.createOrReplaceTempView('ratings')
# rating.show()


r = spark.sql('select count(1) from ratings')

# movie = spark.read.format('csv').option('header',True).option('multiline',True).load('/Users/yibo/PycharmProjects/xingshulin/spark测试/video/data/movies.csv')
# movie.createOrReplaceTempView('movies')


# 获取被打分电影个数
r1 = spark.sql('select count(distinct movieId) from ratings')
r1.show()

# 预期：取点评数前7000的电影作为训练集
# 实际：最多的点评的电影只有300多条点评。比预期较少（如果电影点评较少，那么矩阵过于稀疏），所以，取点评数大于150的电影 共计43部
# train_movies = spark.sql("select movieId, count(1) from ratings group by movieId order by count(1) desc")
train_movies = spark.sql("select movieId from ratings group by movieId having count(1) >= " + str(movie_least_num))
train_movies.show()
movies_list = train_movies.toPandas()['movieId'].tolist()
# print(movies_list)
print('涉及 ', train_movies.count(), ' 部电影')
movies_str = '"' + '","'.join(train_movies.toPandas()['movieId'].tolist()) + '"'

# 行转列 行：用户id  列：电影id
sql_content = """select * from ratings
                 pivot
                 (
                     sum(`rating`) for
                     `movieId` in (""" + movies_str + """)
                 )
              """

df_pivot = spark.sql(sql_content).dropna(how='all')
df_pivot.show()
df_pivot.createOrReplaceTempView('rating_matrix')
print('对 ', df_pivot.count(), ' 用户进行分类')

""" 数据获取 end  """

""" ************** 相关性计算 ************** """

data = df_pivot.collect()
for i in range(0, len(data)):
    for j in range(i, len(data)):
        # 如果a用户的用户id 不等于 b用户的用户id

        if data[i][0] != data[j][0]:
            a = data[i][:]
            b = data[j][:]
            cal_loss(a, b)

corr['person_a'] = person_a
corr['person_b'] = person_b
corr['value'] = value

""" 相关性计算 end """

""" ************** 聚类 ************** """

# 研究的用户群体
user_list = corr.iloc[:, 0].unique().tolist()
user_list.extend(corr.iloc[:, 1].unique().tolist())
user_list = set(user_list)

print('尽管 有610名用户针对43部电影进行了打分，但是有打分重合3部以上的用户 有', len(user_list))

corr = corr.sort_values(by='value', ascending=True)
# plt.figure('相关性分布-直方图')
# plt.hist(corr['value'].values.tolist(), bins=100, facecolor='blue', alpha=0.6)
# plt.show()


cluster = my_cluster(corr.iloc[:60000,:])


""" 聚类 end"""



"""
结果
"""

for i in cluster:
    i[1] = i[1].values.tolist()
    # print(i)

dddd= pd.DataFrame(cluster)

print(dddd)



train_movies_test = spark.sql("select movieId from ratings group by movieId having count(1) between " + str(75) + " and " + str(movie_least_num))
train_movies_test.show()
# print(movies_list)
print('涉及 ', train_movies_test.count(), ' 部电影')
movies_str_test = '"' + '","'.join(train_movies_test.toPandas()['movieId'].tolist()) + '"'


sql_content_test = """select * from ratings
                 pivot
                 (
                     sum(`rating`) for
                     `movieId` in (""" + movies_str_test + """)
                 )
              """
df_pivot_test = spark.sql(sql_content_test).dropna(how='all')
df_pivot_test.show()
df_pivot_test.createTempView('test_data')




var_list = []
for u in cluster:
    test = df_pivot_test.filter(F.col('userId').isin(u[0]))

    #print("cluster:")
    #test.show()

    loss_list = []
    data = test.collect()
    for i in range(0, len(data)):
        for j in range(i, len(data)):
            # 如果a用户的用户id 不等于 b用户的用户id

            if data[i][0] != data[j][0]:
                a = data[i][:]
                b = data[j][:]
                var = cal_loss(a, b)
                if var is not None:
                    loss_list.append(var)
    # print(loss_list)
    print("平均方差:", np.average(np.array(loss_list)))
    var_list.append(np.average(np.array(loss_list)))



print(var_list)
spark.stop()
