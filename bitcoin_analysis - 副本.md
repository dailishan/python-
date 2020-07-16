# -*- coding: utf-8 -*-
# 比特币走势预测，使用时间序列ARMA
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
import warnings
from itertools import product#创建一个迭代器，生成表示item1，item2等中的项目的笛卡尔积的元组，repeat是一个关键字参数，指定重复生成序列的次数。
from datetime import datetime
warnings.filterwarnings('ignore')
# 数据加载
df = pd.read_csv('./bitcoin_2012-01-01_to_2018-10-31.csv')
# print(df)
# print(df.Timestamp)
# 将时间作为df的索引
df.Timestamp = pd.to_datetime(df.Timestamp)#_pd.to_datetime方法可以解析多种不同的日期表示形式。
# print(df.Timestamp)
df.index = df.Timestamp
# print(df.index)
# 数据探索
# print(df.head())
# 按照月，季度，年来统计
df_month = df.resample('M').mean()#Pandas中的resample，重新采样，是对原样本重新处理的一个方法，是一个对常规时间序列数据重新采样和频率转换的便捷的方法。
df_Q = df.resample('Q-DEC').mean()
df_year = df.resample('A-DEC').mean()
# print('按月',df_month)
# print(df_Q)
# print(df_year)
# 按照天，月，季度，年来显示比特币的走势
# print('aaaaaaa',df.Weighted_Price)#结果与下面一样
# print('bbbbbbbbbbb',df['Weighted_Price'])#结果与上面一样
fig = plt.figure(figsize=[15,15])
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.suptitle('比特币金额（美金）', fontsize=20)
plt.subplot(221)#一个figure对象包含了多个子图，可以使用subplot（）函数来绘制子图：那么这个figure就是个2*2的矩阵图，也就是总共有4个图，1就代表了第一幅图
plt.plot(df.Weighted_Price, '-', label='按天')
plt.legend()#给图像加上图例。
plt.subplot(222)#指2行2列的第2幅图 ；
plt.plot(df_month.Weighted_Price, '-', label='按月')
plt.legend()
plt.subplot(223)
plt.plot(df_Q.Weighted_Price, '-', label='按季度')
plt.legend()
plt.subplot(224)
plt.plot(df_year.Weighted_Price, '-', label='按年')
plt.legend()
# plt.show()
# 设置参数范围
ps = range(0, 3)
qs = range(0, 3)
parameters = product(ps, qs)
# print(parameters)#无法打印
parameters_list = list(parameters)
# print(parameters_list)#打印列表，列表内有元组
# 寻找最优ARMA模型参数，即best_aic最小
results = []
best_aic = float("inf") # 正无穷
for param in parameters_list:
    try:
        model = ARMA(df_month.Weighted_Price,order=(param[0], param[1])).fit()
    except ValueError:
        print('参数错误:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
        print('最优模型',best_model)
        print('最优a',best_aic)
        print('最优参数',best_param)
    results.append([param, model.aic])#.append只接受一个参数
    print('结果',results)
# # 输出最优模型
# result_table = pd.DataFrame(results)
# print(result_table)
# result_table.columns = ['parameters', 'aic']
# print(result_table)
print('最优模型: ', best_model.summary())#通过model.summary()输出模型各层的参数状况
# 比特币预测
df_month2 = df_month[['Weighted_Price']]
print(df_month2)
date_list = [datetime(2018, 11, 30), datetime(2018, 12, 31), datetime(2019, 1, 31), datetime(2019, 2, 28), datetime(2019, 3, 31),
             datetime(2019, 4, 30), datetime(2019, 5, 31), datetime(2019, 6, 30)]
print(date_list)
future = pd.DataFrame(index=date_list, columns= df_month.columns)
#
print(future)
df_month2 = pd.concat([df_month2, future])#.concat为什么是（[]）
print(df_month2)
df_month2['forecast'] = best_model.predict(start=0, end=91)
print(df_month2)
print(df_month2.info())
# 比特币预测结果显示
plt.figure(figsize=(20,7))
df_month2.Weighted_Price.plot(label='实际金额')
df_month2.forecast.plot(color='r', ls='--', label='预测金额')
plt.legend()
plt.title('比特币金额（月）')
plt.xlabel('时间')
plt.ylabel('美金')
plt.show()
