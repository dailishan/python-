
# coding:utf-8
# 用ARMA进行时间序列预测
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm #Statsmodels 是 Python 中一个强大的统计分析包，包含了回归分析、时间序列分析、假设检
# 验等等的功能。
from statsmodels.tsa.arima_model import ARMA
# from statsmodels.graphics.api import qqplot
# 创建数据
data = [5922, 5308, 5546, 5975, 2704, 1767, 4111, 5542, 4726, 5866, 6183, 3199, 1471, 1325, 6618, 6644, 5337, 7064, 2912, 1456, 4705, 4579, 4990, 4331, 4481, 1813, 1258, 4383, 5451, 5169, 5362, 6259, 3743, 2268, 5397, 5821, 6115, 6631, 6474, 4134, 2728, 5753, 7130, 7860, 6991, 7499, 5301, 2808, 6755, 6658, 7644, 6472, 8680, 6366, 5252, 8223, 8181, 10548, 11823, 14640, 9873, 6613, 14415, 13204, 14982, 9690, 10693, 8276, 4519, 7865, 8137, 10022, 7646, 8749, 5246, 4736, 9705, 7501, 9587, 10078, 9732, 6986, 4385, 8451, 9815, 10894, 10287, 9666, 6072, 5418]
data=pd.Series(data)
data_index = sm.tsa.datetools.dates_from_range('1901','1990')
# 绘制数据图
data.index = pd.Index(data_index)
data.plot(figsize=(12,8))
plt.show()
# 创建ARMA模型# 创建ARMA模型
arma = ARMA(data,(7,0)).fit()
print('AIC: %0.4lf' %arma.aic)
# 模型预测
predict_y = arma.predict('1990', '2000')
#在matplotlib一般使用plt.figure来设置窗口尺寸。plt.figure(figsize=(a, b))
# 预测结果绘制,plt.subplots()是一个函数，返回一个包含figure和axes对象的元组。
# fig代表绘图窗口(Figure)；ax代表这个绘图窗口上的坐标系(axis)，一般会继续对ax进行操作。
fig, ax = plt.subplots(figsize=(12, 8))
print('图',fig)#图长宽大小比率
print('斧',ax)#
# ax = data.ix['1901':].plot(ax=ax)
ax = data.loc['1901':].plot(ax=ax)#ix / loc 可以通过行号和行标签进行索引，比如 df.loc['a'] , df.loc[1], df.ix['a'] , df.ix[1]
# 而iloc只能通过行号索引 , df.iloc[0] 是对的, 而df.iloc['a'] 是错误的
# 建议：当用行号索引的时候, 尽量用 il oc 来进行索引; 而用标签索引的时候用 loc ,  ix 尽量别用。
predict_y.plot(ax=ax)
plt.show()