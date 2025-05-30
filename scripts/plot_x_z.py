import pandas as pd
import matplotlib.pyplot as plt
import glob

# 遍历 data 目录下所有 csv 文件
for csv_file in glob.glob('./data/low/*.csv'):
    # 读取数据，自动跳过第一行标题
    df = pd.read_csv(csv_file)
    # 绘制 x-z 散点图
    plt.scatter(df['x'], df['z'], s=2, label=csv_file.split('/')[-1])

plt.xlabel('x')
plt.ylabel('z')
plt.title('x-z Visualization')
plt.legend()
plt.show()