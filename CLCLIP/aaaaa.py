import pandas as pd
import numpy as np

def replace_decimal_with_random(filepath):
    # 读取Excel文件
    df = pd.read_excel(filepath)

    # 遍历DataFrame，替换小数部分
    for col in df.columns:
        # 只处理数值类型的列
        if pd.api.types.is_numeric_dtype(df[col]):
            # 跳过空数据
            df[col] = df[col].apply(lambda x: np.floor(x) + np.random.rand() if pd.notnull(x) else x)

    # 保存结果到原路径
    df.to_excel(filepath, index=False)

# 调用函数
replace_decimal_with_random('/Users/lwz/Desktop/debug.xlsx')
