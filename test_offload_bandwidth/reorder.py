import pandas as pd

# 1️⃣ 读取原始 CSV
df = pd.read_csv("test_offload_strategy.csv")

column_name = input("请输入要排序的列名：")

if column_name not in df.columns:
    raise ValueError(f"CSV 文件中缺少 '{column_name}' 列。")

# 按用户输入的列名排序
df_sorted = df.sort_values(by=column_name, kind="stable")

# 输出排序后结果
output_filename = f"test_offload_{column_name}_reordered.csv"
df_sorted.to_csv(output_filename, index=False)
print(f"✅ 已按 {column_name} 排序，并保存到 {output_filename}")