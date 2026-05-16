import pandas as pd
import os

# 1. 设置路径
csv_path = 'dataset/all_historical_daily_data.csv'
parquet_path = 'dataset/all_historical_daily_data.parquet'

print(f"正在读取 CSV: {csv_path} ...")
df = pd.read_csv(csv_path)

# 2. 预处理：清洗与预计算（Pre-computation）
print("正在进行数据清洗与预计算...")

# (A) 确保日期格式正确
if 'trade_date' in df.columns:
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))

# (B) 提前计算收益率 'ret' (省去后端每次运行时的重复计算)
if 'ret' not in df.columns:
    if {'close', 'pre_close'}.issubset(df.columns):
        print("正在预计算收益率 'ret'...")
        df['ret'] = (df['close'] / df['pre_close'] - 1.0).fillna(0.0)
    else:
        print("⚠️ 警告：缺失 close/pre_close，无法预计算 ret")

# (C) 设置索引并排序 (非常关键！Parquet 会记住这个结构)
# 这能让 shift/rolling 等算子速度飞快
if 'ts_code' in df.columns and 'trade_date' in df.columns:
    df = df.set_index(['trade_date', 'ts_code']).sort_index()

# 3. 保存为 Parquet
print(f"正在保存为 Parquet (Snappy 压缩): {parquet_path} ...")
df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')

print("✅ 转换完成！")
print(f"CSV 体积: {os.path.getsize(csv_path) / 1024 / 1024:.2f} MB")
print(f"Parquet 体积: {os.path.getsize(parquet_path) / 1024 / 1024:.2f} MB")