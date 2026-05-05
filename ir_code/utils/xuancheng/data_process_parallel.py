import pandas as pd
import json
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager, Lock
from functools import partial
import numpy as np
import os
import subprocess
import itertools

def generate_random_number():
    """生成随机数的函数，需要根据你的实际情况实现"""
    return np.random.randint(1, 10)

def parse_mixed_datetime(series):
    """解析混合格式的时间数据"""
    def parse_single_time(time_str):
        try:
            # 尝试解析为Unix时间戳（秒）
            if str(time_str).isdigit():
                return pd.to_datetime(int(time_str), unit='s')
            else:
                # 尝试解析为标准日期时间格式
                return pd.to_datetime(time_str, format='%Y-%m-%d %H:%M:%S')
        except:
            # 如果都失败，尝试自动推断格式
            return pd.to_datetime(time_str)
    
    return series.apply(parse_single_time)

# 全局包装函数，用于传递时间参数
def process_chunk_with_time_params(args):
    """包装函数，用于传递时间参数给process_chunk_efficient_global"""
    chunk, start_time, end_time = args
    return process_chunk_efficient_global(chunk, start_time, end_time)

# 高效处理单个chunk的函数 - 简化版本，不使用共享状态
def process_chunk_efficient_global(chunk, start_time, end_time):
    """高效处理单个chunk - 简化版本，带时间筛选"""
    # 使用列表收集数据，避免频繁的DataFrame操作
    data_rows = []
    
    # 记录本chunk中新增的映射
    new_mappings = {}
    
    # 时间筛选 - 先过滤符合时间条件的数据
    # 智能解析混合格式的时间数据
    chunk['from_time'] = pd.to_datetime(chunk['from_time'], format='%Y-%m-%d %H:%M:%S')
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    filtered_chunk = chunk[(chunk['from_time'] >= start_time) & (chunk['from_time'] <= end_time)]
    
    # 批量处理所有行
    for _, row in filtered_chunk.iterrows():
        cllx_lhy = row['cllx_lhy']
        
        if pd.isna(cllx_lhy):
            hphm = row['hphm']
            if hphm in new_mappings:
                cllx_lhy = new_mappings[hphm]
            else:
                # 生成新的随机数
                cllx_lhy = generate_random_number()
                new_mappings[hphm] = cllx_lhy
        
        # 直接添加到列表，避免DataFrame的loc操作
        data_rows.append([
            row['roadcl_id'], row['fnode'], row['tnode'], row['hphm'], 
            row['from_time'], row['to_time'], row['tripno'], cllx_lhy
        ])
    
    # 一次性创建DataFrame
    if data_rows:
        return pd.DataFrame(data_rows, columns=['roadcl_id', 'fnode', 'tnode', 'hphm',
                    'from_time', 'to_time', 'tripno', 'cllx_lhy'])
    else:
        return pd.DataFrame(columns=['roadcl_id', 'fnode', 'tnode', 'hphm',
                    'from_time', 'to_time', 'tripno', 'cllx_lhy'])



# 优化后的高效并发版本
def process_chunks_parallel_optimized(traffic_path, output_path, start_time, end_time, chunk_size=10000, batch_size=20):
    """优化的并行处理版本 - 高效批量处理，带时间筛选"""
    
    if os.path.exists(output_path):
        os.remove(output_path)
    
    def chunk_reader():
        """生成chunk生成器"""
        return pd.read_csv(
            traffic_path, 
            chunksize=chunk_size, 
            usecols=['roadcl_id', 'fnode', 'tnode', 'hphm',
                    'from_time', 'to_time', 'tripno', 'cllx_lhy'], 
            encoding='utf-8', 
            iterator=True,
            low_memory=False
        )
    
    # 并行处理chunks
    with Pool(processes=cpu_count()//2) as pool:
        total_rows = 0
        first_write = True
        batch_data = []
        
        # 创建参数生成器
        def param_generator():
            for chunk in chunk_reader():
                yield (chunk, start_time, end_time)
        
        # 使用imap进行流式并行处理，使用全局包装函数
        for processed_chunk in tqdm(
            pool.imap(process_chunk_with_time_params, param_generator()), 
            desc="Processing chunks", 
            unit="chunk"
        ):
            if not processed_chunk.empty:
                batch_data.append(processed_chunk)
            
            # 当达到批量大小时，写入文件
            if len(batch_data) >= batch_size:
                # 使用concat一次性合并所有数据
                combined_data = pd.concat(batch_data, ignore_index=True)
                
                # 写入文件
                if first_write:
                    combined_data.to_csv(output_path, mode='w', encoding='gbk', sep=',', index=False)
                    first_write = False
                else:
                    combined_data.to_csv(output_path, mode='a', encoding='gbk', sep=',', index=False, header=False)
                
                total_rows += len(combined_data)
                batch_data = []  # 清空批量数据
        
        # 处理剩余的数据
        if batch_data:
            combined_data = pd.concat(batch_data, ignore_index=True)
            if first_write:
                combined_data.to_csv(output_path, mode='w', encoding='gbk', sep=',', index=False)
            else:
                combined_data.to_csv(output_path, mode='a', encoding='gbk', sep=',', index=False, header=False)
            total_rows += len(combined_data)
    
    print(f"处理完成，结果保存到: {output_path}")
    print(f"总共处理 {total_rows} 行数据")
    print(f"时间范围: {start_time} 到 {end_time}")
    print(f"并行处理完成")

def process_trip(traffic_csv, output_csv=None):
    """优化的trip处理函数 - 使用最高效的向量化操作"""
    # 读取数据
    df1 = pd.read_csv(traffic_csv, usecols=['hphm','roadcl_id', 'tripno','from_time','to_time','cllx_lhy'],encoding='gbk', low_memory=False)
    
    # 向量化处理日期
    df1['date'] = pd.to_datetime(df1['from_time']).dt.strftime("%Y%m%d")
    
    # 排序 - 使用更高效的排序
    df1 = df1.sort_values(by='hphm').reset_index(drop=True)

    # 使用更高效的分组方法 - 直接使用多列分组
    grouped = df1.groupby(['hphm', 'tripno', 'cllx_lhy', 'date']).agg({
        'roadcl_id': list,
        'from_time': list,
        'to_time': list
    }).reset_index()
    
    # 向量化处理tripno - 使用字符串连接
    grouped['tripno'] = grouped['date'] + '_' + grouped['tripno'].astype(str)
    
    # 向量化处理time列表 - 使用zip和list comprehension
    grouped['time'] = [list(zip(from_times, to_times)) for from_times, to_times in zip(grouped['from_time'], grouped['to_time'])]
    
    # 创建最终DataFrame - 直接选择列
    df2 = grouped[['hphm', 'tripno', 'time', 'roadcl_id', 'cllx_lhy']].copy()
    df2.columns = ['hphm', 'tripno', 'time', 'route', 'cllx_lhy']
    

    df2.to_csv(output_csv, mode='w', encoding='gbk', sep=',', index=False)   

# 使用示例
if __name__ == "__main__":
    traffic_path = r'F:/BaiduNetdiskDownload/fw/postgres/data_file/s_xc_rt_segm_inout_new_temp_20240416.csv'
    for i in range(10,35,7):
        if i<0:
            start_time = f'2023-04-01 00:00:00'
        elif 0<i <30:
            start_time = f'2023-04-{i:02d} 00:00:00'
        else:
            start_time = f'2023-05-{i-30:02d} 00:00:00'
        if i<0:
            end_time = f'2023-04-03 00:00:00'
        elif 7<i+7<30:
            end_time = f'2023-04-{i+7:02d} 00:00:00'
        else:
            end_time = f'2023-05-{i+7-30:02d} 00:00:00'
        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)
        start_day = f'{start_time.date().month :02d}_{start_time.date().day :02d}'
        end_day = f'{end_time.date().month :02d}_{end_time.date().day :02d}'
        
        data_dir = 'I:/data/fw/postgres/data_file'
        output_path = os.path.join(data_dir, f'data_2023_{start_day}_2023_{end_day}.csv')
        
        start = time.time()

        # process_chunks_parallel_optimized(traffic_path, output_path, start_time, end_time, chunk_size=10000, batch_size=20)

        
        # 然后处理trip，使用生成的过滤后文件作为输入
        trip_path = os.path.join(data_dir, f'trip_2023_{start_day}_2023_{end_day}.csv')
        process_trip(output_path, trip_path)  # 使用output_path作为输入，输出到trip_path
        end = time.time()
        print(f"总耗时: {end - start:.2f} 秒")
    