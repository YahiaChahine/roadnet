import pandas as pd
import ast
import orjson
import json
import re
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
import os
import pandas as pd
from tqdm import tqdm

def extract_times_from_filename(csv_path):
    """从CSV文件名中提取开始时间和结束时间（月-日格式）"""
    # 获取文件名（不包含路径）
    filename = os.path.basename(csv_path)
    
    # 使用正则表达式匹配时间格式
    # 匹配 data_YYYY_MM_DD_YYYY_MM_DD.csv 格式
    pattern = r'trip_(\d{4})_(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})\.csv'
    match = re.search(pattern, filename)
    
    if match:
        start_year, start_month, start_day, end_year, end_month, end_day = match.groups()
        
        # 构建完整的时间字符串（包含年份）
        start_time_str = f"{start_year}-{start_month}-{start_day} 00:00:00"
        end_time_str = f"{end_year}-{end_month}-{end_day} 23:59:59"
        
        # 返回月-日格式的日期字符串（用于文件名）
        start_date_str = f"{start_month}_{start_day}"
        end_date_str = f"{end_month}_{end_day}"
        
        return start_time_str, end_time_str, start_date_str, end_date_str
    else:
        # 如果无法从文件名提取，返回默认值
        print(f"警告：无法从文件名 {filename} 中提取时间信息")
        return "2023-04-01 00:00:00", "2023-04-03 23:59:59", "04_01", "04_03"

def load_roads():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, 'roads.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        roads_data = json.load(f)
    # 去除roads中的非数字部分并转换为int
    return [int(re.sub(r'\D', '', road)) for road in roads_data]

def filter_route_and_time(route, time):
    # 1️⃣ **筛选 `route` 只保留 `roads` 中的部分，同时同步 `time`**
    filtered_route = [r for r in route if r in ROADS]
    filtered_time = [t for r, t in zip(route, time) if r in ROADS]  # 确保 `time` 同步变化
    return filtered_route, filtered_time if filtered_route else ([], [])

def sort_route_and_time(route, time):
    combined = list(zip(route, time))
    combined.sort(key=lambda x: x[1][0])
    # 解压缩回 route 和 time
    sorted_route, sorted_time = zip(*combined)
    return list(sorted_route), list(sorted_time)

def remove_duplicate_routes(route, time):
    if not route or not time:
        return [], []
    
    seen = {}  # 记录每个路径第一次出现的位置
    to_remove = set()  # 记录需要删除的索引
    for i, r in enumerate(route):
        if r in seen:
            first_index = seen[r]
            current_index = i
            for j in range(first_index, current_index):
                to_remove.add(j)
            seen[r] = current_index
        else:
            seen[r] = i
    filtered_route = [route[i] for i in range(len(route)) if i not in to_remove]
    filtered_time = [time[i] for i in range(len(time)) if i not in to_remove]
    
    return filtered_route, filtered_time


def filter_chunk_in_time(chunk,start_time):
    # 过滤符合时间范围的数据
    chunk['start_time'] = chunk['time'].str.extract(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    chunk['start_time'] = pd.to_datetime(chunk['start_time'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    filtered_chunk = chunk[(chunk['start_time'] >= start_time) & (chunk['start_time'] <= start_time + pd.Timedelta(days=1))].copy()
    filtered_chunk.loc[:, 'start_time'] = (filtered_chunk['start_time'] - start_time).dt.total_seconds()
    filtered_chunk.loc[:, 'route'] = filtered_chunk['route'].apply(ast.literal_eval)
    filtered_chunk.loc[:, 'time'] = filtered_chunk['time'].apply(ast.literal_eval)
    filtered_chunk.loc[:, 'time'] = filtered_chunk['time'].apply(
        lambda x: [pd.to_datetime(i, format="%Y-%m-%d %H:%M:%S", errors='coerce') for i in x])
    filtered_chunk.loc[:, 'time'] = filtered_chunk['time'].apply(
        lambda x: [(i - start_time).total_seconds() for i in x])
    filtered_chunk.loc[:, 'time'] = filtered_chunk['time'].apply(lambda x: [list(i) for i in x])

    # 1. 筛选路段数据
    filtered_values = filtered_chunk.apply(
        lambda row: filter_route_and_time(row['route'], row['time']), axis=1, result_type='expand'
    )

    filtered_chunk.loc[:, 'route'] = filtered_values[0]
    filtered_chunk.loc[:, 'time'] = filtered_values[1]
    filtered_chunk = filtered_chunk[filtered_chunk['route'].str.len() > 0]

    # 2. 对每行数据内的time进行排序
    result = filtered_chunk.apply(lambda row: sort_route_and_time(row['route'], row['time']), axis=1, result_type='expand')
    filtered_chunk.loc[:, 'route'] = result[0]
    filtered_chunk.loc[:, 'time'] = result[1]

    # 3. 去除重复的路段
    res = filtered_chunk.apply(
    lambda row: remove_duplicate_routes(row['route'], row['time']), axis=1, result_type='expand'
    )
    filtered_chunk.loc[:, 'route'] = res[0]
    filtered_chunk.loc[:, 'time'] = res[1]
    return filtered_chunk



def process_chunks_in_parallel(csv_path, chunk_size, output_csv, start_time):
    def reader():
        # 依据文件名选择合适编码：trip_* 使用 gbk，其它默认 utf-8
        enc = 'gbk' if os.path.basename(csv_path).startswith('trip_') else 'utf-8'
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, encoding=enc, low_memory=False):
            yield chunk

    # 使用 imap 流式并行处理
    with Pool(processes=max(1, cpu_count()//2)) as pool:
        func = partial(filter_chunk_in_time, start_time=start_time)
        results = []
        for filtered_chunk in tqdm(pool.imap(func, reader()), desc="Processing", unit="chunk"):
            if not filtered_chunk.empty:
                results.append(filtered_chunk)

    # 合并所有非空结果
    if results:
        filtered_data = pd.concat(results, ignore_index=True)
        filtered_data['sort_key'] = filtered_data['time'].apply(lambda x: x[0][0] if x and len(x) > 0 and len(x[0]) > 0 else 0).copy()
        filtered_data = filtered_data.sort_values(by='sort_key', ascending=True)
        filtered_data = filtered_data.drop(columns=['hphm', 'tripno','sort_key'])
    else:
        filtered_data = pd.DataFrame(columns=['route','time','cllx_lhy'])

    filtered_data.to_csv(output_csv, index=False)


    

## 全局变量
ROADS = set(load_roads())  
VEHICLE_TYPES = ['K33','K31','H31','K39','Q11','K32']
VEHICLE_TYPE_DICT = {
    'K33': {
                "length": 5,
                "width": 1.8,
                "maxPosAcc":2.6 ,
                "maxNegAcc": 4.5,
                "usualPosAcc": 2.6,
                "usualNegAcc": 4.5,
                "minGap": 2.5,
                "maxSpeed": 27.78,
                "headwayTime": 2
                },
    'K31': {
                "length": 5,
                "width": 1.8,
                "maxPosAcc":2.6 ,
                "maxNegAcc": 4.5,
                "usualPosAcc": 2.6,
                "usualNegAcc": 4.5,
                "minGap": 2.5,
                "maxSpeed": 27.78,
                "headwayTime": 2
                },
    'H31': {
                "length": 7.1,
                "width": 2.4,
                "maxPosAcc":1.3 ,
                "maxNegAcc": 4,
                "usualPosAcc": 1.3,
                "usualNegAcc": 4,
                "minGap": 2.5,
                "maxSpeed": 27.78,
                "headwayTime": 2
                },
    'K39': {
                "length": 5,
                "width": 1.8,
                "maxPosAcc":2.6 ,
                "maxNegAcc": 4.5,
                "usualPosAcc": 2.6,
                "usualNegAcc": 4.5,
                "minGap": 2.5,
                "maxSpeed": 27.78,
                "headwayTime": 2
                },
    'Q11': {
                "length": 7.1,
                "width": 2.4,
                "maxPosAcc":1.3 ,
                "maxNegAcc": 4,
                "usualPosAcc": 1.3,
                "usualNegAcc": 4,
                "minGap": 2.5,
                "maxSpeed": 27.78,
                "headwayTime": 2
                },
    'K32': {
                "length": 5,
                "width": 1.8,
                "maxPosAcc":2.6 ,
                "maxNegAcc": 4.5,
                "usualPosAcc": 2.6,
                "usualNegAcc": 4.5,
                "minGap": 2.5,
                "maxSpeed": 27.78,
                "headwayTime": 2
                },
    'k11': {
                "length": 12,
                "width": 2.5,
                "maxPosAcc": 1.2,
                "maxNegAcc": 4,
                "usualPosAcc": 1.2,
                "usualNegAcc": 4,
                "minGap": 2.5,
                "maxSpeed": 27.78,
                "headwayTime": 2
    },
    'others': {
                "length": 5,
                "width": 1.8,
                "maxPosAcc":2.6 ,
                "maxNegAcc": 4.5,
                "usualPosAcc": 2.6,
                "usualNegAcc": 4.5,
                "minGap": 2.5,
                "maxSpeed": 27.78,
                "headwayTime": 2
                }
}



if __name__ == "__main__":
    # 设定 Pandas 显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # 设定文件路径和参数
    csv_path = 'I:/data/fw/postgres/data_file/trip_2023_04_01_2023_04_03.csv'
    # csv_path = 'I:/data/fw/postgres/data_file/data_2023_04_03_2023_04_09.csv'
    # csv_path = 'I:/data/fw/postgres/data_file/data_2023_04_10_2023_04_17.csv'
    # csv_path = 'I:/data/fw/postgres/data_file/data_2023_04_17_2023_04_24.csv'
    # csv_path = 'I:/data/fw/postgres/data_file/data_2023_04_24_2023_05_01.csv'
    chunk_size = 20000

    # 从CSV文件名中提取开始时间和结束时间
    start_time_str, end_time_str, start_date_str, end_date_str = extract_times_from_filename(csv_path)
    print(f"从文件名提取的时间范围: {start_time_str} 到 {end_time_str}")
    
    # 转换为pandas时间戳
    start_time = pd.Timestamp(start_time_str)
    end_time = pd.Timestamp(end_time_str)
    
    # 生成输出文件名（使用提取的时间信息）
    output_csv = f'I:/data/fw/postgres/data_file/data_{start_date_str}_{end_date_str}_filtered.csv'
    output_json = f"E:/MyCityFlow/data/xuancheng/data_{start_date_str}_{end_date_str}_type_filtered.json"
    
    print(f"输出CSV文件: {output_csv}")
    print(f"输出JSON文件: {output_json}")

    # 生成日期范围
    date_range = pd.date_range(start=start_time.date(), end=end_time.date(), freq='D')
    
    for current_date in date_range:
        month = current_date.month
        day = current_date.day
        
        # 使用月-日格式命名每天的输出文件
        day_output_csv = f'I:/data/fw/postgres/data_file/data_{month:02d}_{day:02d}_filtered.csv'
        day_output_json = f"E:/MyCityFlow/data/xuancheng/data_{month:02d}_{day:02d}_type_filtered.json"
        
        print(f"处理日期: {current_date.strftime('%Y-%m-%d')}, 输出文件: {day_output_csv}")
 
        process_chunks_in_parallel(csv_path, chunk_size, day_output_csv, pd.Timestamp(current_date))

        with open(day_output_json, "wb") as f:
            f.write(b"[")  # 写入 JSON 数组的起始符
            first_item = True  # 控制逗号的添加

            for chunk in pd.read_csv(day_output_csv, chunksize=chunk_size, low_memory=False):
                chunk["route"] = chunk["route"].apply(lambda x: [str(i) for i in ast.literal_eval(str(x))] if isinstance(x, str) and x else [])
                chunk["route"] = chunk["route"].astype(object)
                chunk['time'] = chunk['time'].apply(ast.literal_eval)
                json_list = chunk.to_dict(orient="records")

                for row in json_list:
                    if len(row["route"])<2:
                        continue
                    json_object = {
                        "vehicle": VEHICLE_TYPE_DICT[row["cllx_lhy"]] if row["cllx_lhy"] in VEHICLE_TYPES else VEHICLE_TYPE_DICT["others"],
                        "interval": 1,
                        "startTime": int(row["time"][0][0]),
                        "endTime": int(row["time"][0][0]),
                        "route": row["route"]
                    }

                    if not first_item:
                        f.write(b",\n  ")  # 在每个 JSON 对象之间加逗号
                    else:
                        f.write(b"\n  ")  # 第一个元素前也加缩进
                    f.write(orjson.dumps(json_object,option=orjson.OPT_INDENT_2))
                    first_item = False  # 只有第一个对象不加逗号

            f.write(b"\n]")  # 写入 JSON 数组的结束符

        # 读取json文件
        with open(day_output_json, "rb") as f:
            data = orjson.loads(f.read())
            print(f"完成处理 {current_date.strftime('%Y-%m-%d')} 的数据")






