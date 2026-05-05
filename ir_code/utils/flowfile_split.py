import orjson

json_path = './data/xuancheng/data_2023_04_04_type_filtered.json'
# Read the JSON file
with open(json_path, "r", encoding="utf-8") as f:
    data = orjson.loads(f.read())

# Define time range
min_time = 3600*17 # adjust as needed
max_time = 3600*18  # adjust as needed

# Filter and adjust startTime
vehicle_time = []
for item in data:
    if min_time <= item.get("startTime", 0) <= max_time:
        adjusted_item = item.copy()
        adjusted_item["startTime"] = int(item["startTime"] - min_time)
        adjusted_item["endTime"] = int(item["endTime"] - min_time)
        vehicle_time.append(adjusted_item)

save_path = "./data/xuancheng/trip_2023_04_04_17.json"
with open(save_path, "w", encoding="utf-8") as f:     
    f.write(orjson.dumps(vehicle_time, option=orjson.OPT_INDENT_2).decode("utf-8"))
    # Read the saved JSON file for route analysis
    with open(save_path, "r", encoding="utf-8") as f:
        filtered_data = orjson.loads(f.read())

    # Count road frequency
    road_frequency = {}
    for item in filtered_data:
        route = item.get("route", [])
        for road in route:
            road_frequency[road] = road_frequency.get(road, 0) + 1

    # Sort by frequency (descending)
    sorted_roads = sorted(road_frequency.items(), key=lambda x: x[1], reverse=True)

    # Print top 10 most frequent roads
    print("Top 30 most frequent roads:")
    for road, count in sorted_roads[:30]:
        print(f"{road}: {count}")

    # Save road frequency analysis
    frequency_save_path = "./utils/validate/road_frequency_16.json"
    with open(frequency_save_path, "w", encoding="utf-8") as f:
        f.write(orjson.dumps(dict(sorted_roads), option=orjson.OPT_INDENT_2).decode("utf-8"))