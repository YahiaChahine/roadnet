import unittest
import cityflow
import sys
import tempfile
from io import StringIO
import re
import os
from tqdm import trange
import orjson
from cityflow_env import CityFlowEnv

class CaptureCppOutput:
    def __enter__(self):
        # 创建临时文件用于存储标准错误输出
        self.temp_file = tempfile.TemporaryFile(mode='w+')
        
        # 保存原始的标准错误文件描述符
        self.original_stderr_fd = os.dup(2)  # 2 是标准错误的文件描述符
        
        # 将标准错误重定向到临时文件
        os.dup2(self.temp_file.fileno(), 2)
        return self
    
    def __exit__(self, *args):
        # 恢复原始的标准错误
        os.dup2(self.original_stderr_fd, 2)
        os.close(self.original_stderr_fd)
        
        # 读取临时文件中的输出
        self.temp_file.seek(0)
        self.output = self.temp_file.read()
        self.temp_file.close()

def vehilce_departure(vehicle_id):
    pass



class TestAPI(unittest.TestCase):

    config_file = "./cfg/xuancheng/config_xuancheng_test_save.json"
    with open(config_file, "rb") as f:
        data = orjson.loads(f.read())
        flow = data["flowFile"]
    with open(flow, "rb") as f:
        flow_data = orjson.loads(f.read())
        trip = {k:v for k, v in enumerate(flow_data)}
    period = 1800
    invalid = []
    valid = []


    def test_data_api(self):
        """Single save and single load with single threading engine"""
        eng = cityflow.Engine(config_file=self.config_file, thread_num=1)
        env = CityFlowEnv(self.config_file)

        for i in trange(self.period, desc="Running simulation"):
      
            with CaptureCppOutput() as output:
                eng.next_step()
            if output.output:  
                # 处理捕获的警告信息
                if "Invalid route" in output.output:
                    # 提取无效路由的ID
                    cur_invalid = sorted(re.findall(r"Invalid route '(.*?)'", output.output))
                    cur_invalid_id =  [int(id[5:]) for id in cur_invalid]
                    cur_invalid_route = {id:self.trip[id]['route'] for id in cur_invalid_id}
                    # env.check_valid_route(cur_invalid_route)
                    # self.invalid.extend(cur_invalid)
                    
            cur = sorted(list(set([v[:-2] for v in eng.get_vehicles()])))
            self.valid.extend(cur)
            cur_id  =  [int(id[5:]) for id in cur]
            cur_vec_departure = [self.trip[id]['route'] for id in cur_id]

          

            running_count = len(eng.get_vehicles())
            total_count = len(eng.get_vehicles(include_waiting=True))
            self.assertTrue(running_count <= total_count)
            # self.assertTrue(running_count, eng.get_vehicle_count())
            eng.get_lane_vehicle_count()
            eng.get_lane_waiting_vehicle_count()
            eng.get_lane_vehicles()
            eng.get_vehicle_speed()
            eng.get_vehicle_distance()
            eng.get_current_time()
            # eng.update_Log()
        print(f"valid route: {len(set(self.valid))}" )
        print(f"invalid route: {len(set(self.invalid))}" )
        del eng

    # def test_set_replay(self):
    #     """change replay path on the fly"""
    #     eng = cityflow.Engine(config_file=self.config_file, thread_num=1)

    #     for _ in range(100):
    #         eng.next_step()

    #     eng.set_replay_file("replay2.txt")

    #     for _ in range(100):
    #         eng.next_step()

    #     del eng

if __name__ == '__main__':
    unittest.main(verbosity=2)
    # TestAPI()