from cityflow_env import CityFlowEnv, datetime_utc8
import numpy as np
from agent.dqn_agent import DQNAgent
from model.model_tf.baseline.FRAP_model import FRAPModel

class FRAP(DQNAgent):
    name='FRAP'
    def __init__(self, env, decision_interval, output_dir=None):
        super().__init__(env, decision_interval, output_dir)
        self.switch = True
        ####  norm
        self.cell_len = 50  # space/m
        self.car_len = 5
        self.max_speed_lim = 16.6666
        self.max_vehicle_num = 50
        """memory parameter"""
        self.buffer_eps = 4
        self.reuse_time = 8

        """exploration parameters"""
        self.explore_strategy = 'epsilon_greedy'     # epsilon_greedy_logit
        self.epsilon = 1  # exploration rate
        self.epsilon_max = self.epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

        """learning parameter"""
        self.gamma = 0.9  # discount rate
        self.update_target_ratio = 1
        self.lr = 1e-3
        self.lr_max = self.lr
        self.lr_min = 1e-5
        self.lr_decay = 0.9
        self.episodes = 100
        self.update_target_freq = 64

        """sampling parameter"""
        self.batch_size = 256
        self.sample_mode = 'uniform'    # balance
        self.balance_sample_size = 32
        self.bin_width = 1

        """feature selections"""
        self.feature_list = ['up_lane_dens', 'up_lane_speed', 'up_lane_num', 'down_lane_dens', 'down_lane_speed', 'down_lane_num','inlane_throughput',\
                              'phase_mask','cur_phase','action']

        """build model"""
        self.model = FRAPModel(self.max_phase,env.env_config)
        self.target_model = FRAPModel(self.max_phase,env.env_config)

        self._record_config()

    def observe(self, env, cal_reward=False, dyna={}):
        state = {}
        self.lane_mask={}
        if cal_reward:
            reward = {}
        lane_delay = dyna.get('lane_delay', {})
        lane_in = dyna.get('lane_in', {})
        lane_out = dyna.get('lane_out', {})
        lane_veh_id = env.get_lane_veh_id(True)
        lane_density = env.get_lane_density()
        lane_avg_speed = env.get_lane_avg_speed()
        lane_veh_num = env.get_lane_veh_num()
        veh_dist = env.get_veh_distance(to_end=True)
        veh_speed = env.get_veh_speed()
        lane_queue = env.get_lane_queue()
        #### 无需分割为cell
        for agent_id in self.agent_list:
            agent = env.intersections[agent_id]
            agent_lane = env.get_inter_sort_lane(agent_id)
            inter_exit_veh_id = set().union(*[lane_in.get(l, set()) for l in agent_lane[12:]])
            phase_inlane_mask = []
            phase_mask = env.env_phase_map(agent_id, mask=True)
            for i, pm in enumerate(phase_mask):
                if pm:
                    phase_inlane = env.get_inter_phase_lane(agent_id, i, 'in')
                    phase_inlane_mask.append([int(lane_id in phase_inlane) for lane_id in agent_lane[:12]])
                else:
                    phase_inlane_mask.append([0] * 12)
            self.lane_mask[agent_id] = list(map(lambda x: 0 if x ==None else 1,  agent_lane))
            # 转为列表
            lane_queue_list = [lane_queue.get(l,0)/(self.cell_len/self.car_len) for l in agent_lane]
            lane_density_list = [lane_density.get(l, 0)* self.car_len for l in agent_lane]
            lane_avg_speed_list = [lane_avg_speed.get(l, -1)/self.max_speed_lim  for l in agent_lane]
            lane_veh_num_list = [lane_veh_num.get(l, 0)/self.max_vehicle_num for l in agent_lane]
            #### 下游数据处理
            down_lane_queue = self.downstream_average(self.lane_mask[agent_id], lane_queue_list [12:])
            down_lane_queue = self.out_index(env.env_config.out_road,self.lane_mask[agent_id], down_lane_queue)
            down_lane_dens = self.downstream_average(self.lane_mask[agent_id],lane_density_list[12:])
            down_lane_dens = self.out_index(env.env_config.out_road,self.lane_mask[agent_id], down_lane_dens)
            down_lane_speed = self.downstream_average(self.lane_mask[agent_id],lane_avg_speed_list[12:])
            down_lane_speed = self.out_index(env.env_config.out_road,self.lane_mask[agent_id], down_lane_speed)    
            down_lane_num = self.downstream_average(self.lane_mask[agent_id],lane_veh_num_list[12:])
            down_lane_num = self.out_index(env.env_config.out_road,self.lane_mask[agent_id], down_lane_num)   
            
            state[agent_id] = {
                'up_lane_dens':lane_density_list[:12],
                'up_lane_speed':lane_avg_speed_list[:12],
                'up_lane_num': lane_veh_num_list [:12],
                'down_lane_dens': down_lane_dens,
                'down_lane_speed': down_lane_speed,
                'down_lane_num':down_lane_num,
                'inlane_throughput': [len(lane_out.get(l, set()) & inter_exit_veh_id) for l in agent_lane[:12]],
                'cur_phase': self.cur_phase[agent_id],
                'phase_inlane_mask': phase_inlane_mask,
                'phase_mask': phase_mask,
                'action':phase_mask
                # 'total_queue_length': -np.mean( np.array(self.env_config.action_lane) * agent_lane_queue[:12])
            }

            if cal_reward:
              reward[agent_id] = -np.mean(np.array(env.env_config.action_lane)*[lane_queue.get(l, 0) for l in agent_lane[:12]])
        if cal_reward:
            return state, reward
        else:
            return state


    def _get_bin_index(self, bin_value, bin_width):
        bin_index = {}
        for i, left in enumerate(range(0, int(bin_value.max()), bin_width)):

            # if i == 0:
            #     continue

            right = left + bin_width
            bin_index[i], *_ = np.where((bin_value >= left) & (bin_value < right))

        self._bin_index = bin_index
        return bin_index

    def _buffer_sampler(self, state, action, reward, next_state, target):
        if self.sample_mode == 'balance':
            try:
                bin_index = self._bin_index
            except AttributeError:
                inlane_throughput = np.sum(next_state['inlane_throughput'], axis=1)
                bin_index = self._get_bin_index(inlane_throughput, self.bin_width)

            balance_sample_size = min(self.balance_sample_size, np.median([len(v) for v in bin_index.values()]).astype(int))

            index = []
            for v in bin_index.values():
                index.extend(np.random.choice(v, min(len(v), balance_sample_size), replace=False).tolist())
        else:   # uniform
            num_exp = len(reward)
            num_sample = min(self.batch_size, num_exp)

            index = np.random.choice(list(range(num_exp)), num_sample, replace=False)

        batch_state = {}
        batch_next_state = {}
        for f in state:
            batch_state[f] = state[f][index]
            batch_next_state[f] = next_state[f][index]

        batch_action = action[index]

        batch_reward = reward[index]
        batch_target = target[index]

        return batch_state, batch_action, batch_reward, batch_next_state, batch_target

    def _reward_normalize(self, batch_reward):
        return np.divide(batch_reward, 10)

    def _create_buffer(self):
        batch_action=[]
        batch_state, batch_action, batch_reward, batch_next_state = super()._create_buffer()
        inlane_throughput = np.sum(batch_next_state['inlane_throughput'], axis=1)
        valid_index, *_ = np.where(inlane_throughput > 0)
        batch_action = batch_action[valid_index]
        batch_reward = batch_reward[valid_index]
        batch_state = {f: batch_state[f][valid_index] for f in batch_state.keys()}
        batch_next_state = {f: batch_next_state[f][valid_index]  for f in batch_state.keys()}

        print(f'{datetime_utc8()} Filted buffer size: ', len(batch_reward))

        return batch_state, batch_action, batch_reward, batch_next_state

    # def TD_learning(self):
    #     state, action, reward, next_state = self._create_buffer()
    #     num_exp = len(reward)
    #     num_sample = min(self.batch_size, num_exp)

    #     one_hot_action = np.eye(self.max_phase)[action].reshape(-1, self.max_phase)
    #     state['phase_mask'] *= one_hot_action   # make training loss 0 for not-actions' q value

    #     value_loss_list = []
    #     for n in range(num_exp * self.reuse_time // num_sample):
    #         ############################## Update 1-step value function ##################################
    #         if n % self.update_target_freq == 0:
    #             self._update_target_model()
    #             target = self.target_model.eval(next_state)
    #             target = reward + self.gamma * target.max(axis=1).reshape(-1, 1)
    #             target = np.tile(target, (1, self.max_phase))
    #             # make training loss 0 for not-actions' q value
    #             target *= one_hot_action
    #             target += (one_hot_action - 1) * 1e5

    #         batch_state, batch_action, batch_reward, batch_next_state, batch_target = self._buffer_sampler(state, one_hot_action, reward, next_state, target)
    #         batch_target = np.expand_dims(batch_target,2)
    #         loss = self.model.train(batch_state, batch_target, lr=self.lr * (0.99) ** n)

    #         loss = np.sqrt(loss)
    #         value_loss_list.append(loss)
    #         # if n % 10 == 0:
    #         print(f'Sample: {len(batch_reward)} Loss: {format(loss, ".3f")}', end='\r')

    #     print(' ' * 24 + f'{datetime_utc8()} \rAvg loss: {np.mean(value_loss_list)}')
    #     np.save('/'.join((self.cache_dir, 'loss.npy')), value_loss_list)

    #     return value_loss_list
    