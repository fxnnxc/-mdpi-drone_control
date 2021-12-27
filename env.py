from operator import mul
from pickle import NONE
import warnings
from drone.control.BaseControl import BaseControl
from drone.envs.WindAviary import WindAviary
from drone.train.predict_model.model import Predictor
from ray.rllib.env.multi_agent_env import MultiAgentEnv  
import numpy as np 
from gym.spaces import Discrete, Box
import torch 
import copy
from scipy.stats import mode 

class RLCorrectionEnv(WindAviary, MultiAgentEnv):
    def __init__(self, config={}):
        np.random.seed(123)

        d = 1.0
        super().__init__(wind_magnitude = config.get("wind_magnitude",0.04), 
                    wind_field=config.get("wind_field"),
                    freq=240, 
                    initial_xyzs=np.array([[1,0,1]]), 
                    num_drones=1, 
                    gui=config.get("gui", None), 
                    trajectory=config.get("trajectory", [np.array([1,1,1]), np.array([1-d,1,1]), np.array([1-d, 1-d, 1]), np.array([1, 1-d , 1]), np.array([1,1,1])]),
                    trajectory_epsilon=config.get("trajectory_epsilon", 0.05),
                    trajectory_num_in_sec=config.get("trajectory_num_in_sec", 0.3),
                    debug_mode=config.get("debug_mode"))
        
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        self.EPISODE_LEN_SEC = config.get("EPISODE_LEN_SEC", 50)
        
        self.predict_model = torch.load(config.get("predict_model_path", "C:/Users/sail/Desktop/git/mdpi_drone_paper/drone/train/predict_model/predictor.pt"))
        self.predict_model.eval()
        
        self.init_position_freq_ratio = config.get("position_freq_ratio", 0.3)
        self.ctrl = BaseControl(self.DRONE_MODEL, gamma=self.init_position_freq_ratio)
        self.reset_counter = 0 
        self.h_mode_list = [] 
        self.config = config 

    def _observationSpace(self):
        return Box(-np.inf, np.inf, shape=(26,))

    def _actionSpace(self):
        return Discrete(5)

    def reset(self):
        self.reset_counter+=1
        self.INIT_XYZS = np.array([self.trajectory[0]]) 
        self.ctrl = BaseControl(self.DRONE_MODEL, gamma=self.init_position_freq_ratio)
        obs = super().reset()
        # self.trajectory = self.trajectory + self.trajectory
        self.wind_error = np.zeros(3)
        self.step_counter = 0
        self.done_condition = None 
        
        #return {0:self._obs()}
        self.h = np.random.randint(1, 240)//10 * 10
        self.positions_for_store = [] 
        self.infos = [] 
        obs_ = [0 for _ in range(26)]
        self.h_list = []
        if self.reset_counter ==1:
            self.wind_field = self.config.get("wind_field")
        else:
            self.wind_field = lambda pos: self.config.get("wind_field")(pos) + (np.random.random(3,)-0.5)/20

        return {0:obs_}

    def step(self, action):
        # --- compute the action
        action = action[0]
        if action==0:
            action = -1 
        elif action ==2:
            action = 1
        else:
            action = 0
        self.ctrl.h = max(2, self.ctrl.h + action * self.ctrl.H // 10)
        self.ctrl.h = np.clip(self.ctrl.h, 1, self.ctrl.H) 
        self.h_list.append(self.ctrl.h)
        
        control_obs = []
        wind_errors = []

        done={"__all__":False}
        # self.ctrl.control_counter = 0
        for i in range(self.ctrl.H):
            # --- predict the error of PID before wind effect
            self.position_before_control = self._getDroneStateVector(0)[:3]
            with torch.no_grad():
                relative_sp = self.sp - self.position_before_control
                model_input = Predictor.get_model_input_from_env(self)
                # model_input = self.predict_model.get_model_input_from_env(self)
                self.predicted_position_difference = self.predict_model(torch.tensor(model_input, dtype=torch.float32)).detach().numpy().reshape(3,)

            state, reward, done, info = super().step(self.sp)

            self.wind_error = (self._getDroneStateVector(0)[:3] - self.position_before_control) - self.predicted_position_difference
            self.wind_error =  np.zeros(3,) #self.get_wind_direction(0)

            wind_errors.append(self.wind_error)
            current_state = self._getDroneStateVector(0)
            current_state[:3] = 0 
            control_obs.append(current_state)
            self.positions_for_store.append(self.current_position)
            self.infos.append(self._getDroneStateVector(0).copy())
            
            done = {0:self._computeDone()}
            done['__all__'] = self._computeDone()
            if done[0]:
                break     
        # --- pooling the control observation ---
        control_obs = self.pooling(control_obs, wind_errors)

        if done['__all__']:
            if self.done_condition == 'die':
                reward = -2
            elif self.done_condition =="reach_all":
                reward = (self.SIM_FREQ*self.EPISODE_LEN_SEC)/(self.step_counter)
            else:
                reward = -1
            print(self.h_list, end=" : ")
            print(reward)
            self.h_mode_list.append(mode(self.h_list)[0][0])
            self.h_mode_list.append(np.mean(self.h_list))
        
            # np.save("../../../../../../../analysis/data/2h_mode.npy", np.array(self.h_mode_list))
            if self.reset_counter%1==0: 
                name = "wo"
                np.save(f"../../../../../../../analysis/data/1_position_{name}QFP.npy", np.array(self.positions_for_store))
                np.save(f"../../../../../../../analysis/data/1_state_{name}QFP.npy", np.array(self.infos))
        else:
            reward = 0            
        self.step_counter += 1
        return {0:control_obs}, {0:reward}, done, info

    def pooling(self, control_obs, wind_errors):
        # return the pooled control observation 
        pooled = np.mean(control_obs, 0) # mean pooling over the timesteps.
        if self.wind_magnitude !=0:
            mean, std = np.mean(wind_errors), np.std(wind_errors)
        else:
            mean, std = 0,0
        pooled = np.append(pooled, [mean, std, self.ctrl.h/self.ctrl.H])
        if self.trajectory_position < len(self.trajectory)-1:
            next_way_points = (self.trajectory[self.trajectory_position+1:,:] -self.current_position)
            for i in range(next_way_points.shape[0]):
                next_way_points[i,:] *= 1/(1+i)
            pooled = np.append(pooled,  next_way_points.mean(axis=0))
        else:
            pooled = np.append(pooled, [0,0,0])
        return pooled

    # def _obs(self):
    #     # wind error + drone kinematic information + current ratio
    #     new_state = np.hstack([self.wind_error, self._getDroneStateVector(0), np.array(self.ctrl.gamma)])
    #     return new_state

    def get_predictor_input(self, relatvie_sp):
        # kinematic information without current position 
        # + relative setpoint 
        # + current pid position error
        cur_state = self._getDroneStateVector(0)[3:].reshape(1,-1)
        pos_error = self.ctrl.pos_err.reshape(1, -1)
        relatvie_sp = relatvie_sp.reshape(1,-1)
        ratio = np.array([self.ctrl.gamma])
        if len(ratio.shape)==1:
            ratio = np.expand_dims(ratio, 0)

        x = np.hstack([cur_state, relatvie_sp, pos_error, ratio])

        return x

    def _computeDone(self):
        done=False
        if self._getDroneStateVector(0)[2] < 0.1:
            done = True
            self.done_condition = 'die'
        if len(self.trajectory) == len(self.trajectory_timesteps):
            self.done_condition = 'reach_all'
            done =True
        if self.step_counter >= self.EPISODE_LEN_SEC * self.SIM_FREQ:
            done=True 
        return done



if __name__ == "__main__":
    import pybullet as p 
    trajectory_ = [np.array([1/2*(-1)**(i%2),1/2*(-1)**(i%2),1])  for i in range(10) ]
    trajectory_ = [np.array([((-1)**i)/5, ((-1)**i)*np.cos(i)+np.random.random(), 2+ np.sin(i/18)+ np.cos(i/8)])  for i in range(5)]
    d = 1.0
    trajectory1 = [np.array([1,1,1]), np.array([1-d,1,1]), np.array([1-d, 1-d, 1]), np.array([1, 1-d , 1]), np.array([1,1,1])]
    trajectory2 = [np.array([1,1,1]), np.array([1+d,1,1]), np.array([1,1,1]), np.array([1+d,1,1]), np.array([1,1,1])]
    trajectory3 = [np.array([1,1,1]), np.array([-1,1+d,1]), np.array([-1, 1+2*d, 1+d]), np.array([1, 1+3*d, 1+2*d]), np.array([1,1+3*d, 1+3*d])]
    trajectory4 = [np.array([1,1.2,1]), np.array([1,1,1+d]), np.array([1,1,1]), np.array([1,1,1+d]), np.array([1,1,1])]
    random_int_trajectory = np.array([[np.random.randint(-3,3), np.random.randint(-3,3), np.random.randint(1,5)] for i in range(np.random.randint(6,7))])
    
    print("----Trajectory-------")
    print(random_int_trajectory)
    print("LEN TRAJECTORY: %d"%len(random_int_trajectory))
    print("--------------------")

    random_max_dist = np.random.random()
    for num in [1,2,3,4]:
        print("WAY POINT IN SEC: %d"%num, end= " | ")

        for t in  [0.1, 0.4 , 0.5,  0.99, 1.0]:
            env = RLCorrectionEnv({"gui": False, 
                                    "wind_magnitude":0.04, 
                                    "predict_model_path":"../../predict_model/predictor.pt", 
                                    "use_prediction":True,
                                    #"trajectory": np.array(random_int_trajectory),
                                    "trajectory": np.array(trajectory1),
                                    "wind_schedule":[np.array([(-1)**(i%2),(-1)**(i%2),1]) for i in range(480)],
                                    "trajectory_epsilon":0.05,
                                    "EPISODE_LEN_SEC":50,
                                    "position_freq_ratio":t,
                                    "trajectory_num_in_sec":num})
            env = RLCorrectionEnv({"gui": False, 
                                    
                                    
                                    #"position_freq_ratio":t,
                                    "trajectory_num_in_sec":num})
            if t==0.05:
                print("LEN TRAJECTORY: %d"%len(env.trajectory))
            env.reset()
            print(env.ctrl.h)
            summ = 0 
            count = 0
            total_reward = 0 
            while True:
                action = env.action_space.sample()
                #action = 1
                state, reward, done, info = env.step({0:action})
                total_reward += reward[0]
                if env.wind_error is not None:
                    summ += np.linalg.norm(env.wind_error)
                count += 1
                if done['__all__'] is True:
                    break 
                # print("action: %d | postion hertz: %d"%(action, env.ctrl.h), end=" | ")
                # print("pos:", env._getDroneStateVector(0)[:3])
        
            print("positon control hertz : ", env.ctrl.h," | reward:%3.6f"%total_reward, " | reach time:", env.trajectory_timesteps)
