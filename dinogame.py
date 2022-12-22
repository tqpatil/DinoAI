from mss import mss
import pydirectinput
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
from gym import Env
from gym.spaces import Box, Discrete
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
from stable_baselines3 import DQN
from PIL import ImageGrab
class WebGame(Env):
    def __init__(self):
        super().__init__()
        self.observation_space=Box(low=0,high=255, shape=(1,83,100),dtype=np.uint8)
        self.action_space=Discrete(3)
        self.cap=mss()
        self.game_location={'top':300,'left':0,'width':600,'height':500}
        self.done_location={'top':390,'left':630,'width':660,'height':150}
    def obstacle_exists(self):
        image = ImageGrab.grab(bbox=(220,550,650,750))
        exists=False
        for i in range(430):
            for j in range(200):
                color=image.getpixel((i,j))
                if color==(83,83,83):
                    exists=True
                    break
            if color[2]==83:
                exists=True
                break
        return exists

        # image=np.array(self.cap.grab({'top':300,'left':216,'width':500,'height':465}))
        # gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # contains_ob=False
        # for i in range(len(gray)):
        #     if 83 in gray[i]:
        #         contains_ob=True
        # return contains_ob

    def step(self,action): 
        if self.obstacle_exists():
            winrew=0.5
        else:
            winrew=0
        if action==0:
            pydirectinput.press('space')
        elif action==1:
            pydirectinput.press('down')
        elif action==2:
            pydirectinput.press('no_op')
        else:
            print("Invalid Action")
            raise ValueError
        #if game over
        done,render=self.get_done()
        if done==False and winrew==0.5 and action==0:
            win=2.05
        elif done==False:
            win=2
        else:
            win=0

        new_obs=self.get_observation()
        

        info={"act":action}
        return new_obs,win,done,info

    def render(self):
        cv2.imshow("Game",np.array(self.cap.grab(self.game_location))[:,:,:3])
        if cv2.waitKey(0) and 0xFF == ord('q'):
            self.close()
    
    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=150,y=150)
        pydirectinput.press('space')
        return self.get_observation()


    


    
    def close(self):
        cv2.destroyAllWindows()
    def get_observation(self):
        r=np.array(self.cap.grab(self.game_location))[:,:,:3]
        grey=cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)
        rSize=cv2.resize(grey,(100,83))
        channel=np.reshape(rSize,(1,83,100))
        return channel
        # [::3].astype(np.uint8)

    def get_done(self):
        done=False
        image = ImageGrab.grab(bbox=(600,450,700,550))

        for i in range(100):
            for j in range(100):
                color=image.getpixel((i,j))
                if color==(168,168,168):
                    done=True
                    break
            if color==(168,168,168):
                break
        r=np.array(self.cap.grab(self.done_location))[:,:,:3]
        return done,r
        # for y in range():
        #     for x in range():
        #         color = image.getpixel((x, y))
        # print(color)

        # r=np.array(self.cap.grab(self.done_location))[:,:,:3]
        # text= (pytesseract.image_to_string(r))
        # done=False
        # if "GAME" in text or "OVER" in text or "GAME OVER" in text or "GAHE" in text:
        #     done=True
        # return done,r
        
if __name__=="__main__":
    env=WebGame()

    # for i in range(0):
    #     obv=env.reset()
    #     done=False 
    #     total_win=0
    #     while not done:
    #         obs,reward,done,info=env.step(env.action_space.sample())
    #         total_win+=reward
    #     print(f"Total Reward for epoch {i} is {total_win}") 
    env_checker.check_env(env)
    class TrainCallback(BaseCallback):
        def __init__(self,check_freq,save_path,verbose=1):
            super(TrainCallback,self).__init__(verbose)
            self.check_freq=check_freq
            self.save_path=save_path
        def _init_callback(self):
            if self.save_path !=None:
                os.makedirs(self.save_path,exist_ok=True)
        def _on_step(self):
            if self.n_calls% self.check_freq==0:
                model_path=os.path.join(self.save_path,"best_model_{}".format(self.n_calls))
                self.model.save(model_path)
            return True
    CHECKPOINT='C:/Users/tqpat/OneDrive/Documents/Python/Dino/Checks'
    LOG_DIR='C:/Users/tqpat/OneDrive/Documents/Python/Dino/Logs'
    callbacker=TrainCallback(check_freq=10000,save_path=CHECKPOINT)
    model=DQN('CnnPolicy',env,tensorboard_log=LOG_DIR,verbose=1,buffer_size=100000,learning_starts=0)
    # model.learn(total_timesteps=50000,callback= callbacker)

    model=DQN.load(os.path.join('C:\\Users\\tqpat\\OneDrive\\Documents\\Python\\Dino\\Checks','best_model_40000'))
    for ep in range(10):     
        obs=env.reset()
        done=False
        total_reward=0
        while not done:
            action,_=model.predict(obs)
            obs,reward,done,info=env.step(int(action))
            total_reward+=reward
        print(f'Total Reward for epoch {ep} is {total_reward}')


    


        
             

        

