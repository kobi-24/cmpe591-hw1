import numpy as np 
from homework1 import Hw1Env 
import os 

def collect_data(num_samples = 1000):
    print(f"Starting Data colection for {num_samples} samples")

    env = Hw1Env(render_mode="gui")

    imgs_before = []
    action = []
    position_after = []
    imgs_after = []

    for i in range(num_samples):
        env.reset()
        action_id = np.random.randint(4)
        _, img_before = env.state()


        env.step(action_id)
        pos_after, img_after = env.state()

        imgs_before.append(img_before)
        action.append(action_id)
        position_after.append(pos_after)
        imgs_after.append(img_after)

        if (i + 1) % 100 == 0:
            print(f"Collected {i + 1} / {num_samples} samples")

        
    

    print("saving the datasetr to disk....")
    np.savez_compressed("hw1_dataset.npz", 
        imgs_before=np.array(imgs_before), 
        action=np.array(action), 
        position_after=np.array(position_after), 
        imgs_after=np.array(imgs_after))
    
    print("Done")

if __name__ == "__main__":
    collect_data(1000)