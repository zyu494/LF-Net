import numpy as np
from typing import Optional
def predict(obs,prediction_time,step_time,velocity:Optional[float]=2.0):
    trajectory=[]
    for id in range(len(obs)):
        trajectory.append([])
        for i in np.arange(0,prediction_time,step_time):
            x=obs[id][0]+velocity*i
            y=obs[id][1]
            trajectory[id].append([x,y])
    return trajectory

if __name__ == '__main__':
    trajectory=predict([[1,0],[2,0],[3,0]],2,0.1,1)
    print(trajectory)
