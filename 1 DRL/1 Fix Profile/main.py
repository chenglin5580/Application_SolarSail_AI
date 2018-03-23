"""
DDPG is Actor-Critic based algorithm
Designer: Lin Cheng  17.08.2017
"""

########################### Package  Input  #################################

from SolarSail2 import SolarSail as Objective_AI
from FixedProfile import FixedProfile as Method
import numpy as np
import matplotlib.pyplot as plt

############################ Hyper Parameters #################################

max_Episodes = 20000
max_Ep_Steps = 20000
rendering = False
############################ Object and Method  ####################################

env = Objective_AI()

RL_method = Method()


# reset
observation = env.reset()


ob_profile = np.empty((0, 4))
time_profile = np.empty(1)

ep_r = 0
for j in range(max_Ep_Steps):

    action = RL_method.choose_action(env.t)
    action = 0.79

    observation_, reward, done, info = env.step(action)

    ep_r += reward


    # memorize the profile
    ob_profile = np.vstack((ob_profile, observation))
    time_profile = np.vstack((time_profile, env.t))

    observation = observation_

    if done:
        break

print('转移轨道时间%d天' % env.t)
print('reward', ep_r)
print(env.state)
plt.subplot(111, polar=True)
theta = np.arange(0, 2 * np.pi, 0.02)
plt.plot(theta, 1 * np.ones_like(theta),'m')
plt.plot(theta, 1.524 * np.ones_like(theta),'b')
plt.plot(ob_profile[:, 1], ob_profile[:, 0],'r')
plt.show()




