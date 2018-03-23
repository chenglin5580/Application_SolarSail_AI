

# from SmallStateControl import SSCPENV as Object_AI
from SolarSail2 import SolarSail as Objective_AI
import A3C
env = Objective_AI()
train_flag = True
# train_flag = False
para = A3C.Para(env,
                MAX_GLOBAL_EP=3000,
                UPDATE_GLOBAL_ITER=50,
                GAMMA=0.9,
                ENTROPY_BETA=0.01,
                LR_A=0.0001,
                LR_C=0.001,
                train=train_flag  # 表示训练
                )

RL = A3C.A3C(para)
if para.train:
    RL.run()
else:
    # 1 stable
    # 2 random
    # 3 multi
    RL.display(1)
#