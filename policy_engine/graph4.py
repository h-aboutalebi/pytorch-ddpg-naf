import matplotlib.pyplot as plt
import numpy as np
import pickle
string_directory="/Users/hosseinaboutalebi/Desktop/Git_hub/pytorch-ddpg-naf/policy_engine/files3/result_reward"
epoch_list=[]
ratio=[]
colors=["orange","b",'g']
labels=[r'$\beta=0.99$',r'$\beta=0.1$',r'$\beta=0.01$']

for i in range(3):
    infile = open(string_directory+str(i) + ".pkl", 'rb')
    example_dict = pickle.load(infile)
    epoch_list.append(example_dict["poly_rl_ratio"]["epoch"])
    ratio.append([x/y for x, y in zip(example_dict["poly_rl_ratio"]["ratio"], example_dict["poly_rl_ratio"]["step_number"])])
    plt.plot(epoch_list[i],ratio[i], colors[i], label=labels[i])
    plt.xscale('log')
    plt.xlabel('Episodes')
    plt.ylabel('Exploitation Percentage')

plt.legend(loc='lower right')
plt.show()
