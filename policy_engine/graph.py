import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
from scipy.stats import sem
from scipy import stats
from scipy.interpolate import make_interp_spline, BSpline


class Create_Graph:

    def __init__(self, Number_of_iteration, max_x_number=1000, min_x_number=0):
        self.Number_of_iteration =  Number_of_iteration
        self.max_x_number = max_x_number
        self.min_x_number = min_x_number
        self.x_values = [i for i in range(0, max_x_number, 10)]
        PolyRL_result = [[] for _ in range(Number_of_iteration)]
        ParamNoise_result = [[] for _ in range(Number_of_iteration)]
        DDPG_result = [[] for _ in range(Number_of_iteration)]
        self.get_results_from_file(PolyRL_result, string="files/PolyRL")
        self.get_results_from_file(ParamNoise_result, string="files3/param_noise")
        self.get_results_from_file(DDPG_result, string="files/DDPG")
        # self.get_results_from_file(DDPG_result, string="files2/DDPG")
        self.plot_figure(PolyRL_result, DDPG_result,ParamNoise_result)

    def get_results_from_file(self, list, string):
        for i in range(self.Number_of_iteration):
            infile = open(string + str(i+1) + ".pkl", 'rb')
            example_dict = pickle.load(infile)
            self.x_new_values, power_smooth = self.make_smooth_line(example_dict['modified_reward'])
            list[i].append(power_smooth)

    # implementation from https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
    def make_smooth_line(self, list):
        xnew = np.linspace(self.min_x_number, self.max_x_number,
                           len(list))  # 300 represents number of points to make between T.min and T.max
        spl = make_interp_spline(self.x_values, list, k=3)  # BSpline object
        power_smooth = spl(xnew)
        return xnew, power_smooth

    def plot_figure(self, PolyRL_result, DDPG_result,ParamNoise_result):
        y_PolyRL_result = np.mean(PolyRL_result, axis=0)
        error_PolyRL_result = stats.sem(PolyRL_result, axis=0)
        y_ParamNoise_result = np.mean(ParamNoise_result, axis=0)
        error_ParamNoise_result = stats.sem(ParamNoise_result, axis=0)
        y_DDPG_result = np.mean(DDPG_result, axis=0)
        error_DDPG_result = stats.sem(DDPG_result, axis=0)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.plot(self.x_new_values, y_PolyRL_result[0], "b",label='DDPG with PolyRL')
        plt.plot(self.x_new_values, y_ParamNoise_result[0], "r", label='DDPG with Parameter Noise')
        plt.plot( self.x_new_values, y_DDPG_result[0], "g", label='DDPG')
        plt.fill_between(self.x_new_values, y_PolyRL_result[0] - error_PolyRL_result[0], y_PolyRL_result[0] + error_PolyRL_result[0], edgecolor='#0000FF',
                         facecolor='#0000FF', alpha=0.5,
                         linewidth=0)
        plt.legend(loc='upper left')
        plt.fill_between(self.x_new_values, y_ParamNoise_result[0] - error_ParamNoise_result[0], y_ParamNoise_result[0] + error_ParamNoise_result[0], edgecolor='#3F7F4C',
                         facecolor='lightcoral', alpha=0.5,
                         linewidth=0)
        plt.fill_between(self.x_new_values, y_DDPG_result[0] - error_DDPG_result[0], y_DDPG_result[0] + error_DDPG_result[0], edgecolor='#3F7F4C', facecolor='#3F7F4C', alpha=0.5,
                         linewidth=0)

        plt.show()

Create_Graph(Number_of_iteration=5)