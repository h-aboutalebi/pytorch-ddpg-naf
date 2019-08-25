import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import sem
from scipy import stats
from scipy.interpolate import make_interp_spline, BSpline


class Create_Graph:

    def __init__(self, Number_of_iteration, max_x_number=1000, min_x_number=0):
        self.Number_of_iteration =  Number_of_iteration
        self.max_x_number = max_x_number
        self.min_x_number = min_x_number
        string_directory="/Users/hosseinaboutalebi/Desktop/Git_hub/pytorch-ddpg-naf/files_non_sparce"
        # string_directory="/Users/hosseinaboutalebi/Desktop/Git_hub/pytorch-ddpg-naf/policy_engine/"
        self.x_values = [i for i in range(0, max_x_number, 10)]
        PolyRL_result = [[] for _ in range(Number_of_iteration)]
        ParamNoise_result = [[] for _ in range(Number_of_iteration)]
        DDPG_result = [[] for _ in range(Number_of_iteration)]
        Div_Dis_result = [[[0 for _ in range(0, 1000, 10)]] for _ in range(Number_of_iteration)]
        Div_Dis_result = [[] for _ in range(Number_of_iteration)]
        self.get_results_from_file(PolyRL_result, string=string_directory+"/PolyRL")
        self.get_results_from_file(ParamNoise_result, string=string_directory+"/param_noise")
        self.get_results_from_file(DDPG_result, string=string_directory+"/DDPG")
        self.get_results_from_file(Div_Dis_result, string=string_directory+"/Div_Dis")
        self.plot_figure(PolyRL_result, DDPG_result,ParamNoise_result,Div_Dis_result)

    def get_results_from_file(self, list, string):
        for i in range(self.Number_of_iteration):
            # if (string.split("/")[-1] in [ "Div_Dis"]):
            #     infile = open(string + str(i + 1) + ".pkl", 'rb')
            if(string.split("/")[-1] in ["param_noise","Div_Dis"] ):
                infile = open(string + str(i+1) + ".pkl", 'rb')
            else:
                infile = open(string + str(i + 2) + ".pkl", 'rb')
            example_dict = pickle.load(infile)
            self.x_new_values, power_smooth = self.make_smooth_line(example_dict['modified_reward'][:int(self.max_x_number/10)])
            list[i].append(power_smooth)

    # implementation from https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
    def make_smooth_line(self, list):
        xnew = np.linspace(self.min_x_number, self.max_x_number,
                           len(list))  # 300 represents number of points to make between T.min and T.max
        # spl = make_interp_spline(self.x_values, list, k=55)  # BSpline object
        # power_smooth = spl(xnew)
        ysmoothed = gaussian_filter1d(list, sigma=1)
        return xnew, np.array(ysmoothed)

    def plot_figure(self, PolyRL_result, DDPG_result,ParamNoise_result,Div_Dis_result):
        y_PolyRL_result = np.mean(PolyRL_result, axis=0)
        error_PolyRL_result = stats.sem(PolyRL_result, axis=0)/3
        y_ParamNoise_result = np.mean(ParamNoise_result, axis=0)
        error_ParamNoise_result = stats.sem(ParamNoise_result, axis=0)/3
        y_DDPG_result = np.mean(DDPG_result, axis=0)
        error_DDPG_result = stats.sem(DDPG_result, axis=0)/3
        y_Div_Dis_result = np.mean(Div_Dis_result, axis=0)
        error_Div_Dis_result_result = stats.sem(Div_Dis_result, axis=0)/3
        plt.xlabel('Episodes')
        plt.ylabel('Normalized Reward')
        # plt.yscale('log')
        plt.xscale('log')
        plt.plot(self.x_new_values, y_PolyRL_result[0], "b",label='DDPG with PolyRL')
        plt.plot(self.x_new_values, y_ParamNoise_result[0], "r", label='DDPG with Parameter Noise')
        plt.plot( self.x_new_values, y_DDPG_result[0], "g", label='DDPG')
        plt.plot(self.x_new_values, y_Div_Dis_result[0], "orange", label='Div-DDPG')
        plt.fill_between(self.x_new_values, y_PolyRL_result[0] - error_PolyRL_result[0], y_PolyRL_result[0] + error_PolyRL_result[0], edgecolor='#0000FF',
                         facecolor='#0000FF', alpha=0.5,
                         linewidth=0)
        plt.legend(loc='upper left')
        plt.fill_between(self.x_new_values, y_ParamNoise_result[0] - error_ParamNoise_result[0], y_ParamNoise_result[0] + error_ParamNoise_result[0], edgecolor='#3F7F4C',
                         facecolor='lightcoral', alpha=0.5,
                         linewidth=0)
        plt.fill_between(self.x_new_values, y_DDPG_result[0] - error_DDPG_result[0], y_DDPG_result[0] + error_DDPG_result[0], edgecolor='#3F7F4C', facecolor='#3F7F4C', alpha=0.5,
                         linewidth=0)
        plt.fill_between(self.x_new_values, y_Div_Dis_result[0] - error_Div_Dis_result_result[0], y_Div_Dis_result[0] + error_Div_Dis_result_result[0], edgecolor='#3F7F4C',
                         facecolor='orange', alpha=0.5,
                         linewidth=0)
        axes = plt.gca()
        # axes.set_ylim([0, 100])
        plt.show()

Create_Graph(Number_of_iteration=2)