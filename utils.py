import typing
import pickle
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


def save_pickle(filepath: str, data: np.ndarray) -> bool:
    try:
        with open(filepath, 'wb') as fw:
            pickle.dump(data, fw)
        return True
    except Exception as e:
        return False


def read_pickle(filepath: str) -> np.ndarray:
    try:
        with open(filepath, 'rb') as fr:
            data = pickle.load(fr)
        return data
    except Exception as e:
        return np.ndarray([])


def plot_anything(data_to_plot: list) -> None:
    """
    As you see, plot anything in one figure.
    data_to_plot: e.g.: [(x1_list, y1_list), (y2_list), (x3_list, y3_list), ...]
    block:        block the main program or not
    """
    data = deepcopy(data_to_plot)  # preserve original data
    amount = len(data)
    if amount >= 10:
        data = data[:9]
    fig = plt.figure(10 + amount)
    ax = fig.add_subplot(amount*100+11)
    
    each = data[0]
    if type(each) == tuple:
        if len(each) == 1:
            ax.plot(each[0])
        elif len(each) == 2:
            ax.plot(each[0], each[1])
    elif type(each) == list:
        ax.plot(each)
    elif type(each) == np.ndarray:
        ax.plot(each)
    else:
        raise TypeError("things in data_to_plot must be tuple or list!")
    
    for index, each in enumerate(data[1:]):
        ax2 = fig.add_subplot(amount*100+10+index+2, sharex=ax)
        if type(each) == tuple:
            if len(each) == 1:
                ax2.plot(each[0])
            elif len(each) == 2:
                ax2.plot(each[0], each[1])
        elif type(each) == list or type(each) == np.ndarray:
            ax2.plot(each)
        else:
            raise TypeError("things in data_to_plot must be tuple or list!")
    plt.show()


def plot_imfs(t, IMFs: typing.List[list], title: str=""):
    plt.figure()
    for i in range(len(IMFs)):
        plt.subplot(len(IMFs), 1, i+1)
        plt.plot(t, IMFs[i])
        if i==0:
            plt.rcParams['font.sans-serif']='Times New Roman'
            plt.title(f'{title} Decomposition Signal',fontsize=14)
        elif i==len(IMFs)-1:
            plt.rcParams['font.sans-serif']='Times New Roman'
            plt.xlabel('Time/s')
    plt.show()
