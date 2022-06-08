'''
序列处理工具函数集
@author: Junyi Lee
@createDate: 06/07/22
@updateDate: 06/08/22
'''
import math
import typing
import pickle
import numpy as np
from tqdm import tqdm
from scipy import signal
from typing import Union
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from vmdpy import VMD
from PyEMD import EMD, EEMD, Visualisation

# define annotation type
SP = typing.TypeVar('SP', bound='SeriesProcessor')


class SeriesProcessor(object):
    """
    WARNING:
    所有以get开头的函数不会改变自身的self.data
    除此之外的函数均会改变自身的self.data
    =======================================

    示例：
    >>> s = SeriesProcessor()
    >>> s.load_data_from_filepath("test.pkl")
    >>> s[500:1000].z_score_normalize().plot()
    """

    def __init__(self, init_data: typing.Union[list, np.ndarray]=np.array([]), lite_mode: bool = True):
        """[summary]

        Args:
            init_data (Union[list, np.ndarray]): 用于初始化的数据
            lite_mode (bool, optional): 是否保存原始数据的副本，默认不保存. Defaults to True.
        """
        self.lite_mode = lite_mode

        self.load_data_from_list(init_data)
        self.__raw_data = []    # 原始数据，永不变动
        self.__data_length = 0
        self.__raw_data_length = 0

    def __str__(self):
        return f"<SeriesProcessor object at {hex(id(self))}, raw data length is {self.__raw_data_length}, now data length is {self.__data_length}>"

    def __repr__(self):
        return f"<SeriesProcessor object at {hex(id(self))}, raw data length is {self.__raw_data_length}, now data length is {self.__data_length}>"

    def __len__(self):
        return self.__data_length

    def __add__(self, other):
        return self.data + other

    def __sub__(self, other):
        return self.data - other

    def __mul__(self, other):
        return self.data * other

    def __truediv__(self, other):
        return self.data / other

    def __floordiv__(self, other):
        return self.data // other

    def __iter__(self):
        """enable use `for` to walk through self.data"""
        return iter(self.data)

    def __getitem__(self, index):
        """define `s[::] | s[] | s[[...]]` operations"""

        if isinstance(index, int) or isinstance(index, np.int64):
            return self.data[index]
        elif isinstance(index, slice):
            return self.get_sliced_data(index.start, index.stop, index.step)
        elif isinstance(index, list):
            return get_SeriesProcessor_instance().load_data_from_list(self.data[index])
        else:
            raise TypeError(f"Not support type {type(index)}")

    def __setitem__(self, index, value):
        if isinstance(index, int) or isinstance(index, np.int64):
            self.data[index] = value
        elif isinstance(index, list):
            for i, each_index in index:
                self.data[each_index] = value[i]

    def load_data_from_pkl(self, filepath: str) -> SP:
        """从pkl文件中读取pickle序列化的数据"""
        self.data = read_pickle(filepath)
        self.__data_length = len(self.data)
        if self.lite_mode is False:
            self.__raw_data = deepcopy(self.data)
            self.__raw_data_length = self.__data_length
        return self

    def load_data_from_list(self, raw_data: typing.Union[list, np.ndarray]) -> SP:
        if isinstance(raw_data, np.ndarray):
            self.data = deepcopy(raw_data)
        elif isinstance(raw_data, list):
            self.data = np.array(raw_data)
        else:
            raise TypeError("Only list and ndarray are valid!")
        self.__data_length = len(raw_data)
        if self.lite_mode is False:
            self.__raw_data = deepcopy(raw_data)
            self.__raw_data_length = self.__data_length
        return self

    def set_data(self, data: Union[list, SP, np.ndarray]) -> SP:
        if isinstance(data, list):
            self.data = np.array(data)
            self.__data_length = len(data)
        elif isinstance(data, SeriesProcessor):
            self.data = data.get_data()
            self.__data_length = data.get_data_length()
        elif isinstance(data, np.ndarray):
            self.data = deepcopy(data)
            self.__data_length = data.size
        else:
            raise TypeError(f"set_data() donot support type {type(data)}")
        return self

    def get_data(self, start: int = None, end: int = None) -> np.ndarray:
        if start is None:
            start = 0
        if end is None:
            end = self.__data_length
        return self.data[start:end]

    def get_data_length(self) -> int:
        return self.__data_length

    def reset_data(self) -> SP:
        self.data = self.set_data(self.__raw_data)
        return self

    def get_raw_data(self) -> np.ndarray:
        return deepcopy(self.__raw_data)

    def get_raw_data_length(self) -> int:
        return self.__raw_data_length

    def to_list(self) -> list:
        return self.data.tolist()

    def copy(self) -> SP:
        s = get_SeriesProcessor_instance()
        return s.load_data_from_list(self.data)

    def plot(
        self, start: int = 0, end: int = 0,
        with_raw_data: bool = False, exit_after_plot: bool = False
    ) -> None:
        end = len(self.data) if end == 0 else end
        if with_raw_data is True:
            plt.plot(self.__raw_data)
        plt.plot(self.data[start:end])
        plt.show()
        if exit_after_plot is True:
            exit(0)

    def process_with_func(self, my_function, args: tuple = (), kwargs: dict = {}) -> SP:
        """
        用自定义函数处理数据，自定义的函数至少接收一个list类型的数据，并返回一个list类型的数据
        自定义函数的参数传递可以使用args(元组)，kwargs(字典)传入
        """
        processed_data = my_function(self.get_data(), *args, **kwargs)
        self.set_data(processed_data)
        return self

    def scale(self, scale: float) -> SP:
        """对data进行乘法缩放操作"""
        scaled_data = self.get_scaled_data(self.data, scale)
        self.set_data(scaled_data)
        return self

    def flip_value(self) -> SP:
        """以0为对称轴，对所有的数取相反数

        Returns:
            SP: [description]
        """
        self.data = [-data for data in self.data]
        return self

    def detrend(self) -> SP:
        """对data进行去趋势化操作"""
        detrended_data = signal.detrend(self.data).tolist()
        self.set_data(detrended_data)
        return self

    def remove_mean(self) -> SP:
        """将数据的baseline设为0附近"""
        mean_val = sum(self.data) / self.__data_length
        data_mean_value_as_baseline = [i-mean_val for i in self.data]
        self.set_data(data_mean_value_as_baseline)
        return self

    @staticmethod
    def __z_score_normalize(data: list) -> list:
        data_length = len(data)
        mean_val = sum(data) / data_length
        standard_deviation = math.sqrt(
            sum([(x-mean_val)**2 for x in data])/data_length)
        z_score_normalized_data = [
            (x-mean_val)/standard_deviation for x in data]
        return z_score_normalized_data

    def z_score_normalize(self) -> SP:
        """
        Z-score标准化
        z=x-mean/standard_deviation
        """
        z_score_normalized_data = self.__z_score_normalize(self.data)
        self.set_data(z_score_normalized_data)
        return self

    @staticmethod
    def __range_normalize(data: list, low_threshold: float = 0, high_threshold: float = 1) -> list:
        d_max = max(data)
        d_min = min(data)
        if d_max == d_min:
            return [0] * len(data)
        data = np.array(data)
        normalized_data = (data-d_min) / (d_max-d_min) * \
            (high_threshold-low_threshold) + low_threshold
        return normalized_data.tolist()

    def range_normalize(self, low_threshold: float = 0, high_threshold: float = 1) -> SP:
        """
        给定目标区间，进行归一化操作
        :param low_threshold: 目标数据最小值
        :param high_threshold: 目标数据最小值
        :return:
        """
        normalized_data = self.__range_normalize(
            self.data, low_threshold, high_threshold)
        self.set_data(normalized_data)
        return self

    def spline_interpolate(self, amount: int, kind: str = 'cubic') -> SP:
        """spiline interpolate self.data样条插值

        Args:
            amount (int): 插值后的序列长度
            kind (str, optional): 插值类型. Defaults to 'cubic'.

        Returns:
            SP: [description]
        """
        res = self.get_spline_interpolate_curve(self.data, amount, kind)
        self.set_data(res)
        return self

    def difference_waveform(self) -> SP:
        """求差分曲线

        Returns:
            SP: [description]
        """
        self.set_data(self.get_difference_waveform(self.data))
        return self

    def get_sliced_data(self, start: int = None, end: int = None, step: int = None) -> SP:
        return get_SeriesProcessor_instance(self.sample_rate).load_data_from_list(self.data[start:end:step])

    @classmethod
    def get_range_normalized_data(cls, data: list, low_threshold: float = 0, high_threshold: float = 1) -> list:
        return cls.__range_normalize(data, low_threshold, high_threshold)

    @classmethod
    def get_z_score_normalized_data(cls, data: list):
        return cls.__z_score_normalize(data)

    @staticmethod
    def get_self_correlate_waveform(data_list: list) -> list:
        """计算self.data自相关曲线"""
        data_length = len(data_list)
        acf = np.correlate(data_list, data_list, mode='full')  # 自相关
        acf = acf[data_length-1:]
        acf = acf / acf[0]
        return acf.tolist()

    @staticmethod
    def get_difference_waveform(data_list: list) -> list:
        """求差分曲线"""
        data_length = len(data_list)
        gn1 = []
        for i in range(1, data_length - 1):
            gn1.append((data_list[i + 1] - data_list[i - 1]) / 2)

        gn2 = []
        for i in range(2, data_length - 2):
            gn2.append((2 * data_list[i + 1] + data_list[i + 2] -
                        2 * data_list[i - 1] - data_list[i - 2]) / 8)

        Gn = []
        for i in range(0, len(gn2)):
            Gn.append(gn1[i + 1] * gn1[i + 1] + gn2[i] * gn2[i])

        return Gn

    @staticmethod
    def get_hilbert_envelope(data_list: list) -> list:
        """求hilbert包络线"""
        analytical_signal = signal.hilbert(np.array(data_list))
        amplitude_envelope = np.abs(analytical_signal)
        # instantaneous_phase = np.unwrap(np.angle(analytical_signal))
        return amplitude_envelope.tolist()

    @staticmethod
    def get_EMD_imfs(t: list, data_list: list) -> typing.Tuple[list, typing.List[list]]:
        # if t does not matter, can pass range(len(data_list))
        emd = EMD()
        IMFs = emd.emd(np.array(data_list))
        return t, IMFs
    
    @staticmethod
    def get_EEMD_imfs(t: list, data_list: list) -> typing.Tuple[list, typing.List[list]]:
        eemd = EEMD()
        emd = eemd.EMD
        emd.extrema_detection="parabol"
        IMFs= eemd.eemd(np.array(data_list), np.array(t))
        return t, IMFs
    
    @staticmethod
    def get_VMD_imfs(t: list, data_list: list, amount: int) -> typing.Tuple[list, typing.List[list]]:
        alpha = 2000       # moderate bandwidth constraint
        tau = 0.           # noise-tolerance (no strict fidelity enforcement)
        DC = 0             # no DC part imposed
        init = 1           # initialize omegas uniformly
        tol = 1e-7
        # Run actual VMD code
        IMFs, _, _ = VMD(np.array(data_list), alpha, tau, amount, DC, init, tol)
        return t, IMFs

    @staticmethod
    def get_spline_interpolate_curve(data_list: list, amount: int, kind='cubic') -> list:
        """获取经过样条插值之后的曲线
        @param amount: 插值之后的目标序列长度
        """
        x = [i for i in range(len(data_list))]
        res = interp1d(x, data_list, kind=kind)(
            np.linspace(0, len(data_list)-1, amount)).tolist()
        return res

    @staticmethod
    def get_median_filtered_data(data_list: list, order: int = 3) -> list:
        """
        中值滤波. Replace each RRi value by the median of its ⌊N/2⌋
        neighbors. The first and the last ⌊N/2⌋ RRi values are not filtered

        Parameters
        ----------
        rri : array_like
            sequence containing the RRi series
        order : int, optional
            Strength of the filter. Number of adjacent RRi values used to calculate
            the median value to replace the current RRi. Defaults to 3.

        .. math::
            considering movinge average of order equal to 3:
                RRi[j] = np.median([RRi[j-2], RRi[j-1], RRi[j+1], RRi[j+2]])

        Returns
        -------
        results : RRi list
            instance of the RRi class containing the filtered RRi values
        """
        if len(data_list) < order-1:
            return []
        results = signal.medfilt(data_list, kernel_size=order).tolist()
        return results

    @staticmethod
    def get_quartile_values(data_list: list) -> typing.Tuple[int, int, int]:
        """计算三个四等分点"""
        sorted_data_list = sorted(data_list)
        length = len(sorted_data_list)
        first_quartile_index = int((length+1) * 0.25)
        second_quartile_index = int((length+1) * 0.5)
        third_quartile_index = int((length+1) * 0.75)
        return sorted_data_list[first_quartile_index], sorted_data_list[second_quartile_index], sorted_data_list[third_quartile_index]

    @staticmethod
    def get_detrended_data(data_list: list) -> list:
        """对data进行去趋势化操作"""
        detrended_data = signal.detrend(data_list).tolist()
        return detrended_data

    @staticmethod
    def get_scaled_data(data_list: list, scale: float) -> list:
        """输出放大指定倍数后的序列

        Args:
            data_list (list): 数据
            scale (float): 放大倍数

        Returns:
            list: [description]
        """
        return [i*scale for i in data_list]

    def get_processed_data_with_func(self, my_function, args: tuple = (), kwargs: dict = {}) -> typing.Any:
        """
        用自定义函数处理数据，自定义的函数至少接收一个list类型的数据
        自定义函数的参数传递可以使用args(元组)，kwargs(字典)传入
        """
        processed_data = my_function(self.get_data(), *args, **kwargs)
        return processed_data

    def get_data_values_from_indexes(self, indexes: list) -> np.ndarray:
        """
        根据下标列表从self.data中取值，返回对应的值列表
        """
        return self.data[indexes]


def get_SeriesProcessor_instance() -> SeriesProcessor:
    return SeriesProcessor()


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
