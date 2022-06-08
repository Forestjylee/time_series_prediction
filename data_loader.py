import random
import typing
import numpy as np
from copy import deepcopy
from series_toolkit import SeriesProcessor
from utils import read_pickle, save_pickle


def add_gauss_noise(original_signal: list, sigma: float=0.2) -> np.ndarray:
    """
    对输入数据加入高斯噪声
    :param original_signal: 原始信号
    :param sigma: 噪声强度，越大噪声越大
    :return: 加入高斯噪声之后的信号
    """
    mu = 0
    noisy_signal = deepcopy(original_signal)
    for i in range(len(original_signal)):
        noisy_signal[i] += random.gauss(mu, sigma)
    return noisy_signal


def generate_fake_signal_to_test_prediction(t: float, A=1, f=5, fs=100, phi=0) -> np.ndarray:
    '''
    :params A:    振幅
    :params f:    信号频率
    :params fs:   采样频率
    :params phi:  相位
    :params t:    时间长度
    '''
    # 若时间序列长度为 t=1s, 
    # 采样频率 fs=1000 Hz, 则采样时间间隔 Ts=1/fs=0.001s
    # 对于时间序列采样点个数为 n=t/Ts=1/0.001=1000, 即有1000个点,每个点间隔为 Ts
    Ts = 1 / fs
    n = t / Ts
    n = np.arange(n)
    y = A*np.sin(2*np.pi*f*n*Ts + phi*(np.pi/180))
    return y


def generate_fake_signal_to_test_decompose(fs: int=1000) -> np.ndarray:
    """根据采样率生成用于测试分解的一段信号

    Args:
        fs (int, optional): 采样率. Defaults to 1000.

    Returns:
        _type_: _description_
    """
    t = np.arange(0, 1.0, 1.0 /fs)
    f1, f2, f3 = 100, 200, 300
    signal = np.piecewise(
        t,
        [t < 1, t < 0.8, t < 0.3],
        [
            lambda t: np.sin(2 * np.pi * f1 * t),
            lambda t: np.sin(2 * np.pi * f2 * t),
            lambda t: np.sin(2 * np.pi * f3 * t),
        ],
    )
    return signal


def save_data(filepath: str, data: np.ndarray) -> bool:
    try:
        save_pickle(filepath, data)
        return True
    except Exception as e:
        return False


def load_data(filepath: str) -> np.ndarray:
    try:
        return read_pickle(filepath)
    except Exception as e:
        return np.ndarray([])


def slide_window_train_test_split(data: np.ndarray, test_ratio=0.2) -> typing.Tuple:
    train_test_boundary = int(len(data) * (1-test_ratio))
    train_data, test_data = data[:train_test_boundary], data[train_test_boundary:]
    
    train_data_segments = []
    for i in range(len(train_data)-11):
        train_data_segments.append(
            SeriesProcessor.get_range_normalized_data(train_data[i:i+11].tolist()))
    random.shuffle(train_data_segments)
    
    test_data_segments = []
    for i in range(len(test_data)-11):
        test_data_segments.append(
            SeriesProcessor.get_range_normalized_data(test_data[i:i+11].tolist()))
    
    train_data_segments = np.array(train_data_segments).astype('float64')
    test_data_segments = np.array(test_data_segments).astype('float64')

    X_train = train_data_segments[:, :-1]
    y_train = train_data_segments[:, -1]
    X_test = test_data_segments[:, :-1]
    y_test = test_data_segments[:, -1]
    return X_train, y_train, X_test, y_test


def get_train_and_test_data() -> typing.Tuple:
    data = load_data("test.pkl")
    X_train, y_train, X_test, y_test = slide_window_train_test_split(data, test_ratio=0.2)
    return X_train, y_train, X_test, y_test
