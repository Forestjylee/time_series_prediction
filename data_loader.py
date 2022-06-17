import typing
import numpy as np
from copy import deepcopy
from series_toolkit import SeriesProcessor
from utils import read_pickle, save_pickle


def add_gauss_noise(original_signal: list, sigma: float = 0.2) -> np.ndarray:
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


def generate_fake_signal_to_test_prediction(
    t: float, A=1, f=5, fs=100, phi=0
) -> np.ndarray:
    """
    :params A:    振幅
    :params f:    信号频率
    :params fs:   采样频率
    :params phi:  相位
    :params t:    时间长度
    """
    # 若时间序列长度为 t=1s,
    # 采样频率 fs=1000 Hz, 则采样时间间隔 Ts=1/fs=0.001s
    # 对于时间序列采样点个数为 n=t/Ts=1/0.001=1000, 即有1000个点,每个点间隔为 Ts
    Ts = 1 / fs
    n = t / Ts
    n = np.arange(n)
    y = A * np.sin(2 * np.pi * f * n * Ts + phi * (np.pi / 180))
    return y


def generate_fake_signal_to_test_decompose(fs: int = 1000) -> np.ndarray:
    """根据采样率生成用于测试分解的一段信号

    Args:
        fs (int, optional): 采样率. Defaults to 1000.

    Returns:
        _type_: _description_
    """
    t = np.arange(0, 1.0, 1.0 / fs)
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


def load_electrical_data(filepath: str) -> np.ndarray:
    raw_electrical_data = []
    with open(filepath, "r") as fr:
        for line in fr.readlines():
            raw_electrical_data.append(int(line.strip()))
    interpolated_electrical_data = SeriesProcessor.get_spline_interpolate_curve(
        raw_electrical_data, 1000
    )
    return interpolated_electrical_data


def _slide_window_split(data: typing.Union[np.ndarray, list], is_shuffle: bool = False) -> np.ndarray:
    if not data:
        return np.array([])
    if isinstance(data, np.ndarray):
        data = data.tolist()
    data_segments = []
    data = SeriesProcessor.get_range_normalized_data(data)
    for i in range(len(data) - 11):
        data_segments.append(data[i : i + 11])

    data_segments = np.array(data_segments).astype("float64")
    if is_shuffle is True:
        np.random.shuffle(data_segments)

    return data_segments


def slide_window_train_test_split(data: np.ndarray, test_ratio=0.2) -> typing.Tuple:
    train_test_boundary = int(len(data) * (1 - test_ratio))
    train_data, test_data = data[:train_test_boundary], data[train_test_boundary:]

    train_data_segments = _slide_window_split(train_data, is_shuffle=True)
    test_data_segments = _slide_window_split(test_data, is_shuffle=False)

    X_train, y_train = train_data_segments[:, :-1], train_data_segments[:, -1]
    X_test, y_test = test_data_segments[:, :-1], test_data_segments[:, -1]

    return X_train, y_train, X_test, y_test


def get_train_and_test_data() -> typing.Tuple:
    # data = load_data("test.pkl")
    # X_train, y_train, X_test, y_test = slide_window_train_test_split(data, test_ratio=0.2)
    electrical_data1 = load_electrical_data("500kV250电力变压器.txt")
    electrical_data2 = load_electrical_data("500kV334电力变压器.txt")
    train_test_boundary = int(0.2 * len(electrical_data2))

    # train_data_segments1 = _slide_window_split(electrical_data1)
    # train_data_segments2 = _slide_window_split(electrical_data2[:train_test_boundary])
    
    # train_data_segments = np.concatenate((train_data_segments1, train_data_segments2), axis=0)
    # test_data_segments = _slide_window_split(electrical_data2[train_test_boundary:], is_shuffle=False)
    
    train_data_segments = _slide_window_split(electrical_data1)
    test_data_segments = _slide_window_split(electrical_data2, is_shuffle=False)
    
    np.random.shuffle(train_data_segments)
    
    X_train, y_train = train_data_segments[:, :-1], train_data_segments[:, -1]
    X_test, y_test = test_data_segments[:, :-1], test_data_segments[:, -1]
    
    return X_train, y_train, X_test, y_test
