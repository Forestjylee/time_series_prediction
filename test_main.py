from utils import *
from data_loader import *
from series_toolkit import SeriesProcessor


if __name__ == '__main__':
    fake_sin_signal = generate_fake_signal_to_test_prediction(t=50, fs=100)
    noisy_fake_sin_signal = add_gauss_noise(fake_sin_signal, sigma=0.5)
    save_data("test.pkl", noisy_fake_sin_signal)
    plot_anything([fake_sin_signal, noisy_fake_sin_signal])
    
    # save_data("test.pkl", noisy_fake_sin_signal)
    # loaded_data = load_data("test.pkl")
    # plot_anything([loaded_data])
    
    # fake_mixed_signal = generate_fake_signal_to_test_decompose(1000)
    # plot_anything([fake_mixed_signal])
    
    # Decompose
    # t = list(range(len(fake_mixed_signal)))
    # sp = SeriesProcessor()
    # t, IMFs = sp.get_EMD_imfs(t, fake_mixed_signal)
    # # t, IMFs = sp.get_EEMD_imfs(t, fake_mixed_signal)
    # # t, IMFs = sp.get_VMD_imfs(t, fake_mixed_signal, 6)
    # plot_imfs(t, IMFs, title="EMD")
    
    # Get tain and test data
    # get_train_and_test_data()
