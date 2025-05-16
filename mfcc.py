import numpy as np
from scipy.fftpack.realtransforms import dct
import librosa
import scipy.signal
import matplotlib.pyplot as plt

LEN = 13

def triangular_filter_bank(fs, nfft, lowfreq=133.33, linc=200/3, logsc=1.0711703, lin_filt=LEN, log_filt=27):
    """
    Triangular filterbank for MFCC computation, adjusted for rfft (nfft//2 + 1 bins).
    lin_filt = No. of linear filters
    log_filt = No. of log filters

    """
    num_filts = lin_filt + log_filt

    # Số bins thực tế từ rfft: nfft//2 + 1
    nfreq_bins = nfft // 2 + 1

    # start, mid and end points of the filters in spectral domain
    freqs = np.zeros(num_filts + 2)
    freqs[0:lin_filt] = lowfreq + np.arange(lin_filt) * linc
    freqs[lin_filt:] = freqs[lin_filt - 1] * logsc ** np.arange(1, log_filt + 3)
    heights = 2.0 / (freqs[2:] - freqs[0:-2])

    # filterbank coeff (fft bins Hz), chỉ lấy tần số dương
    fbank = np.zeros((num_filts, nfreq_bins))
    nfreqs = np.linspace(0, fs/2, nfreq_bins)  # Tần số từ 0 đến fs/2

    for i in range(num_filts):
        f_low = freqs[i]
        f_center = freqs[i + 1]
        f_high = freqs[i + 2]

        left_bin = np.searchsorted(nfreqs, f_low)
        center_bin = np.searchsorted(nfreqs, f_center)
        right_bin = np.searchsorted(nfreqs, f_high)

        # Tính slope cho các đoạn tuyến tính
        for j in range(left_bin, center_bin):
            if j < nfreq_bins:
                fbank[i, j] = heights[i] * (nfreqs[j] - f_low) / (f_center - f_low)
        for j in range(center_bin, right_bin):
            if j < nfreq_bins:
                fbank[i, j] = heights[i] * (f_high - nfreqs[j]) / (f_high - f_center)

    return fbank, freqs

def compute_mfcc(fft_magnitude, fbank, num_mfcc_feats):
    """
    Returns the MFCCs of a signal frame.

    """
    mspec = np.log10(np.dot(fft_magnitude, fbank.T) + 1e-8)
    ceps = dct(mspec, type=2, norm="ortho", axis=-1)[:num_mfcc_feats]
    return ceps

def get_mfcc(fs, nfft, n_mfcc_feats, signal):
    """
    Returns mfcc features for a signal.

    """
    [fbank, freqs] = triangular_filter_bank(fs, nfft)
    feature = compute_mfcc(signal, fbank, n_mfcc_feats)
    return feature

def mfcc_function(file_path, sr=16000, frame_length=0.025, hop_length=0.010, n_mfcc=13, nfft=2048):
    # Đọc tín hiệu âm thanh
    y, sr = librosa.load(file_path, sr=sr)
    duration = len(y) / sr
    print(f"File âm thanh: {duration:.2f} giây, số mẫu: {len(y)}")

    # Chia khung
    frame_length_samples = int(frame_length * sr)
    hop_length_samples = int(hop_length * sr)
    frames = librosa.util.frame(y, frame_length=frame_length_samples, hop_length=hop_length_samples).T.copy()
    num_frames = frames.shape[0]
    print(f"Số khung: {num_frames}")

    # Áp dụng cửa sổ Hamming
    window = scipy.signal.get_window("hamming", frame_length_samples)
    frames *= window

    # Tính FFT magnitude
    fft_magnitude = np.abs(np.fft.rfft(frames, n=nfft, axis=1)) ** 2

    # Tính MFCC cho tất cả khung
    mfcc_features = np.zeros((num_frames, n_mfcc))
    for i in range(num_frames):
        if np.sum(fft_magnitude[i, :]) < 1e-6:
            mfcc_features[i, :] = 0
        else:
            mfcc_features[i, :] = get_mfcc(sr, nfft, n_mfcc, fft_magnitude[i, :])

    # Xác định các khung không lặng
    non_silent_indices = np.where(np.sum(fft_magnitude, axis=1) >= 1e-6)[0]
    print(f"Số khung không lặng: {len(non_silent_indices)}")

    # Chuẩn hóa MFCC về (-1, 1) cho tất cả khung (kể cả lặng)
    mfcc_features_normalized = np.zeros_like(mfcc_features)
    if len(non_silent_indices) > 0:
        non_silent_mfcc = mfcc_features[non_silent_indices, :]
        
        # Tính min/max cho từng hệ số MFCC
        min_vals = np.min(non_silent_mfcc, axis=0, keepdims=True)
        max_vals = np.max(non_silent_mfcc, axis=0, keepdims=True)
        ranges = max_vals - min_vals + 1e-8
        
        # Chuẩn hóa các khung không lặng
        normalized = 2 * (non_silent_mfcc - min_vals) / ranges - 1
        normalized = np.clip(normalized, -1.0, 1.0)
        
        # Gán giá trị đã chuẩn hóa
        mfcc_features_normalized[non_silent_indices, :] = normalized

    # Tính vector trung bình từ dữ liệu ĐÃ CHUẨN HÓA
    if len(non_silent_indices) > 0:
        mfcc_mean_normalized = np.mean(mfcc_features_normalized[non_silent_indices, :], axis=0)
    else:
        mfcc_mean_normalized = np.zeros(n_mfcc)

    # Chuyển vị ma trận
    return mfcc_mean_normalized, mfcc_features_normalized.T, mfcc_features_normalized.T

def plot_mfcc_heatmap(mfcc_features, hop_length, sr, title="Hình. Minh họa MFCCs"):
    """
    Vẽ heatmap MFCC trên tất cả các khung.

    """
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc_features, aspect='auto', origin='lower', cmap='jet', interpolation='nearest')
    plt.colorbar(label='Hệ số MFCC')

    # Đặt nhãn trục x (Thời gian)
    num_frames = mfcc_features.shape[1]
    time_axis = np.arange(num_frames) * hop_length
    plt.xticks(ticks=np.linspace(0, num_frames-1, 5), labels=np.round(np.linspace(0, time_axis[-1], 5), 2))
    plt.xlabel('Thời gian (s)')

    # Đặt nhãn trục y (giá trị MFCC)
    plt.ylabel('Hệ số MFCC')
    plt.ylim(-0.5, 12.5)  # Đảm bảo hiển thị đầy đủ 13 hệ số

    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_mfcc_mean(mfcc_mean, title="Biểu đồ Vector MFCC Trung bình"):
    """
    Vẽ biểu đồ cột cho vector MFCC trung bình (13 chiều).

    """
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(mfcc_mean)), mfcc_mean, color='skyblue')
    plt.xlabel('Hệ số MFCC')
    plt.ylabel('Giá trị MFCC trung bình')
    plt.title(title)
    plt.xticks(range(len(mfcc_mean)), [f'MFCC{i}' for i in range(len(mfcc_mean))])
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_signal(y, sr, title="Tín hiệu âm thanh"):
    """
    Vẽ tín hiệu âm thanh để kiểm tra.
    """
    plt.figure(figsize=(10, 3))
    plt.plot(np.arange(len(y)) / sr, y)
    plt.xlabel('Thời gian (s)')
    plt.ylabel('Biên độ')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def test_mfcc(file_path):
    """
    Hàm test MFCC với một file âm thanh.
    
    """
    try:
        # Đọc tín hiệu âm thanh để kiểm tra
        y, sr = librosa.load(file_path, sr=16000)
        plot_signal(y, sr)

         # Tính MFCC
        mfcc_mean, mfcc_features, mfcc_features_normalized = mfcc_function(file_path)

        # Kiểm tra giá trị
        print("Giá trị MFCC đã chuẩn hóa (min, max):", 
              np.min(mfcc_features_normalized), 
              np.max(mfcc_features_normalized))
        print("Vector MFCC trung bình (đã chuẩn hóa):", mfcc_mean)

        # Vẽ heatmap MFCC (dùng bản đã chuẩn hóa để tăng độ tương phản)
        plot_mfcc_heatmap(mfcc_features_normalized, hop_length=0.010, sr=16000)

        # Vẽ biểu đồ cột cho vector MFCC trung bình
        plot_mfcc_mean(mfcc_mean)

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file âm thanh '{file_path}'.")
    except Exception as e:
        print(f"Lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    # Đường dẫn đến file âm thanh
    file_path = "sound-fixed_Cello_F1-Arioso_in_C_for_Cello_Solo.wav"  # Thay bằng đường dẫn file âm thanh của bạn
    test_mfcc(file_path)