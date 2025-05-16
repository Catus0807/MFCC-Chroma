import numpy as np
import librosa
import scipy.signal
import matplotlib.pyplot as plt

def chroma(signal, fs):
 
    signal_len = len(signal)
    # Tính tần số của từng bin và ánh xạ sang nốt nhạc
    f = np.log2(
        np.array(
            [((1 + inx) * fs) / (27.5 * (2 * signal_len)) for inx in range(0, signal_len)]
        )
    )
    num_chroma = np.round(f * 12).astype(int)
    num_f_chroma = np.zeros((num_chroma.shape[0],))
    
    # Tổng hợp số bin cho từng nốt
    for n in np.unique(num_chroma):
        inx = np.nonzero(num_chroma == n)
        num_f_chroma[inx] = inx[0].shape

    # Tính năng lượng cho từng nốt
    if num_chroma.max() < num_chroma.shape[0]:
        c = np.zeros((num_chroma.shape[0],))
        c[num_chroma] = signal ** 2
        c = c / num_f_chroma[num_chroma]
    else:
        i = np.nonzero(num_chroma > num_chroma.shape[0])[0][0]
        c = np.zeros((num_chroma.shape[0],))
        c[num_chroma[0 : i - 1]] = signal ** 2
        c = c / num_f_chroma
    
    # Ánh xạ vào 12 nốt
    final_matrix = np.zeros((12, 1))
    d = int(np.ceil(c.shape[0] / 12.0) * 12)
    c2 = np.zeros((d,))
    c2[0 : c.shape[0]] = c
    c2 = c2.reshape(int(c2.shape[0] / 12), 12)
    final_matrix = np.sum(c2, axis=0).reshape(1, -1).T

    # Chuẩn hóa năng lượng
    signal_energy = np.sum(signal ** 2)
    if signal_energy == 0:
        return final_matrix / 1e-8
    return final_matrix / signal_energy

def chroma_function(file_path, sr=16000, frame_length=0.025, hop_length=0.010, nfft=2048):
    """
    Tính đặc trưng Chroma cho file âm thanh và trả về vector trung bình cùng ma trận Chroma.
    """
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
    fft_magnitude = np.abs(np.fft.rfft(frames, n=nfft, axis=1))

    # Tính Chroma cho tất cả khung
    n_chroma = 12
    chroma_features = np.zeros((num_frames, n_chroma))
    for i in range(num_frames):
        chroma_features[i, :] = chroma(fft_magnitude[i, :], sr).flatten()

    # Xác định các khung không lặng
    non_silent_indices = np.where(np.sum(fft_magnitude, axis=1) >= 1e-6)[0]
    print(f"Số khung không lặng: {len(non_silent_indices)}")

    # Chuẩn hóa Chroma về (-1, 1)
    chroma_features_normalized = np.zeros_like(chroma_features)
    if len(non_silent_indices) > 0:
        non_silent_chroma = chroma_features[non_silent_indices, :]
        
        # Tính min/max cho từng nốt Chroma (axis=0)
        min_vals = np.min(non_silent_chroma, axis=0, keepdims=True)
        max_vals = np.max(non_silent_chroma, axis=0, keepdims=True)
        ranges = max_vals - min_vals + 1e-8
        
        # Áp dụng min-max scaling
        normalized = 2 * (non_silent_chroma - min_vals) / ranges - 1
        normalized = np.clip(normalized, -1.0, 1.0)
        
        chroma_features_normalized[non_silent_indices, :] = normalized

    # Tính vector trung bình từ dữ liệu ĐÃ CHUẨN HÓA
    if len(non_silent_indices) > 0:
        chroma_mean = np.mean(chroma_features_normalized[non_silent_indices, :], axis=0)
    else:
        chroma_mean = np.zeros(n_chroma)

    # Chuyển vị ma trận
    return chroma_mean, chroma_features_normalized.T, chroma_features_normalized.T

def plot_chroma_heatmap(chroma_features, hop_length, sr, title="Hình. Minh họa Chroma"):
    """
    Vẽ heatmap Chroma trên tất cả các khung.
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(chroma_features, aspect='auto', origin='lower', cmap='jet', interpolation='nearest')
    plt.colorbar(label='Năng lượng Chroma')

    # Đặt nhãn trục x (Thời gian)
    num_frames = chroma_features.shape[1]
    time_axis = np.arange(num_frames) * hop_length
    plt.xticks(ticks=np.linspace(0, num_frames-1, 5), labels=np.round(np.linspace(0, time_axis[-1], 5), 2))
    plt.xlabel('Thời gian (s)')

    # Đặt nhãn trục y (Nốt nhạc)
    plt.ylabel('Nốt nhạc')
    plt.yticks(ticks=np.arange(12), labels=['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#'])
    plt.ylim(-0.5, 11.5)  # Đảm bảo hiển thị đầy đủ 12 nốt

    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_chroma_mean(chroma_mean, title="Biểu đồ Vector Chroma Trung bình"):
    """
    Vẽ biểu đồ cột cho vector Chroma trung bình (12 chiều).
    """
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(chroma_mean)), chroma_mean, color='skyblue')
    plt.xlabel('Nốt nhạc')
    plt.ylabel('Năng lượng Chroma trung bình')
    plt.title(title)
    plt.xticks(range(len(chroma_mean)), ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#'])
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

def test_chroma(file_path):
    """
    Hàm test Chroma với một file âm thanh.
    """
    try:
        # Đọc tín hiệu âm thanh để kiểm tra
        y, sr = librosa.load(file_path, sr=16000)
        plot_signal(y, sr)

        # Tính Chroma
        chroma_mean, chroma_features, chroma_features_normalized = chroma_function(file_path)

         # In giá trị Chroma đã chuẩn hóa
        print("Giá trị Chroma đã chuẩn hóa (min, max):", 
              np.min(chroma_features_normalized), 
              np.max(chroma_features_normalized))
        print("Vector Chroma trung bình (đã chuẩn hóa):", chroma_mean)

        # Vẽ heatmap Chroma (dùng bản đã chuẩn hóa để tăng độ tương phản)
        plot_chroma_heatmap(chroma_features_normalized, hop_length=0.010, sr=16000)

        # Vẽ biểu đồ cột cho vector Chroma trung bình
        plot_chroma_mean(chroma_mean)

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file âm thanh '{file_path}'.")
    except Exception as e:
        print(f"Lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    # Đường dẫn đến file âm thanh
    file_path = "sound-fixed_Cello_F1-Arioso_in_C_for_Cello_Solo.wav"  # Thay bằng đường dẫn file âm thanh của bạn
    test_chroma(file_path)