# ==============================================================
#   classifier.py
# ==============================================================

import time
import numpy as np
from collections import deque
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes
from multiprocessing import Queue


# ====== 滤波 ======

def sliding_filter(new_data, fs):
    filtered_data = new_data.copy()

    DataFilter.perform_bandpass(
        filtered_data, fs, 0.5, 40.0, 4,
        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
    )

    DataFilter.remove_environmental_noise(
        filtered_data, fs, NoiseTypes.FIFTY.value
    )

    return filtered_data


# ====== 参考信号 ======

def build_ref_matrix(fs, L, f, Nh=3):
    T = int(fs * L)
    t = np.arange(1, T + 1) / fs
    cols = []

    for h in range(1, Nh + 1):
        cols.append(np.sin(2 * np.pi * h * f * t))
        cols.append(np.cos(2 * np.pi * h * f * t))

    return np.stack(cols, axis=1)


# ====== CCA ======

def cca_rho_regression(X, Y):
    X = X - np.mean(X)
    Y = Y - np.mean(Y, axis=0)

    G = Y.T @ Y + 1e-8 * np.eye(Y.shape[1])

    beta = np.linalg.solve(G, Y.T @ X)
    x_hat = Y @ beta

    rho = np.linalg.norm(x_hat) / np.linalg.norm(X)
    return float(rho)


def classify_ssvep(sig, Y):
    X = sig[:, None]

    rhos = [cca_rho_regression(X, Yf) for Yf in Y.values()]
    best_i = int(np.argmax(rhos))

    best_f = list(Y.keys())[best_i]

    return best_f, rhos


# ====== 主循环 ======

def classifier_loop(q: Queue):
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = "COM4"

    fs = 250
    window_length = 3.0
    step = 0.5

    freqs = [10,12,15,20]
    Nh = 3

    buffer = deque(maxlen=int(fs * window_length))
    Y = {f: build_ref_matrix(fs, window_length, f, Nh) for f in freqs}

    board = None

    try:
        board = BoardShim(BoardIds.CYTON_BOARD.value, params)
        board.prepare_session()
        board.start_stream()

        print("\n🧠 分类进程启动...\n")

        while True:

            data = board.get_current_board_data(int(fs * step))
            if data.shape[1] == 0:
                time.sleep(0.02)
                continue

            eeg_chs = BoardShim.get_eeg_channels(
                BoardIds.CYTON_BOARD.value
            )

            eeg = np.mean(data[eeg_chs], axis=0)

            filtered = sliding_filter(eeg, fs)

            buffer.extend(filtered)

            if len(buffer) == int(fs * window_length):
                sig = np.array(buffer)

                freq, rhos = classify_ssvep(sig, Y)

                result = {
                    "freq": float(freq),
                    "score": float(max(rhos))
                }

                q.put(result)

                print(f" -> 预测: {freq:.2f} Hz | rho={max(rhos):.3f}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n🛑 分类进程停止")

    finally:

        if board:
            board.stop_stream()
            board.release_session()
