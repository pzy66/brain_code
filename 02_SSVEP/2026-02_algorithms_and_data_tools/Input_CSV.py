import argparse
import time
import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


def main():
    BoardShim.enable_dev_board_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument('--serial-port', type=str, default='COM4')
    parser.add_argument('--board-id', type=int, default=0)
    args = parser.parse_args()

##有指定输入的一个类：
    # class BrainFlowInputParams:
    #     def __init__(self):
    #         self.serial_port = ''
    #         self.mac_address = ''
    #         self.ip_address = ''
    #         self.ip_port = 0
    #         self.other_info = ''
    #         self.serial_number = ''
    #         self.ip_protocol = 0
    #         self.timeout = 0
    #         self.file = ''

    params = BrainFlowInputParams()
    params.serial_port = args.serial_port

    try:
        board = BoardShim(args.board_id, params)
        board.prepare_session()
        board.start_stream()

        print("开始采集数据，持续10秒...")
        time.sleep(10)

        data = board.get_board_data()
        board.stop_stream()
        board.release_session()

        print(f"原始数据形状: {data.shape}")


        eeg_data = data[1:9]
        timestamp_data = data[-2:-1]

        # 合并数据
        selected_data = np.vstack([eeg_data, timestamp_data])

        # 创建列名
        channel_names =[ ] + [f'EEG_Channel_{i + 1}' for i in range(8)] + ['Timestamp']

        # 创建DataFrame
        df = pd.DataFrame(selected_data.T, columns=channel_names)

        # 保存CSV
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"eeg_data_8ch_{timestamp}.csv"
        df.to_csv(filename, index=False)


    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()