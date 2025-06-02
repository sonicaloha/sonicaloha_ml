import os,sys
import time
import h5py
import argparse
import h5py_cache
import numpy as np
from tqdm import tqdm
import cv2,queue
import threading
import sounddevice as sd
import wavio
from constants import DT, START_ARM_POSE, TASK_CONFIGS, DT, Audio_Lenght_For_Learning
from constants import MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
from robot_utils import Recorder, ImageRecorder, get_arm_gripper_positions
from robot_utils import move_arms, torque_on, torque_off, move_grippers
from real_env import make_real_env, get_action
import pyaudio, wave
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from sleep_plus import sleep_all_robots,shut_down_all_robots
from pynput.keyboard import Key, Listener
import IPython

e = IPython.embed

AUDIO = True  # Temp flag, disable audio for debugging

def opening_ceremony(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right, dataset_name):
    """ Move all 4 robots to a pose where it is easy to start demonstration """
    # reboot gripper motors, and set operating modes for all motors

    puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)

    # puppet_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    # puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    # master_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    # master_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
    # # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit
    # puppet_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    # puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    # master_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    # master_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
    # # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit

    # threading
    import threading
    def set_operating_modes(robot, arm_mode, gripper_mode):
        robot.dxl.robot_set_operating_modes("group", "arm", arm_mode)
        robot.dxl.robot_set_operating_modes("single", "gripper", gripper_mode)
    def configure_robots(puppet_bot_left, master_bot_left, puppet_bot_right, master_bot_right):
        threads = [
            threading.Thread(target=set_operating_modes, args=(puppet_bot_left, "position", "current_based_position")),
            threading.Thread(target=set_operating_modes, args=(master_bot_left, "position", "position")),
            threading.Thread(target=set_operating_modes, args=(puppet_bot_right, "position", "current_based_position")),
            threading.Thread(target=set_operating_modes, args=(master_bot_right, "position", "position"))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    configure_robots(puppet_bot_left, master_bot_left, puppet_bot_right, master_bot_right)

    torque_on(puppet_bot_left)
    torque_on(master_bot_left)
    torque_on(puppet_bot_right)
    torque_on(master_bot_right)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [start_arm_qpos] * 4, move_time=1.5)
    # move grippers to starting position
    move_grippers([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=0.5)


    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement
    master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", False)
    print(f'Close the gripper to start collecting the {dataset_name}.')
    close_thresh = -0.3 #-1.4
    pressed = False
    while not pressed:
        gripper_pos_left = get_arm_gripper_positions(master_bot_left)
        gripper_pos_right = get_arm_gripper_positions(master_bot_right)
        if (gripper_pos_left < close_thresh) and (gripper_pos_right < close_thresh):
            pressed = True
        time.sleep(DT/10)
    torque_off(master_bot_left)
    torque_off(master_bot_right)
    print(f'Started!')


def capture_one_episode(dt, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite):
    print("*"*100)
    print(f'Dataset name: {dataset_name}')
    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    env = make_real_env(init_node=False, setup_robots=False)

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    # move all 4 robots to a starting pose where it is easy to start teleoperation, then wait till both gripper closed
    opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right, dataset_name)

    # Data collection
    ts = env.reset(fake=True)
    timesteps = [ts]
    actions = []
    actual_dt_history = []

    audio_sampling_rate = 48000
    channels = 1
    CHUNK = 256 # 1024
    FORMAT = pyaudio.paInt16

    TARGET_DEVICE_NAME = "USB PnP Audio Device"

    # 初始化 PyAudio
    audio = pyaudio.PyAudio()

    # 遍历所有设备，查找匹配的设备名称
    device_index = None
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        if TARGET_DEVICE_NAME in device_info["name"]:
            device_index = i
            print(f"Found device: {device_info['name']}, Index: {device_index}")
            break

    audio.terminate()

    # 选择特定相机的音频设备
    DEVICE_INDEX = device_index

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=channels,
                        rate=audio_sampling_rate, input=True,
                        frames_per_buffer=CHUNK, input_device_index=DEVICE_INDEX)

    audio_queue = queue.Queue()
    recording = True
    # 录音数据存储
    audio_chunks = [[] for _ in range(max_timesteps)]  # 预分配列表

    def record_audio():
        """后台线程：不断采集音频数据，并存入 `audio_queue`"""
        while recording:
            audio_block = stream.read(CHUNK, exception_on_overflow=False)
            audio_queue.put(audio_block)

    recording_thread = threading.Thread(target=record_audio, daemon=True)
    recording_thread.start()

    print("Recording started...")

    time0 = time.time()

    for t in tqdm(range(max_timesteps)):
        t0 = time.time()

        # 机器人执行操作
        action = get_action(master_bot_left, master_bot_right)
        t1 = time.time()

        ts = env.step(action)
        t2 = time.time()

        timesteps.append(ts)
        actions.append(action)
        actual_dt_history.append([t0, t1, t2])
        time.sleep(max(0, DT - (time.time() - t0)))

        while not audio_queue.empty():
            audio_chunks[t].append(audio_queue.get())

    print("Recording finished.")
    print(f"Length of audio_chunks: {len(audio_chunks)}")

    recording = False
    recording_thread.join()
    stream.stop_stream()
    stream.close()
    audio.terminate()
    #
    # with wave.open("recorded_audio.wav", "wb") as wf:1024
    #     wf.setnchannels(channels)
    #     wf.setsampwidth(audio.get_sample_size(FORMAT))
    #     wf.setframerate(audio_sampling_rate)
    #     for step_chunks in audio_chunks:
    #         wf.writeframes(b"".join(step_chunks))  # 合并当前 step 的音频数据
    # print("Audio saved as recorded_audio.wav")

    # Torque on both master bots
    torque_on(master_bot_left)
    torque_on(master_bot_right)
    # Open puppet grippers
    env.puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
    env.puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)

    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 30:
        print(f'\n\nfreq_mean is {freq_mean}, lower than 30, re-collecting... \n\n\n\n')
        return False

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        # '/base_action': [],
        # '/base_action_t265': [],
    }

    # if AUDIO:
    #     data_dict['/audio'] = audio_dataset
    #     data_dict['/audio_sampling_rate'] = audio_sampling_rate  # 存入采样率
    #     data_dict['/audio_channels'] = channels  # 存入通道数


    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/action'].append(action)

        # data_dict['/base_action'].append(ts.observation['base_vel'])
        # data_dict['/base_action_t265'].append(ts.observation['base_vel_t265'])
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

    # plot /base_action vs /base_action_t265
    # import matplotlib.pyplot as plt
    # plt.plot(np.array(data_dict['/base_action'])[:, 0], label='base_action_linear')
    # plt.plot(np.array(data_dict['/base_action'])[:, 1], label='base_action_angular')
    # plt.plot(np.array(data_dict['/base_action_t265'])[:, 0], '--', label='base_action_t265_linear')
    # plt.plot(np.array(data_dict['/base_action_t265'])[:, 1], '--', label='base_action_t265_angular')
    # plt.legend()
    # plt.savefig('record_episodes_vel_debug.png', dpi=300)



    COMPRESS = True

    if COMPRESS:
        # JPEG compression
        t0 = time.time()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] # tried as low as 20, seems fine
        compressed_len = []
        for cam_name in camera_names:
            image_list = data_dict[f'/observations/images/{cam_name}']
            compressed_list = []
            compressed_len.append([])
            for image in image_list:
                result, encoded_image = cv2.imencode('.jpg', image, encode_param) # 0.02 sec # cv2.imdecode(encoded_image, 1)
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))
            data_dict[f'/observations/images/{cam_name}'] = compressed_list
        print(f'compression: {time.time() - t0:.2f}s')

        # pad so it has same length
        t0 = time.time()
        compressed_len = np.array(compressed_len)
        padded_size = compressed_len.max()
        for cam_name in camera_names:
            compressed_image_list = data_dict[f'/observations/images/{cam_name}']
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
        print(f'padding: {time.time() - t0:.2f}s')



    # HDF5
    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = COMPRESS
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            if COMPRESS:
                _ = image.create_dataset(cam_name, (max_timesteps, padded_size), dtype='uint8',
                                         chunks=(1, padded_size), )
            else:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
        _ = obs.create_dataset('qpos', (max_timesteps, 14))
        _ = obs.create_dataset('qvel', (max_timesteps, 14))
        _ = obs.create_dataset('effort', (max_timesteps, 14))
        _ = root.create_dataset('action', (max_timesteps, 14))
        _ = root.create_dataset('base_action', (max_timesteps, 2))
        # _ = root.create_dataset('base_action_t265', (max_timesteps, 2))

        if AUDIO:
            max_substeps = 2048   # may need to modify
            audio_chunks_np = np.zeros((max_timesteps, max_substeps), dtype=np.int16)
            for t in range(max_timesteps):
                # Merge all chunks of this time step
                audio_data = np.frombuffer(b"".join(audio_chunks[t]), dtype=np.int16)
                if len(audio_data) > max_substeps - 1:
                    raise ValueError(
                        f"Time step {t} has too many audio samples ({len(audio_data)} > {max_substeps})! Please check the data.")
                audio_chunks_np[t, :len(audio_data)] = audio_data
                audio_chunks_np[t, -1] = len(audio_data)
            root.create_dataset("audio", data=audio_chunks_np, dtype=np.int16)

            # audio_chunks_int16 = [np.frombuffer(b"".join(step), dtype=np.int16) for step in audio_chunks]
            # vlen_dtype = h5py.special_dtype(vlen=np.int16)
            # root.create_dataset("audio", (max_timesteps,), dtype=vlen_dtype, data=audio_chunks_int16)

            root.attrs['audio_sampling_rate'] = audio_sampling_rate  # 记录采样率
            root.attrs['audio_channels'] = channels  # 记录音频通道数
            print(f"Audio saved as int16")

        for name, array in data_dict.items():
            root[name][...] = array

        if COMPRESS:
            _ = root.create_dataset('compress_len', (len(camera_names), max_timesteps))
            root['/compress_len'][...] = compressed_len

    print(f'Saving {dataset_name} all information: {time.time() - t0:.1f} secs')


    return True


def main(args):
    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']

    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)
    overwrite = True

    dataset_name = f'episode_{episode_idx}'
    print(dataset_name + '\n')
    while True:
        is_healthy = capture_one_episode(DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite)
        if is_healthy:
            break


def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}')
    return freq_mean

def debug():
    print(f'====== Debug mode ======')
    recorder = Recorder('right', is_debug=True)
    image_recorder = ImageRecorder(init_node=False, is_debug=True)
    while True:
        time.sleep(1)
        recorder.print_diagnostics()
        image_recorder.print_diagnostics()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
#     parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
#     main(vars(parser.parse_args())) # TODO
    # debug()
def on_press(key):
    global user_input
    try:
        # 处理键盘字符键
        # if key.char == 'c' or key.char == 'C':
        #     user_input = 'C'
        if key.char == 'c' or key.char == 'C':
            user_input = 'C'
        elif key.char == 'r' or key.char == 'R':
            user_input = 'R'
        elif key.char == 'q' or key.char == 'Q':
            user_input = 'Q'
    except AttributeError:
        # 处理其他键类型，例如功能键等
        pass

def listen_for_key():
    # 监听键盘输入直到获取有效输入
    with Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    parser.add_argument('--start_idx', action='store', type=int, help='Start index.', required=True) # @gnq
    parser.add_argument('--end_idx', action='store', type=int, help='Start index.', required=True)
    # main(vars(parser.parse_args()))
    # debug()

    args = vars(parser.parse_args())

    current_episode = args['start_idx']

    while current_episode <= args['end_idx']:
        args['episode_idx'] = current_episode

        # success = print("eposid", current_episode)
        success = main(args)

        current_episode += 1

        print("*" * 100)
        print(f"The {current_episode-1} episode finished.")
        print("Press 'C' to continue to the next episode, 'R' to repeat this episode, 'Q' to quit:")

        while True:
            user_input = None
            listener = Listener(on_press=on_press)
            listener.start()
            while user_input is None:
                pass
            listener.stop()
            if user_input == 'C':   # left button
                sleep_all_robots()
                # shut_down_all_robots()
                break
            elif user_input == 'R':     # middle button
                sleep_all_robots()
                # shut_down_all_robots()
                current_episode -= 1
                break  # repetition
            elif user_input == 'Q':     # right button
                # sleep_all_robots()
                shut_down_all_robots()
                # delete the quit data
                task_config = TASK_CONFIGS[args['task_name']]
                dataset_dir = task_config['dataset_dir']
                dataset_name = f'episode_{current_episode - 1}.hdf5'
                dataset_path = os.path.join(dataset_dir, dataset_name)
                if os.path.exists(dataset_path):
                    os.remove(dataset_path)
                    print(f"File {dataset_path} has been deleted successfully.")
                else:
                    print(f"File {dataset_path} does not exist.")
                print(f"The data is saved until {current_episode - 2} episode.")
                sys.exit("Quitting the process.")
            else:
                print("Invalid input, please try again.")

        if current_episode > args['end_idx']:
            shut_down_all_robots()
            print("Completed all episodes.")
            print(f"The data is saved to {current_episode - 1} episode.")
            break