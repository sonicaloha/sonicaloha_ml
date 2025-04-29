import os
import numpy as np
import cv2
import h5py
import argparse

import matplotlib.pyplot as plt
from constants import DT
import cv2
import numpy as np
import soundfile as sf
import subprocess
import os
import librosa
import time

import IPython
e = IPython.embed

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]
BASE_STATE_NAMES = ["linear_vel", "angular_vel"]

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        compressed = root.attrs.get('compress', False)
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        if 'effort' in root.keys():
            effort = root['/observations/effort'][()]
        else:
            effort = None
        action = root['/action'][()]
        base_action = root['/base_action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        if compressed:
            compress_len = root['/compress_len'][()]

    if compressed:
        for cam_id, cam_name in enumerate(image_dict.keys()):
            # un-pad and uncompress
            padded_compressed_image_list = image_dict[cam_name]
            image_list = []
            for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list): # [:1000] to save memory
                image_len = int(compress_len[cam_id, frame_id])
                compressed_image = padded_compressed_image
                image = cv2.imdecode(compressed_image, 1)
                image_list.append(image)
            image_dict[cam_name] = image_list

        # **读取音频数据**
    audio_data, audio_sampling_rate, audio_channels = None, None, None
    with h5py.File(dataset_path, 'r') as root:
        if 'audio' in root:
            audio_chunks_np = root['/audio'][()]  # **读取音频数据（可能是 (N,) 或 (N, 2)）**
            audio_sampling_rate = root.attrs.get('audio_sampling_rate')  # 采样率
            # audio_data = np.concatenate(audio_data_chunk, axis=0)
            # audio_data = audio_data.astype(np.float32) / 32768.0
            # audio_channels = root.attrs.get('audio_channels')  # 通道数（1=单声道, 2=双声道）
    audio_reconstructed = []
    for t in range(audio_chunks_np.shape[0]):  # Iterate over timesteps
        length = audio_chunks_np[t, -1]  # Get the actual length of this timestep
        # print(f"t: {t}, length: {length}")
        audio_reconstructed.append(audio_chunks_np[t, :length])
    audio_reconstructed = np.concatenate(audio_reconstructed, axis=0)
    audio_reconstructed = audio_reconstructed.astype(np.float32) / 32768.0
    audio_data = audio_reconstructed

    return qpos, qvel, effort, action, base_action, image_dict, audio_data, audio_sampling_rate, audio_channels


def load_hdf5_half(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        compressed = root.attrs.get('compress', False)

        # **读取轨迹数据**
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        effort = root['/observations/effort'][()] if 'effort' in root.keys() else None
        action = root['/action'][()]
        base_action = root['/base_action'][()]

        # **读取图像数据**
        image_dict = {}
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

        if compressed:
            compress_len = root['/compress_len'][()]

    if compressed:
        for cam_id, cam_name in enumerate(image_dict.keys()):
            padded_compressed_image_list = image_dict[cam_name]
            image_list = []
            for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list):
                image_len = int(compress_len[cam_id, frame_id])
                compressed_image = padded_compressed_image
                image = cv2.imdecode(compressed_image, 1)
                image_list.append(image)
            image_dict[cam_name] = image_list

    # **读取音频数据**
    audio_data, audio_sampling_rate, audio_channels = None, None, None
    with h5py.File(dataset_path, 'r') as root:
        if 'audio' in root:
            audio_data_chunk = root['/audio'][()]  # **音频数据（多个时间步）**
            mid_idx = len(audio_data_chunk) // 2
            audio_data_chunk = audio_data_chunk[mid_idx:]
            audio_data = np.concatenate(audio_data_chunk, axis=0)  # **合并所有时间步的音频**
            audio_data = audio_data.astype(np.float32) / 32768.0
            audio_sampling_rate = root.attrs.get('audio_sampling_rate')  # 采样率
            audio_channels = root.attrs.get('audio_channels')  # 通道数（1=单声道, 2=立体声）

    # **截取最后一半数据**
    mid_idx = len(qpos) // 2  # 计算中点索引
    qpos = qpos[mid_idx:]
    qvel = qvel[mid_idx:]
    effort = effort[mid_idx:] if effort is not None else None
    action = action[mid_idx:]
    base_action = base_action[mid_idx:]

    # **截取图像数据**
    for cam_name in image_dict:
        image_dict[cam_name] = image_dict[cam_name][mid_idx:]

    # **截取音频数据**

    return qpos, qvel, effort, action, base_action, image_dict, audio_data, audio_sampling_rate, audio_channels

def parse_episode_indices(episode_input):
    """
    Parse the input episode indices, which can be a single number,
    a comma-separated list, or a range.
    Examples:
    - "2" -> [2]
    - "2,3,7-8" -> [2, 3, 7, 8]
    - "5-9" -> [5, 6, 7, 8, 9]
    """
    indices = set()  # Use a set to avoid duplicates
    parts = episode_input.split(",")  # Split the input by commas
    for part in parts:
        if "-" in part:  # Handle ranges
            start, end = map(int, part.split("-"))
            indices.update(range(start, end + 1))  # Add all numbers in the range
        else:  # Handle single numbers
            indices.add(int(part))
    return sorted(indices)  # Return a sorted list


def main(args):
    dataset_dir = args['dataset_dir']
    episode_input = args['episode_idx']  # Can be a single number, list, or range, e.g., "2,3,7-8"
    ismirror = args['ismirror']

    # Parse episode_idx to support single or multiple indices
    episode_indices = parse_episode_indices(episode_input)
    print(f"Processing episodes: {episode_indices}")

    for episode_idx in episode_indices:
        if ismirror:
            dataset_name = f'mirror_episode_{episode_idx}'
        else:
            dataset_name = f'episode_{episode_idx}'

        # Load the data
        qpos, qvel, effort, action, base_action, image_dict, audio_data, audio_sampling_rate, audio_channels = load_hdf5(dataset_dir, dataset_name)
        # qpos, qvel, effort, action, base_action, image_dict, audio_data, audio_sampling_rate, audio_channels = load_hdf5_half(dataset_dir, dataset_name)
        print(f'HDF5 for {dataset_name} loaded!!')

        # Save the video
        save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'), audio_data = audio_data, audio_sampling_rate=audio_sampling_rate)
        # save_only_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))

        # Visualize joint data
        visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))

        # Other visualizations (uncomment if needed)
        # visualize_single(effort, 'effort', plot_path=os.path.join(dataset_dir, dataset_name + '_effort.png'))
        # visualize_single(action - qpos, 'tracking_error', plot_path=os.path.join(dataset_dir, dataset_name + '_error.png'))
        visualize_base(base_action, plot_path=os.path.join(dataset_dir, dataset_name + '_base_action.png'))

        # TODO: Add timestamp visualization if needed
        # visualize_timestamp(t_list, dataset_path)

def save_only_videos(video, dt, video_path="output.mp4"):
    """
    保存视频到指定路径，不包含音频。

    参数:
        video: 视频帧列表（每帧为 dict）或多摄像头帧组成的 dict。
        dt: 每帧之间的时间间隔（秒）。
        video_path: 保存视频的路径。
    """
    if isinstance(video, list):
        # video 是一个包含每一帧的列表，每一帧是一个摄像头名称到图像的 dict
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        for image_dict in video:
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]]  # BGR to RGB
                images.append(image)
            images = np.concatenate(images, axis=1)  # 横向拼接
            out.write(images)
        out.release()

    elif isinstance(video, dict):
        # video 是一个 dict，key 是摄像头，value 是帧序列
        cam_names = list(video.keys())
        all_cam_videos = [video[cam_name] for cam_name in cam_names]
        all_cam_videos = np.concatenate(all_cam_videos, axis=2)  # width 方向拼接

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # BGR to RGB
            out.write(image)
        out.release()

    print(f"Video saved to: {video_path}")


def save_videos(video, dt, video_path=None, audio_data=None, audio_sampling_rate=48000):
    temp_video_path = "/home/robot/Dataset_and_Checkpoint/temp_video.mp4"  # 临时视频文件路径
    temp_audio_path = "/home/robot/Dataset_and_Checkpoint/temp_audio.wav"  # 临时音频文件路径

    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1 / dt)
        out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]]  # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()

    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = [video[cam_name] for cam_name in cam_names]
        all_cam_videos = np.concatenate(all_cam_videos, axis=2)  # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()

    print(f'Temporary video saved to: {temp_video_path}')

    # **如果提供了音频数据，合成音视频**
    if audio_data is not None:
        # **计算视频和音频的时长**
        video_duration = n_frames / fps
        # print("len(audio_data) is",len(audio_data))
        audio_duration = len(audio_data) / audio_sampling_rate

        # print("audio_data shape:", audio_data.shape)

        # **匹配音频时长**
        if not np.isclose(video_duration, audio_duration, atol=0.01):  # 允许 0.01 秒误差
            print(f"Adjusting audio duration: Video={video_duration:.2f}s, Audio={audio_duration:.2f}s")
            audio_data = librosa.effects.time_stretch(audio_data, rate=video_duration / audio_duration)

        # **保存调整后的音频**
        print("Saving adjusted audio...")
        # sf.write(temp_audio_path, audio_data, audio_sampling_rate)  # 保存音频为 WAV 格式
        audio_data = librosa.resample(audio_data, orig_sr=audio_sampling_rate, target_sr=16000)
        sf.write(temp_audio_path, audio_data, 16000)  # 保存音频为 WAV 格式
        # exit()
        # **使用 FFmpeg 合成音视频**
        final_video_path = video_path if video_path else "output_with_audio.mp4"
        print("Merging video and audio using FFmpeg...")
        command = [
            "ffmpeg", "-y",  # 覆盖已有文件
            "-i", temp_video_path,  # 输入视频
            "-i", temp_audio_path,  # 输入音频
            "-c:v", "copy",  # 复制视频流，不重新编码
            "-c:a", "aac",  # 设置 AAC 音频编码
            "-strict", "experimental",  # 允许实验性 AAC
            final_video_path
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Saved video with audio to: {final_video_path}")
        # input("Press Enter to continue...")

        # **删除临时文件**
        os.remove(temp_video_path)
        os.remove(temp_audio_path)
    else:
        print("No audio provided, saved video without audio.")
        os.rename(temp_video_path, video_path if video_path else "output.mp4")


def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list) # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()

def visualize_single(efforts_list, label, plot_path=None, ylim=None, label_overwrite=None):
    efforts = np.array(efforts_list) # ts, dim
    num_ts, num_dim = efforts.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(efforts[:, dim_idx], label=label)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved effort plot to: {plot_path}')
    plt.close()

def visualize_base(readings, plot_path=None):
    readings = np.array(readings) # ts, dim
    num_ts, num_dim = readings.shape
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = BASE_STATE_NAMES
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(readings[:, dim_idx], label='raw')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(20)/20, mode='same'), label='smoothed_20')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(10)/10, mode='same'), label='smoothed_10')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(5)/5, mode='same'), label='smoothed_5')
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # if ylim:
    #     for dim_idx in range(num_dim):
    #         ax = axs[dim_idx]
    #         ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved effort plot to: {plot_path}')
    plt.close()


def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace('.pkl', '_timestamp.png')
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h*2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10E-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title(f'Camera frame timestamps')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    ax = axs[1]
    ax.plot(np.arange(len(t_float)-1), t_float[:-1] - t_float[1:])
    ax.set_title(f'dt')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved timestamp plot to: {plot_path}')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=str, help='Episode index.', required=False)
    parser.add_argument('--ismirror', action='store_true')
    main(vars(parser.parse_args()))