import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange
import wandb, queue
try:
    import pyaudio
except ImportError:
    pass
import wave
import time, threading
from torchvision import transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from aloha_scripts.constants import FPS
from aloha_scripts.constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, postprocess_base_action # helper functions
from policy import SonicACTPolicy
from visualize_episodes import save_videos
from utils import calculate_weights_exponential, calculate_weights_aloha, calculate_weights_gaussian, apply_weights_to_all_l1, calculate_weighted_action
from detr.models.latent_model import Latent_Model_Transformer
from aloha_scripts.constants import Audio_Lenght_For_Learning

# from sim_env import BOX_POSE
try:
    from aloha_scripts.sleep_plus import sleep_puppet_robots, shut_down_puppet_robots
except ImportError:
    pass

import IPython
e = IPython.embed

def get_auto_index(dataset_dir):
    max_idx = 1000
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'qpos_{i}.npy')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

def main(args):
    set_seed(1)

    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_steps = args['num_steps']
    eval_every = args['eval_every']
    validate_every = args['validate_every']
    save_every = args['save_every']
    resume_ckpt_path = args['resume_ckpt_path']
    # num_rollouts = 16
    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim or task_name == 'all':
        from aloha_scripts.constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 0.99)
    name_filter = task_config.get('name_filter', lambda n: True)

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'vq': args['use_vq'],
                         'vq_class': args['vq_class'],
                         'vq_dim': args['vq_dim'],
                         'action_dim': 16,
                         'no_encoder': args['no_encoder'],
                         }
    elif policy_class == 'SonicACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'vq': args['use_vq'],
                         'vq_class': args['vq_class'],
                         'vq_dim': args['vq_dim'],
                         'action_dim': 16,
                         'no_encoder': args['no_encoder'],
                         }

    else:
        raise NotImplementedError

    actuator_config = {
        'actuator_network_dir': args['actuator_network_dir'],
        'history_len': args['history_len'],
        'future_len': args['future_len'],
        'prediction_len': args['prediction_len'],
    }

    config = {
        'num_steps': num_steps,
        'eval_every': eval_every,
        'validate_every': validate_every,
        'save_every': save_every,
        'ckpt_dir': ckpt_dir,
        'resume_ckpt_path': resume_ckpt_path,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_ensemble': args['temporal_ensemble'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'load_pretrain': args['load_pretrain'],
        'actuator_config': actuator_config,
        'num_selected': args['num_selected'],
    }
    num_rollouts = args['num_rollouts']
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    config_path = os.path.join(ckpt_dir, 'config.pkl')
    expr_name = ckpt_dir.split('/')[-1]
    if not is_eval:
        wandb.init(project="sonicaloha", reinit=True, entity="guningquan", name=expr_name)
        wandb.config.update(config)
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)

    if is_eval:
        ckpt_names = [f'policy_last.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            # print("num_rollouts",num_rollouts)
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True, num_rollouts = num_rollouts)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')

        shut_down_puppet_robots()
        print()
        exit()

    # 加载数据集
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, name_filter, camera_names, batch_size_train, batch_size_val, args['chunk_size'], args['skip_mirrored_data'], config['load_pretrain'], policy_class, stats_dir_l=stats_dir, sample_weights=sample_weights, train_ratio=train_ratio)

    # save dataset stats
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # 训练
    policy = make_policy(policy_class, policy_config).cuda()

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config, policy)
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # 保存最优模型
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    if isinstance(policy, torch.nn.DataParallel):
        torch.save(best_state_dict, ckpt_path)
    else:
        torch.save(best_state_dict, ckpt_path)
    print(f'Best checkpoint saved at step {best_step} with val loss {min_val_loss:.6f}')

    wandb.finish()


def make_policy(policy_class, policy_config):
    if policy_class == 'SonicACT':
        policy = SonicACTPolicy(policy_config)
    else:
        raise NotImplementedError


    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        policy = torch.nn.DataParallel(policy)

    return policy



def make_optimizer(policy_class, policy):
    if policy_class == 'SonicACT':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    if rand_crop_resize:
        # print('rand crop resize is used!')
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)

    return curr_image


def get_image_gray(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')


        if 'gel' in cam_name:

            gray_image = 0.299 * curr_image[0] + 0.587 * curr_image[1] + 0.114 * curr_image[2]

            curr_image = torch.stack([gray_image, gray_image, gray_image], dim=0)

        curr_images.append(curr_image)

    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    if rand_crop_resize:
        # print('rand crop resize is used!')
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)

    return curr_image

def save_audio_chunks(audio_chunk, sample_rate=48000, save_dir="./audio"):
    os.makedirs(save_dir, exist_ok=True)

    # 2️⃣ 计算下一个文件名
    existing_files = [f for f in os.listdir(save_dir) if f.endswith(".wav")]
    existing_indices = sorted([int(f.split(".")[0]) for f in existing_files if f.split(".")[0].isdigit()])
    next_index = existing_indices[-1] + 1 if existing_indices else 0  # 找到下一个编号

    # 3️⃣ 保存为 WAV 文件
    file_path = os.path.join(save_dir, f"{next_index}.wav")

    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)  # 采样率
        wf.writeframes(audio_chunk.tobytes())

    print(f"✅ 音频已保存: {file_path}")
    return file_path

def eval_bc(config, ckpt_name, save_episode=True, num_rollouts=1):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_ensemble = config['temporal_ensemble']

    onscreen_cam = 'angle'
    vq = config['policy_config']['vq']
    actuator_config = config['actuator_config']
    use_actuator_net = actuator_config['actuator_network_dir'] is not None
    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)

    # checkpoint = torch.load(ckpt_path, map_location="cpu")
    # print("\n=== Checkpoint Keys (包含 'fusion_transformer') ===")
    # fusion_keys = [k for k in checkpoint.keys() if "fusion_transformer" in k]
    # for i, key in enumerate(fusion_keys):
    #     print(f"{i + 1}: {key}")
    #     if i >= 9:  # 只打印前10个
    #         break
    #
    # print("\n=== Policy Model Keys (包含 'fusion_transformer') ===")
    # policy_keys = [k for k in policy.state_dict().keys() if "fusion_transformer" in k]
    # for i, key in enumerate(policy_keys):
    #     print(f"{i + 1}: {key}")
    #     if i >= 9:  # 只打印前10个
    #         break
    # exit()

    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    if vq:
        vq_dim = config['policy_config']['vq_dim']
        vq_class = config['policy_config']['vq_class']
        latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
        latent_model_ckpt_path = os.path.join(ckpt_dir, 'latent_model_last.ckpt')
        latent_model.deserialize(torch.load(latent_model_ckpt_path))
        latent_model.eval()
        latent_model.cuda()
        print(f'Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}')
    else:
        print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    # if use_actuator_net:
    #     prediction_len = actuator_config['prediction_len']
    #     future_len = actuator_config['future_len']
    #     history_len = actuator_config['history_len']
    #     actuator_network_dir = actuator_config['actuator_network_dir']

    #     from act.train_actuator_network import ActuatorNetwork
    #     actuator_network = ActuatorNetwork(prediction_len)
    #     actuator_network_path = os.path.join(actuator_network_dir, 'actuator_net_last.ckpt')
    #     loading_status = actuator_network.load_state_dict(torch.load(actuator_network_path))
    #     actuator_network.eval()
    #     actuator_network.cuda()
    #     print(f'Loaded actuator network from: {actuator_network_path}, {loading_status}')

    #     actuator_stats_path  = os.path.join(actuator_network_dir, 'actuator_net_stats.pkl')
    #     with open(actuator_stats_path, 'rb') as f:
    #         actuator_stats = pickle.load(f)
        
    #     actuator_unnorm = lambda x: x * actuator_stats['commanded_speed_std'] + actuator_stats['commanded_speed_std']
    #     actuator_norm = lambda x: (x - actuator_stats['observed_speed_mean']) / actuator_stats['observed_speed_mean']
    #     def collect_base_action(all_actions, norm_episode_all_base_actions):
    #         post_processed_actions = post_process(all_actions.squeeze(0).cpu().numpy())
    #         norm_episode_all_base_actions += actuator_norm(post_processed_actions[:, -2:]).tolist()

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    if policy_class == 'Diffusion' or policy_class == 'DiffusionAudio':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        # env = make_real_env(init_node=True, setup_robots=True, setup_base=True)
        env = make_real_env(init_node=True)
        env_max_reward = 0

    query_frequency = policy_config['num_queries']
    if temporal_ensemble:
        query_frequency = 1
        num_queries = policy_config['num_queries']
    if real_robot:
        BASE_DELAY = 0  # 13 -> 0
        query_frequency -= BASE_DELAY

    Length_multiple = 1.0 # may increase or decrease for real-world tasks 1.0 modify time
    max_timesteps = int(max_timesteps * Length_multiple)

    episode_returns = []
    highest_rewards = []


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


    # # 选择特定相机的音频设备
    # DEVICE_INDEX = device_index
    # audio = pyaudio.PyAudio()
    # stream = audio.open(format=FORMAT, channels=channels,
    #                     rate=audio_sampling_rate, input=True,
    #                     frames_per_buffer=CHUNK, input_device_index=DEVICE_INDEX)
    # audio_queue = queue.Queue()
    # recording = True
    # audio_length = Audio_Lenght_For_Learning
    # DT = 1 / FPS
    # max_audio_length = int(audio_sampling_rate * DT * audio_length)
    # def record_audio():
    #     """后台线程：不断采集音频数据，并存入 `audio_queue`"""
    #     while recording:
    #         audio_block = stream.read(CHUNK, exception_on_overflow=False)
    #         audio_queue.put(audio_block)


    for rollout_id in range(num_rollouts):
        # if real_robot:  # gnq comment
        #     e()   # @gnq



        # 选择特定相机的音频设备
        DEVICE_INDEX = device_index
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=channels,
                            rate=audio_sampling_rate, input=True,
                            frames_per_buffer=CHUNK, input_device_index=DEVICE_INDEX)
        audio_queue = queue.Queue()
        recording = True
        audio_length = Audio_Lenght_For_Learning
        DT = 1 / FPS
        max_audio_length = int(audio_sampling_rate * DT * audio_length)

        def record_audio():
            """后台线程：不断采集音频数据，并存入 `audio_queue`"""
            while recording:
                audio_block = stream.read(CHUNK, exception_on_overflow=False)
                audio_queue.put(audio_block)




        input(f"Press Enter to start the {rollout_id} manipulation...")  # @gnq
        Audio_Chunks_Original = []  # 预分配列表
        rollout_id += 0

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_ensemble:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, 16]).cuda()

        # qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        qpos_history_raw = np.zeros((max_timesteps, state_dim))
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        # if use_actuator_net:
        #     norm_episode_all_base_actions = [actuator_norm(np.zeros(history_len, 2)).tolist()]

        recording_thread = threading.Thread(target=record_audio, daemon=True)
        recording_thread.start()
        print("Audio Recording started...")

        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / FPS
            culmulated_delay = 0
            for t in range(max_timesteps):
                time1 = time.time()
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                time2 = time.time()
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos_history_raw[t] = qpos_numpy
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                # qpos_history[:, t] = qpos
                if t % query_frequency == 0:
                    curr_image = get_image(ts, camera_names, rand_crop_resize=(config['policy_class'] == 'Diffusion' or config['policy_class'] == 'DiffusionAudio'))
                    # curr_image = get_image_gray(ts, camera_names, rand_crop_resize=(config['policy_class'] == 'Diffusion'))  # @gnq
                # print('get image: ', time.time() - time2)
                    while not audio_queue.empty():
                        Audio_Chunks_Original.append(audio_queue.get())
                    Audio_Chunks = np.frombuffer(b"".join(Audio_Chunks_Original), dtype=np.int16)
                    # print("len(Audio_Chunks)", len(Audio_Chunks))
                    if len(Audio_Chunks) > max_audio_length:
                        audio_chunk = Audio_Chunks[-max_audio_length:]
                    elif len(Audio_Chunks) < max_audio_length:
                        pad_length = max_audio_length - len(Audio_Chunks)
                        zero_padding = np.zeros((pad_length,), dtype=Audio_Chunks.dtype)
                        audio_chunk = np.concatenate((zero_padding, Audio_Chunks))
                    else:
                        audio_chunk = Audio_Chunks

                    # save_audio_chunks(audio_chunk, sample_rate=48000) # add
                    audio_chunk = audio_chunk.astype(np.float32) / 32768.0
                    device = curr_image.device
                    audio_chunk = torch.tensor(audio_chunk, device=device)
                    audio_chunk = audio_chunk.unsqueeze(0)
                    # print("curr_image.shape", curr_image.shape)
                    # print("audio_chunk.shape", audio_chunk.shape)
                    # exit()

                if t == 0:
                    # warm up
                    for _ in range(10):
                        policy(qpos, curr_image, audio_chunk, audio_sampling_rate)  # todo gnq
                        # pass
                    print('network warm up done')
                    time1 = time.time()

                ### query policy
                time3 = time.time()
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        if vq:
                            if rollout_id == 0:
                                for _ in range(10):
                                    vq_sample = latent_model.generate(1, temperature=1, x=None)
                                    print(torch.nonzero(vq_sample[0])[:, 1].cpu().numpy())
                            vq_sample = latent_model.generate(1, temperature=1, x=None)
                            all_actions = policy(qpos, curr_image, vq_sample=vq_sample)
                        else:
                            # e()
                            all_actions = policy(qpos, curr_image)
                        # if use_actuator_net:
                        #     collect_base_action(all_actions, norm_episode_all_base_actions)
                        if real_robot and BASE_DELAY == 0:  # gnq added
                            all_actions = torch.cat([all_actions[:, :, :-2], all_actions[:, :, -2:]], dim=2) # @gnq
                        else:
                            all_actions = torch.cat(
                                [all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)
                    if temporal_ensemble:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]


                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)

                    else:
                        raw_action = all_actions[:, t % query_frequency]
                        # if t % query_frequency == query_frequency - 1:
                        #     # zero out base actions to avoid overshooting
                        #     raw_action[0, -2:] = 0
                elif config['policy_class'] == "SonicACT":
                    if t % query_frequency == 0:
                        if vq:
                            if rollout_id == 0:
                                for _ in range(10):
                                    vq_sample = latent_model.generate(1, temperature=1, x=None)
                                    print(torch.nonzero(vq_sample[0])[:, 1].cpu().numpy())
                            vq_sample = latent_model.generate(1, temperature=1, x=None)
                            all_actions = policy(qpos, curr_image, vq_sample=vq_sample)
                        else:
                            # e()
                            # print("here 1")
                            # all_actions = policy(qpos, curr_image)
                            all_actions = policy(qpos, curr_image, audio_chunk, audio_sampling_rate)
                            # print("here 2")
                        # if use_actuator_net:
                        #     collect_base_action(all_actions, norm_episode_all_base_actions)
                        if real_robot and BASE_DELAY == 0:  # gnq added
                            all_actions = torch.cat([all_actions[:, :, :-2], all_actions[:, :, -2:]], dim=2)  # @gnq
                        else:
                            all_actions = torch.cat(
                                [all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)
                    # print("here 3")
                    if temporal_ensemble:

                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]

                        # k = 0.01  # 0.002  0.01
                        # num_selected = config['num_selected']
                        # if len(actions_for_curr_step) < num_selected:
                        #     num_selected = len(actions_for_curr_step)
                        # exp_weights = np.exp(k * np.arange(num_selected))
                        # exp_weights = exp_weights / exp_weights.sum()
                        # exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        # raw_action = (actions_for_curr_step[-num_selected:] * exp_weights).sum(dim=0, keepdim=True)

                        k =   -0.01 # -0.01 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)

                        # print("raw_action", raw_action)
                        # input("enter to continue")
                    else:
                        raw_action = all_actions[:, t % query_frequency]

                else:
                    raise NotImplementedError
                # print('query policy: ', time.time() - time3)

                ### post-process actions
                time4 = time.time()
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action[:-2]


                # print("target_qpos is:", target_qpos)

                # if use_actuator_net:
                #     assert(not temporal_ensemble)
                #     if t % prediction_len == 0:
                #         offset_start_ts = t + history_len
                #         actuator_net_in = np.array(norm_episode_all_base_actions[offset_start_ts - history_len: offset_start_ts + future_len])
                #         actuator_net_in = torch.from_numpy(actuator_net_in).float().unsqueeze(dim=0).cuda()
                #         pred = actuator_network(actuator_net_in)
                #         base_action_chunk = actuator_unnorm(pred.detach().cpu().numpy()[0])
                #     base_action = base_action_chunk[t % prediction_len]
                # else:
                base_action = action[-2:]
                # base_action = calibrate_linear_vel(base_action, c=0.19)
                # base_action = postprocess_base_action(base_action)
                # print('post process: ', time.time() - time4)

                ### step the environment
                time5 = time.time()
                if real_robot:
                    # ts = env.step(target_qpos, base_action)
                    ts = env.step(target_qpos)  # gnq comment
                else:
                    ts = env.step(target_qpos)
                # print('step env: ', time.time() - time5)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
                duration = time.time() - time1
                sleep_time = max(0, DT - duration)
                # print(sleep_time)
                time.sleep(sleep_time)
                # time.sleep(max(0, DT - duration - culmulated_delay))
                if duration >= DT:
                    culmulated_delay += (duration - DT)
                    # print(f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {DT} s, culmulated delay: {culmulated_delay:.3f} s')
                # else:
                #     culmulated_delay = max(0, culmulated_delay - (DT - duration))
                # if t % query_frequency == 0:
                #     print("t ",t)
                #     input("enter to continue")
            recording = False
            recording_thread.join()
            stream.stop_stream()
            stream.close()
            audio.terminate()

            print(f'Avg fps {max_timesteps / (time.time() - time0)}')
            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            sleep_puppet_robots()  # @gnq
            # save qpos_history_raw
            log_id = get_auto_index(ckpt_dir)
            np.save(os.path.join(ckpt_dir, f'qpos_{log_id}.npy'), qpos_history_raw)
            plt.figure(figsize=(10, 20))
            # plot qpos_history_raw for each qpos dim using subplots
            for i in range(state_dim):
                plt.subplot(state_dim, 1, i+1)
                plt.plot(qpos_history_raw[:, i])
                # remove x axis
                if i != state_dim - 1:
                    plt.xticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, f'qpos_{log_id}.png'))
            plt.close()


        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        # if save_episode:  # gnq
        #     save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))
        if save_episode:
            video_path = os.path.join(ckpt_dir, f'video{rollout_id}.mp4')
            t = threading.Thread(target=save_videos, args=(image_list, DT, video_path))
            t.start()
            t.join()

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad, audio_data, audio_sampling_rate = data
    # image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad= image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    audio_data = audio_data.cuda()
    return policy(qpos_data, image_data, audio_data, audio_sampling_rate, action_data, is_pad) # TODO remove None
    # return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config, policy):
    num_steps = config['num_steps']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    eval_every = config['eval_every']
    validate_every = config['validate_every']
    save_every = config['save_every']

    set_seed(seed)


    if config['load_pretrain']:
        pretrain_path = os.path.join('/home/your/pretrain/path',
                                     'policy_step_50000_seed_0.ckpt')
        try:
            if isinstance(policy, torch.nn.DataParallel):
                loading_status = policy.module.deserialize(torch.load(pretrain_path))
            else:
                loading_status = policy.deserialize(torch.load(pretrain_path))
            print(f'Loaded pre-trained weights from {pretrain_path}, Status: {loading_status}')
        except Exception as e:
            print(f"Error loading pre-trained weights from {pretrain_path}: {e}")
            raise e

    if config['resume_ckpt_path'] is not None:
        try:
            if isinstance(policy, torch.nn.DataParallel):
                loading_status = policy.module.deserialize(torch.load(config['resume_ckpt_path']))
            else:
                loading_status = policy.deserialize(torch.load(config['resume_ckpt_path']))
            print(f'Resumed policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
        except Exception as e:
            print(f"Error resuming policy from {config['resume_ckpt_path']}: {e}")
            raise e

    optimizer = make_optimizer(config['policy_class'],
                               policy.module if isinstance(policy, torch.nn.DataParallel) else policy)
    min_val_loss = np.inf
    best_ckpt_info = None

    train_dataloader = repeater(train_dataloader)
    for step in tqdm(range(num_steps + 1)):

        if step % validate_every == 0:
            print('Validating')

            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 50:
                        break

                validation_summary = compute_dict_mean(validation_dicts)

                epoch_val_loss = validation_summary['loss']
                if isinstance(epoch_val_loss, torch.Tensor):
                    epoch_val_loss = epoch_val_loss.mean().item()


                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.module.serialize() if isinstance(policy, torch.nn.DataParallel) else policy.serialize()))


            for k in list(validation_summary.keys()):
                if isinstance(validation_summary[k], torch.Tensor):
                    validation_summary[k] = validation_summary[k].mean().item()  # 将多 GPU 输出的张量取均值
                validation_summary[f'val_{k}'] = validation_summary.pop(k)

            wandb.log(validation_summary, step=step)
            print(f'Validation loss: {epoch_val_loss:.5f}')

        # 训练过程
        policy.train()
        optimizer.zero_grad()
        data = next(train_dataloader)
        forward_dict = forward_pass(data, policy)
        loss = forward_dict['loss']


        if isinstance(loss, torch.Tensor):
            loss = loss.mean()

        loss.backward()
        optimizer.step()
        wandb.log(forward_dict, step=step)


        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            if isinstance(policy, torch.nn.DataParallel):
                torch.save(policy.module.serialize(), ckpt_path)
            else:
                torch.save(policy.serialize(), ckpt_path)


    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    if isinstance(policy, torch.nn.DataParallel):
        torch.save(policy.module.serialize(), ckpt_path)
    else:
        torch.save(policy.serialize(), ckpt_path)

    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_step_{best_step}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished: Seed {seed}, val loss {min_val_loss:.6f} at step {best_step}')

    return best_ckpt_info

def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_steps', action='store', type=int, help='num_steps', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--eval_every', action='store', type=int, default=500, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=500, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_ensemble', action='store_true')
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class')
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim')
    parser.add_argument('--no_encoder', action='store_true')

    parser.add_argument('--num_selected', action='store', type=int, default=50, help='the action selected for temporal agg', required=False)
    parser.add_argument('--num_rollouts', action='store', type=int, default=1, help='num_rollouts', required=False)

    main(vars(parser.parse_args()))
