### Task parameters
import os
if os.getlogin() == 'guningquan':
    DATA_DIR = '/mnt/ssd1/guningquan/Programs_server/act_dataset_checkpoint/dataset'
elif os.getlogin() == 'ubuntu20':
    DATA_DIR = '/home/robot/Dataset_and_Checkpoint/dataset'
else:
    raise ValueError(f"Unknown user: {os.getlogin()}")

TASK_CONFIGS = {

    'alarm_random_pos_dataset': {
        'dataset_dir': DATA_DIR + '/alarm_random_pos',
        'episode_len': 1000,  # 900
        'camera_names': ['cam_high',
                         'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },

    'boxlockdown_dataset': {
        'dataset_dir': DATA_DIR + '/boxlockdown',
        'episode_len':750, # 900
        'camera_names': ['cam_high',
                         'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },


    'stapler_checking_dataset': {
        'dataset_dir': DATA_DIR + '/stapler_checking',
        'episode_len': 600,
        'camera_names': ['cam_high',
                         'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },

    'alarm_random_pos_cnn14_plus_cross': {
        'dataset_dir': DATA_DIR + '/alarm_random_pos',
        'episode_len': 1000,  # 900
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },

    'boxlockdown_cnn14_plus_cross_200audio': {
        'dataset_dir': DATA_DIR + '/boxlockdown',
        'episode_len': 750,  # 900
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },


    'stapler_checking_150audio': {
        'dataset_dir': DATA_DIR + '/stapler_checking',
        'episode_len': 600,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },

    'stapler_checking_150audio_mlp': {
        'dataset_dir': DATA_DIR + '/stapler_checking',
        'episode_len': 600,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },

    'stapler_checking_150audio_no_att': {
        'dataset_dir': DATA_DIR + '/stapler_checking',
        'episode_len': 600,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },

    'test': {
        'dataset_dir': DATA_DIR + '/stapler_simple',
        'episode_len': 600,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },


}

Audio_Lenght_For_Learning = 100   # 100 200 150

### ALOHA fixed constants
DT = 0.02
FPS = 50

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.2  # @gnq -0.6213 -> 0

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
