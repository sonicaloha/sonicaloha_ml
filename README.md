# SonicAloha_ML
#### Main Project Website: https://sonicaloha.github.io/

This repository is used for robot teleoperation, dataset collection, and imitation learning algorithms. It can be placed anywhere on your computer. 

## 📂 Repo Structure
- ``aloha_scripts`` Folders for controlling the robot, camera and microphone. You can use it to test teleoperation, datasets, put the robot to sleep, and visualize the data. You can define your task in ``aloha_scripts/constants.py`` 
- ``detr`` Model definitions
- ``imitate_episodes_multi_gpu.py`` Train and Evaluate policy with multiple GPUs
- ``policy.py`` An adaptor for policy
- ``utils.py`` Utils such as data loading and helper functions

---


## 🏗️ Quick Start Guide

### 🖥️ Software Selection – OS Compatibility

This project has been tested and confirmed to work with the following configuration:  

- ✅ **Ubuntu 20.04 + ROS 1 Noetic** (Fully tested and verified)  

Other configurations may work as well, but they have not been tested yet. If you successfully run this project on a different setup, feel free to contribute by sharing your experience! 🚀

## 🔧 Hardware Setting
1. Install the 4 robots and 4 cameras according to the original [ALOHA](https://github.com/tonyzhaozh/aloha). 
2. Plug in your USB microphone and mount it beside the top-view camera.
3. ❗ Camera Focus Configuration (Not described in ALOHA):

   The cameras in the ALOHA series are set to **fixed focus** in ROS launch. 
The focus value is configured through `aloha.launch` in `aloha/launch`:  
    ```xml
    <param name="focus" value="40"/>
    ```
   It is necessary to determine the appropriate focus value for each camera; otherwise, the camera image may appear blurry during manipulation.

    The recommended procedure to find a suitable focus value is:
    - To check the available video devices, run the following command: 
   ``` bash
     ls /dev/video*
   ```
    - Use the following command to open the camera and adjust the focus: 
   ``` bash
     guvcview -d /dev/video0
   ```
    - Test and note the appropriate focus value for each camera ;

3. `❗` Disable Auto Focus:

   You must disable the continuous autofocus by setting the focus_automatic_continuous control parameter as follows:
    ```bash
      v4l2-ctl -d /dev/video0 --set-ctrl focus_automatic_continuous=0
    ```
    For other cameras please modifiy `/dev/video0` to  `/dev/video2`, `/dev/video4`, etc.
    The way to check whether we have set the camera correctly is to run `roslaunch aloha aloha.launch` and ensure that no warning like this appears:
    ```bash
    Error setting controls: Permission denied
    VIDIOC_S_EXT_CTRLS: failed: Permission denied
    ```
   Note: You will need to reapply the focus_automatic_continuous=0 setting whenever you reboot the computer or unplug and replug the cameras.

## 🛠️ Installation
```sh    
    git clone https://github.com/sonicaloha/sonicaloha_ml.git
    cd sonicaloha_ml
    conda create -n sonicaloha python=3.8.10
    conda activate sonicaloha
    pip install -r requirements.txt
    cd detr && pip install -e .
```
## 📑 Dataset Collection
1. 🤖 **TactileAloha robot system launch:**
We assume you have installed your robot system according to [ALOHA](https://github.com/tonyzhaozh/aloha). This step launches the four robot arms, four cameras.
    ``` ROS
    # ROS terminal
    conda deactivate
    source /opt/ros/noetic/setup.sh && source ~/interbotix_ws/devel/setup.sh
    roslaunch aloha aloha.launch
    ```

2. 📝 **Define the type of robotic manipulation dataset:**  
Including the task name, `dataset_dir`, length of each episode, and the cameras used.
You can set this information in `TASK_CONFIGS` of `aloha_scripts/constants.py`. An example is as follows:
    ```python
    'alarm_shutting': {
        'dataset_dir': DATA_DIR + '/saved_folder_name',
        'episode_len': 900, # This value may be modified according to the length of your task.
        'camera_names': ['cam_high', 
                           'cam_left_wrist', 
                           'cam_right_wrist', 
                           'cam_low']}
    ```
3. 🚀 **Star to teleoperation to task manipulation**: 
    ```
   cd sonicaloha_ml
   source ~/interbotix_ws/devel/setup.sh
    python aloha_scripts/record_episodes_compress_audio.py \
    --task_name Task_name \
   --start_idx 0 --end_idx 50
    ```
   After each episode collection, you can enter `c` to save this episode and continue, or enter `r` to recollect this episode. If you want to quit this process, you can enter `q`. We use [foot pedals](https://www.amazon.co.jp/-/en/gp/product/B07FRMY4XB/ref=ox_sc_act_title_1?smid=A35GGB9A6044W2&psc=1) to assist with this confirmation, which can facilitate this work.
4. 📊 **Data Visualization and Listening** :
    ```
    python aloha_scripts/visualize_episodes_audio.py --dataset_dir <data save dir> --episode_idx 0
    ```
    If you want to visualize multiple data points, you can input the `--episode_idx` parameter like this: `--episode_idx 3, 5, 8` or `--episode_idx 5-19`.
5. 🔄 **Robot shut down or sleep**:
    ```
    python aloha_scripts/sleep_plus.py --shut_down       # All robots will move to zero position and turn off the torque.
   python aloha_scripts/sleep_plus.py --shut_down_puppet   # Only the puppet robots will move to zero position and turn off the torque.
   python aloha_scripts/sleep_plus.py --sleep      # All robots will move to zero position but don't turn off the torque.
    ```
## 🧠 **Policy Training**  
   1. ✅ Set up your training configuration in ``aloha_scripts/constants.py``:

      ```python
      'alarm_cnn14_plus_cross_100audio': {
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
   
   
       'stapler_checking_cnn14_plus_cross_150audio': {
           'dataset_dir': DATA_DIR + '/stapler_checking',
           'episode_len': 600,
           'camera_names': ['cam_high',
                            # 'cam_low',
                            'cam_left_wrist',
                            'cam_right_wrist',
                            ]
       },
      ```
3. Set the audio lenght  in ``aloha_scripts/constants.py``:
```Audio_Lenght_For_Learning = 100   # 100 200 150```
2. 🚀 Train your policy:
   ``` sh
   export CUDA_VISIBLE_DEVICES= 0, 1
   python imitate_episodes_multi_gpu.py  \
   --task_name alarm_cnn14_plus_cross_100audio \
   --ckpt_dir  <data save dir>  \
   --policy_class SonicAloha \
   --kl_weight 10 --chunk_size 100 \
   --hidden_dim 512 --batch_size 16 \
   --dim_feedforward 3200 --lr 1e-5 --seed 0 \
   --num_steps 100000 --eval_every 2000 \
   --validate_every 2000 --save_every 2000
   ```

## 📡 **Policy Deployment**
1. You may set the `Length_multiple` value to control the maximum manipulation timesteps, and the `k` value for temporal ensemble in `imitate_episodes_multi_gpu.py`.

2. Run your trained policy:
   ```sh
   export CUDA_VISIBLE_DEVICES= 0
   python imitate_episodes_multi_gpu.py  \
   --task_name alarm_cnn14_plus_cross_100audio \
   --ckpt_dir  <data save dir>  \
   --policy_class SonicAloha \
   --kl_weight 10 --chunk_size 100 \
   --hidden_dim 512 --batch_size 16 \
   --dim_feedforward 3200 --lr 1e-5 --seed 0 \
   --num_steps 100000 --eval_every 2000 \
   --validate_every 2000 --save_every 2000 \
   --temporal_ensemble --eval --num_rollouts 20
   ```

## 🙏 Acknowledgements
   This project codebase is built based on [ALOHA](https://github.com/tonyzhaozh/aloha) and [ACT](https://github.com/tonyzhaozh/act).