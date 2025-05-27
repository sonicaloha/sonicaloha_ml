import pyaudio
import wave

# 音频参数
FORMAT = pyaudio.paInt16  # 16-bit 音频
CHANNELS = 1  # 单声道
RATE = 48000  # 采样率 (Hz)
CHUNK = 512  # 每个缓冲区大小
RECORD_SECONDS = 5  # 录音时长
OUTPUT_FILENAME = "selected_audio.wav"  # 输出文件名


# TARGET_DEVICE_NAME = "C922 Pro Stream Webcam: USB Audio (hw:5,0)"
TARGET_DEVICE_NAME = "USB PnP Audio Device"

# 初始化 PyAudio
audio = pyaudio.PyAudio()

# 遍历所有设备，查找匹配的设备名称
device_index = None
for i in range(audio.get_device_count()):
    device_info = audio.get_device_info_by_index(i)
    if TARGET_DEVICE_NAME in device_info["name"]:
        device_index = i
        print(f"找到设备: {device_info['name']}, 索引: {device_index}")
        break  # 找到后立即退出循环

audio.terminate()


# 选择特定相机的音频设备
DEVICE_INDEX = device_index

audio = pyaudio.PyAudio()

# 打开流
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK, input_device_index=DEVICE_INDEX)

print("开始录音...")
frames = []

for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("录音完成！")

# 停止并关闭流
stream.stop_stream()
stream.close()
audio.terminate()

# 保存音频
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"音频已保存到 {OUTPUT_FILENAME}")

