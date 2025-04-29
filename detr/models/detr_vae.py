# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from .backbone import build_backbone, build_sonic_backbone_mlp, build_sonic_backbone_vggish
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import torchaudio.transforms as T
from torch_vggish_yamnet import yamnet
from torch_vggish_yamnet.input_proc import *
from torch.nn import TransformerEncoder as TE
from torch.nn import TransformerEncoderLayer as TEL
import copy

import IPython
e = IPython.embed

sonic_decoder = "vgg" # "vgg" "mlp"
fusion = "plus_cross" #  False, "plus_cross"

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAEAUDIO(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, vq, vq_class, vq_dim, action_dim):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        if fusion :
            self.transformer_audio = copy.deepcopy(transformer)
        self.encoder = encoder
        self.vq, self.vq_class, self.vq_dim = vq, vq_class, vq_dim
        self.state_dim, self.action_dim = state_dim, action_dim
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if fusion :
            self.query_embed_audio = nn.Embedding(num_queries, hidden_dim)

        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            if fusion:
                self.input_proj_robot_state_audio = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None
            if fusion:
                self.input_proj_robot_state_audio = nn.Linear(state_dim, hidden_dim)

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding

        print(f'Use VQ: {self.vq}, {self.vq_class}, {self.vq_dim}')
        if self.vq:
            self.latent_proj = nn.Linear(hidden_dim, self.vq_class * self.vq_dim)
        else:
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # self.audio_feature_mlp = nn.Sequential()
        # self.audio_pos_mlp = nn.Sequential()

        # decoder extra parameters
        if self.vq:
            self.latent_out_proj = nn.Linear(self.vq_class * self.vq_dim, hidden_dim)
        else:
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent
        if fusion :
            self.additional_pos_embed_audio = nn.Embedding(2, hidden_dim)
        if fusion == "weight":
            pass
        elif fusion == "MLP":
            self.fusion_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
        elif fusion == "cross":
            self.cross_attention = CrossAttention(hidden_dim=hidden_dim, num_heads=8)

        elif fusion == "transformer":  # gnq
            self.fusion_transformer = FusionTransformer(hidden_dim=hidden_dim, num_layers=2, num_heads=8, dropout=0.1)

        elif fusion == "plus_cross":
            # self.plus_cross_attention = RobotCrossPolicy(num_heads=8)
            self.plus_cross_attention = BidirectionalCrossAttention (embed_dim = hidden_dim, num_heads=8)

        elif fusion == "single_cross":
            self.single_cross_attention = CrossAttention_plus(embed_dim=hidden_dim, num_heads=8)

    def encode(self, qpos, actions=None, is_pad=None, vq_sample=None):
        bs, _ = qpos.shape
        if self.encoder is None:
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
            probs = binaries = mu = logvar = None
        else:
            # cvae encoder
            is_training = actions is not None # train or val
            ### Obtain latent z from action sequence
            if is_training:
                # project action sequence to embedding dim, and concat with a CLS token
                action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
                qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
                qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
                cls_embed = self.cls_embed.weight # (1, hidden_dim)
                cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
                encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
                # do not mask cls token
                cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
                is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
                # obtain position embedding
                pos_embed = self.pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
                # query model
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
                # encoder_output = self.encoder(encoder_input)  # gnq
                # encoder_output = self.encoder(encoder_input,pos_embed=pos_embed, mask=is_pad, query_embed = self.query_embed)  # gnq
                encoder_output = encoder_output[0] # take cls output only
                latent_info = self.latent_proj(encoder_output)
                
                if self.vq:
                    logits = latent_info.reshape([*latent_info.shape[:-1], self.vq_class, self.vq_dim])
                    probs = torch.softmax(logits, dim=-1)
                    binaries = F.one_hot(torch.multinomial(probs.view(-1, self.vq_dim), 1).squeeze(-1), self.vq_dim).view(-1, self.vq_class, self.vq_dim).float()
                    binaries_flat = binaries.view(-1, self.vq_class * self.vq_dim)
                    probs_flat = probs.view(-1, self.vq_class * self.vq_dim)
                    straigt_through = binaries_flat - probs_flat.detach() + probs_flat
                    latent_input = self.latent_out_proj(straigt_through)
                    mu = logvar = None
                else:
                    probs = binaries = None
                    mu = latent_info[:, :self.latent_dim]
                    logvar = latent_info[:, self.latent_dim:]
                    latent_sample = reparametrize(mu, logvar)
                    latent_input = self.latent_out_proj(latent_sample)


                    # print("encoder_output.shape", encoder_output.shape)
                    # print("encoder_output (first 10 elements):", encoder_output.flatten()[:10].tolist())
                    #
                    # print("latent_info.shape", latent_info.shape)
                    # print("latent_info (first 10 elements):", latent_info.flatten()[:10].tolist())
                    #
                    # print("latent_sample.shape", latent_sample.shape)
                    # print("latent_sample (first 10 elements):", latent_sample.flatten()[:10].tolist())
                    #
                    # print("latent_input.shape", latent_input.shape)
                    # print("latent_input (first 10 elements):", latent_input.flatten()[:10].tolist())
                    #
                    # print("mu.shape", mu.shape)
                    # print("mu (first 10 elements):", mu.flatten()[:10].tolist())
                    #
                    # print("logvar.shape", logvar.shape)
                    # print("logvar (first 10 elements):", logvar.flatten()[:10].tolist())
                    #
                    # exit()

            else:
                mu = logvar = binaries = probs = None
                if self.vq:
                    latent_input = self.latent_out_proj(vq_sample.view(-1, self.vq_class * self.vq_dim))
                else:
                    latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                    latent_input = self.latent_out_proj(latent_sample)


        return latent_input, probs, binaries, mu, logvar

    def forward(self, qpos, image, audio, audio_sampling_rate, env_state, actions=None, is_pad=None, vq_sample=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """

        latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)
        # cvae decoder
        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            all_audio_features = []
            all_audio_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                # print("image",image.shape)
                features, pos = self.backbones[cam_id](image[:, cam_id])
                features = features[0] # take the last layer feature
                pos = pos[0]
                # print("the image size is",image[:, cam_id].shape)
                # print("camera features is",features.shape)
                # print("camera features project is",self.input_proj(features).shape)
                # print("camera pos is", pos.shape)

                # if 'gel' in cam_name:
                #     print(cam_name)
                #     visualize_features(features)
                # visualize_features_resized(features)
                # print('cam_name',cam_name)
                # print('features.shape',features.shape)
                # print("self.input_proj(features).shape", self.input_proj(features).shape)
                # print("pos.shape", pos.shape)
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)


            # sonic_decoder = "resnet38" # "vgg"  "resnet" "resnet38" "yamnet"
            # n_fft = 1024  # new add
            if sonic_decoder == "vgg" :
                device = audio.device
                try:
                    audio_sampling_rate = audio_sampling_rate[0]  # 尝试执行
                except Exception as e:
                    pass  # 如果报错，就跳过

                target_sample_rate = 16000
                resample_transform = T.Resample(orig_freq=audio_sampling_rate, new_freq=target_sample_rate).to(device)
                audio = resample_transform(audio)
                audio_sampling_rate = target_sample_rate

                n_mels = 80  # 80  96
                win_length = int(0.025 * audio_sampling_rate)
                # hop_length = int(0.010 * audio_sampling_rate)
                hop_length = int(0.010 * audio_sampling_rate)
                # n_fft = max(2 ** (win_length - 1).bit_length(), win_length)
                # n_fft = 1200  # 2048
                n_fft = 1024
                mel_transform = T.MelSpectrogram(
                    sample_rate=audio_sampling_rate,
                    n_fft=n_fft,
                    n_mels=n_mels,
                    win_length=win_length,
                    hop_length=hop_length,
                    center=False
                ).to(device)  # ✅ 确保在和 audio 相同的 GPU 上
                # print("audio shape is",audio.shape)
                mel_spec = mel_transform(audio)  # 形状: [batch, n_mels, time]
                # print("mel_spec shape is", mel_spec.shape)
                mel_to_db = T.AmplitudeToDB(stype="power", top_db=80).to(device)
                mel_spec_db = mel_to_db(mel_spec)  # 形状: [batch, n_mels, time]

                mel_spec_db = mel_spec_db.unsqueeze(1)
                # print("mel_spec_db shape is", mel_spec_db.shape)
                audio_features, audio_pos = self.backbones[-1](mel_spec_db)
                audio_features = audio_features[0]
                audio_pos = audio_pos[0]
                # print("audio_features befor pro shape is",audio_features.shape)
                audio_features = self.input_proj(audio_features)
                # exit()

            elif sonic_decoder == "mlp" :
                device = audio.device
                try:
                    audio_sampling_rate = audio_sampling_rate[0]  # 尝试执行
                except Exception as e:
                    pass  # 如果报错，就跳过

                target_sample_rate = 16000
                resample_transform = T.Resample(orig_freq=audio_sampling_rate, new_freq=target_sample_rate).to(device)
                audio = resample_transform(audio)
                audio_sampling_rate = target_sample_rate

                n_mels = 80  # 80  96
                win_length = int(0.025 * audio_sampling_rate)
                # hop_length = int(0.010 * audio_sampling_rate)
                hop_length = int(0.010 * audio_sampling_rate)
                # n_fft = max(2 ** (win_length - 1).bit_length(), win_length)
                # n_fft = 1200  # 2048
                n_fft = 1024
                mel_transform = T.MelSpectrogram(
                    sample_rate=audio_sampling_rate,
                    n_fft=n_fft,
                    n_mels=n_mels,
                    win_length=win_length,
                    hop_length=hop_length,
                    center=False
                ).to(device)  # ✅ 确保在和 audio 相同的 GPU 上
                # print("audio shape is",audio.shape)
                mel_spec = mel_transform(audio)  # 形状: [batch, n_mels, time]
                # print("mel_spec shape is", mel_spec.shape)
                mel_to_db = T.AmplitudeToDB(stype="power", top_db=80).to(device)
                mel_spec_db = mel_to_db(mel_spec)  # 形状: [batch, n_mels, time]

                mel_spec_db = mel_spec_db.unsqueeze(1)
                # print("mel_spec_db shape is", mel_spec_db.shape)
                audio_features, audio_pos = self.backbones[-1](mel_spec_db)
                audio_features = audio_features[0]
                audio_pos = audio_pos[0]
                # print("audio_features befor pro shape is",audio_features.shape)
                audio_features = self.input_proj(audio_features)
                # exit()


            if fusion == False:
                # target_size = (15, 20)  # may need to modify
                target_size = all_cam_features[-1].shape[-2:]
                # print("the audio_features shape is ", audio_features.shape)
                # print("the all_cam_features shape is",all_cam_features[-1].shape)
                # exit()

                audio_features = F.interpolate(audio_features, size=target_size, mode="bilinear", align_corners=False) # modify
                audio_pos = F.interpolate(audio_pos, size=target_size, mode="bilinear", align_corners=False)

                all_cam_features.append(audio_features)
                all_cam_pos.append(audio_pos)

                proprio_input = self.input_proj_robot_state(qpos)

                src = torch.cat(all_cam_features, axis=3)
                pos = torch.cat(all_cam_pos, axis=3)
                hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input,
                                      self.additional_pos_embed.weight)[0]

            elif fusion in {"self", "plus_cross", "self_plus_cross", "single_cross"}:
                if fusion == "plus_cross":
                    # target_size = (15, 20)  # may need to modify
                    target_size = all_cam_features[-1].shape[-2:]

                    audio_features = F.interpolate(audio_features, size=target_size, mode="bilinear", align_corners=False)
                    audio_pos = F.interpolate(audio_pos, size=target_size, mode="bilinear", align_corners=False)

                    all_audio_features.append(audio_features)
                    all_cam_pos.append(audio_pos)

                    proprio_input = self.input_proj_robot_state(qpos)

                    src = torch.cat(all_cam_features, axis=3)
                    src_audio = torch.cat(all_audio_features, dim=3)

                    pos = torch.cat(all_cam_pos, axis=3)

                    # print("the src shape is {}".format(src.shape))
                    # print("the src_audio shape is {}".format(src_audio.shape))
                    # print("the pos shape is {}".format(pos.shape))
                    # exit()
                    src, src_audio = self.plus_cross_attention(src, src_audio)
                    # print("the src shape is {}".format(src.shape))
                    # print("the src_audio shape is {}".format(src_audio.shape))
                    src = torch.cat((src, src_audio), dim=-1)
                    # exit()

                    hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input,
                                          self.additional_pos_embed.weight)[0]

                elif fusion == "single_cross":
                    # target_size = (15, 20)  # may need to modify
                    target_size = all_cam_features[-1].shape[-2:]
                    # print("the all_cam_features shape is", all_cam_features[-1].shape[-2:])
                    # exit()
                    audio_features = F.interpolate(audio_features, size=target_size, mode="bilinear",align_corners=False)
                    # print("the shape of audio_features is {}".format(audio_features.shape))
                    # exit()
                    all_audio_features.append(audio_features)
                    proprio_input = self.input_proj_robot_state(qpos)
                    src = torch.cat(all_cam_features, axis=3)
                    src_audio = torch.cat(all_audio_features, dim=3)
                    pos = torch.cat(all_cam_pos, axis=3)
                    src = self.single_cross_attention(src, src_audio)
                    hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input,
                                          self.additional_pos_embed.weight)[0]

        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]


        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar], probs, binaries

class CrossAttention_plus(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention_plus, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.norm = nn.LayerNorm(embed_dim)  # LayerNorm 提高稳定性

    def forward(self, query_feat, key_value_feat):
        """
        query_feat: (B, C, H_q, W_q) -> 查询特征 (如图片或音频)
        key_value_feat: (B, C, H_k, W_k) -> 作为 Key/Value 的特征 (如音频或图片)
        """
        B, C, H_q, W_q = query_feat.shape
        B, C, H_k, W_k = key_value_feat.shape

        # print("query_feat.shape",query_feat.shape)
        # print("key_value_feat.shape",key_value_feat.shape)

        # 变换形状: (B, C, H, W) → (H*W, B, C)
        query_feat = query_feat.view(B, C, -1).permute(2, 0, 1)  # (H_q * W_q, B, C)
        key_value_feat = key_value_feat.view(B, C, -1).permute(2, 0, 1)  # (H_k * W_k, B, C)

        # 计算交叉注意力
        attn_output, _ = self.multihead_attn(query=query_feat, key=key_value_feat, value=key_value_feat)

        # print("attn_output before.shape", attn_output.shape)
        # 残差连接 + LayerNorm
        attn_output = self.norm(attn_output + query_feat)
        # print("attn_output after.shape", attn_output.shape)
        # 变回 (B, C, H_q, W_q)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H_q, W_q)

        # print("attn_output_resize.shape", attn_output.shape)
        # exit()
        return attn_output


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(BidirectionalCrossAttention, self).__init__()
        self.cross_attn_img_to_audio = CrossAttention_plus(embed_dim, num_heads)  # 图片 → 音频
        self.cross_attn_audio_to_img = CrossAttention_plus(embed_dim, num_heads)  # 音频 → 图片

    def forward(self, img_feat, audio_feat):
        """
        img_feat: (B, C, H_img, W_img) -> 图片特征
        audio_feat: (B, C, H_audio, W_audio) -> 音频特征
        """
        # 1. 图片融合音频
        img_feat_updated = self.cross_attn_img_to_audio(img_feat, audio_feat)

        # 2. 音频融合图片
        audio_feat_updated = self.cross_attn_audio_to_img(audio_feat, img_feat)

        return img_feat_updated, audio_feat_updated


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return self.norm(query + self.dropout(attn_output))


class FusionTransformer(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        # Transformer Encoder Layer
        encoder_layer = TEL(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = TE(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        self.linear_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, hs, audio_hs):

        assert hs.shape[1] == audio_hs.shape[1]

        fusion_input = torch.cat([hs, audio_hs], dim=-1)  # (batch, seq_len, hidden_dim * 2)

        fusion_input = self.linear_proj(fusion_input)  # (batch, seq_len, hidden_dim)

        fusion_output = self.transformer_encoder(fusion_input.permute(1, 0, 2))  # (seq_len, batch, hidden_dim)
        fusion_output = self.norm(fusion_output.permute(1, 0, 2))  # (batch, seq_len, hidden_dim)

        return fusion_output


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, vq, vq_class, vq_dim,
                 action_dim):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.vq, self.vq_class, self.vq_dim = vq, vq_class, vq_dim
        self.state_dim, self.action_dim = state_dim, action_dim
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)  # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding

        print(f'Use VQ: {self.vq}, {self.vq_class}, {self.vq_dim}')
        if self.vq:
            self.latent_proj = nn.Linear(hidden_dim, self.vq_class * self.vq_dim)
        else:
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)  # project hidden state to latent std, var
        self.register_buffer('pos_table',
                             get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))  # [CLS], qpos, a_seq

        # decoder extra parameters
        if self.vq:
            self.latent_out_proj = nn.Linear(self.vq_class * self.vq_dim, hidden_dim)
        else:
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)  # learned position embedding for proprio and latent

    def encode(self, qpos, actions=None, is_pad=None, vq_sample=None):
        bs, _ = qpos.shape
        if self.encoder is None:
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
            probs = binaries = mu = logvar = None
        else:
            # cvae encoder
            is_training = actions is not None  # train or val
            ### Obtain latent z from action sequence
            if is_training:
                # project action sequence to embedding dim, and concat with a CLS token
                action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
                qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
                qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
                cls_embed = self.cls_embed.weight  # (1, hidden_dim)
                cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)  # (bs, seq+1, hidden_dim)
                encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
                # do not mask cls token
                cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)  # False: not a padding
                is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
                # obtain position embedding
                pos_embed = self.pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
                # query model
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
                # encoder_output = self.encoder(encoder_input)  # gnq
                # encoder_output = self.encoder(encoder_input,pos_embed=pos_embed, mask=is_pad, query_embed = self.query_embed)  # gnq
                encoder_output = encoder_output[0]  # take cls output only
                latent_info = self.latent_proj(encoder_output)

                if self.vq:
                    logits = latent_info.reshape([*latent_info.shape[:-1], self.vq_class, self.vq_dim])
                    probs = torch.softmax(logits, dim=-1)
                    binaries = F.one_hot(torch.multinomial(probs.view(-1, self.vq_dim), 1).squeeze(-1),
                                         self.vq_dim).view(-1, self.vq_class, self.vq_dim).float()
                    binaries_flat = binaries.view(-1, self.vq_class * self.vq_dim)
                    probs_flat = probs.view(-1, self.vq_class * self.vq_dim)
                    straigt_through = binaries_flat - probs_flat.detach() + probs_flat
                    latent_input = self.latent_out_proj(straigt_through)
                    mu = logvar = None
                else:
                    probs = binaries = None
                    mu = latent_info[:, :self.latent_dim]
                    logvar = latent_info[:, self.latent_dim:]
                    latent_sample = reparametrize(mu, logvar)
                    latent_input = self.latent_out_proj(latent_sample)

            else:
                mu = logvar = binaries = probs = None
                if self.vq:
                    latent_input = self.latent_out_proj(vq_sample.view(-1, self.vq_class * self.vq_dim))
                else:
                    latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                    latent_input = self.latent_out_proj(latent_sample)

        return latent_input, probs, binaries, mu, logvar

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, vq_sample=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)

        # cvae decoder
        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                # print("image",image.shape)
                features, pos = self.backbones[cam_id](image[:, cam_id])
                features = features[0]  # take the last layer feature
                pos = pos[0]
                # if 'gel' in cam_name:
                #     print(cam_name)
                #     visualize_features(features)
                # visualize_features_resized(features)
                # print('cam_name',cam_name)
                # print('features.shape',features.shape)
                # print("self.input_proj(features).shape", self.input_proj(features).shape)
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            # for i, feature in enumerate(all_cam_features):
            #     print(f"Tensor {i}: {feature.shape}")
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input,
                                  self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar], probs, binaries


class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + state_dim
            # self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=self.action_dim, hidden_depth=2) # gnq
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=16, hidden_depth=2)  # gnq add
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    transformer = build_transformer(args)

    if args.no_encoder:
        encoder = None
    else:
        # encoder = build_transformer(args) # gnq comment
        encoder = build_encoder(args) # gnq add

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        vq=args.vq,
        vq_class=args.vq_class,
        vq_dim=args.vq_dim,
        action_dim=args.action_dim,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_sonic(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    if sonic_decoder == "vgg" : # "vgg"  "resnet" "resnet38" "yamnet"
        backbones.append(build_sonic_backbone_vggish(args)) # vgg
    elif sonic_decoder == "mlp" : # "vgg"  "resnet" "resnet38" "yamnet"
        backbones.append(build_sonic_backbone_mlp(args)) # vgg

    transformer = build_transformer(args)

    if args.no_encoder:
        encoder = None
    else:
        # encoder = build_transformer(args) # gnq comment
        encoder = build_encoder(args) # gnq add

    model = DETRVAEAUDIO(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        vq=args.vq,
        vq_class=args.vq_class,
        vq_dim=args.vq_dim,
        action_dim=args.action_dim,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("*"*150)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_tactile(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for camera_name in args.camera_names:
        # print("building tactile model for camera {}".format(camera_name))
        # # backbone = build_backbone(args)
        # # backbones.append(backbone)
        # print("cbam for all")
        # backbone = build_CBAM_backbone_mask(args, window_pos=None)  # attention + mask
        # backbones.append(backbone)

        if 'cam' in camera_name:
            # backbone = build_CBAM_backbone(args) # attention
            # window_pos = (0, 0, 640, 120)
            # backbone = build_CBAM_backbone_mask(args, window_pos = None) # attention + mask
            backbone = build_backbone(args)
            backbones.append(backbone)
        elif 'gel' in camera_name:
            # print("unet for tactile info.")
            # backbone = build_tactile_backbone(args)  # unet
            print("resnet for tactile info.")
            backbone = build_backbone(args)
            backbones.append(backbone)

    transformer = build_transformer(args)

    if args.no_encoder:
        encoder = None
    else:
        # encoder = build_transformer(args) # gnq comment
        encoder = build_encoder(args) # gnq add

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        vq=args.vq,
        vq_class=args.vq_class,
        vq_dim=args.vq_dim,
        action_dim=args.action_dim,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("*"*150)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def visualize_features(features):
    """
    可视化 CNN 提取的特征张量
    Args:
        features (torch.Tensor): 特征张量，形状为 [B, C, H, W]
    """
    # 提取第一个样本
    features = features[0]  # 提取第一个 Batch，形状 [C, H, W]
    print("Features shape:", features.shape)

    # 方法 1：可视化单个通道
    plt.figure(figsize=(15, 15))
    num_channels = min(16, features.shape[0])  # 显示前 16 个通道
    for i in range(num_channels):
        plt.subplot(4, 4, i + 1)
        plt.imshow(features[i].detach().cpu().numpy(), cmap='viridis')
        plt.axis('off')
        plt.title(f"Channel {i}")
    plt.suptitle("Single Channel Visualizations", fontsize=16)
    plt.show()

    # # 方法 2：可视化通道均值
    # avg_feature_map = torch.mean(features, dim=0).detach().cpu().numpy()  # 对通道取均值
    # plt.figure(figsize=(6, 6))
    # plt.imshow(avg_feature_map, cmap='viridis')
    # plt.title("Average Feature Map", fontsize=16)
    # plt.axis('off')
    # plt.show()
    #
    # 方法 3：可视化通道最大值
    # max_feature_map = torch.max(features, dim=0)[0].detach().cpu().numpy()  # 对通道取最大值
    # plt.figure(figsize=(6, 6))
    # plt.imshow(max_feature_map, cmap='viridis')
    # plt.title("Max Feature Map", fontsize=16)
    # plt.axis('off')
    # plt.show()

def visualize_features_resized(features):
    """
    将 CNN 提取的特征张量恢复为 [3, 480, 640] 并显示
    Args:
        features (torch.Tensor): 特征张量，形状为 [B, C, H, W]
    """
    # 提取第一个样本
    features = features[0]  # 提取第一个 Batch，形状 [C, H, W]
    print("Original Features shape:", features.shape)  # [512, 15, 20]

    # 步骤 1：上采样到 [512, 480, 640]
    upsampled_features = F.interpolate(features.unsqueeze(0), size=(480, 640), mode='bilinear', align_corners=False)
    upsampled_features = upsampled_features[0]  # 移除 batch 维度
    print("Upsampled Features shape:", upsampled_features.shape)  # [512, 480, 640]

    # 步骤 2：降维到 [3, 480, 640]
    # 使用 1x1 卷积将通道数从 512 降到 3
    conv = torch.nn.Conv2d(in_channels=512, out_channels=3, kernel_size=1)
    reduced_features = conv(upsampled_features.unsqueeze(0))  # [1, 3, 480, 640]
    reduced_features = reduced_features[0]  # 移除 batch 维度
    print("Reduced Features shape:", reduced_features.shape)  # [3, 480, 640]

    # 步骤 3：可视化
    # 将张量转换为 NumPy 格式并归一化到 [0, 1]（为了显示为图像）
    img = reduced_features.detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())  # 归一化到 [0, 1]
    img = img.transpose(1, 2, 0)  # 转换为 [H, W, C] 格式

    # 显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Restored Image (3, 480, 640)")
    plt.show()

