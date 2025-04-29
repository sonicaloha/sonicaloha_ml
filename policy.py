import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
from detr.main import  build_SonicACT_model_and_optimizer
import IPython
e = IPython.embed



import torchaudio.transforms as T

def compute_log_mel_spectrogram(audio, audio_sampling_rate, device, n_mels=96):
    target_sample_rate = 16000
    if audio_sampling_rate != target_sample_rate:
        resample_transform = T.Resample(orig_freq=audio_sampling_rate,
                                        new_freq=target_sample_rate).to(device)
        audio = resample_transform(audio)

    win_length = int(0.025 * target_sample_rate)
    hop_length = int(0.010 * target_sample_rate)
    n_fft = 1024

    mel_transform = T.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        win_length=win_length,
        hop_length=hop_length,
        center=False
    ).to(device)

    mel_spec = mel_transform(audio)  # [B, n_mels, T]
    mel_to_db = T.AmplitudeToDB(stype="power", top_db=80).to(device)
    mel_spec_db = mel_to_db(mel_spec)

    mel_spec_db = mel_spec_db.unsqueeze(1)  # [B, 1, n_mels, T]
    return mel_spec_db



class SonicACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_SonicACT_model_and_optimizer(args_override)
        # model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.vq = args_override['vq']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, audio, audio_samping_rate, actions=None, is_pad=None, vq_sample=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            loss_dict = dict()
            a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.model(qpos, image, audio, audio_samping_rate, env_state, actions, is_pad, vq_sample)
            if self.vq or self.model.encoder is None:
                total_kld = [torch.tensor(0.0)]
            else:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            if self.vq:
                loss_dict['vq_discrepancy'] = F.l1_loss(probs, binaries, reduction='mean')
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')

            # print("the all_l1 shape is",all_l1.shape)
            # exit()
            # all_l1 = apply_weights_to_all_l1(  # TODO @gnq
            #     all_l1,
            #     weight_function=calculate_weights_aloha,  # @gnq
            #     actions_for_curr_step=self.model.num_queries,
            #     k=0.01 # k=0.008
            # ) * self.model.num_queries

            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else:  # inference time
            # a_hat, _, (_, _), _, _ = self.model(qpos, image, env_state,vq_sample=vq_sample)  # no action, sample from prior
            a_hat, _, (_, _), _, _ = self.model(qpos, image, audio, audio_samping_rate, env_state,vq_sample=vq_sample)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

    @torch.no_grad()
    def vq_encode(self, qpos, actions, is_pad):
        actions = actions[:, :self.model.num_queries]
        is_pad = is_pad[:, :self.model.num_queries]

        _, _, binaries, _, _ = self.model.encode(qpos, actions, is_pad)

        return binaries

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
