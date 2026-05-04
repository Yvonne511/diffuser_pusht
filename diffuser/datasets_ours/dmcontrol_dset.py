import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Callable, Optional
from einops import rearrange
from .traj_dset import TrajDataset, TrajSlicerDataset, get_train_val_sliced

class DummyNormalizer:
    def fit(self, data):
        pass
    def normalize(self, data):
        return data
    def unnormalize(self, data):
        return data


class DMControlDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "data/dmcontrol_reacher",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalizer_type: str = "linear",
        state_based: bool = False,
        action_normalizer=None,
        state_normalizer=None,
        proprio_normalizer=None,
        linear_action_normalizer=None,
        linear_state_normalizer=None,
        linear_proprio_normalizer=None,
    ):
        assert normalizer_type == "dummy", "Only dummy normalizer is currently supported for DMControlDataset for diffuser"
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalizer_type = normalizer_type
        self.state_based = state_based
        self.states = torch.load(self.data_path / "states.pth").float()
        self.actions = torch.load(self.data_path / "actions.pth").float()
        self.seq_lengths = [self.states.shape[1]] * self.states.shape[0]

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.proprios = self.states.clone()
        print(f"Loaded {n} rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        self.action_normalizer = action_normalizer
        self.state_normalizer = state_normalizer
        self.proprio_normalizer = proprio_normalizer
        self.linear_action_normalizer = linear_action_normalizer
        self.linear_state_normalizer = linear_state_normalizer
        self.linear_proprio_normalizer = linear_proprio_normalizer
        if self.action_normalizer is None:
            self.initialize_normalizers()

        self.actions = self.action_normalizer.normalize(self.actions)
        self.proprios = self.proprio_normalizer.normalize(self.proprios)

        self.normalized_states = self.state_normalizer.normalize(self.states.clone())

    def initialize_normalizers(self):

        self.action_normalizer = DummyNormalizer()
        self.state_normalizer = DummyNormalizer()
        self.proprio_normalizer = DummyNormalizer()

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        if not self.state_based:
            obs_dir = self.data_path / "obses"
            image = torch.load(obs_dir / f"episode_{idx:03d}.pth")
            image = image[frames]  # THWC
            image = image / 255.0
            image = rearrange(image, "T H W C -> T C H W")
            if self.transform:
                image = self.transform(image)
            obs = {
                "visual": image,
                "proprio": proprio
            }
        else:
            normalized_state = self.normalized_states[idx, frames]
            obs = {
                "visual": normalized_state,
                "proprio": proprio
            }
        return obs, act, state, {}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0


def load_dmcontrol_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/dmcontrol_reacher",
    normalizer_type="linear",
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    state_based=False,
    dset_type="traj",
):
    dset = DMControlDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        normalizer_type=normalizer_type,
        state_based=state_based,
    )
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
    )

    datasets = {}
    datasets['train'] = train_slices
    datasets['valid'] = val_slices
    traj_dset = {}
    traj_dset['train'] = dset_train
    traj_dset['valid'] = dset_val
    return datasets, traj_dset