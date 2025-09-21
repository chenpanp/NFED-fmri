from collections import deque
from typing import Generator, Optional
import cv2
import torch
from torch import Tensor
from torch.nn import Module
from torchvision import transforms
import random



def resize(vid, size, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid,
        size=size,
        scale_factor=scale,
        mode=interpolation,
        align_corners=False)
def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255

class ToFloatTensorInZeroOne(object):

    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)

class FeatureExtractor(Module):
    def __init__(self, model, model_path, device,n_frames):
        super().__init__()
        self.device = device
        self.clip_frames = n_frames
        state_dict = torch.load(model_path, map_location='cpu')
        for model_key in ['model', 'module']:
            if model_key in state_dict:
                state_dict = state_dict[model_key]
                break
        model.load_state_dict(state_dict)
        # Set to eval mode and move to desired device
        self.model = model.eval()
        self.model.to(device)

        # videomaev2 transforms
        self.transform = transforms.Compose(
            [ToFloatTensorInZeroOne(),
             Resize((224, 224))])

    def extract_features(self, x: Tensor, keep_seq: bool = True):
        """Extract features for one video clip (v)"""
        with torch.no_grad():
            return self.model(x)

    @torch.no_grad()
    def extract_video(self, video_path: str,  sample_rate: int = 2,
                      stride: int = 16,
                      reduction: str = "none"
                      ) -> Tensor:
        features = []
        for v in self._load_video(video_path, sample_rate, stride):
            # v: (1, C, T, H, W)
            assert v.shape[3:] == (224, 224)
            features.append(self.extract_features(v))

        # features = torch.cat(features)  # (N, 768)
        #
        # if reduction == "mean":
        #     return features.mean(dim=0)
        # elif reduction == "max":
        #     return features.max(dim=0)[0]

        return features

    def _load_video(self, video_path: str, sample_rate: int, stride: int) -> Generator[Tensor, None, None]:


# extract features based on sliding window
        cap = cv2.VideoCapture(video_path)
        deq = deque(maxlen=self.clip_frames)
        deq_index = deque(maxlen=90)
        # clip_start_indexes = list(range(0, total_frames - self.clip_frames * sample_rate, stride * sample_rate))
        # clip_end_indexes = [i + self.clip_frames * sample_rate - 1 for i in clip_start_indexes]
        clip_start_indexes = [0, 30, 58]
        clip_end_indexes = [31, 61, 89]

        for start_frame, end_frame in zip(clip_start_indexes, clip_end_indexes):
            current_index = start_frame - 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                current_index += 1
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = cv2.resize(frame, (224, 224))
                # frame_tmp = frame.copy()
                frame = torch.from_numpy(frame)  # (C, H, W)

                for _ in range(sample_rate - 1):
                    cap.read()
                    current_index += 1

                if current_index >= start_frame and current_index <= end_frame:
                    deq.append(frame)
                    deq_index.append(current_index)

                if current_index == end_frame:
                    # cv2.imwrite("{}.jpg".format(end_frame), frame_tmp)
                    v = torch.stack(list(deq))  # (T, H, W，C)
                    yield self.transform(v).unsqueeze(0).to(self.device)

        cap.release()
