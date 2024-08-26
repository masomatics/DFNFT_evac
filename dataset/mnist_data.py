import os
import numpy as np
import cv2
import torch
import torchvision
import math
import colorsys
from skimage.transform import resize
from copy import deepcopy
import pdb
import copy
import sys


class SequentialMNIST:
    # Rotate around z axis only.

    def __init__(
        self,
        root,
        train=True,
        transforms=torchvision.transforms.ToTensor(),
        T=3,
        max_angle_velocity_ratio=[-0.2, 0.2],
        max_angle_accl_ratio=[-0.0, 0.0],
        max_color_velocity_ratio=[-0.2, 0.2],
        max_color_accl_ratio=[-0.0, 0.0],
        max_pos=[-10, 10],
        max_trans_accl=[-0.0, 0.0],
        label=False,
        label_velo=False,
        label_accl=False,
        max_T=9,
        only_use_digit4=False,
        backgrnd=False,
        shared_transition=False,
        rng=None,
        **kwargs,
    ):
        self.T = T
        self.max_T = max_T
        self.rng = rng if rng is not None else np.random
        self.transforms = transforms
        self.data = torchvision.datasets.MNIST(root, train, download=True)

        self.angle_velocity_range = (
            (-max_angle_velocity_ratio, max_angle_velocity_ratio)
            if isinstance(max_angle_velocity_ratio, (int, float))
            else max_angle_velocity_ratio
        )
        self.color_velocity_range = (
            (-max_color_velocity_ratio, max_color_velocity_ratio)
            if isinstance(max_color_velocity_ratio, (int, float))
            else max_color_velocity_ratio
        )
        self.angle_accl_range = (
            (-max_angle_accl_ratio, max_angle_accl_ratio)
            if isinstance(max_angle_accl_ratio, (int, float))
            else max_angle_accl_ratio
        )
        self.color_accl_range = (
            (-max_color_accl_ratio, max_color_accl_ratio)
            if isinstance(max_color_accl_ratio, (int, float))
            else max_color_accl_ratio
        )

        self.max_pos = max_pos
        self.max_trans_accl = max_trans_accl
        self.label = label
        self.label_velo = label_velo
        self.label_accl = label_accl
        if backgrnd:
            print("""
                  =============
                  background ON
                  =============
                  """)
            fname = "MNIST/train_dat.pt" if train else "MNIST/test_dat.pt"
            self.backgrnd_data = torch.load(os.path.join(root, fname))

        if only_use_digit4:
            datas = []
            for pair in self.data:
                if pair[1] == 4:
                    datas.append(pair)
            self.data = datas
        self.shared_transition = shared_transition
        if self.shared_transition:
            self.init_shared_transition_parameters()

    def init_shared_transition_parameters(self):
        self.angles_v = self.rng.uniform(
            math.pi * self.angle_velocity_range[0],
            math.pi * self.angle_velocity_range[1],
            size=1,
        )
        self.angles_a = self.rng.uniform(
            math.pi * self.angle_accl_range[0],
            math.pi * self.angle_accl_range[1],
            size=1,
        )
        self.color_v = 0.5 * self.rng.uniform(
            self.color_velocity_range[0], self.color_velocity_range[1], size=1
        )
        self.color_a = 0.5 * self.rng.uniform(
            self.color_accl_range[0], self.color_accl_range[1], size=1
        )
        pos0 = self.rng.uniform(self.max_pos[0], self.max_pos[1], size=[2])
        pos1 = self.rng.uniform(self.max_pos[0], self.max_pos[1], size=[2])
        self.pos_v = (pos1 - pos0) / (self.max_T - 1)
        self.pos_a = self.rng.uniform(
            self.max_trans_accl[0], self.max_trans_accl[1], size=[2]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = np.array(self.data[i][0], np.float32).reshape(28, 28)
        image = resize(image, [24, 24])
        image = cv2.copyMakeBorder(
            image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        angles_0 = self.rng.uniform(0, 2 * math.pi, size=1)
        color_0 = self.rng.uniform(0, 1, size=1)
        pos0 = self.rng.uniform(self.max_pos[0], self.max_pos[1], size=[2])
        pos1 = self.rng.uniform(self.max_pos[0], self.max_pos[1], size=[2])
        if self.shared_transition:
            (angles_v, angles_a) = (self.angles_v, self.angles_a)
            (color_v, color_a) = (self.color_v, self.color_a)
            (pos_v, pos_a) = (self.pos_v, self.pos_a)
        else:
            angles_v = self.rng.uniform(
                math.pi * self.angle_velocity_range[0],
                math.pi * self.angle_velocity_range[1],
                size=1,
            )
            angles_a = self.rng.uniform(
                math.pi * self.angle_accl_range[0],
                math.pi * self.angle_accl_range[1],
                size=1,
            )
            color_v = 0.5 * self.rng.uniform(
                self.color_velocity_range[0], self.color_velocity_range[1], size=1
            )
            color_a = 0.5 * self.rng.uniform(
                self.color_accl_range[0], self.color_accl_range[1], size=1
            )
            pos_v = (pos1 - pos0) / (self.max_T - 1)
            pos_a = self.rng.uniform(
                self.max_trans_accl[0], self.max_trans_accl[1], size=[2]
            )
        images = []
        for t in range(self.T):
            angles_t = 0.5 * angles_a * t**2 + angles_v * t + angles_0
            pos_t = 0.5 * pos_a * t**2 + pos_v * t + pos0
            color_t = (0.5 * color_a * t**2 + t * color_v + color_0) % 1
            mat = get_RTmat(0, 0, float(angles_t), 32, 32, pos_t[0], pos_t[1])
            _image = cv2.warpPerspective(image.copy(), mat, (32, 32))

            rgb = np.asarray(colorsys.hsv_to_rgb(color_t[0], 1, 1), dtype=np.float32)
            _image = np.concatenate([_image[:, :, None]] * 3, axis=-1) * rgb[None, None]
            _image = _image / 255.0

            if hasattr(self, "backgrnd_data"):
                _imagemask = np.sum(_image, axis=2, keepdims=True) < 3e-1
                _image = torch.tensor(_image) + self.backgrnd_data[i].permute(
                    [1, 2, 0]
                ) * (_imagemask)
                _image = np.array(torch.clip(_image, max=1.0))

            images.append(torch.tensor(_image.astype(np.float32)).permute([2, 0, 1]))

            # THIS IMAGE IS NOT PIL IMAGE ANYMORE, why apply this??? Torchvision shape problem
            # images.append(self.transforms(_image.astype(np.float32)))

        if self.label or self.label_velo:
            ret = [images]
            if self.label:
                ret += [self.data[i][1]]
            if self.label_velo:
                ret += [
                    freq_to_wave(angles_v.astype(np.float32)),
                    freq_to_wave((2 * math.pi * color_v).astype(np.float32)),
                    pos_v.astype(np.float32),
                ]
            if self.label_accl:
                ret += [
                    freq_to_wave(angles_a.astype(np.float32)),
                    freq_to_wave((2 * math.pi * color_a).astype(np.float32)),
                    pos_a.astype(np.float32),
                ]
            return ret
        else:
            return torch.stack(images)


"""
Sequential MNIST with two objects.

If fixpos is True, there won't be translation.
If pair_translation is True, same action will be applied to both objects.

By default, the initial positioning is fixed.
"""


class SequentialMNIST_double(SequentialMNIST):
    def __init__(
        self,
        root,
        train=True,
        transforms=torchvision.transforms.ToTensor(),
        T=3,
        max_angle_velocity_ratio=[-0.2, 0.2],
        max_angle_accl_ratio=[-0.0, 0.0],
        max_color_velocity_ratio=[-0.2, 0.2],
        max_color_accl_ratio=[-0.0, 0.0],
        max_pos=[-8, 8],
        max_trans_accl=[-10.0, 10.0],
        label=False,
        label_velo=False,
        label_accl=False,
        max_T=9,
        only_use_digit4=False,
        backgrnd=False,
        shared_transition=False,
        pair_transition=False,
        same_object=True,
        rng=None,
        fixpos=True,
        param_debug=False,
        align_initial=False,
    ):
        super().__init__(
            root,
            train=train,
            transforms=transforms,
            T=T,
            max_angle_velocity_ratio=max_angle_velocity_ratio,
            max_angle_accl_ratio=max_angle_accl_ratio,
            max_color_velocity_ratio=max_color_velocity_ratio,
            max_color_accl_ratio=max_color_accl_ratio,
            max_pos=max_pos,
            max_trans_accl=max_trans_accl,
            label=label,
            label_velo=label_velo,
            label_accl=label_accl,
            max_T=max_T,
            only_use_digit4=only_use_digit4,
            backgrnd=backgrnd,
            shared_transition=shared_transition,
            rng=rng,
        )
        np.random.seed(0)
        self.first_indices = np.random.choice(len(self.data), len(self.data))
        self.second_indices = np.random.choice(len(self.data), len(self.data))
        self.pair_transition = pair_transition
        self.same_object = same_object
        self.fixpos = fixpos
        self.param_debug = param_debug

        # DEBUGGING PURPOSE: Default is False
        self.align_initial = align_initial

        if True:
            # if fixpos == True:
            # pos0 : pos0_obj0, pos0_obj1
            # pos1 : pos1_obj0, pos1_obj1

            # np.random.seed(1)

            self.pairpos0 = np.array(
                [[-4.01038611, -6.21961682], [-4.44810926, 5.9317169]]
            )
            self.pairpos1 = np.array(
                [[-4.69249351, 6.69777453], [8.62280464, 3.90062484]]
            )
        #
        # else:
        #    self.pairpos0 , self.pairpos1 = self.initial_poss()

    def initial_poss(self):
        pos0 = []  # pos0_obj0, pos0_obj1
        pos1 = []  # pos1_obj0, pos1_obj1

        pos0.append(self.rng.uniform(self.max_pos[0], self.max_pos[1], size=[2]))
        pos0.append(self.rng.uniform(self.max_pos[1], self.max_pos[1], size=[2]))

        pos1.append(self.rng.uniform(pos0[0], self.max_pos[0] - pos0[0], size=[2]))
        pos1.append(self.rng.uniform(pos0[1], self.max_pos[1] - pos0[1], size=[2]))

        return pos0, pos1

    def __getitem__(self, i):
        pos0, pos1 = self.initial_poss()
        if self.fixpos == True:
            # pos0, pos1 = self.pairpos0, self.pairpos1
            pos0 = self.pairpos0
        # pos0, pos1 = self.initial_poss()
        # pos0 = self.pairpos0

        if self.same_object == True:
            first_idx = 1
            second_idx = 4
        else:
            first_idx = self.first_indices[i]
            second_idx = self.second_indices[i]

        if self.pair_transition != True:
            params0 = self.otain_action_parameters(pos0[0], pos1[0])
            params1 = self.otain_action_parameters(pos0[1], pos1[1])

        else:
            # If running experiment with all sequences consisting of the same object
            params0 = self.otain_action_parameters(pos0[0], pos1[0])
            params1 = copy.deepcopy(params0)

            # param1 is the same as param0 all except at initial position, angle and color
            params1["pos0"], params1["pos1"] = pos0[1], pos1[1]

            angle_0 = self.rng.uniform(0, 2 * math.pi, size=1)
            color_0 = self.rng.uniform(0, 1, size=1)
            params1["angles_0"], params1["color_0"] = angle_0, color_0
            params1["pos_v"] = params0["pos_v"]

        if self.align_initial == True:
            params1["angles_0"], params1["color_0"] = (
                params0["angles_0"],
                params0["color_0"],
            )

        if self.param_debug == True:
            return params0, params1

        images0 = self.get_image(first_idx, **params0)
        images1 = self.get_image(second_idx, **params1)

        images = []

        for t in range(len(images0)):
            _image = images0[t] + images1[t]

            if hasattr(self, "backgrnd_data"):
                _imagemask = np.sum(_image, axis=2, keepdims=True) < 3e-1
                _image = torch.tensor(_image) + self.backgrnd_data[i].permute(
                    [1, 2, 0]
                ) * (_imagemask)
            image = torch.clip(_image, max=1.0)
            # image = torch.tensor((torch.clip(_image, max=1.)))

            images.append(image)

        return torch.stack(images)

    def one_obj_immobile(self, i, mode=1):
        if self.same_object:
            first_idx = 1
            second_idx = 4
        else:
            first_idx = self.first_indices[i]
            second_idx = self.second_indices[i]

        pos0, pos1 = self.pairpos0, self.pairpos1
        params0 = self.otain_action_parameters(pos0[0], pos1[0])

        params1 = copy.deepcopy(params0)
        params1["pos0"], params1["pos1"] = pos0[1], pos1[1]
        params1["pos_v"] = params0["pos_v"]

        angle_0 = self.rng.uniform(0, 2 * math.pi, size=1)
        color_0 = self.rng.uniform(0, 1, size=1)
        params1["angles_0"], params1["color_0"] = angle_0, color_0

        if mode == 1:
            images0 = self.get_image(first_idx, **params0)
            images1 = self.get_image(second_idx, immobile=True, **params1)
        else:
            images0 = self.get_image(first_idx, immobile=True, **params0)
            images1 = self.get_image(second_idx, **params1)

        images = []

        for t in range(len(images0)):
            _image = images0[t] + images1[t]
            # _image = images0[t]

            if hasattr(self, "backgrnd_data"):
                _imagemask = np.sum(_image, axis=2, keepdims=True) < 3e-1
                _image = torch.tensor(_image) + self.backgrnd_data[i].permute(
                    [1, 2, 0]
                ) * (_imagemask)
            image = np.array(torch.clip(_image, max=1.0))

            images.append(image)

        return images

    def otain_action_parameters(self, pos0, pos1):
        angles_0 = self.rng.uniform(0, 2 * math.pi, size=1)
        color_0 = self.rng.uniform(0, 1, size=1)

        if self.shared_transition:
            (angles_v, angles_a) = (self.angles_v, self.angles_a)
            (color_v, color_a) = (self.color_v, self.color_a)
            (pos_v, pos_a) = (self.pos_v, self.pos_a)
        else:
            angles_v = self.rng.uniform(
                math.pi * self.angle_velocity_range[0],
                math.pi * self.angle_velocity_range[1],
                size=1,
            )
            angles_a = self.rng.uniform(
                math.pi * self.angle_accl_range[0],
                math.pi * self.angle_accl_range[1],
                size=1,
            )
            color_v = 0.5 * self.rng.uniform(
                self.color_velocity_range[0], self.color_velocity_range[1], size=1
            )
            color_a = 0.5 * self.rng.uniform(
                self.color_accl_range[0], self.color_accl_range[1], size=1
            )
            pos_v = (pos1 - pos0) / (self.max_T - 1)
            pos_a = self.rng.uniform(
                self.max_trans_accl[0], self.max_trans_accl[1], size=[2]
            )

        action_params = {}
        action_params["angles_0"] = angles_0
        action_params["angles_v"] = angles_v
        action_params["angles_a"] = angles_a
        action_params["color_0"] = color_0
        action_params["color_v"] = color_v
        action_params["color_a"] = color_a
        action_params["pos0"] = pos0
        action_params["pos_v"] = pos_v
        action_params["pos_a"] = 0
        action_params["pos1"] = pos1

        return action_params

    def get_image(
        self,
        i,
        pos0,
        pos1,
        angles_0,
        angles_v,
        angles_a,
        color_0,
        color_v,
        color_a,
        pos_v,
        pos_a,
        immobile=False,
    ):
        original_size = 28
        digit_size = 14
        margin = int((32 - digit_size) / 2)

        image = np.array(self.data[i][0], np.float32).reshape(
            original_size, original_size
        )
        image = resize(image, [digit_size, digit_size])
        image = cv2.copyMakeBorder(
            image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        images = []

        for t in range(self.T):
            if immobile == False:
                angles_t = 0.5 * angles_a * t**2 + angles_v * t + angles_0

                if self.fixpos == False:
                    ###
                    # Freeze translation
                    ###
                    pos_t = 0.5 * pos_a * t**2 + pos_v * t + pos0
                else:
                    pos_t = pos0
                color_t = (0.5 * color_a * t**2 + t * color_v + color_0) % 1
            if immobile == True:
                angles_t = angles_0
                pos_t = pos0
                color_t = color_0

            mat = get_RTmat(0, 0, float(angles_t), 32, 32, pos_t[0], pos_t[1])
            _image = cv2.warpPerspective(image.copy(), mat, (32, 32))
            rgb = np.asarray(colorsys.hsv_to_rgb(color_t[0], 1, 1), dtype=np.float32)
            _image = np.concatenate([_image[:, :, None]] * 3, axis=-1) * rgb[None, None]
            _image = _image / 255.0

            images.append(self.transforms(_image.astype(np.float32)))

        if self.label or self.label_velo:
            ret = [images]
            if self.label:
                ret += [self.data[i][1]]
            if self.label_velo:
                ret += [
                    freq_to_wave(angles_v.astype(np.float32)),
                    freq_to_wave((2 * math.pi * color_v).astype(np.float32)),
                    pos_v.astype(np.float32),
                ]
            if self.label_accl:
                ret += [
                    freq_to_wave(angles_a.astype(np.float32)),
                    freq_to_wave((2 * math.pi * color_a).astype(np.float32)),
                    pos_a.astype(np.float32),
                ]
            return ret
        else:
            return images


########## Misc ######


def freq_to_wave(freq, is_radian=True):
    _freq_rad = 2 * math.pi * freq if not is_radian else freq
    return torch.hstack([torch.cos(_freq_rad), torch.sin(_freq_rad)])


def get_RTmat(theta, phi, gamma, w, h, dx, dy):
    d = np.sqrt(h**2 + w**2)
    f = d / (2 * np.sin(gamma) if np.sin(gamma) != 0 else 1)
    # Projection 2D -> 3D matrix
    A1 = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1], [0, 0, 1]])

    # Rotation matrices around the X, Y, and Z axis
    RX = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )

    RY = np.array(
        [
            [np.cos(phi), 0, -np.sin(phi), 0],
            [0, 1, 0, 0],
            [np.sin(phi), 0, np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    )

    RZ = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0, 0],
            [np.sin(gamma), np.cos(gamma), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)

    # Translation matrix
    T = np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, f], [0, 0, 0, 1]])
    # Projection 3D -> 2D matrix
    A2 = np.array([[f, 0, w / 2, 0], [0, f, h / 2, 0], [0, 0, 1, 0]])
    return np.dot(A2, np.dot(T, np.dot(R, A1)))
