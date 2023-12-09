import torch
import torch.nn as nn
import torch.nn.functional as F

from . import regist_model
from ..util.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling, pixel_shuffle_up_sampling_pd, pixel_shuffle_down_sampling_pd, DeformConv2d
from PIL import Image
import numpy as np
import math
import torch.nn.parallel as P
from src.model.restormer_arch import DTB


## Crop: [128, 128]

## WHen testing, pd_pad = 0


"""
This code is modified from Spatially Adaptive SSID.

The code of BNN is modified from https://github.com/COMP6248-Reproducability-Challenge/selfsupervised-denoising/blob/master-with-report/ssdn/ssdn/models/noise_network.py
"""

from typing import Tuple


def rotate(x, angle):
    """Rotate images by 90 degrees clockwise. Can handle any 2D data format.
    Args:
        x (Tensor): Image or batch of images.
        angle (int): Clockwise rotation angle in multiples of 90.
        data_format (str, optional): Format of input image data, e.g. BCHW,
            HWC. Defaults to BCHW.
    Returns:
        Tensor: Copy of tensor with rotation applied.
    """
    h_dim, w_dim = 2, 3

    if angle == 0:
        return x
    elif angle == 90:
        return x.flip(w_dim).transpose(h_dim, w_dim)
    elif angle == 180:
        return x.flip(w_dim).flip(h_dim)
    elif angle == 270:
        return x.flip(h_dim).transpose(h_dim, w_dim)
    else:
        raise NotImplementedError("Must be rotation divisible by 90 degrees")


class Crop2d(nn.Module):
    """Crop input using slicing. Assumes BCHW data.

    Args:
        crop (Tuple[int, int, int, int]): Amounts to crop from each side of the image.
            Tuple is treated as [left, right, top, bottom]/
    """

    def __init__(self, crop: Tuple[int, int, int, int]):
        super().__init__()
        self.crop = crop
        assert len(crop) == 4

    def forward(self, x):
        (left, right, top, bottom) = self.crop
        x0, x1 = left, x.shape[-1] - right
        y0, y1 = top, x.shape[-2] - bottom
        return x[:, :, y0:y1, x0:x1]


class Shift2d(nn.Module):
    """Shift an image in either or both of the vertical and horizontal axis by first
    zero padding on the opposite side that the image is shifting towards before
    cropping the side being shifted towards.

    Args:
        shift (Tuple[int, int]): Tuple of vertical and horizontal shift. Positive values
            shift towards right and bottom, negative values shift towards left and top.
    """

    def __init__(self, shift: Tuple[int, int]):
        super().__init__()
        self.shift = shift
        vert, horz = self.shift
        y_a, y_b = abs(vert), 0
        x_a, x_b = abs(horz), 0
        if vert < 0:
            y_a, y_b = y_b, y_a
        if horz < 0:
            x_a, x_b = x_b, x_a
        # Order : Left, Right, Top Bottom
        self.pad = nn.ZeroPad2d((x_a, x_b, y_a, y_b))
        self.crop = Crop2d((x_b, x_a, y_b, y_a))
        self.shift_block = nn.Sequential(self.pad, self.crop)

    def forward(self, x):
        return self.shift_block(x)


class ShiftConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """Custom convolution layer as defined by Laine et al. for restricting the
        receptive field of a convolution layer to only be upwards. For a h × w kernel,
        a downwards offset of k = [h/2] pixels is used. This is applied as a k sized pad
        to the top of the input before applying the convolution. The bottom k rows are
        cropped out for output.
        """
        super().__init__(*args, **kwargs)
        self.shift_size = (self.kernel_size[0] // 2, 0)
        # Use individual layers of shift for wrapping conv with shift
        shift = Shift2d(self.shift_size)
        self.pad = shift.pad
        self.crop = shift.crop

    def forward(self, x):
        x = self.pad(x)
        x = super().forward(x)
        x = self.crop(x)
        return x


class SSID_BNN(nn.Module):
    def __init__(self, blindspot, in_ch=3, out_ch=3, dim=48):
        super(SSID_BNN, self).__init__()
        in_channels = in_ch
        out_channels = out_ch
        self.blindspot = blindspot

        ####################################
        # Encode Blocks
        ####################################

        # Layers: enc_conv0, enc_conv1, pool1
        self.encode_block_1 = nn.Sequential(
            ShiftConv2d(in_channels, dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ShiftConv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Shift2d((1, 0)),
            nn.MaxPool2d(2)
        )

        # Layers: enc_conv(i), pool(i); i=2..5
        def _encode_block_2_3_4_5() -> nn.Module:
            return nn.Sequential(
                ShiftConv2d(dim, dim, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                Shift2d((1, 0)),
                nn.MaxPool2d(2)
            )

        # Separate instances of same encode module definition created
        self.encode_block_2 = _encode_block_2_3_4_5()
        self.encode_block_3 = _encode_block_2_3_4_5()
        self.encode_block_4 = _encode_block_2_3_4_5()
        self.encode_block_5 = _encode_block_2_3_4_5()

        # Layers: enc_conv6
        self.encode_block_6 = nn.Sequential(
            ShiftConv2d(dim, dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        ####################################
        # Decode Blocks
        ####################################
        # Layers: upsample5
        self.decode_block_6 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self.decode_block_5 = nn.Sequential(
            ShiftConv2d(2 * dim, 2 * dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ShiftConv2d(2 * dim, 2 * dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        def _decode_block_4_3_2() -> nn.Module:
            return nn.Sequential(
                ShiftConv2d(3 * dim, 2 * dim, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ShiftConv2d(2 * dim, 2 * dim, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

        # Separate instances of same decode module definition created
        self.decode_block_4 = _decode_block_4_3_2()
        self.decode_block_3 = _decode_block_4_3_2()
        self.decode_block_2 = _decode_block_4_3_2()

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self.decode_block_1 = nn.Sequential(
            ShiftConv2d(2 * dim + in_channels, 2 * dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ShiftConv2d(2 * dim, 2 * dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        ####################################
        # Output Block
        ####################################

        # Shift blindspot pixel down
        self.shift = Shift2d(((self.blindspot + 1) // 2, 0))

        # nin_a,b,c, linear_act
        self.output_conv = ShiftConv2d(2 * dim, out_channels, 1)
        self.output_block = nn.Sequential(
            ShiftConv2d(8 * dim, 8 * dim, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ShiftConv2d(8 * dim, out_channels, 1)

            # ShiftConv2d(8 * dim, 2 * dim, 1),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # self.output_conv,
        )

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initializes weights using Kaiming  He et al. (2015).

        Only convolution layers have learnable weights. All convolutions use a leaky
        relu activation function (negative_slope = 0.1) except the last which is just
        a linear output.
        """
        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()
        # Initialise last output layer
        nn.init.kaiming_normal_(self.output_conv.weight.data, nonlinearity="linear")

    def forward(self, x, shift=None):

        if shift is not None:
            self.shift = Shift2d((shift, 0))
        else:
            self.shift = Shift2d(((self.blindspot + 1) // 2, 0))

        rotated = [rotate(x, rot) for rot in (0, 90, 180, 270)]

        x = torch.cat((rotated), dim=0)

        # Encoder
        pool1 = self.encode_block_1(x)
        pool2 = self.encode_block_2(pool1)
        pool3 = self.encode_block_3(pool2)
        pool4 = self.encode_block_4(pool3)
        pool5 = self.encode_block_5(pool4)
        encoded = self.encode_block_6(pool5)

        # Decoder
        upsample5 = self.decode_block_6(encoded)
        
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.decode_block_5(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.decode_block_4(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.decode_block_3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.decode_block_2(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        x = self.decode_block_1(concat1)

        # Apply shift
        shifted = self.shift(x)
        # Unstack, rotate and combine
        rotated_batch = torch.chunk(shifted, 4, dim=0)
        aligned = [
            rotate(rotated, rot)
            for rotated, rot in zip(rotated_batch, (0, 270, 180, 90))
        ]
        x = torch.cat(aligned, dim=1)

        x = self.output_block(x)

        return x

    @staticmethod
    def input_wh_mul() -> int:
        """Multiple that both the width and height dimensions of an input must be to be
        processed by the network. This is devised from the number of pooling layers that
        reduce the input size.

        Returns:
            int: Dimension multiplier
        """
        max_pool_layers = 5
        return 2 ** max_pool_layers



class global_branch(nn.Module):
    def __init__(self, stride, in_ch, num_module, group=1, head_ch=None, SIDD=True):
        super().__init__()

        kernel = 21
        pad = kernel // 2

        if head_ch is None:
            head_ch = in_ch

        self.Maskconv = DSPMC_21(head_ch, in_ch, kernel_size=kernel, stride=1, padding=pad, groups=group, padding_mode='reflect')

        self.body = DTB(stride=stride, num_blocks=num_module, dim=in_ch)
        self.SIDD = SIDD

    def forward(self, x, refine=False, dict=None):
        
        if self.SIDD:
            pd_train_br2 = 5
            pd_test_br2 = 4
            pd_refine_br2 = 4
        else:
            pd_train_br2 = 5
            pd_test_br2 = 4
            pd_refine_br2 = 2


        if dict is not None:
            pd_test_br2 = dict['pd_test_br2']
            pd_refine_br2 = dict['pd_refine_br2']

        b, c, h, w = x.shape

        pad = 0
        if self.training:
            if h % pd_train_br2 != 0:
                x = F.pad(x, (0, 0, 0, pd_train_br2 - h % pd_train_br2), mode='reflect')
                pad = pd_train_br2 - h % pd_train_br2
            if w % pd_train_br2 != 0:
                x = F.pad(x, (0, pd_train_br2 - w % pd_train_br2, 0, 0), mode='reflect')
        elif not refine:
            if h % pd_test_br2 != 0:
                x = F.pad(x, (0, 0, 0, pd_test_br2 - h % pd_test_br2), mode='reflect')
                pad = pd_test_br2 - h % pd_test_br2
            if w % pd_test_br2 != 0:
                x = F.pad(x, (0, pd_test_br2 - w % pd_test_br2, 0, 0), mode='reflect')
        else:
            if h % pd_refine_br2 != 0:
                x = F.pad(x, (0, 0, 0, pd_refine_br2 - h % pd_refine_br2), mode='reflect')
                pad = pd_refine_br2 - h % pd_refine_br2
            if w % pd_refine_br2 != 0:
                x = F.pad(x, (0, pd_refine_br2 - w % pd_refine_br2, 0, 0), mode='reflect')

        x = self.Maskconv(x, refine, dict=dict, SIDD=self.SIDD)
        if self.training:
            x = pixel_shuffle_down_sampling(x, f=pd_train_br2, pad=0)
            x = self.body(x)
            x = pixel_shuffle_up_sampling(x, f=pd_train_br2, pad=0)
        elif not refine:
            x = pixel_shuffle_down_sampling_pd(x, f=pd_test_br2, pad=7)
            x = self.body(x)
            x = pixel_shuffle_up_sampling_pd(x, f=pd_test_br2, pad=7)
        else:
            if pd_refine_br2 > 1:
                x = pixel_shuffle_down_sampling_pd(x, f=pd_refine_br2, pad=7)
                x = self.body(x)
                x = pixel_shuffle_up_sampling_pd(x, f=pd_refine_br2, pad=7)
            else:
                x = self.body(x)

        if pad != 0:
            x = x[:, :, :-pad, :-pad]

        return x


class DSPMC_21(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO:
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        # self.mask.fill_(0)

        kwargs_test = kwargs
        kwargs_test['stride'] = kW
        kwargs_test['padding'] = (0, 0)
        self.test_conv = nn.Conv2d(*args, **kwargs_test)
        # y = F.conv2d(input=x_offset, weight=weight, bias=bias, stride=kW, groups=1)

        self.mask.fill_(0)

        stride = 2
        self.mask[:, :, ::stride, ::stride] = 1

        dis = 9 // 2

        for i in range(kH):
            for j in range(kW):
                if abs(i-kH//2) + abs(j-kW//2) <= dis:
                    self.mask[:, :, i, j] = 0

        a = 1
        # self.mask.detach().cpu().numpy()[0,0,...]


    def forward(self, x, refine=False, dict=None, SIDD=True):

        if SIDD:
            pd_test_ratio = 0.8
            pd_refine_ratio = 0.43
        else:
            pd_test_ratio = 0.65
            pd_refine_ratio = 0.35


        if dict is not None:
            pd_test_ratio = dict['pd_test_ratio_br2']
            pd_refine_ratio = dict['pd_refine_ratio_br2']

        if self.training:
            self.weight.data *= self.mask
            return super().forward(x)

        elif not refine:
            x_out = self.forward_chop(x, ratio=pd_test_ratio)
            return x_out

        else:
            x_out = self.forward_chop(x, ratio=pd_refine_ratio)
            return x_out

    # 之前是30
    def forward_chop(self, *args, shave=11, min_size=80000, n_GPUs=1, ratio=1):
        # scale = 1 if self.input_large else self.scale[self.idx_scale]
        scale = 1
        n_GPUs = torch.cuda.device_count()
        n_GPUs = min(n_GPUs, 4)
        # height, width
        h, w = args[0].size()[-2:]

        top = slice(0, h // 2 + shave)
        bottom = slice(h - h // 2 - shave, h)
        left = slice(0, w // 2 + shave)
        right = slice(w - w // 2 - shave, w)
        x_chops = [torch.cat([
            a[..., top, left],
            a[..., top, right],
            a[..., bottom, left],
            a[..., bottom, right]
        ]) for a in args]

        y_chops = []
        if h * w < 4 * min_size:
            for i in range(0, 4, n_GPUs):
                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
                weight = self.weight
                bias = self.bias
                inc, outc, kH, kW = self.weight.size()

                deform_conv = DeformConv2d(inc=inc, outc=outc, kernel_size=kW, stride=1, padding=kW//2, ratio=ratio)

                x_offset = P.data_parallel(deform_conv, *x, range(n_GPUs))
                # x_offset = deform_conv(x[0])

                # y = F.conv2d(input=x_offset, weight=weight, bias=bias, stride=kW, groups=1)
                self.test_conv.weight = weight
                self.test_conv.bias = bias
                y = P.data_parallel(self.test_conv, x_offset, range(n_GPUs))

                # y = out
                del x_offset

                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in zip(*x_chops):
                y = self.forward_chop(*p, shave=shave, min_size=min_size, ratio=ratio)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

        h *= scale
        w *= scale
        top = slice(0, h // 2)
        bottom = slice(h - h // 2, h)
        bottom_r = slice(h // 2 - h, None)
        left = slice(0, w // 2)
        right = slice(w - w // 2, w)

        if w % 2 != 0:
            right_r = slice(w // 2 - w + 1, None)
            bottom_r = slice(h // 2 - h + 1, None)
        else:
            right_r = slice(w // 2 - w, None)
            bottom_r = slice(h // 2 - h, None)

        # batch size, number of color channels
        b, c = y_chops[0][0].size()[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if len(y) == 1: y = y[0]

        return y


@regist_model
class LGBNN(nn.Module):

    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9, pattern='baseline', group=1, head_ch=None, br2_blc=6, SIDD=True):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        assert base_ch % 2 == 0, "base channel should be divided with 2"

        if head_ch is None:
            head_ch = base_ch

        ly = []
        ly += [nn.Conv2d(in_ch, head_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.head = nn.Sequential(*ly)

        self.branch2 = global_branch(stride=3, num_module=[br2_blc], in_ch=base_ch, head_ch=head_ch, SIDD=SIDD)
        self.branch1 = SSID_BNN(blindspot=9, out_ch=base_ch)

        ly = []
        ly += [nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, out_ch, kernel_size=1)]
        self.tail = nn.Sequential(*ly)


    def forward(self, x, refine=False, dict=None):
        pad = 0
        b, c, h, w = x.shape

        # print((h,w))


        br1 = self.branch1(x)

        x_br2 = self.head(x)
        br2 = self.branch2(x_br2, refine, dict=dict)

        x = torch.cat([br1, br2], dim=1)
        if pad != 0:
            x = x[:, :, :-pad, :-pad]

        return self.tail(x)

    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)
