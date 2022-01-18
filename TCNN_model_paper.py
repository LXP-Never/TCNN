# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2021/12/6
"""

"""
import torch
import torch.nn.functional as F
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, causal=False):
        super(DepthwiseSeparableConv, self).__init__()
        # Use `groups` option to implement depthwise convolution
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels, bias=False)

        pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        if causal:
            self.net = nn.Sequential(depthwise_conv,
                                     Chomp1d(padding),
                                     nn.PReLU(),
                                     nn.BatchNorm1d(in_channels),
                                     pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv,
                                     nn.PReLU(),
                                     nn.BatchNorm1d(in_channels),
                                     pointwise_conv)

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResBlock, self).__init__()

        self.TCM_net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.PReLU(num_parameters=1),
            nn.BatchNorm1d(num_features=out_channels),
            DepthwiseSeparableConv(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size,
                                   stride=1,
                                   padding=(kernel_size - 1) * dilation, dilation=dilation, causal=True)
        )

    def forward(self, input):
        x = self.TCM_net(input)
        return x + input


class TCNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, init_dilation=3, num_layers=6):
        super(TCNN_Block, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation_size = init_dilation ** i
            # in_channels = in_channels if i == 0 else out_channels

            layers += [ResBlock(in_channels, out_channels,
                                kernel_size, dilation=dilation_size)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DConv2d_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(DConv2d_block, self).__init__()
        self.DConv2d = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU()
        )
        self.drop = nn.Dropout(0.2)

    def forward(self, encode, decode):
        encode = self.drop(encode)
        skip_connection = torch.cat((encode, decode), dim=1)
        DConv2d = self.DConv2d(skip_connection)

        return DConv2d


# input: (B, 1, T, 320)    T为帧数，320为帧长
class TCNN(nn.Module):
    def __init__(self):
        super(TCNN, self).__init__()
        self.Conv2d_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 5), stride=(1, 1), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.prelu1 = nn.PReLU()

        self.Conv2d_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5), stride=(1, 2), padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.prelu2 = nn.PReLU()

        self.Conv2d_3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5), stride=(1, 2), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(num_features=16)
        self.prelu3 = nn.PReLU()

        self.Conv2d_4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 5), stride=(1, 2), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.prelu4 = nn.PReLU()

        self.Conv2d_5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 5), stride=(1, 2), padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(num_features=32)
        self.prelu5 = nn.PReLU()

        self.Conv2d_6 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 5), stride=(1, 2), padding=(1, 1))
        self.bn6 = nn.BatchNorm2d(num_features=64)
        self.prelu6 = nn.PReLU()

        self.Conv2d_7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 5), stride=(1, 2), padding=(1, 1))
        self.bn7 = nn.BatchNorm2d(num_features=64)
        self.prelu7 = nn.PReLU()

        self.TCNN_Block_1 = TCNN_Block(in_channels=256, out_channels=512, kernel_size=3, init_dilation=2, num_layers=6)
        self.TCNN_Block_2 = TCNN_Block(in_channels=256, out_channels=512, kernel_size=3, init_dilation=2, num_layers=6)
        self.TCNN_Block_3 = TCNN_Block(in_channels=256, out_channels=512, kernel_size=3, init_dilation=2, num_layers=6)

        self.DConv2d_7 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 5), stride=(1, 2),
                                            padding=(1, 1),
                                            output_padding=(0, 0))
        self.bn7_t = nn.BatchNorm2d(num_features=64)
        self.prelu7_t = nn.PReLU()

        self.DConv2d_6 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(2, 5), stride=(1, 2),
                                            padding=(1, 1),
                                            output_padding=(0, 0))
        self.bn6_t = nn.BatchNorm2d(num_features=32)
        self.prelu6_t = nn.PReLU()

        self.DConv2d_5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2, 5), stride=(1, 2),
                                            padding=(1, 1),
                                            output_padding=(0, 0))
        self.bn5_t = nn.BatchNorm2d(num_features=32)
        self.prelu5_t = nn.PReLU()

        self.DConv2d_4 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(2, 5), stride=(1, 2),
                                            padding=(1, 1),
                                            output_padding=(0, 0))
        self.bn4_t = nn.BatchNorm2d(num_features=16)
        self.prelu4_t = nn.PReLU()

        self.DConv2d_3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(2, 5), stride=(1, 2),
                                            padding=(1, 1),
                                            output_padding=(0, 1))
        self.bn3_t = nn.BatchNorm2d(num_features=16)
        self.prelu3_t = nn.PReLU()

        self.DConv2d_2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(2, 5), stride=(1, 2),
                                            padding=(1, 2),
                                            output_padding=(0, 1))
        self.bn2_t = nn.BatchNorm2d(num_features=16)
        self.prelu2_t = nn.PReLU()

        self.DConv2d_1 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(2, 5), stride=(1, 1),
                                            padding=(1, 2),
                                            output_padding=(0, 0))
        self.bn1_t = nn.BatchNorm2d(num_features=1)
        self.prelu1_t = nn.PReLU()

        self.dropout = nn.Dropout(p=0.3, inplace=True)

    def forward(self, input):
        Conv2d_1 = self.prelu1(self.bn1(self.Conv2d_1(input)[:, :, :-1, :].contiguous()))
        print("Conv2d_1", Conv2d_1.shape)  # [64, 16, 5, 320]
        Conv2d_2 = self.prelu2(self.bn2(self.Conv2d_2(Conv2d_1)[:, :, :-1, :].contiguous()))
        print("Conv2d_2", Conv2d_2.shape)  # [64, 16, 5, 160]
        Conv2d_3 = self.prelu3(self.bn3(self.Conv2d_3(Conv2d_2)[:, :, :-1, :].contiguous()))
        print("Conv2d_3", Conv2d_3.shape)  # [64, 16, 5, 79]
        Conv2d_4 = self.prelu4(self.bn4(self.Conv2d_4(Conv2d_3)[:, :, :-1, :].contiguous()))
        print("Conv2d_4", Conv2d_4.shape)  # [64, 32, 5, 39]
        Conv2d_5 = self.prelu5(self.bn5(self.Conv2d_5(Conv2d_4)[:, :, :-1, :].contiguous()))
        print("Conv2d_5", Conv2d_5.shape)  # [64, 32, 5, 19]
        Conv2d_6 = self.prelu6(self.bn6(self.Conv2d_6(Conv2d_5)[:, :, :-1, :].contiguous()))
        print("Conv2d_6", Conv2d_6.shape)  # [64, 64, 5, 9]
        Conv2d_7 = self.prelu7(self.bn7(self.Conv2d_7(Conv2d_6)[:, :, :-1, :].contiguous()))
        print("Conv2d_7", Conv2d_7.shape)  # [64, 64, 5, 4] (B, 1, T, 320)
        reshape_1 = Conv2d_7.permute(0, 1, 3, 2)  # [64, 64, 4, 5] (B,C,帧长,帧数)
        batch_size, C, frame_len, frame_num = reshape_1.shape
        reshape_1 = reshape_1.reshape(batch_size, C * frame_len, frame_num)
        # print("reshape_1", reshape_1.shape)  # [64, 256, 5]

        TCNN_Block_1 = self.TCNN_Block_1(reshape_1)
        TCNN_Block_2 = self.TCNN_Block_2(TCNN_Block_1)
        TCNN_Block_3 = self.TCNN_Block_3(TCNN_Block_2)

        reshape_2 = TCNN_Block_3.reshape(batch_size, C, frame_len, frame_num)
        reshape_2 = reshape_2.permute(0, 1, 3, 2)
        print("reshape_2", reshape_2.shape)  # [64, 64, 5, 4]

        DConv2d_7 = self.prelu7_t(self.bn7_t(F.pad(self.DConv2d_7(torch.cat((self.dropout(Conv2d_7), reshape_2), dim=1)),[0, 0, 1, 0]).contiguous()))
        print("DConv2d_7", DConv2d_7.shape)     # [64, 64, 5, 9]
        DConv2d_6 = self.prelu6_t(self.bn6_t(F.pad(self.DConv2d_6(torch.cat((self.dropout(Conv2d_6), DConv2d_7), dim=1)),[0, 0, 1, 0]).contiguous()))
        print("DConv2d_6", DConv2d_6.shape)     # [64, 32, 5, 19]
        DConv2d_5 = self.prelu5_t(self.bn5_t(F.pad(self.DConv2d_5(torch.cat((self.dropout(Conv2d_5), DConv2d_6), dim=1)),[0, 0, 1, 0]).contiguous()))
        print("DConv2d_5", DConv2d_5.shape)     # [64, 32, 5, 39]
        DConv2d_4 = self.prelu4_t(self.bn4_t(F.pad(self.DConv2d_4(torch.cat((self.dropout(Conv2d_4), DConv2d_5), dim=1)),[0, 0, 1, 0]).contiguous()))
        print("DConv2d_4", DConv2d_4.shape)     # [64, 16, 5, 79]
        DConv2d_3 = self.prelu3_t(self.bn3_t(F.pad(self.DConv2d_3(torch.cat((self.dropout(Conv2d_3), DConv2d_4), dim=1)),[0, 0, 1, 0]).contiguous()))
        print("DConv2d_3", DConv2d_3.shape)     # [64, 16, 5, 160]
        DConv2d_2 = self.prelu2_t(self.bn2_t(F.pad(self.DConv2d_2(torch.cat((self.dropout(Conv2d_2), DConv2d_3), dim=1)),[0, 0, 1, 0]).contiguous()))
        print("DConv2d_2", DConv2d_2.shape)     # [64, 16, 5, 320]
        DConv2d_1 = self.prelu1_t(self.bn1_t(F.pad(self.DConv2d_1(torch.cat((self.dropout(Conv2d_1), DConv2d_2), dim=1)),[0, 0, 1, 0]).contiguous()))
        print("DConv2d_1", DConv2d_1.shape)     # [64, 1, 5, 320]

        return DConv2d_1


if __name__ == "__main__":
    # x = torch.randn(64, 32, 8192)
    # model = TCNN_Block(in_channels=32, out_channels=64, kernel_size=3, init_dilation=3, num_layers=6)  # 输出 (64, 1, 8192)

    x = torch.randn(64, 1, 5, 320)
    model = TCNN()

    # x = torch.randn(64, 32, 256)
    # model = DepthwiseSeparableConv(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=4, dilation=2,
    #                                causal=True)

    y = model(x)
    print("output", y.shape)  # output torch.Size([64, 1, 8192])
