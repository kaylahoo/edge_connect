import torch
import torch.nn as nn
from src.partialconv2d import PartialConv2d #加
from src.partialconv2d import PConvBNActiv #加
import torch.nn.functional as F # 加
#from timm.models.layers import DropPath




class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)  #初始化正态分布N ( 0 , std=1 )
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain) #初始化为 正态分布~ N ( 0 , std )

                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') #kaiming针对relu函数提出的初始化方法
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0) #初始化为常数

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class InpaintGenerator(BaseNetwork):
    #class Generator(nn.Module):

        def __init__(self, image_in_channels=3, edge_in_channels=2, out_channels=3, init_weights=True):
            super(InpaintGenerator, self).__init__()

            self.freeze_ec_bn = False
            #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

            # -----------------------
            # small encoder-decoder
            # -----------------------
            self.ec_texture_1 = PConvBNActiv(image_in_channels, 64, bn=False, sample='down-7')
            self.ec_texture_2 = PConvBNActiv(64, 128, sample='down-5', )
            self.ec_texture_3 = PConvBNActiv(128, 256, sample='down-5')
            self.ec_texture_4 = PConvBNActiv(256, 512, sample='down-3')
            self.ec_texture_5 = PConvBNActiv(512, 512, sample='down-3')
            self.ec_texture_6 = PConvBNActiv(512, 512, sample='down-3')
            self.ec_texture_7 = PConvBNActiv(512, 512, sample='down-3')

            self.dc_texture_7 = PConvBNActiv(512 + 512, 512, activ='leaky')
            self.dc_texture_6 = PConvBNActiv(512 + 512, 512, activ='leaky')
            self.dc_texture_5 = PConvBNActiv(512 + 512, 512, activ='leaky')
            self.dc_texture_4 = PConvBNActiv(512 + 256, 256, activ='leaky')
            self.dc_texture_3 = PConvBNActiv(256 + 128, 128, activ='leaky')
            self.dc_texture_2 = PConvBNActiv(128 + 64, 64, activ='leaky')
            self.dc_texture_1 = PConvBNActiv(64 + out_channels, 64, activ='leaky')

            # -------------------------
            # large encoder-decoder
            # -------------------------
            self.ec_structure_1 = PConvBNActiv(edge_in_channels, 64, bn=False, sample='down-31')
            self.ec_structure_2 = PConvBNActiv(64, 128, sample='down-29')
            self.ec_structure_3 = PConvBNActiv(128, 256, sample='down-27')
            self.ec_structure_4 = PConvBNActiv(256, 512, sample='down-13')
            self.ec_structure_5 = PConvBNActiv(512, 512, sample='down-13')
            self.ec_structure_6 = PConvBNActiv(512, 512, sample='down-13')
            self.ec_structure_7 = PConvBNActiv(512, 512, sample='down-13')

            self.dc_structure_7 = PConvBNActiv(512 + 512, 512, activ='leaky', type='large')
            self.dc_structure_6 = PConvBNActiv(512 + 512, 512, activ='leaky', type='large')
            self.dc_structure_5 = PConvBNActiv(512 + 512, 512, activ='leaky', type='large')
            self.dc_structure_4 = PConvBNActiv(512 + 256, 256, activ='leaky', type='large')
            self.dc_structure_3 = PConvBNActiv(256 + 128, 128, activ='leaky', type='large')
            self.dc_structure_2 = PConvBNActiv(128 + 64, 64, activ='leaky', type='large')
            self.dc_structure_1 = PConvBNActiv(64 + 2, 64, activ='leaky', type='large')

            self.fusion_layer1 = nn.Sequential(
                nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.2)
            )
            self.fusion_layer2 = nn.Sequential(
                nn.Conv2d(64 , 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.2),
            )
            self.out_layer = nn.Sequential(
                nn.Conv2d(64, 3, kernel_size=1),
                nn.Tanh()
            )


            if init_weights:
                self.init_weights()


        def forward(self, input_image, input_edge, mask):

            ec_textures = {}
            ec_structures = {}

            input_texture_mask = torch.cat((mask, mask, mask), dim=1)
            ec_textures['ec_t_0'], ec_textures['ec_t_masks_0'] = input_image, input_texture_mask
            ec_textures['ec_t_1'], ec_textures['ec_t_masks_1'] = self.ec_texture_1(ec_textures['ec_t_0'],
                                                                                   ec_textures['ec_t_masks_0'])
            ec_textures['ec_t_2'], ec_textures['ec_t_masks_2'] = self.ec_texture_2(ec_textures['ec_t_1'],
                                                                                   ec_textures['ec_t_masks_1'])
            ec_textures['ec_t_3'], ec_textures['ec_t_masks_3'] = self.ec_texture_3(ec_textures['ec_t_2'],
                                                                                   ec_textures['ec_t_masks_2'])
            ec_textures['ec_t_4'], ec_textures['ec_t_masks_4'] = self.ec_texture_4(ec_textures['ec_t_3'],
                                                                                   ec_textures['ec_t_masks_3'])
            ec_textures['ec_t_5'], ec_textures['ec_t_masks_5'] = self.ec_texture_5(ec_textures['ec_t_4'],
                                                                                   ec_textures['ec_t_masks_4'])
            ec_textures['ec_t_6'], ec_textures['ec_t_masks_6'] = self.ec_texture_6(ec_textures['ec_t_5'],
                                                                                   ec_textures['ec_t_masks_5'])
            ec_textures['ec_t_7'], ec_textures['ec_t_masks_7'] = self.ec_texture_7(ec_textures['ec_t_6'],
                                                                                   ec_textures['ec_t_masks_6'])

            input_structure_mask = torch.cat((mask, mask), dim=1)
            ec_structures['ec_s_0'], ec_structures['ec_s_masks_0'] = input_edge, input_structure_mask
            ec_structures['ec_s_1'], ec_structures['ec_s_masks_1'] = self.ec_structure_1(ec_structures['ec_s_0'],
                                                                                         ec_structures['ec_s_masks_0'])
            ec_structures['ec_s_2'], ec_structures['ec_s_masks_2'] = self.ec_structure_2(ec_structures['ec_s_1'],
                                                                                         ec_structures['ec_s_masks_1'])
            ec_structures['ec_s_3'], ec_structures['ec_s_masks_3'] = self.ec_structure_3(ec_structures['ec_s_2'],
                                                                                         ec_structures['ec_s_masks_2'])
            ec_structures['ec_s_4'], ec_structures['ec_s_masks_4'] = self.ec_structure_4(ec_structures['ec_s_3'],
                                                                                         ec_structures['ec_s_masks_3'])
            ec_structures['ec_s_5'], ec_structures['ec_s_masks_5'] = self.ec_structure_5(ec_structures['ec_s_4'],
                                                                                         ec_structures['ec_s_masks_4'])
            ec_structures['ec_s_6'], ec_structures['ec_s_masks_6'] = self.ec_structure_6(ec_structures['ec_s_5'],
                                                                                         ec_structures['ec_s_masks_5'])
            ec_structures['ec_s_7'], ec_structures['ec_s_masks_7'] = self.ec_structure_7(ec_structures['ec_s_6'],
                                                                                         ec_structures['ec_s_masks_6'])

            dc_texture, dc_tecture_mask = ec_structures['ec_s_7'], ec_structures['ec_s_masks_7']  # 2x2

            for _ in range(7, 0, -1):
                ec_texture_skip = 'ec_t_{:d}'.format(_ - 1)  # ec_t_6
                ec_texture_masks_skip = 'ec_t_masks_{:d}'.format(_ - 1)  # ec_t_masks_6
                dc_conv = 'dc_texture_{:d}'.format(_)  # dc_texture_7

                dc_texture = F.interpolate(dc_texture, scale_factor=2, mode='bilinear')  # dc_texture 4x4
                dc_tecture_mask = F.interpolate(dc_tecture_mask, scale_factor=2, mode='nearest')  # dc_tecture_mask 4x4

                dc_texture = torch.cat((dc_texture, ec_textures[ec_texture_skip]),
                                       dim=1)  # dc_texture 4x4 ec_textures['ec_t_6'] 4x4
                dc_tecture_mask = torch.cat((dc_tecture_mask, ec_textures[ec_texture_masks_skip]),
                                            dim=1)  # dc_tecture_mask 4x4 ec_textures['ec_t_masks_6'] 4x4

                dc_texture, dc_tecture_mask = getattr(self, dc_conv)(dc_texture,
                                                                     dc_tecture_mask)
                # self.dc_texture_7 = PConvBNActiv(512 + 512, 512, activ='leaky')  self.dc_texture_7(dc_texture, dc_tecture_mask)

            dc_structure, dc_structure_masks = ec_textures['ec_t_7'], ec_textures['ec_t_masks_7']
            for _ in range(7, 0, -1):
                ec_structure_skip = 'ec_s_{:d}'.format(_ - 1)
                ec_structure_masks_skip = 'ec_s_masks_{:d}'.format(_ - 1)
                dc_conv = 'dc_structure_{:d}'.format(_)

                dc_structure = F.interpolate(dc_structure, scale_factor=2, mode='bilinear')
                dc_structure_masks = F.interpolate(dc_structure_masks, scale_factor=2, mode='nearest')

                dc_structure = torch.cat((dc_structure, ec_structures[ec_structure_skip]), dim=1)
                dc_structure_masks = torch.cat((dc_structure_masks, ec_structures[ec_structure_masks_skip]), dim=1)

                dc_structure, dc_structure_masks = getattr(self, dc_conv)(dc_structure, dc_structure_masks)

            output1 = torch.cat(dc_texture,dc_structure)
            output2 = self.fusion_layer1(output1)
            output3 = self.fusion_layer2(output2)
            output4 = self.out_layer(output3)


            return output4




    # def __init__(self, residual_blocks=8, init_weights=True):
    #     super(InpaintGenerator, self).__init__() #在子类中调用父类方法
    #
    #     self.encoder = nn.Sequential(
    #         nn.ReflectionPad2d(3),#镜像填充
    #         nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
    #         nn.InstanceNorm2d(64, track_running_stats=False), #一个channel内做归一化，算H*W的均值
    #         nn.ReLU(True),
    #
    #         nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
    #         nn.InstanceNorm2d(128, track_running_stats=False),
    #         nn.ReLU(True),
    #
    #         nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
    #         nn.InstanceNorm2d(256, track_running_stats=False),
    #         nn.ReLU(True)
    #     )
    #
    #     blocks = []
    #     for _ in range(residual_blocks):
    #         block = ResnetBlock(256, 2)
    #         blocks.append(block)
    #
    #     self.middle = nn.Sequential(*blocks)
    #
    #     self.decoder = nn.Sequential(
    #         nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
    #         nn.InstanceNorm2d(128, track_running_stats=False),
    #         nn.ReLU(True),
    #
    #         nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
    #         nn.InstanceNorm2d(64, track_running_stats=False),
    #         nn.ReLU(True),
    #
    #         nn.ReflectionPad2d(3),
    #         nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
    #     )
    #
    #     if init_weights:
    #         self.init_weights()
    # def forward(self, x):
    #     x = self.encoder(x)
    #     x = self.middle(x)
    #     x = self.decoder(x)
    #     x = (torch.tanh(x) + 1) / 2
    #
    #     return x

    #csy
    # def __init__(self, residual_blocks=8, init_weights=True):
    #     super(InpaintGenerator, self).__init__()
    #
    #     n = 32
    #     # rough
    #     self.enc_r1 = nn.Sequential(
    #         nn.ReflectionPad2d(3),
    #         nn.Conv2d(in_channels=7, out_channels=n, kernel_size=7, stride=1, padding=0),
    #         nn.InstanceNorm2d(n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 256
    #
    #     self.Pconv_r2 = PartialConv2d(in_channels=n, out_channels=2*n, kernel_size=7, stride=2, padding=3,
    #                                   return_mask=True)
    #     self.enc_r2 = nn.Sequential(
    #         nn.InstanceNorm2d(2*n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 128
    #
    #     self.Pconv_r3 = PartialConv2d(in_channels=2*n, out_channels=4*n, kernel_size=3, stride=2, padding=1,
    #                                   return_mask=True)
    #     self.enc_r3 = nn.Sequential(
    #         nn.InstanceNorm2d(4*n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 64
    #
    #     self.enc_r4 = nn.Sequential(
    #         nn.Conv2d(in_channels=4*n, out_channels=8*n, kernel_size=3, stride=2, padding=1),
    #         nn.InstanceNorm2d(8*n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 32
    #
    #
    #     # fine
    #     self.enc_f1 = nn.Sequential(
    #         nn.ReflectionPad2d(2),
    #         nn.Conv2d(in_channels=7, out_channels=n, kernel_size=5, stride=1, padding=0),
    #         nn.InstanceNorm2d(n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 256
    #
    #     self.Pconv_f2 = PartialConv2d(in_channels=n, out_channels=2*n, kernel_size=5, stride=2, padding=2, return_mask=True)
    #     self.enc_f2 = nn.Sequential(
    #         nn.InstanceNorm2d(2*n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 128
    #
    #     self.Pconv_f3 = PartialConv2d(in_channels=2*n, out_channels=4*n, kernel_size=3, stride=2, padding=1, return_mask=True)
    #     self.enc_f3 = nn.Sequential(
    #         nn.InstanceNorm2d(4*n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 64
    #
    #     self.enc_f4 = nn.Sequential(
    #         nn.Conv2d(in_channels=4*n, out_channels=8*n, kernel_size=3, stride=2, padding=1),
    #         nn.InstanceNorm2d(8*n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 32
    #
    #     # bottleneck
    #     blocks = []
    #     for i in range(residual_blocks-1):
    #         block = ResnetBlock(dim=8*n, dilation=2, use_attention_norm=False)
    #         blocks.append(block)
    #     self.middle = nn.Sequential(*blocks)
    #     self.chan_att_norm = ResnetBlock(dim=8*n, dilation=2, use_attention_norm=True)
    #
    #     # decoder
    #     self.dec_1 = nn.Sequential(
    #         nn.ConvTranspose2d(in_channels=8*n, out_channels=4*n, kernel_size=4, stride=2, padding=1),
    #         nn.InstanceNorm2d(4*n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 64
    #
    #     self.dec_2 = nn.Sequential(
    #         nn.ConvTranspose2d(in_channels=4*n, out_channels=2*n, kernel_size=4, stride=2, padding=1),
    #         nn.InstanceNorm2d(2*n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 128
    #
    #     self.dec_3 = nn.Sequential(
    #         nn.ConvTranspose2d(in_channels=2*n+3, out_channels=n, kernel_size=4, stride=2, padding=1),
    #         nn.InstanceNorm2d(n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 256
    #
    #     self.dec_4 = nn.Sequential(
    #         nn.Conv2d(in_channels=n, out_channels=3, kernel_size=1, padding=0)
    #     )  # 256
    #
    #
    #
    #     if init_weights:
    #         self.init_weights()
    #
    # def forward(self, x, mask, stage):
    #
    #     if stage is 0:
    #         structure = x[:, -3:, ...]
    #         structure = F.interpolate(structure, scale_factor=0.5, mode='bilinear')
    #
    #         x = self.enc_r1(x)
    #         x, m = self.Pconv_r2(x, mask)
    #         x = self.enc_r2(x)
    #         f1 = x.detach()
    #         x, _ = self.Pconv_r3(x, m)
    #         x = self.enc_r3(x)
    #         x = self.enc_r4(x)
    #         x = self.middle(x)
    #         x, l_c1 = self.chan_att_norm(x)
    #         x = self.dec_1(x)
    #         x = self.dec_2(x)
    #         x, l_p1 = self.pos_attention_norm1(x, f1, m)
    #         att_structure = self.pos_attention_norm1(structure, structure, m, reuse=True)
    #         att_structure = att_structure.detach()
    #         x = torch.cat((x, att_structure), dim=1)
    #         x = self.dec_3(x)
    #         x = self.dec_4(x)
    #         x_rough = (torch.tanh(x) + 1) / 2
    #         orth_loss = (l_c1 + l_p1) / 2
    #         return x_rough, orth_loss
    #
    #     if stage is 1:
    #         residual = x[:, -3:, ...]
    #         residual = F.interpolate(residual, scale_factor=0.5, mode='bilinear')
    #
    #         x = self.enc_f1(x)
    #         x, m = self.Pconv_f2(x, mask)
    #         x = self.enc_f2(x)
    #         f2 = x.detach()
    #         x, _ = self.Pconv_f3(x, m)
    #         x = self.enc_f3(x)
    #         x = self.enc_f4(x)
    #         x = self.middle(x)
    #         x, l_c2 = self.chan_att_norm(x)
    #         x = self.dec_1(x)
    #         x = self.dec_2(x)
    #         x, l_p2 = self.pos_attention_norm2(x, f2, m)
    #         att_residual = self.pos_attention_norm2(residual, residual, m, reuse=True)
    #         att_residual = att_residual.detach()
    #         x = torch.cat((x, att_residual), dim=1)
    #         x = self.dec_3(x)
    #         x = self.dec_4(x)
    #         x_fine = (torch.tanh(x) + 1) / 2
    #         orth_loss = (l_c2 + l_p2) / 2
    #         return x_fine, orth_loss







class EdgeGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0)
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        inplace = True
        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=inplace),# nn.LeakyReLU给非负值赋予一个非零斜率
        )# spectral_norm利用pytorch自带的频谱归一化函数，给设定好的网络进行频谱归一化，主要用于生成对抗网络的鉴别器

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=inplace),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=inplace),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=inplace),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            #  谱归一化，为了约束GAN的鉴别器映射函数
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
