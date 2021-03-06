import torch
import torch.nn as nn
import torch.nn.functional as F
from .module_denoise import *


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1) 

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x

class ConvReLU_Double(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConvReLU_Double, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.block = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
                                    nn.ReLU())
        self.conv_1x1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride)

    def forward(self, x):
        x1 = self.block(x)
        x2 = self.conv_1x1(x)
        x = x1 + x2
        return x


class FeatureNet_New(nn.Module):
    def __init__(self):
        super(FeatureNet_New, self).__init__()
        self.block1 = ConvReLU_Double(3, 8, 1)
        self.block2 = ConvReLU_Double(8, 16, 2)
        self.block3 = ConvReLU_Double(16, 32, 2)
        self.feature = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.feature(x)
        return x

class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))
        
        self.add_module1 = ConvBnReLU3D(32, 32)

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))
        
        self.add_module2 = ConvBnReLU3D(16, 16)

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.add_module3 = ConvBnReLU3D(8, 8)

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = self.conv7(x)
        x = self.add_module1(x)
        x = conv4 + x
        x = self.conv9(x)
        x = self.add_module2(x)
        x = conv2 + x
        x = self.conv11(x)
        x = self.add_module3(x)
        x = conv0 + x
        x = self.prob(x)
        return x


class Downblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Downblock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.block = nn.Sequential(nn.Conv3d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(self.out_channels, self.out_channels, kernel_size=1, stride=1),
                                )
        self.conv_1x1 = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride)
    
    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x = self.block(x)
        x = x + x_1x1
        x = F.relu(x, inplace=True)
        return x

class Upblock(nn.Module):
    def __init__(self, in_channels):
        super(Upblock, self).__init__()
        self.in_channels = in_channels
        self.block = nn.Sequential(nn.Conv3d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(self.in_channels, self.in_channels, kernel_size=1, stride=1),
                                )
        self.conv_1x1 = nn.Conv3d(self.in_channels, self.in_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x = self.block(x)
        x = x + x_1x1
        return x


class CostRegNet_NewUnet(nn.Module):
    def __init__(self):
        super(CostRegNet_NewUnet, self).__init__()
        self.downblock1 = Downblock(32, 8, 1)
        self.downblock2 = Downblock(8, 16, 2)
        self.downblock3 = Downblock(16, 32, 2)
        self.downblock4 = Downblock(32, 64, 2)

        self.deconv1 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose3d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.upblock1 = Upblock(32)
        self.upblock2 = Upblock(16)
        self.upblock3 = Upblock(8)
        self.prob = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.downblock1(x)
        conv1 = self.downblock2(conv0)
        conv2 = self.downblock3(conv1)
        conv3 = self.downblock4(conv2)

        x = self.deconv1(conv3)
        x = x + conv2
        x = F.relu(x, inplace=True)
        x = self.upblock1(x)
        x = self.deconv2(x)
        x = x + conv1
        x = F.relu(x, inplace=True)
        x = self.upblock2(x)
        x = self.deconv3(x)
        x = x + conv0
        x = F.relu(x, inplace=True)
        x = self.prob(x)
        return x


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = torch.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class MVSNet_Denoise(nn.Module):
    def __init__(self, refine=True):
        super(MVSNet_Denoise, self).__init__()
        self.refine = refine

        self.feature = FeatureNet_New()
        self.cost_regularization = CostRegNet_NewUnet()
        if self.refine:
            self.refine_network = RefineNet()

    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1) # unbind ????????????1
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # step 3. cost volume regularization
        cost_reg = self.cost_regularization(volume_variance)
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        cost_reg = cost_reg.squeeze(1)
        prob_volume = F.softmax(cost_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        # step 4. depth map refinement
        if not self.refine:
            return {"depth": depth, "photometric_confidence": photometric_confidence}
        else:
            # hanjiawei: concatnation preparation
            origin_size = imgs[0].size()
            # batch_size = origin_size[0]
            # channel = origin_size[1]
            h = origin_size[2]
            w = origin_size[3]
            img_resize = F.interpolate(imgs[0], (int(h/4), int(w/4)), mode='area')
            depth = depth.unsqueeze(1)
            refined_depth = self.refine_network(img_resize, depth)
            depth = depth.squeeze(1)
            refined_depth = refined_depth.squeeze(1)
            return {"depth": depth, "refined_depth": refined_depth, "photometric_confidence": photometric_confidence}



