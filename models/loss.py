import torch
import torch.nn as nn
import torch.nn.functional as F

def mvsnet_loss(depth_est, depth_gt, mask):
    mask = mask > 0.5
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)

#SSIM_LOSS 
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 4
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)]) 
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(self.window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.autograd.Variable(_2D_window.expand(channel, 1, self.window_size, window_size).contiguous())
        return window 
    
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True, full=False):
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = v1 / v2  # contrast * sensitivity

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            cs = cs.mean()
            ret = ssim_map.mean()
        else:
            cs = cs
            ret = ssim_map

        if full:
            return ret, cs
        return ret
        

    def forward(self, img1, img2, retain_grad=True):    
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
            window = window.cuda(img1.get_device())
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        ssim_value = self._ssim(img1, img2, self.window, self.window_size, channel, self.size_average)
        ssim_loss = 1 - ssim_value
        if retain_grad:
            ssim_loss.retain_grad()
        return ssim_loss
    
# Multi_Scale SSIM Loss
class MS_SSIMLoss(SSIMLoss):
    def __init__(self, window_size=11, size_average=True):
        super(MS_SSIMLoss, self).__init__(window_size=window_size, size_average=size_average)

    def msssim(self, img1, img2, window_size=11, size_average=True):
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda()
        levels = weights.size()[0]
        ssims = []
        mcs = []
        for _ in range(levels):
            sim, cs = self._ssim(img1=img1, img2=img2, window=self.window, window_size=window_size,\
                 channel=self.channel, size_average=size_average, full=True)
            ssims.append(sim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        ssims = torch.stack(ssims)
        mcs = torch.stack(mcs)

        pow1 = mcs ** weights
        pow2 = ssims ** weights

        output = torch.prod(pow1[:-1]) * pow2[-1]
        return output

    def forward(self, img1, img2, retain_grad=True):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
            window = window.cuda(img1.get_device())
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        msssim_value = self.msssim(img1=img1, img2=img2, window_size=self.window_size, size_average=self.size_average)
        msssim_loss = 1 - msssim_value
        if retain_grad:
            msssim_loss.retain_grad()
        return msssim_loss

# L1 and multi_ssim fusion loss function
class L1_MSSSIM(MS_SSIMLoss):
    def __init__(self, alpha=0.84, size_average=True, retain_grad=True):
        super(L1_MSSSIM, self).__init__()
        self.alpha = alpha

    def forward(self, img1, img2, retain_grad=True):
        msssim_loss = MS_SSIMLoss()
        msssim_loss_res = msssim_loss(img1, img2, retain_grad=True)
        l1_loss = nn.L1Loss(reduction='none')
        l1_loss_res = l1_loss(img1, img2)
        self.window = self.window.cuda(img1.get_device())
        l1_loss_gaussian = F.conv2d(l1_loss_res, self.window, padding=self.window_size//2, groups=self.channel)
        l1_loss_gaussian = l1_loss_gaussian.mean()
        l1_msssim_loss = self.alpha * msssim_loss_res + (1 - self.alpha) * l1_loss_gaussian

        if retain_grad:
            l1_msssim_loss.retain_grad()
        return l1_msssim_loss   



# test code, please ignore it
if __name__ == "__main__":
    a = torch.rand(size=(10,4,512,960), requires_grad=True, device="cuda:0")
    b = torch.rand(size=(10,4,512,960), requires_grad=False, device="cuda:0")
    # a = torch.autograd.Variable(a)

    loss = MS_SSIMLoss()
    optimizer = torch.optim.Adam([a], lr=0.001)

    loss_value = loss(a,b)
    while loss_value > 0.05:
        optimizer.zero_grad()
        loss_value = loss(a,b)
        loss_value.backward()
        optimizer.step()
        print(loss_value)
