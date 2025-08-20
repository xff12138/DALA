import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
from localAttention import (similar_forward,
                            similar_backward,
                            weighting_forward,
                            weighting_backward_ori,
                            weighting_backward_weight)

class similarFunction(Function):
    @staticmethod
    def forward(ctx, x_ori, x_loc, kH, kW):
        ctx.save_for_backward(x_ori, x_loc)
        ctx.kHW = (kH, kW)
        output = similar_forward(x_ori, x_loc, kH, kW)

        return output

    @staticmethod
    #@once_differentiable
    def backward(ctx, grad_outputs):
        x_ori, x_loc = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = similar_backward(x_loc, grad_outputs, kH, kW, True)
        grad_loc = similar_backward(x_ori, grad_outputs, kH, kW, False)

        return grad_ori, grad_loc, None, None


class weightingFunction(Function):
    @staticmethod
    def forward(ctx, x_ori, x_weight, kH, kW):
        ctx.save_for_backward(x_ori, x_weight)
        ctx.kHW = (kH, kW)
        output = weighting_forward(x_ori, x_weight, kH, kW)

        return output

    @staticmethod
    #@once_differentiable
    def backward(ctx, grad_outputs):
        x_ori, x_weight = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = weighting_backward_ori(x_weight, grad_outputs, kH, kW)
        grad_weight = weighting_backward_weight(x_ori, grad_outputs, kH, kW)

        return grad_ori, grad_weight, None, None

f_similar = similarFunction.apply
f_weighting = weightingFunction.apply

def f_similar_cpu(query, key, kH, kW, key_uf):
    unfold = nn.Unfold(kernel_size=(kH, kW), padding=(kH//2, kW//2))
    # N, C, kH*kW, H, W
    key_uf = unfold(key).view(key.shape[0],key.shape[1],kH*kW, key.shape[-2], key.shape[-1])
    # N, 1, C, H, W
    query_uf = torch.unsqueeze(query,1)

    query_uf = query_uf.transpose(2,3).transpose(3,4).transpose(1,2).transpose(2,3)
    key_uf = key_uf.transpose(2,3).transpose(3,4).transpose(1,2).transpose(2,3)
    # N, H, W, 1, kH*kW
    import time
    start = time.time()
    weight = torch.matmul(
        query_uf, key_uf[0,0,0]
        )
    print(time.time() - start)
    import pdb; pdb.set_trace()
    # N, H, W, kH*kW
    return weight[:,:,:,0,:]

def f_weighting_cpu(value, weight, kH, kW):
    # import pdb; pdb.set_trace()
    unfold = nn.Unfold(kernel_size=(kH, kW), padding=(kH//2, kW//2))
    # N, C, kH*kW, H, W
    value_uf = unfold(value).view(value.shape[0],value.shape[1],kH*kW, value.shape[-2], value.shape[-1])
    # N, 1, kH*kW, H, W
    weight_uf = torch.unsqueeze(weight,1).transpose(3,4).transpose(2,3)
    # N, C, kH*kW, H, W
    weight = value_uf * weight_uf
    # N, C, H, W
    return weight.sum(dim=2)


class LocalAttention(nn.Module):
    def __init__(self, feat_dim, kW, kH):
        super(LocalAttention, self).__init__()

        self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1,
                                       groups=feat_dim)  # computation can be reduced by setting groups=feat_dim
        self.hr_key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1,
                                     groups=feat_dim)  # computation can be reduced by setting groups=feat_dim

        self.hr_value_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim)
        # self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=1)
        # self.f_similar = f_similar
        # self.f_weighting = f_weighting

        self.softmax = nn.Softmax(dim=3)

        self.kW = kW
        self.kH = kH

        self.init_weight()

        # self.placeholder = torch.empty(1, feat_dim, kH*kW, 720, 960)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, hr_feat, lr_feat):
        # # - hr_feat [N, C, H, W]
        # # - lr_feat [N, C, h, w]

        N, C, H, W = hr_feat.shape
        N, C, h, w = lr_feat.shape

        lr_feat = F.interpolate(lr_feat, (H, W), mode='bilinear', align_corners=True)
        # hr_feat = F.interpolate(hr_feat,(h,w),mode='bilinear', align_corners=True)

        hr_value = self.hr_value_conv(hr_feat)
        # hr_value = hr_feat
        hr_key = self.hr_key_conv(hr_feat)
        lr_query = self.lr_query_conv(lr_feat)

        weight = f_similar(lr_query, hr_key, self.kH, self.kW)
        # weight = f_similar_cpu(lr_query, hr_key, self.kH, self.kW, self.placeholder)
        # print(weight.shape)
        # weight = F.softmax(weight, dim=3)
        weight = self.softmax(weight)
        # np.save('./image_test_result/test-038-weight.npy',weight.cpu().detach().numpy())
        # import pdb; pdb.set_trace()

        attention_result = f_weighting(hr_value, weight, self.kH, self.kW)

        result = lr_feat + attention_result
        # result = attention_result

        return result

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # print("1", name, module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    # print("2", name, module)
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # print("3", name, module)
                # nowd_params += list(module.parameters())
                ## The params in attention modules are nn.Linear
                continue
        return wd_params, nowd_params

if __name__ == "__main__":
    a1 = torch.randn(1, 64, 256, 256).cuda() 
    a2 = torch.randn(1, 64, 256, 256).cuda() 
    mobile = LocalAttention(feat_dim=64, kW=5, kH=5).cuda()
    print(mobile(a1,a2).shape)