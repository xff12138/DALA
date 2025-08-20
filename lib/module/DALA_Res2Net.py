import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.module.LightRFB import LightRFB
from lib.module.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.module.PCDAmodule import PCDAlignment
from lib.module.LA import LocalAttention

class conbine_feature(nn.Module):
    def __init__(self):
        super(conbine_feature, self).__init__()
        self.fea_low = nn.Conv2d(32, 32, 1, stride=1, padding=0, bias=False)
        self.fea_bn2 = nn.BatchNorm2d(32)
        self.fea_act = nn.PReLU(32)
        self.refine = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU())

    def forward(self, low_fea, att_fea):
        low_fea = self.fea_act(self.fea_bn2(self.fea_low(low_fea)))
        refine_feature = self.refine(torch.cat([low_fea, att_fea], dim=1))
        return refine_feature

class DALA(nn.Module):
    def __init__(self):
        super(DALA, self).__init__()
        self.feature_extractor = res2net50_v1b_26w_4s(pretrained=True)
        self.RFB1 = LightRFB(channels_in=256, channels_mid=128, channels_out=32)
        self.RFB2 = LightRFB(channels_in=512, channels_mid=128, channels_out=32)
        self.RFB3 = LightRFB(channels_in=1024, channels_mid=128, channels_out=32)
        self.PCDA = PCDAlignment(num_feat=32, deformable_groups=8, pyramid_layer=3)
        self.LA = LocalAttention(feat_dim=32, kW=5, kH=5)
        self.decoder = conbine_feature()
        self.SegNIN = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.PReLU(),
            nn.Dropout2d(0.1), nn.Conv2d(16, 1, kernel_size=1, bias=False)
        )


    def forward(self, x):

        origin_shape = x.shape
        x = x.view(-1, *origin_shape[2:])
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(x)
        
        #extract features
        feature_layer1 = self.feature_extractor.layer1(x)
        feature_layer2 = self.feature_extractor.layer2(feature_layer1)
        feature_layer3 = self.feature_extractor.layer3(feature_layer2)

        # Reduce the channel dimension.
        feature_reduce1 = self.RFB1(feature_layer1)
        feature_reduce2 = self.RFB2(feature_layer2)
        feature_reduce3 = self.RFB3(feature_layer3)

        # Reshape into temporal formation.
        feature_reduce1 = feature_reduce1.view(*origin_shape[:2], *feature_reduce1.shape[1:])
        feature_reduce2 = feature_reduce2.view(*origin_shape[:2], *feature_reduce2.shape[1:])
        feature_reduce3 = feature_reduce3.view(*origin_shape[:2], *feature_reduce3.shape[1:])

        # Feature Separation.
        feature1_ref = feature_reduce1[:, 0, ...]
        feature1_pred = feature_reduce1[:, 1, ...]
        feature2_ref = feature_reduce2[:, 0, ...]
        feature2_pred = feature_reduce2[:, 1, ...]
        feature3_ref = feature_reduce3[:, 0, ...]
        feature3_pred = feature_reduce3[:, 1, ...]
        feature_ref = [feature1_ref, feature2_ref, feature3_ref]
        feature_pred = [feature1_pred, feature2_pred, feature3_pred]

        #Alignment feature
        feature_alignment = self.PCDA(feature_ref, feature_pred)
        
        #local attention
        feature_attention = self.LA(feature_alignment, feature1_pred)
        
        # decoder
        out = self.decoder(feature1_pred, feature_attention)
        out = torch.sigmoid(
            F.interpolate(self.SegNIN(out), size=(origin_shape[-2], origin_shape[-1]), mode="bilinear",
                          align_corners=False))

        return out
    
if __name__ == "__main__":
    a1 = torch.randn(12, 2, 3, 256, 448).cuda() 
    mobile = DALA().cuda()
    print(mobile(a1).shape)
    print('Total params: %.2fM' % (sum(p.numel() for p in mobile.parameters()) / 1000000.0))