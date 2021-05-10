import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from .seg_hrnet import BatchNorm2d, BN_MOMENTUM

class MSAttention(nn.Module):
    def __init__(self, config, backbone, **kwargs):
        super(MSAttention, self).__init__()
        self.backbone = backbone
        convs = []
        att_heads = []
        seg_heads = []
        for idx, in_feature in enumerate(config.MODEL.ATTENTION.IN_FEATURES):
            convs += [nn.Sequential(nn.Conv2d(in_channels=in_feature,
                            out_channels=config.MODEL.ATTENTION.INTER_CHANNEL,
                            kernel_size=1,
                            stride=1,
                            padding=0), 
                            BatchNorm2d(config.MODEL.ATTENTION.INTER_CHANNEL, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))]
            if idx == 0:
                att_heads += [None]
            else:
                att_heads += [nn.Sequential(nn.Conv2d(in_channels=config.MODEL.ATTENTION.INTER_CHANNEL * 2,
                                out_channels=1,
                                kernel_size=1,
                                stride=1,
                                padding=0), nn.Sigmoid())]
            seg_heads += [nn.Conv2d(in_channels=config.MODEL.ATTENTION.INTER_CHANNEL,
                                out_channels=config.DATASET.NUM_CLASSES,
                                kernel_size=config.MODEL.EXTRA.FINAL_CONV_KERNEL,
                                stride=1,
                                padding=1 if config.MODEL.EXTRA.FINAL_CONV_KERNEL == 3 else 0)]
        self.convs = nn.ModuleList(convs)
        self.att_heads = nn.ModuleList(att_heads)
        self.seg_heads = nn.ModuleList(seg_heads)

        self.backbone.init_weights(config.MODEL.ATTENTION.PRETRAINED)

    def forward(self, x, return_attention=False):
        features = self.backbone(x)
        previous_feature, uncertainty_score = None, None
        outputs = []
        attention = []
        for conv, att_head, seg_head, feature in zip(self.convs, self.att_heads, self.seg_heads, features):
            
            feature = conv(feature)
            
            # Attention and aggregation
            if att_head is not None:
                
                previous_feature = F.interpolate(previous_feature, size=(feature.shape[2], feature.shape[3]), mode='bilinear', align_corners=False)
                alpha = att_head(torch.cat([previous_feature, feature], dim=1))
                feature = previous_feature * ( 1.0 - alpha) + feature * alpha
                attention += [alpha]

            # Prediction
            logits = seg_head(feature)
            outputs += [logits]
            previous_feature = feature
            
        if return_attention:
            return outputs, attention
        return outputs
    
    def init_weights(self, pretrained='',):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            print('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            #for k, _ in pretrained_dict.items():
            #    logger.info(
            #        '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
    
def get_seg_model(cfg, backbone, **kwargs):
    model = MSAttention(cfg, backbone, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model
