import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from .seg_hrnet import BatchNorm2d, BN_MOMENTUM
from utils.certainty import calculate_certainty

class CascadeAttention(nn.Module):
    def __init__(self, config, backbone, **kwargs):
        super(CascadeAttention, self).__init__()
        self.backbone = backbone
        convs = []
        heads = []
        for in_feature in config.MODEL.ATTENTION.IN_FEATURES:
            convs += [nn.Conv2d(in_channels=in_feature,
                            out_channels=config.MODEL.ATTENTION.INTER_CHANNEL,
                            kernel_size=1,
                            stride=1,
                            padding=0)]
            heads += [nn.Sequential(BatchNorm2d(config.MODEL.ATTENTION.INTER_CHANNEL, momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(
                                        in_channels=config.MODEL.ATTENTION.INTER_CHANNEL,
                                        out_channels=config.DATASET.NUM_CLASSES,
                                        kernel_size=config.MODEL.EXTRA.FINAL_CONV_KERNEL,
                                        stride=1,
                                        padding=1 if config.MODEL.EXTRA.FINAL_CONV_KERNEL == 3 else 0))]
        self.convs = nn.ModuleList(convs)
        self.heads = nn.ModuleList(heads)

        self.backbone.init_weights(config.MODEL.ATTENTION.PRETRAINED)

    def forward(self, x, return_attention=False):
        features = self.backbone(x)
        previous_feature, uncertainty_score = None, None
        outputs = []
        attention = []
        for conv, head, feature in zip(self.convs, self.heads, features):
            
            feature = conv(feature)
            
            # Aggregation
            if uncertainty_score is not None:
                
                uncertainty_score = F.interpolate(uncertainty_score, size=(feature.shape[2], feature.shape[3]), mode='bilinear', align_corners=False)
                certainty_score = 1.0 - uncertainty_score
                previous_feature = F.interpolate(previous_feature, size=(feature.shape[2], feature.shape[3]), mode='bilinear', align_corners=False)
                feature = feature * uncertainty_score + certainty_score * previous_feature
            previous_feature = feature

            # Prediction
            logits = head(feature)
            outputs.append(logits)

            pred = logits.softmax(1)
            uncertainty_score = 1.0 - calculate_certainty(pred)
            attention += [uncertainty_score]

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
    model = CascadeAttention(cfg, backbone, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model
