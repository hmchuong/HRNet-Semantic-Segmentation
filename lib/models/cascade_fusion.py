import torch.nn as nn
import torch.nn.functional as F
from .seg_hrnet import BatchNorm2d, BN_MOMENTUM
from utils.certainty import calculate_certainty, get_uncertain_point_coords_on_grid, point_sample
from .cascade_attention import CascadeAttention

class CascadeFusion(CascadeAttention):
    def __init__(self, config, backbone, **kwargs):
        super(CascadeFusion, self).__init__(config, backbone)
        self.refinement_points = config.MODEL.ATTENTION.POINTS

    def forward(self, x):
        features = self.backbone(x)
        final_logits, previous_feature, uncertainty_score = None, None, None
        outputs = []
        for conv, head, feature, n_points in zip(self.convs, self.heads, features, self.refinement_points):
            
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

            if final_logits is not None:
                B, C, H, W = pred.shape

                # Calculate certainty score
                certainty_score = calculate_certainty(pred)
                
                # Calculate refine score
                refine_score = uncertainty_score * certainty_score

                # Replace points
                if n_points >= 1.0:
                    n_points = int(n_points)
                else:
                    n_points = int(H * W * n_points)
                
                error_point_indices, error_point_coords = get_uncertain_point_coords_on_grid(refine_score, n_points)
                error_point_indices = error_point_indices.unsqueeze(1).expand(-1, C, -1)
                
                refined_logits = point_sample(logits, error_point_coords, align_corners=False)
                
                final_logits = F.interpolate(final_logits, size=(H, W), mode='bilinear', align_corners=False)
                final_logits = (
                    final_logits.reshape(B, C, H * W)
                    .scatter_(2, error_point_indices, refined_logits)
                    .view(B, C, H, W)
                )
            else:
                final_logits = logits

            # Calculate uncertainty
            uncertainty_score = 1.0 - calculate_certainty(final_logits.softmax(1))
            
        outputs += [final_logits]
        return outputs
    
def get_seg_model(cfg, backbone, **kwargs):
    model = CascadeFusion(cfg, backbone, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model