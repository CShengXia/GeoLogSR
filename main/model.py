import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HF_ENHANCEMENT

class DilatedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate=1):
        super(DilatedConv1D, self).__init__()
        padding = (dilation_rate * (kernel_size - 1)) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               dilation=dilation_rate, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Noise Injection layer
class NoiseInjectionLayer(nn.Module):
    def __init__(self, channels, intensity=0.1):
        super(NoiseInjectionLayer, self).__init__()
        self.intensity = intensity
        self.scale_factor = nn.Parameter(torch.ones(1))
    def forward(self, x):
        if self.training:
            batch_size, channels, seq_len = x.shape
            noise = torch.randn_like(x) * self.intensity * self.scale_factor
            if seq_len > 2:
                try:
                    diff = x[:, :, 1:] - x[:, :, :-1]
                    padding = torch.zeros(batch_size, channels, 1, device=x.device)
                    grad = torch.cat([diff, padding], dim=2).abs()
                    threshold = grad.mean() + grad.std() * 0.8
                    mask = torch.sigmoid((grad - threshold) * 5)
                    return x + noise * mask
                except Exception as e:
                    return x + noise * 0.5
            else:
                return x + noise * 0.3
        else:
            return x

# Channel Attention module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16, k_size=3):
        super(ChannelAttention, self).__init__()
        reduced_channels = max(1, in_channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.se_branch = nn.Sequential(
            nn.Conv1d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )
        self.eca_branch = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.channel_weights = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        se_out = self.se_branch(self.avg_pool(x)) + self.se_branch(self.max_pool(x))
        y = self.avg_pool(x)
        y_eca = self.eca_branch(y.transpose(-1, -2)).transpose(-1, -2)
        attention = self.sigmoid(se_out + y_eca)
        enhanced_x = x * attention
        channel_weight = self.channel_weights(x)
        return enhanced_x * channel_weight + x * (1 - channel_weight)

# Graph Interaction Module (GIM)
class GraphInteractionModule(nn.Module):

    def __init__(self, feature_dim, num_branches=4):
        super(GraphInteractionModule, self).__init__()
        self.feature_dim = feature_dim
        self.num_branches = num_branches
        self.query_proj = nn.ModuleList([nn.Conv1d(feature_dim, feature_dim, kernel_size=1) for _ in range(num_branches)])
        self.key_proj   = nn.ModuleList([nn.Conv1d(feature_dim, feature_dim, kernel_size=1) for _ in range(num_branches)])
        self.value_proj = nn.ModuleList([nn.Conv1d(feature_dim, feature_dim, kernel_size=1) for _ in range(num_branches)])
        self.edge_weights = nn.Parameter(torch.ones(num_branches, num_branches))
        self.output_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_branches)
        ])
        self.scale_factor = torch.sqrt(torch.tensor(feature_dim, dtype=torch.float))
        self.branch_importance = nn.Parameter(torch.ones(num_branches))
    def forward(self, branch_features):

        batch_size = branch_features[0].shape[0]
        queries = [self.query_proj[i](feat) for i, feat in enumerate(branch_features)]
        keys    = [self.key_proj[i](feat)   for i, feat in enumerate(branch_features)]
        values  = [self.value_proj[i](feat) for i, feat in enumerate(branch_features)]
        q_desc = [torch.cat([q.mean(dim=2, keepdim=True), q.max(dim=2, keepdim=True)[0]], dim=2) for q in queries]
        k_desc = [torch.cat([k.mean(dim=2, keepdim=True), k.max(dim=2, keepdim=True)[0]], dim=2) for k in keys]
        enhanced_features = []
        for i in range(self.num_branches):
            attention_weights = []
            for j in range(self.num_branches):
                q = q_desc[i].view(batch_size, self.feature_dim, 2)
                k = k_desc[j].view(batch_size, self.feature_dim, 2)
                attn_score = torch.bmm(q.transpose(1, 2), k) / self.scale_factor
                attn_score = attn_score.mean(dim=(1, 2)) * self.edge_weights[i, j]
                attention_weights.append(attn_score.view(batch_size, 1))
            attn_matrix = torch.cat(attention_weights, dim=1)
            attn_matrix = F.softmax(attn_matrix, dim=1) * F.softmax(self.branch_importance, dim=0)
            aggregated_feature = torch.zeros_like(values[i])
            for j in range(self.num_branches):
                aggregated_feature += values[j] * attn_matrix[:, j].view(batch_size, 1, 1)
            enhanced = self.output_proj[i](aggregated_feature) + branch_features[i]
            enhanced_features.append(enhanced)
        return enhanced_features

# Transformer
class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        self.attention_scale = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        self.conv_branch = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim)
        )
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )
        # Dynamic routing weights
        self.dynamic_router = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        # Input x: [B, C, L]
        B, C, L = x.shape
        assert C == self.dim, "TransformerEncoderBlock: input channel dim mismatch."
        identity = x  # for residual
        conv_features = self.conv_branch(x)
        x_seq = x.permute(0, 2, 1)
        norm_x = self.norm1(x_seq)
        attn_weights = self.attention_scale(norm_x)
        scaled_norm_x = norm_x * attn_weights
        attn_out, _ = self.mha(scaled_norm_x, scaled_norm_x, scaled_norm_x)
        x_seq = x_seq + attn_out
        x_seq = x_seq + self.mlp(self.norm2(x_seq))
        trans_features = x_seq.permute(0, 2, 1)
        fused = self.feature_fusion(torch.cat([trans_features, conv_features], dim=1))
        route_weights = self.dynamic_router(fused)
        trans_w = route_weights[:, 0:1, :]
        conv_w  = route_weights[:, 1:2, :]
        output = trans_w * trans_features + conv_w * conv_features + identity
        return output

# Low-Frequency Branch
class LowFrequencyBranch(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32):
        super(LowFrequencyBranch, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.low_freq_layers = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),  # pointwise conv
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),  # pointwise conv
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.low_freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_dim, max(1, hidden_dim // 4), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(1, hidden_dim // 4), hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.feature_integration = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.output_conv = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
        self.residual = nn.Conv1d(1, 1, kernel_size=1)
    def forward(self, x):
        residual_input = self.residual(x)
        features = self.initial_conv(x)
        low_freq_feat = self.low_freq_layers(features)
        attention = self.low_freq_attention(low_freq_feat)
        attentive_feat = low_freq_feat * attention
        enhanced_feat = self.feature_integration(attentive_feat) + features
        output = self.output_conv(enhanced_feat) + residual_input
        return output

class HighFrequencyBranch(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64):
        super(HighFrequencyBranch, self).__init__()
        assert hidden_dim % 4 == 0, "Hidden dimension must be a multiple of 4"
        self.branch_dim = hidden_dim // 4
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.branch1 = nn.Sequential(
            nn.Conv1d(hidden_dim, self.branch_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.branch_dim),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(hidden_dim, self.branch_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(self.branch_dim),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(hidden_dim, self.branch_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(self.branch_dim),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv1d(hidden_dim, self.branch_dim, kernel_size=1),
            nn.BatchNorm1d(self.branch_dim),
            nn.ReLU(inplace=True)
        )
        self.graph_interaction = GraphInteractionModule(self.branch_dim, num_branches=4)
        self.detail_attention = ChannelAttention(hidden_dim, ratio=8)
        self.edge_enhancement = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        # Spike enhancer
        self.spike_enhancer = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1),
            NoiseInjectionLayer(hidden_dim // 2, intensity=HF_ENHANCEMENT['noise_intensity']),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        with torch.no_grad():
            for i in range(hidden_dim // 2):
                for j in range(min(hidden_dim, (hidden_dim // 2) * 2)):
                    if j % 2 == 0 and i < hidden_dim // 2:
                        self.spike_enhancer[0].weight[i, j, 0] = -1.0
                        self.spike_enhancer[0].weight[i, j, 1] =  2.0
                        self.spike_enhancer[0].weight[i, j, 2] = -1.0

            for i in range(hidden_dim // 2):
                self.spike_enhancer[2].weight[i, i, 0] = -0.5
                self.spike_enhancer[2].weight[i, i, 1] = -0.5
                self.spike_enhancer[2].weight[i, i, 2] =  2.0
                self.spike_enhancer[2].weight[i, i, 3] = -0.5
                self.spike_enhancer[2].weight[i, i, 4] = -0.5

        self.transformer = TransformerEncoderBlock(hidden_dim, num_heads=4)

        self.low_freq_guidance = nn.Sequential(
            nn.Conv1d(hidden_dim + 1, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)

        self.residual_conv = nn.Conv1d(1, 1, kernel_size=1)
    def forward(self, x, low_freq_features=None):
        res = self.residual_conv(x)
        features = self.initial_conv(x)
        b1 = self.branch1(features)
        b2 = self.branch2(features)
        b3 = self.branch3(features)
        b4 = self.branch4(features)
        enhanced_branches = self.graph_interaction([b1, b2, b3, b4])
        b1_e, b2_e, b3_e, b4_e = enhanced_branches
        high_freq_feat = torch.cat([b1_e, b2_e, b3_e, b4_e], dim=1)
        if low_freq_features is not None:
            combined = torch.cat([high_freq_feat, low_freq_features], dim=1)
            guided_feat = self.low_freq_guidance(combined)
        else:
            guided_feat = high_freq_feat
        attended = self.detail_attention(guided_feat)
        spike_features = self.spike_enhancer(attended)
        edge_weights = self.edge_enhancement(attended)
        edge_enhanced = attended * edge_weights + attended + spike_features * HF_ENHANCEMENT['hf_boost_factor']
        transformed = self.transformer(edge_enhanced)
        out = self.conv_out(transformed) + res
        return out

class FusionModule(nn.Module):
    def __init__(self, channels=1):
        super(FusionModule, self).__init__()
        self.freq_interaction = nn.Sequential(
            nn.Conv1d(channels * 2, channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels * 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.adaptive_weights = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels * 2, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.low_freq_enhancer = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.high_freq_enhancer = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.integration_net = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        self.spectrum_modulation = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Tanh()
        )
        self.hf_detail_enhancement = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.1),
            NoiseInjectionLayer(channels, intensity=HF_ENHANCEMENT['noise_intensity'] / 2),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        with torch.no_grad():
            for i in range(channels):
                self.hf_detail_enhancement[0].weight[i, i, 0] = -1.0
                self.hf_detail_enhancement[0].weight[i, i, 1] =  2.0
                self.hf_detail_enhancement[0].weight[i, i, 2] = -1.0
    def forward(self, low_freq, high_freq):
        concat = torch.cat([low_freq, high_freq], dim=1)
        interaction_map = self.freq_interaction(concat)
        weights = self.adaptive_weights(concat)
        low_w = weights[:, 0:1, :]
        high_w = weights[:, 1:2, :]
        enhanced_low = low_freq * self.low_freq_enhancer(low_freq)
        enhanced_high = high_freq * self.high_freq_enhancer(high_freq)
        hf_detail_weights = self.hf_detail_enhancement(high_freq)
        enhanced_high = enhanced_high + high_freq * hf_detail_weights * HF_ENHANCEMENT['hf_boost_factor']
        weighted_low = enhanced_low * low_w
        weighted_high = enhanced_high * high_w * 1.2
        interaction_enhanced = (weighted_low + weighted_high) * interaction_map
        integrated = self.integration_net(torch.cat([weighted_low, weighted_high], dim=1))
        final_output = integrated + self.spectrum_modulation(interaction_enhanced) * integrated
        return final_output

class SimpleFrequencyUpsamplingModule(nn.Module):
    def __init__(self, channels, scale_factor=2):
        super(SimpleFrequencyUpsamplingModule, self).__init__()
        self.channels = channels
        self.scale_factor = scale_factor
        self.transposed_conv = nn.ConvTranspose1d(channels, channels, kernel_size=4,
                                                  stride=scale_factor, padding=1, output_padding=0)
        self.highfreq_branch = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=scale_factor, padding=1, output_padding=0),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.fusion_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels * 2, max(8, channels * 2), kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(max(8, channels * 2), 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.refinement = nn.Sequential(
            nn.Conv1d(channels, max(channels * 2, 4), kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(max(channels * 2, 4), channels, kernel_size=1)
        )
        self._init_highpass_filter()
    def _init_highpass_filter(self):
        with torch.no_grad():
            for i in range(self.channels):
                self.highfreq_branch[0].weight[i, i, 0] = -1.0
                self.highfreq_branch[0].weight[i, i, 1] =  2.0
                self.highfreq_branch[0].weight[i, i, 2] = -1.0
    def forward(self, x):
        up_main = self.transposed_conv(x)
        high_freq = self.highfreq_branch(x)
        high_freq_enhanced = up_main * (1.0 + high_freq)
        combined = torch.cat([up_main, high_freq_enhanced], dim=1)
        weights = self.fusion_attention(combined)
        w_main = weights[:, 0:1, :]
        w_high = weights[:, 1:2, :]
        fused = up_main * w_main + high_freq_enhanced * w_high * 1.2
        output = self.refinement(fused) + fused
        return output

# Dual-Branch Super-Resolution Model
class DualChannelSRModel(nn.Module):
    def __init__(self):
        super(DualChannelSRModel, self).__init__()
        self.low_freq_branch = LowFrequencyBranch()
        self.high_freq_branch = HighFrequencyBranch()
        self.fusion_module = FusionModule()
        self.upsample = SimpleFrequencyUpsamplingModule(channels=1)
        self.input_upsample = SimpleFrequencyUpsamplingModule(channels=1)
        self.refine = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 1, kernel_size=3, padding=1)
        )
        self.adaptive_residual = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.final_hf_enhance = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(8, 1, kernel_size=1),
            nn.Tanh()
        )
    def forward(self, x):
        low_out = self.low_freq_branch(x)
        high_out = self.high_freq_branch(x, low_out)
        fused = self.fusion_module(low_out, high_out)
        up_out = self.upsample(fused)
        up_input = self.input_upsample(x)
        res_weight = self.adaptive_residual(up_input)
        refined = self.refine(up_out) + res_weight * up_input
        hf_detail = self.final_hf_enhance(refined)
        output = refined + hf_detail * 0.25  # add a fraction of high-frequency detail
        return output