from typing import Union, List, Tuple
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from Models.modules import PatchExpand, FinalPatchExpand_X4
from Models.vmamba import VSSMEncoder, load_pretrained_Base, LayerNorm2d, Linear2d, BaseDecoderBlock
from collections import OrderedDict
import math
import torch.nn.functional as F

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

class EPI(nn.Module):
    def __init__(
        self,
        in_channels,
        prototype_dim=512,
        num_classes=6
    ):
        super().__init__()

        self.num_classes = num_classes
        self.prototype_dim = prototype_dim
        self.in_channels = in_channels
        self.visual_prototypes = nn.Parameter(torch.randn(num_classes, prototype_dim) * 0.02)
        self.prototype_proj = nn.Linear(prototype_dim, in_channels)
        self.response_gate = nn.Sequential(
            nn.Conv2d(num_classes,num_classes,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(num_classes),
            nn.GELU(),
            nn.Conv2d(num_classes,1,kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat):
        B, C, H, W = feat.shape
        prototype = self.prototype_proj(self.visual_prototypes)
        query_logits = torch.einsum(
            "bchw,kc->bkhw",
            feat,
            prototype
        ) / math.sqrt(C)
        query_response = torch.sigmoid(query_logits)   # [B, 6, H, W]
        traffic_response = self.response_gate(query_response)   # [B, 1, H, W]
        activated_feat = feat * traffic_response        # [B, C, H, W]
        feat_out = feat + activated_feat
        return feat_out, query_response


class DSPM(nn.Module):
    def __init__(
        self,
        channels,
        sigma_x=0.8,
        sigma_y=0.8,
        reduction=4
    ):
        super().__init__()


        self.sigma_x_raw = nn.Parameter(self.inverse_softplus(sigma_x))
        self.sigma_y_raw = nn.Parameter(self.inverse_softplus(sigma_y))

        hidden = max(channels // reduction, 32)
        self.pre_proj = nn.Sequential(
            nn.Conv2d(channels,hidden,kernel_size=1,bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU()
        )
        # Local fine-detail branch
        self.local_branch = nn.Sequential(
            nn.Conv2d(hidden,hidden,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU()
        )
        # Medium-range branch
        self.dilated_branch = nn.Sequential(
            nn.Conv2d(hidden,hidden,kernel_size=3,padding=2,dilation=2,bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU()
        )
        # Broad-context branch
        self.context_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=5,stride=1,padding=2),
            nn.Conv2d(hidden,hidden,kernel_size=1,bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU()
        )
        # Parallel feature aggregation
        self.context_fusion = nn.Sequential(
            nn.Conv2d(hidden * 3,channels,kernel_size=1,bias=False),
            nn.BatchNorm2d(channels)
        )

    @staticmethod
    def inverse_softplus(x):
        x = torch.tensor(float(x))
        return torch.log(torch.expm1(x))

    def build_distance_prior(self, B, H, W, device, dtype):
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij"
        )
        x = (xx - (W - 1) / 2.0) / ((W - 1) / 2.0 + 1e-6)
        y = (H - 1 - yy) / (H - 1 + 1e-6)
        sigma_x = F.softplus(self.sigma_x_raw).to(device=device, dtype=dtype) + 1e-6
        sigma_y = F.softplus(self.sigma_y_raw).to(device=device, dtype=dtype) + 1e-6
        dist2 = (x / sigma_x) ** 2 + (y / sigma_y) ** 2
        distance_prior = torch.exp(-dist2)
        distance_prior = distance_prior.unsqueeze(0).unsqueeze(0)
        distance_prior = distance_prior.expand(B, 1, H, W)

        return distance_prior

    @torch.no_grad()
    def get_sigma(self):
        sigma_x = F.softplus(self.sigma_x_raw).item()
        sigma_y = F.softplus(self.sigma_y_raw).item()
        return sigma_x, sigma_y

    def forward(self, feat):
        B, C, H, W = feat.shape
        risk_map = self.build_distance_prior(B=B,H=H,W=W,device=feat.device,dtype=feat.dtype)
        x = self.pre_proj(feat)
        local_feat = self.local_branch(x)
        dilated_feat = self.dilated_branch(x)
        context_feat = self.context_branch(x)
        enhanced_feat = torch.cat([local_feat, dilated_feat, context_feat], dim=1)
        enhanced_feat = self.context_fusion(enhanced_feat)
        feat_risk = feat + risk_map * enhanced_feat
        return feat_risk, risk_map

class VSSMDecoder(nn.Module):
    def __init__(
            self,
            deep_supervision,
            features_per_stage: Union[Tuple[int, ...], List[int]] = None,
            drop_path_rate=0.,
            depths=None,
            img_size=384,
            channel_first=True,
    ):
        super().__init__()
        encoder_output_channels = features_per_stage
        self.deep_supervision = deep_supervision
        n_stages_encoder = len(encoder_output_channels)

        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (n_stages_encoder - 1) * 2)]
        depths = [2, 2, 2, 2] if depths is None else depths
        # input_resolution = [img_size // 2 ** len(depths), img_size // 2 ** len(depths)]   

        norm_layer = LayerNorm2d
        self.channel_first = channel_first

        self.stage_layers = nn.ModuleList()
        self.expand_layers = nn.ModuleList()
        # self.guide_layers = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for stage in range(1, n_stages_encoder):
            input_features_below = encoder_output_channels[-stage]
            input_features_skip = encoder_output_channels[-(stage + 1)]
            self.expand_layers.append(PatchExpand(
                dim=input_features_below,
                dim_scale=2,
                norm_layer=norm_layer,
                channel_first=self.channel_first
            ))
            self.stage_layers.append(self._make_layer(
                dims=input_features_skip,
                drop_path=dpr[sum(depths[:stage - 1]):sum(depths[:stage])],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ))
            self.seg_layers.append(nn.Conv2d(input_features_skip, 1, 1, 1, 0, bias=True))

            Linear = Linear2d if self.channel_first else nn.Linear
            self.concat_back_dim.append(Linear(2 * input_features_skip, input_features_skip))

        # for final prediction
        self.expand_layers.append(FinalPatchExpand_X4(
            dim=encoder_output_channels[0],
            dim_scale=4,
            norm_layer=norm_layer,
            channel_first=self.channel_first,
        ))
        self.stage_layers.append(nn.Identity())
        # self.seg_layers.append(nn.Conv2d(input_features_skip, 1, 1, 1, 0, bias=True))
        self.seg_layers.append(nn.Conv2d(encoder_output_channels[0], 1, 1, 1, 0, bias=True))
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or \
                isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
            m.weight = nn.init.kaiming_normal_(m.weight, a=1e-2)
            if m.bias is not None:
                m.bias = nn.init.constant_(m.bias, 0)
    @staticmethod
    def _make_layer(
            dims=128,
            drop_path=[0.1, 0.1],
            norm_layer=LayerNorm2d,
            channel_first=True,
            **kwargs,
    ):
        depth = len(drop_path)
    
        blocks = nn.ModuleList([
            BaseDecoderBlock(
                hidden_dim=dims,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first
            )
            for d in range(depth)
        ])
    
        return blocks

    def forward(self, skips, prototypes=None):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stage_layers)):
            x = self.expand_layers[s](lres_input)
            if s < (len(self.stage_layers) - 1):
                mid = skips[-(s + 2)]

                if self.channel_first:
                    x = torch.cat((x, mid), 1)
                else:
                    x = torch.cat((x, mid.permute(0, 2, 3, 1).contiguous()), -1)
                x = self.concat_back_dim[s](x)
            if s < len(self.stage_layers) - 1:
                prototype_s = prototypes[s]
            
                for block in self.stage_layers[s]:
                    x = block(x, prototype=prototype_s)
            else:
                x = self.stage_layers[s](x)

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stage_layers) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x
        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

class BaseUMamba(nn.Module):
    def __init__(
        self,
        vss_args,
        decoder_args,
        use_pretrain=True,
        pretrained_path='',
    ):

        super().__init__()

        dims = decoder_args["features_per_stage"]   # [128, 256, 512, 1024]

        self.encoder_proto_proj = nn.ModuleList([
            nn.Linear(512, dims[i] * 2)
            for i in range(len(dims))
        ])
        self.decoder_proto_proj = nn.ModuleList([
            nn.Linear(512, dims[2] * 2),   # 512 -> 1024
            nn.Linear(512, dims[1] * 2),   # 512 -> 512
            nn.Linear(512, dims[0] * 2),   # 512 -> 256
        ])

        self.vssm_encoder = VSSMEncoder(**vss_args)
        self.decoder = VSSMDecoder(**decoder_args)

        self.prototype_query_modules = nn.ModuleList([
            EPI(
                in_channels=dims[i],
                prototype_dim=512,
                num_classes=6
            )
            for i in range(0, len(dims))
        ])

        self.risk_modules = nn.ModuleList([
            DSPM(
                channels=dims[i],
                sigma_x=0.7,
                sigma_y=0.8,
                reduction=4
            )
            for i in range(len(dims))
        ])

        if use_pretrain:
            load_pretrained_Base(self.vssm_encoder,
                                 ckpt_path=pretrained_path)

    def forward(self, x):
        # skips = self.vssm_encoder(x)
        encoder_prototypes = []

        for i in range(len(self.prototype_query_modules)):
            raw_proto = self.prototype_query_modules[i].visual_prototypes
            # [6, 512]
        
            proto_i = self.encoder_proto_proj[i](raw_proto)
            # [6, d_inner_i]
        
            encoder_prototypes.append(proto_i)
        
        skips = self.vssm_encoder(x,prototypes=encoder_prototypes)

        fused_skips = [skips[0]]
    
        query_responses_all = []
        risk_maps_all = []
    
        for i in range(1, len(skips)):
            feat = skips[i]
            idx = i - 1
            feat, query_response = self.prototype_query_modules[idx](feat)
            feat, risk_map = self.risk_modules[idx](feat)
            fused_skips.append(feat)
            query_responses_all.append(query_response)
            risk_maps_all.append(risk_map)
        decoder_prototypes = [
            self.decoder_proto_proj[0](
                self.prototype_query_modules[2].visual_prototypes
            ),
            self.decoder_proto_proj[1](
                self.prototype_query_modules[1].visual_prototypes
            ),
            self.decoder_proto_proj[2](
                self.prototype_query_modules[0].visual_prototypes
            ),
        ]
        out = self.decoder(
            fused_skips,
            prototypes=decoder_prototypes
        )
        
        return out



    @torch.no_grad()
    def freeze_encoder(self):
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True


def bulid_model(
        deep_supervision: bool = True,
        use_pretrain: bool = True,
        img_size: int = 384,
        dims=128,
        depths=[2, 2, 2, 2],
        pretrained_path='',
):

    vss_args = dict(
        patch_size=4,
        in_chans=3,
        depths=[2, 2, 15, 2],
        dims=dims,
        # =========================
        drop_path_rate=0.6,
        patch_norm=True,
        norm_layer="LN2D",  # "BN", "LN2D"
        # =========================
        posembed=False,
        imgsize=img_size,
    )

    decoder_args = dict(
        deep_supervision=deep_supervision,
        features_per_stage=[dims, dims * 2, dims * 4, dims * 8],
        depths=depths,
        img_size=img_size,
        drop_path_rate=0.2,
    )

    model = BaseUMamba(
        vss_args,
        decoder_args,
        use_pretrain=use_pretrain,
        pretrained_path=pretrained_path,
    )

    return model


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    model = bulid_model(deep_supervision=True,
                        use_pretrain=False,
                        img_size=384,
                        dims=128,
                        depths=[2, 2, 2, 2]
                        ).cuda()
    model.eval()
    """
    torch.Size([4, 3, 384, 384])
    torch.Size([4, 128, 96, 96])
    torch.Size([4, 256, 48, 48])
    torch.Size([4, 512, 24, 24])
    torch.Size([4, 1024, 12, 12])
    """
    high_res_input_size = (1, 3, 384, 384)
    dummy_input = torch.randn(high_res_input_size).cuda()


    

    print("Warming up...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input,)


    print("Starting timing...")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 500 
    timings = torch.zeros((repetitions,)).cuda()

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input,)

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) 
            timings[rep] = curr_time

    mean_syn = torch.mean(timings).item()
    std_syn = torch.std(timings).item()
    fps = 1000.0 / mean_syn

    print(f"Input shape: {high_res_input_size}")
    print(f"Average latency: {mean_syn:.3f} ms")
    print(f"Standard deviation: {std_syn:.3f} ms")
    print(f"Inference speed: {fps:.2f} FPS")
    # input = torch.randn(1, 3, 384, 384).cuda()
    # out = model(input)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters in the model: {str(total_params / 1000 ** 2)}")
    # flops = FlopCountAnalysis(model, input)
    # print(flops.total() / 1e9)
