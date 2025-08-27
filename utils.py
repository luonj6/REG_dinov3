import os
from torchvision.datasets.utils import download_url
import torch
import torchvision.models as torchvision_models
import timm
from models import mocov3_vit
import math
import warnings


# code from SiT repository
pretrained_models = {'last.pt'}

def download_model(model_name):
    """
    Downloads a pre-trained SiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://www.dl.dropboxusercontent.com/scl/fi/cxedbs4da5ugjq5wg3zrg/last.pt?rlkey=8otgrdkno0nd89po3dpwngwcc&st=apcc645o&dl=0'
        download_url(web_path, 'pretrained_models', filename=model_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model

def fix_mocov3_state_dict(state_dict):
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder'):
            # fix naming bug in checkpoint
            new_k = k[len("module.base_encoder."):]
            if "blocks.13.norm13" in new_k:
                new_k = new_k.replace("norm13", "norm1")
            if "blocks.13.mlp.fc13" in k:
                new_k = new_k.replace("fc13", "fc1")
            if "blocks.14.norm14" in k:
                new_k = new_k.replace("norm14", "norm2")
            if "blocks.14.mlp.fc14" in k:
                new_k = new_k.replace("fc14", "fc2")
            # remove prefix
            if 'head' not in new_k and new_k.split('.')[0] != 'fc':
                state_dict[new_k] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    if 'pos_embed' in state_dict.keys():
        state_dict['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
            state_dict['pos_embed'], [16, 16],
        )
    return state_dict

@torch.no_grad()
def load_encoders(enc_type, device, resolution=256):
    assert (resolution == 256) or (resolution == 512)
    
    enc_names = enc_type.split(',')
    encoders, architectures, encoder_types = [], [], []
    for enc_name in enc_names:
        encoder_type, architecture, model_config = enc_name.split('-')
        # Currently, we only support 512x512 experiments with DINOv2 encoders.
        if resolution == 512:
            if encoder_type != 'dinov2':
                raise NotImplementedError(
                    "Currently, we only support 512x512 experiments with DINOv2 encoders."
                    )

        architectures.append(architecture)
        encoder_types.append(encoder_type)
        if encoder_type == 'mocov3':
            if architecture == 'vit':
                if model_config == 's':
                    encoder = mocov3_vit.vit_small()
                elif model_config == 'b':
                    encoder = mocov3_vit.vit_base()
                elif model_config == 'l':
                    encoder = mocov3_vit.vit_large()
                ckpt = torch.load(f'./ckpts/mocov3_vit{model_config}.pth')
                state_dict = fix_mocov3_state_dict(ckpt['state_dict'])
                del encoder.head
                encoder.load_state_dict(state_dict, strict=True)
                encoder.head = torch.nn.Identity()
            elif architecture == 'resnet':
                raise NotImplementedError()
 
            encoder = encoder.to(device)
            encoder.eval()

        elif 'dinov2' in encoder_type:
            import timm
            if 'reg' in encoder_type:
                try:
                    # encoder = torch.hub.load('your_path/.cache/torch/hub/facebookresearch_dinov2_main',
                    #                         f'dinov2_vit{model_config}14_reg', source='local')
                    encoder = torch.hub.load('/home/zhangjunhao/.cache/torch/hub/facebookresearch_dinov2_main',
                                            f'dinov2_vit{model_config}14_reg', source='local')
                except:
                    encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
            else:
                try:
                    # encoder = torch.hub.load('your_path/.cache/torch/hub/facebookresearch_dinov2_main',
                    #                          f'dinov2_vit{model_config}14', source='local')
                    encoder = torch.hub.load('/home/zhangjunhao/.cache/torch/hub/facebookresearch_dinov2_main',
                                             f'dinov2_vit{model_config}14', source='local')
                except:
                    encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')

            print(f"Now you are using the {enc_name} as the aligning model")
            del encoder.head
            patch_resolution = 16 * (resolution // 256)
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            encoder = encoder.to(device)
            encoder.eval()

        # elif 'dinov3' in encoder_type:
        #     import timm
        #     if 'reg' in encoder_type:
        #         try:
        #             # encoder = torch.hub.load('your_path/.cache/torch/hub/facebookresearch_dinov2_main',
        #             #                         f'dinov2_vit{model_config}14_reg', source='local')
        #             encoder = torch.hub.load('/home/zhangjunhao/.cache/torch/hub/facebookresearch_dinov3_main',
        #                                     f'dinov3_vit{model_config}16_reg', source='local')
        #         except:
        #             encoder = torch.hub.load('facebookresearch/dinov3', f'dinov3_vit{model_config}16_reg')
        #     else:
        #         try:
        #             # encoder = torch.hub.load('your_path/.cache/torch/hub/facebookresearch_dinov2_main',
        #             #                          f'dinov2_vit{model_config}14', source='local')
        #             encoder = torch.hub.load('/home/zhangjunhao/.cache/torch/hub/facebookresearch_dinov3_main',
        #                                      f'dinov3_vit{model_config}16', source='local')
        #         except:
        #             encoder = torch.hub.load('facebookresearch/dinov3', f'dinov2_vit{model_config}16')

        #     print(f"Now you are using the {enc_name} as the aligning model")
        #     del encoder.head
        #     patch_resolution = 16 * (resolution // 256)
        #     encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
        #         encoder.pos_embed.data, [patch_resolution, patch_resolution],
        #     )
        #     encoder.head = torch.nn.Identity()
        #     encoder = encoder.to(device)
        #     encoder.eval()

        #使用github项目来加载dinov3
        elif 'dinov3' in encoder_type:
            import timm
            if 'reg' in encoder_type:
                try:
                    # encoder = torch.hub.load('your_path/.cache/torch/hub/facebookresearch_dinov2_main',
                    #                         f'dinov2_vit{model_config}14_reg', source='local')
                    encoder = torch.hub.load('/home/zhangjunhao/.cache/torch/hub/facebookresearch_dinov3_main',
                                            f'dinov3_vit{model_config}16_reg', source='local')
                except:
                    encoder = torch.hub.load('facebookresearch/dinov3', f'dinov3_vit{model_config}16_reg', weights = f"https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoienI4bzBkbmowN28ycHpnZXdpZml3Zms5IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTY0NTYxNTd9fX1dfQ__&Signature=bZH78U0fqnuLedfq7EsJjTzwNdGowmn1XWmKm3U2TnqCsudCEPcZPd54HVkzCFtg8KQJcBVHKb9-VQi%7E1i9rkk7UPp9loiGNxFHpqgSvTs15bXoBrUYwxC61Uzs50Z6AfRl85H6NO%7EVo%7ENZoylVPaj-WCxmasnyOVcX616kjkMnASZ4Qz825dqprfSEAMQIK0a8D7TTIgNTOzfRu4Mq1KkUVgCi3%7EGSoEHAhun83Qfp1Q1XzCfei4S8GUqtkyNrwYF2IinAhb2DmQgsOQcO-Jr2BeOFreoSWS1R2C8vZpOI%7EejdBwzCf5W-sGiOLDtyn%7EKbv5BO9XltazisQdAV3LQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=631836603304588")
            else:
                #  encoder = torch.hub.load('/home/zhangjunhao/.cache/torch/hub/facebookresearch_dinov3_main',
                #                              f'dinov3_vit{model_config}16', source='local',weights = f"https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoienI4bzBkbmowN28ycHpnZXdpZml3Zms5IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTY0NTYxNTd9fX1dfQ__&Signature=bZH78U0fqnuLedfq7EsJjTzwNdGowmn1XWmKm3U2TnqCsudCEPcZPd54HVkzCFtg8KQJcBVHKb9-VQi%7E1i9rkk7UPp9loiGNxFHpqgSvTs15bXoBrUYwxC61Uzs50Z6AfRl85H6NO%7EVo%7ENZoylVPaj-WCxmasnyOVcX616kjkMnASZ4Qz825dqprfSEAMQIK0a8D7TTIgNTOzfRu4Mq1KkUVgCi3%7EGSoEHAhun83Qfp1Q1XzCfei4S8GUqtkyNrwYF2IinAhb2DmQgsOQcO-Jr2BeOFreoSWS1R2C8vZpOI%7EejdBwzCf5W-sGiOLDtyn%7EKbv5BO9XltazisQdAV3LQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=631836603304588")
                try:
                    # encoder = torch.hub.load('your_path/.cache/torch/hub/facebookresearch_dinov2_main',
                    #                          f'dinov2_vit{model_config}14', source='local')
                    encoder = torch.hub.load('/home/zhangjunhao/.cache/torch/hub/facebookresearch_dinov3_main',
                                             f'dinov3_vit{model_config}16', source='local',weights = f"https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoienI4bzBkbmowN28ycHpnZXdpZml3Zms5IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTY0NTYxNTd9fX1dfQ__&Signature=bZH78U0fqnuLedfq7EsJjTzwNdGowmn1XWmKm3U2TnqCsudCEPcZPd54HVkzCFtg8KQJcBVHKb9-VQi%7E1i9rkk7UPp9loiGNxFHpqgSvTs15bXoBrUYwxC61Uzs50Z6AfRl85H6NO%7EVo%7ENZoylVPaj-WCxmasnyOVcX616kjkMnASZ4Qz825dqprfSEAMQIK0a8D7TTIgNTOzfRu4Mq1KkUVgCi3%7EGSoEHAhun83Qfp1Q1XzCfei4S8GUqtkyNrwYF2IinAhb2DmQgsOQcO-Jr2BeOFreoSWS1R2C8vZpOI%7EejdBwzCf5W-sGiOLDtyn%7EKbv5BO9XltazisQdAV3LQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=631836603304588")
                except:
                    encoder = torch.hub.load('facebookresearch/dinov3', f'dinov3_vit{model_config}16', weights = f"https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoienI4bzBkbmowN28ycHpnZXdpZml3Zms5IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTY0NTYxNTd9fX1dfQ__&Signature=bZH78U0fqnuLedfq7EsJjTzwNdGowmn1XWmKm3U2TnqCsudCEPcZPd54HVkzCFtg8KQJcBVHKb9-VQi%7E1i9rkk7UPp9loiGNxFHpqgSvTs15bXoBrUYwxC61Uzs50Z6AfRl85H6NO%7EVo%7ENZoylVPaj-WCxmasnyOVcX616kjkMnASZ4Qz825dqprfSEAMQIK0a8D7TTIgNTOzfRu4Mq1KkUVgCi3%7EGSoEHAhun83Qfp1Q1XzCfei4S8GUqtkyNrwYF2IinAhb2DmQgsOQcO-Jr2BeOFreoSWS1R2C8vZpOI%7EejdBwzCf5W-sGiOLDtyn%7EKbv5BO9XltazisQdAV3LQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=631836603304588")
                    # encoder = torch.hub.load('/home/zhangjunhao/.cache/torch/hub/facebookresearch_dinov3_main', f'dinov3_vit{model_config}16', weights = f"https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoienI4bzBkbmowN28ycHpnZXdpZml3Zms5IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTY0NTYxNTd9fX1dfQ__&Signature=bZH78U0fqnuLedfq7EsJjTzwNdGowmn1XWmKm3U2TnqCsudCEPcZPd54HVkzCFtg8KQJcBVHKb9-VQi%7E1i9rkk7UPp9loiGNxFHpqgSvTs15bXoBrUYwxC61Uzs50Z6AfRl85H6NO%7EVo%7ENZoylVPaj-WCxmasnyOVcX616kjkMnASZ4Qz825dqprfSEAMQIK0a8D7TTIgNTOzfRu4Mq1KkUVgCi3%7EGSoEHAhun83Qfp1Q1XzCfei4S8GUqtkyNrwYF2IinAhb2DmQgsOQcO-Jr2BeOFreoSWS1R2C8vZpOI%7EejdBwzCf5W-sGiOLDtyn%7EKbv5BO9XltazisQdAV3LQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=631836603304588")
            print(f"Now you are using the {enc_name} as the aligning model")
            del encoder.head
            patch_resolution = 16 * (resolution // 256)
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            encoder = encoder.to(device)
            encoder.eval()



        # #使用huggingface从本地权重加载dinov3
        # elif 'dinov3' in encoder_type:
        #     from transformers import AutoModel  # 使用Hugging Face的AutoModel
        #     import timm  # 用于pos_embed调整

        #     # 映射model_config到DINOv3 ViT模型，假设model_config='b'对应vitb16
        #     # model_name = "facebook/dinov3-vit{model_config}16-pretrain-lvd1689m"
        #     # base_name = "dinov3-vit{model_config}16-pretrain-lvd1689m"
        #     model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        #     base_name = "dinov3-vitb16-pretrain-lvd1689m"

        #     # 如果已用huggingface-cli login设置了token，可忽略hf_token
        #     hf_token = "your huggingface access_token"

        #     if 'reg' in encoder_type:
        #         try:
        #             # 如果模型已下载到本地
        #             encoder = AutoModel.from_pretrained(
        #                 '/mnt/mydisk/zhangjunhao/REG/dinov3_models/dinov3-vitb16-pretrain-lvd1689m',
        #                 torch_dtype=torch.float16,  # 低精度节省内存
        #                 device_map="auto"  # 自动分配GPU/CPU
        #             )
        #         except:
        #             # 自动从Hugging Face下载
        #             encoder = AutoModel.from_pretrained(
        #                 model_name,
        #                 token=hf_token,
        #                 torch_dtype=torch.float16,
        #                 device_map="auto"
        #             )
        #     else:
        #         try:
        #             # 如果模型已下载到本地
        #             encoder = AutoModel.from_pretrained(
        #                 '/mnt/mydisk/zhangjunhao/REG/dinov3_models/dinov3-vitb16-pretrain-lvd1689m',
        #                 torch_dtype=torch.float16,  # 低精度节省内存
        #                 device_map="auto"  # 自动分配GPU/CPU
        #             )
        #         except:
        #             # 自动从Hugging Face下载
        #             encoder = AutoModel.from_pretrained(
        #                 model_name,
        #                 token=hf_token,
        #                 torch_dtype=torch.float16,
        #                 device_map="auto"
        #             )

        #     print(f"Now you are using the {enc_name} as the aligning model")
        #     # del encoder.head
        #     patch_resolution = 16 * (resolution // 256)
        #     # encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
        #     #     encoder.pos_embed.data, [patch_resolution, patch_resolution],
        #     # )

        #     if hasattr(encoder, 'pos_embed'):
        #         encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
        #         encoder.pos_embed.data, [patch_resolution, patch_resolution],
        #         )
        #     else:
        #         print("DINOv3 uses RoPE, skipping pos_embed resampling.")

        #     encoder.head = torch.nn.Identity()
        #     encoder = encoder.to(device)
        #     encoder.eval()



            # # 处理pooler（DINOv3 ViT没有head，但可能有pooler）
            # if hasattr(encoder, 'pooler'):
            #     del encoder.pooler
            # encoder.pooler = torch.nn.Identity()  # 确保输出patch embeddings


            # # 调整pos_embed（DINOv3 ViT的patch_size=16）
            # patch_resolution = resolution // 16  # DINOv3 ViT patch_size=16
            # pos_embed = encoder.embeddings.position_embeddings.data  # ViT的pos_embed路径
            # cls_token_pos = pos_embed[:, :1, :]  # CLS token（如果有）
            # patch_pos = pos_embed[:, 1:, :]  # patch位置嵌入
            # new_patch_pos = timm.layers.pos_embed.resample_abs_pos_embed(
            #     patch_pos.unsqueeze(0),  # 加batch维度
            #     new_size=(patch_resolution, patch_resolution)  # 新网格尺寸
            # ).squeeze(0)
            # encoder.embeddings.position_embeddings.data = torch.cat([cls_token_pos, new_patch_pos], dim=1)

            # encoder = encoder.to(device)
            # encoder.eval()

        
        elif 'dinov1' == encoder_type:
            import timm
            from models import dinov1
            encoder = dinov1.vit_base()
            ckpt =  torch.load(f'./ckpts/dinov1_vit{model_config}.pth') 
            if 'pos_embed' in ckpt.keys():
                ckpt['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                    ckpt['pos_embed'], [16, 16],
                )
            del encoder.head
            encoder.head = torch.nn.Identity()
            encoder.load_state_dict(ckpt, strict=True)
            encoder = encoder.to(device)
            encoder.forward_features = encoder.forward
            encoder.eval()

        elif encoder_type == 'clip':
            import clip
            from models.clip_vit import UpdatedVisionTransformer
            encoder_ = clip.load(f"ViT-{model_config}/14", device='cpu')[0].visual
            encoder = UpdatedVisionTransformer(encoder_).to(device)
             #.to(device)
            encoder.embed_dim = encoder.model.transformer.width
            encoder.forward_features = encoder.forward
            encoder.eval()
        
        elif encoder_type == 'mae':
            from models.mae_vit import vit_large_patch16
            import timm
            kwargs = dict(img_size=256)
            encoder = vit_large_patch16(**kwargs).to(device)
            with open(f"ckpts/mae_vit{model_config}.pth", "rb") as f:
                state_dict = torch.load(f)
            if 'pos_embed' in state_dict["model"].keys():
                state_dict["model"]['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                    state_dict["model"]['pos_embed'], [16, 16],
                )
            encoder.load_state_dict(state_dict["model"])

            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [16, 16],
            )

        elif encoder_type == 'jepa':
            from models.jepa import vit_huge
            kwargs = dict(img_size=[224, 224], patch_size=14)
            encoder = vit_huge(**kwargs).to(device)
            with open(f"ckpts/ijepa_vit{model_config}.pth", "rb") as f:
                state_dict = torch.load(f, map_location=device)
            new_state_dict = dict()
            for key, value in state_dict['encoder'].items():
                new_state_dict[key[7:]] = value
            encoder.load_state_dict(new_state_dict)
            encoder.forward_features = encoder.forward

        encoders.append(encoder)
    
    return encoders, encoder_types, architectures


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def load_legacy_checkpoints(state_dict, encoder_depth):
    new_state_dict = dict()
    for key, value in state_dict.items():
        if 'decoder_blocks' in key:
            parts =key.split('.')
            new_idx = int(parts[1]) + encoder_depth
            parts[0] = 'blocks'
            parts[1] = str(new_idx)
            new_key = '.'.join(parts)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict