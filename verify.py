from custom import VisionTransformer
import torch


custom_config = {
        "img_size": 384,
        "in_chans": 3,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "n_heads": 12,
        "qkv_bias": True,
        "mlp_ratio": 4,
}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

vit = VisionTransformer(**custom_config).to(device)

t1 = torch.randn(5,3,384,384).to(device) #5 samples is the limit

print(vit(t1).shape)



