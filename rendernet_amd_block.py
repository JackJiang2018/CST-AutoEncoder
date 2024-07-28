# %%
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.registry import register_model
import argparse
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

class CSTEncoder(nn.Module):
    def __init__(
        self,
        patch_size=16,
        cluster_num=[512, 2048, 8192],
        embed_dim=768,
    ):
        super().__init__()
        self.cluster_num=cluster_num
        self.register_buffer("cluster_center0", torch.randn(cluster_num[0], 3*patch_size**2))
        self.register_buffer("cluster_center1", torch.randn(cluster_num[1], 3*patch_size**2))
        self.register_buffer("cluster_center2", torch.randn(cluster_num[2], 3*patch_size**2))
        self.embedding_list = nn.ModuleList(
            [nn.Embedding(cluster_num[i], embed_dim) for i in range(3)]
        )
        self.patch_size = patch_size

    @torch.no_grad()
    def ClassPatch(self, data, center):
        patch_cls = torch.cdist(data, center).argmin(dim=-1)
        patch_img = center[patch_cls]
        patch_diff = data - patch_img
        return (patch_cls, patch_diff, patch_img)

    @torch.no_grad()
    def ConstructImage(self, data):
        patch_list = []
        for i in range(len(self.model_list)):
            with torch.no_grad():
                if i == 0:
                    patch_cls, patch_diff = self.model_list[i](data)

                else:
                    patch_cls, patch_diff = self.model_list[i](patch_diff)
                patch = self.model_list[i].cluster_center[patch_cls]
            patch_list.append(patch)
        return sum(patch_list)
    
    def GetEmbedByCls(self,patch_cls):
        # patch_embed=[ for i in patch_cls]
        assert patch_cls[0].max() < self.cluster_num[0], f'the color class number must small than {self.cluster_num[0]}'
        patch_embed0=self.cluster_center0[patch_cls[0]]
        assert patch_cls[1].max() < self.cluster_num[1], f'the shape class number must small than {self.cluster_num[1]}'
        patch_embed1=self.cluster_center1[patch_cls[1]]
        assert patch_cls[2].max() < self.cluster_num[2], f'the texture class number must small than {self.cluster_num[2]}'
        patch_embed2=self.cluster_center2[patch_cls[2]]
        return sum([patch_embed0,patch_embed1,patch_embed2])

    def forward(self, data, combine=True, return_patch_cls=False):
        b, c, h, w = data.shape

        with torch.no_grad():
            # data = rearrange(data, "b c h w->b (h w) c")
            data = rearrange(
                data,
                "b c (h hp) (w wp)->b (h w) (c hp wp)",
                hp=self.patch_size,
                wp=self.patch_size,
            )
            patch_cls0, patch_diff, patch_img0 = self.ClassPatch(
                data, self.cluster_center0
            )
            patch_cls1, patch_diff, patch_img1 = self.ClassPatch(
                patch_diff, self.cluster_center1
            )
            patch_cls2, patch_diff, patch_img2 = self.ClassPatch(
                patch_diff, self.cluster_center2
            )
            patch_img = [patch_img0, patch_img1, patch_img2]
            patch_img = [
                rearrange(
                    i,
                    "b (h w) (c hp wp)->b c (h hp) (w wp)",
                    h=h // self.patch_size,
                    hp=self.patch_size,
                    wp=self.patch_size,
                )
                for i in patch_img
            ]
            patch_diff = rearrange(
                patch_diff, "b (h w) c->b c h w", h=h // self.patch_size
            )
        patch_embed = [
            self.embedding_list[0](patch_cls0),
            self.embedding_list[1](patch_cls1),
            self.embedding_list[2](patch_cls2),
        ]
        patch_embed = [
            rearrange(i, "b (h w) c->b c h w", h=h // self.patch_size)
            for i in patch_embed
        ]
        if return_patch_cls:
            return [patch_cls0, patch_cls1, patch_cls2]
        if combine:
            return sum(patch_embed), sum(patch_img), patch_diff
        else:
            return patch_embed, patch_img, patch_diff



class AMDBlock(nn.Module):
    def __init__(self, dim, expand_ratio, kernel_size, act_layer):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(dim, dim * expand_ratio, kernel_size, 1, 1, bias=False),
            nn.BatchNorm2d(dim * expand_ratio),
            act_layer(),
            nn.Conv2d(dim * expand_ratio, dim, 1, 1, 0),
        )
        self.feature2 = nn.Sequential(
            nn.Conv2d(dim, dim * expand_ratio, kernel_size, 1, 1, bias=False),
            nn.BatchNorm2d(dim * expand_ratio),
            act_layer(),
            nn.Conv2d(dim * expand_ratio, dim, 1, 1, 0),
        )


    def fuse_conv_bn(self):
        self.feature = self.fuse_conv_bn_layer(self.feature)
        self.feature2 = self.fuse_conv_bn_layer(self.feature2)

    def fuse_conv_bn_layer(self, feature):
        with torch.no_grad():
            model = feature[1]
            conv = feature[0]
            gamma = model.weight
            beta = model.bias
            var_std = torch.sqrt(model.running_var + model.eps)
            mean = model.running_mean
            weight = conv.weight
            new_w = weight * (gamma / var_std).reshape(-1, 1, 1, 1)
            new_b = -gamma * mean / var_std + beta
            # new_b = (weight * (-gamma * mean / var_std + beta)).sum((1, 2, 3)) + bias
        new_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
        )
        new_conv.weight = nn.Parameter(new_w)
        new_conv.bias = nn.Parameter(new_b)
        new_feature = nn.Sequential(new_conv, feature[2], feature[3])
        return new_feature

    def forward(self, x):
        x = x + self.feature(x)
        x = x + self.feature2(x)
        return x

class RenderNet(nn.Module):
    def __init__(
        self,
        dim=768,
        depth=12,
        expand_ratio=4,
        cluster_centers=[512, 2048, 8192],
        act_layer=nn.Hardswish,
        patch_size=16,
        img_size=256,
        kernel_size=3,
        loss="l1",
    ) -> None:
        super().__init__()
        self.tokenizer = CSTEncoder(
            patch_size,
            cluster_centers,
            dim,
        )
        self.feature = nn.Sequential(
            *[
                AMDBlock(dim, expand_ratio, kernel_size, act_layer)
                for i in range(depth)
            ],
            nn.Conv2d(dim, 3 * (patch_size) ** 2, 3, 1, 1),
            nn.PixelShuffle(patch_size),
        )
        self.patch_size = patch_size
        self.feature_size = img_size // patch_size
        self.depth = depth
        self.apply(self._init_weights)

    def fuse_conv_bn(self):
        for i in range(self.depth):
            self.feature[i].fuse_conv_bn()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1)
    def encodeImage(self,x,combine=True, return_patch_cls=False):
        if return_patch_cls:
            patch_cls=self.tokenizer(x)
            return patch_cls
        patch_embed,patch_img,patch_res=self.tokenizer(x,combine)
        return patch_embed,patch_img,patch_res
    def renderImage(self,x):
        x=torch.clamp(self.feature(x),-1,1)*0.5+0.5
        return x

    def forward(self, x):
        patch_embed, patch_img, patch_res = self.tokenizer(x)
        out = torch.clamp(self.feature(patch_embed), -1, 1)*0.5+0.5
        return out, patch_img

@register_model
def RenderNet_patch8_256_tiny(pretrained=False, **kwargs):
    model = RenderNet(256, 6, 4, patch_size=8, act_layer=nn.ReLU)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--checkpoint',
        default="rendernet_tiny_amd_patch8_ft.pth",
        help='the path to the float checkpoint ')
    parser.add_argument('--mode', 
        default='encode_render', 
        choices=['encode', 'render', 'encode_render'], 
        help='encode: get the categories or embeddings of color, shape and texure; render: according the latent embeddings to render the image; encode_render: encode the image and reconstruct the image')
    parser.add_argument('--patch_cls', 
        action='store_true',
        help='get the color, shape, and texture categories of each patch')
    parser.add_argument('--no-combine', 
        action='store_true',
        help='get the separate embeddings of CST')
    parser.add_argument('--input-file', 
        default='',
        help='the path to encode or decode')
    parser.add_argument('--output-file', 
        default='',
        help='the path to save the results of encode or decode')
    args, _ = parser.parse_known_args()
    ckpt=torch.load(args.checkpoint)
    model=RenderNet_patch8_256_tiny()
    model.load_state_dict(ckpt['state_dict'],strict=False)
    model.eval()
    if args.mode=='encode_render':
        img=Image.open(args.input_file)
        trans=transforms.Compose([transforms.Resize(256),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(1,1,1))])
        img_t=trans(img)
        with torch.no_grad():
            out,patch_img=model(img_t)
            plt.imshow(out[0].permute(1,2,0).numpy())
            plt.savefig(args.out_file)
    elif args.mode=='encode':
        img=Image.open(args.input_file)
        trans=transforms.Compose([transforms.Resize(256),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(1,1,1))])
        img_t=trans(img)
        with torch.no_grad():
            if args.patch_cls:
                patch_cls=model.encodeImage(img_t,return_patch_cls=True)
                torch.save(patch_cls,args.out_file)
            elif args.no_combine:
                patch_embed,patch_img,patch_res=model.encodeImage(img_t,False)
                torch.save(patch_embed,args.output_file)
            else:
                patch_embed,patch_img,patch_res=model.encodeImage(img_t,True)
                torch.save(patch_embed,args.output_file)
    elif args.mode == 'render':
        patch_embed=torch.load(args.input_file)
        with torch.no_grad():
            out,patch_img=model.renderImage(patch_embed)
            plt.imshow(out[0].permute(1,2,0).numpy())
            plt.savefig(args.out_file)
    else:
        print(f'there is no mode for {args.mode}')




