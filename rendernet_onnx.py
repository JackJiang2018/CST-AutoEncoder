#%%
import onnxruntime
import numpy as np
import torch
from rendernet_amd_block import CSTEncoder
import argparse
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--checkpoint',
        default=r"quantization_results\Sequential_int.onnx",
        help='the path to the quantization onnx ')
    parser.add_argument(
        '--encoder-para',
        default=r"encoder_cst_para.pth",
        help='the path to the encoder parameters ')
    parser.add_argument('--mode', 
        default='encode_render', 
        choices=['encode', 'render', 'encode_render','benchmark'], 
        help='encode: get the categories or embeddings of color, shape and texure; render: according the latent embeddings to render the image; encode_render: encode the image and reconstruct the image')
    parser.add_argument('--target', 
        default='NPU', 
        choices=['CPU', 'NPU'], 
        help='the device to run the onnx model')
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
    parser.add_argument('--img-size', 
        default=256,type=int,
        help='the original shape of images')
    args, _ = parser.parse_known_args()
    # load the encoder
    encoder=CSTEncoder(8,embed_dim=256)
    ckpt=torch.load(args.encoder_para)
    encoder.load_state_dict(ckpt)
    
    # load the decoder
    sess_options=onnxruntime.SessionOptions()
    sess_options.graph_optimization_level=onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.target=='CPU':
        decoder=onnxruntime.InferenceSession(args.checkpoint,sess_options,providers=['CPUExecutionProvider'])
    elif args.target=='NPU':
        decoder = onnxruntime.InferenceSession(
                args.checkpoint,
                sess_options=sess_options,
                providers=["VitisAIExecutionProvider"],
                provider_options=[{"config_file":r"vaip_config.json"}],)
    
    if args.mode=='encode_render':
        img=Image.open(args.input_file)
        trans=transforms.Compose([transforms.Resize((args.img_size,args.img_size)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(1,1,1))])
        img_t=trans(img).unsqueeze(dim=0)
        # print(img_t.shape)
        with torch.no_grad():
            patch_embed, patch_img, patch_res = encoder(img_t)
            ort_inputs={decoder.get_inputs()[0].name:patch_embed.numpy().astype(np.float32)}
            out = np.clip(decoder.run(None,ort_inputs), -1, 1)*0.5+0.5
            plt.imshow(out[0][0].transpose((1,2,0)))
            plt.axis('off')
            plt.savefig(args.output_file)
    elif args.mode=='encode':
        img=Image.open(args.input_file)
        trans=transforms.Compose([transforms.Resize(args.img_size,args.img_size),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(1,1,1))])
        img_t=trans(img).unsqueeze(dim=0)
        with torch.no_grad():
            if args.patch_cls:
                patch_cls=encoder(img_t,return_patch_cls=True)
                torch.save(patch_cls,args.output_file)
            elif args.no_combine:
                patch_embed,patch_img,patch_res=encoder(img_t,False)
                torch.save(patch_embed,args.output_file)
            else:
                patch_embed,patch_img,patch_res=encoder(img_t,True)
                torch.save(patch_embed,args.output_file)
    elif args.mode == 'render':
        patch_embed=torch.load(args.input_file)
        ort_inputs={decoder.get_inputs()[0].name:patch_embed.detach().numpy().astype(np.float32)}
        out = np.clip(decoder.run(None,ort_inputs), -1, 1)*0.5+0.5
        plt.imshow(out[0][0].transpose(1,2,0))
        plt.axis('off')
        plt.savefig(args.output_file)
    elif args.mode == 'benchmark':
        start=time.time()
        for i in range(100):
            inp={decoder.get_inputs()[0].name:np.random.randn(1,256,args.img_size//8,args.img_size//8).astype(np.float32)}
            ort_out=decoder.run(None,inp)
        end=time.time()
        average_time=(end-start)/100
        print(f'Average inference time per run: {average_time:.4f} seconds on {args.target} device')
    else:
        print(f'there is no mode for {args.mode}')