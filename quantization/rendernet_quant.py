#%%
from pytorch_nndct.apis import torch_quantizer
import torch
import torch.nn as nn
from timm.data import create_dataset,create_loader
from rendernet_amd_block import RenderNet_patch8_256_tiny
import argparse
from tqdm import tqdm

def evaluate(model,tokenizer, val_loader, loss_fn):

  model.eval()
  Loss = 0
  total=0
  for iteraction, (images, labels) in tqdm(
      enumerate(val_loader), total=len(val_loader)):
    #pdb.set_trace()
    tokens=tokenizer(images)
    outputs = model(tokens)
    outputs=torch.clamp(outputs,-1,1)
    loss = loss_fn(outputs, images)
    Loss += loss.item()
    total += images.size(0)
  return Loss / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--checkpoint',
        default="rendernet_tiny_amd_patch8_ft.pth",
        help='the path to the float checkpoint ')
    parser.add_argument('--dataset-path', 
        default='/vitis_ai_home/minitrain3',
        help='the path to dataset folder')
    parser.add_argument('--output-path', 
        default='',
        help='the path to save the results of quantization')
    parser.add_argument('--img-size', 
        default=256,type=int,
        help='the original shape of images')
    
    args, _ = parser.parse_known_args()
    # prepare the model
    ckpt=torch.load(args.checkpoint)
    model=RenderNet_patch8_256_tiny()
    model.load_state_dict(ckpt['state_dict'],strict=False)
    model.eval()
    model.fuse_conv_bn()
    print(model)

    # prepare the dataset and dataloader
    input=torch.randn(1,256,args.img_size//8,args.img_size//8)
    dataset = create_dataset(
            "ImageFolder",
            args.dataset_path,
            is_training=False,
            batch_size=1,
            split="",
        )
    loader = create_loader(
            dataset,
            (3,args.img_size,args.img_size),
            8,
            False,
            use_prefetcher=True,
            no_aug=True,
            mean=(0.5, 0.5, 0.5),
            std=(1, 1, 1),
            num_workers=4,device=torch.device('cpu'))
    
    # quantization the model
    quantizer=torch_quantizer('calib',model.feature,(input,),output_dir=args.output_path)
    quant_model=quantizer.quant_model
    loss_fn=nn.L1Loss()
    quantizer.fast_finetune(evaluate,(quant_model,model.tokenizer,loader,loss_fn))
    loss=evaluate(quant_model,model.tokenizer, loader,loss_fn)
    print(loss)
    quantizer.export_quant_config()
    quantizer.export_torch_script(output_dir=args.output_path)
    quantizer.export_onnx_model(output_dir=args.output_path)
    quantizer.export_xmodel(output_dir=args.output_path)