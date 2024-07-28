# Test the float models
Get the reconstructed images from original images
```
python .\rendernet_amd_block.py --input-file .\ILSVRC2012_val_00045087.JPEG --output-file reconstructed_image.png --mode encode_render 
```
Get the patches' categories from original images
```
python .\rendernet_amd_block.py --input-file .\ILSVRC2012_val_00045087.JPEG --output-file test_img_patch_cls.pth --mode encode --patch-cls
```
Get the patches' embeddings from original images
```
python .\rendernet_amd_block.py --input-file .\ILSVRC2012_val_00045087.JPEG --output-file test_img_patch_embed.pth --mode encode
```
Get the reconstructed images from the patch embeddings
```
python .\rendernet_amd_block.py --input-file .\test_img_patch_embed.pth --output-file reconstructed_image.png --mode render
```
# Test the quantization models with onnx session
Get the reconstructed images from original images
```
python .\rendernet_onnx.py --input-file .\ILSVRC2012_val_00045087.JPEG --output-file reconstructed_image.png --mode encode_render --checkpoint quantization\quantization_results_512\Sequential_int.onnx --img 512
```
Get the patches' categories from original images
```
python .\rendernet_onnx.py --input-file .\ILSVRC2012_val_00045087.JPEG --output-file test_img_patch_cls.pth --mode encode --patch-cls --checkpoint quantization\quantization_results_512\Sequential_int.onnx --img 512
```
Get the patches' embeddings from original images
```
python .\rendernet_onnx.py --input-file .\ILSVRC2012_val_00045087.JPEG --output-file test_img_patch_embed.pth --mode encode --checkpoint quantization\quantization_results_512\Sequential_int.onnx --img 512
```
Get the reconstructed images from the patch embeddings
```
python .\rendernet_onnx.py --input-file .\test_img_patch_embed.pth --output-file reconstructed_image.png --mode render --checkpoint quantization\quantization_results_512\Sequential_int.onnx --img 512
```
## Test the benchmark with 256*256 resolution
On NPU device
```
python .\rendernet_onnx.py --mode benchmark --checkpoint quantization\quantization_results_256\Sequential_int.onnx --target NPU --img-size 256
```
You can get the results with 'Average inference time per run: 0.0866 seconds on NPU device'

On CPU device
```
python .\rendernet_onnx.py --mode benchmark --checkpoint quantization\quantization_results_256\Sequential_int.onnx --target CPU --img-size 256
```
You can get the results with 'Average inference time per run: 0.0974 seconds on CPU device'
## Test the benchmark with 512*512 resolution
On NPU device
```
python .\rendernet_onnx.py --mode benchmark --checkpoint quantization\quantization_results_512\Sequential_int.onnx --target NPU --img-size 512
```
You can get the results with 'Average inference time per run: 0.2966 seconds on NPU device'

On CPU device
```
python .\rendernet_onnx.py --mode benchmark --checkpoint quantization\quantization_results_512\Sequential_int.onnx --target CPU --img-size 512
```
You can get the results with 'Average inference time per run: 0.3707 seconds on CPU device'