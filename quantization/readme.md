# Start the Docker
At first, you should start the wsl2 terminal.
You should run following command in wsl terminal or linux bash.
```
 ./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest
```
# Switch virtual environment and install some packages
```
conda activate vitis-ai-pytorch
pip install timm einops tqdm
```
# download the checkpoint and dataset
download the checkpoint

# Quantizate the model
```
python rendernet_quant.py --checkpoint rendernet_tiny_amd_patch8_ft.pth --dataset-path minitrain3 --output-path quantization_results_256 --img-size 256
```
quantizate the model with 512*512 resolutions
```
python rendernet_quant.py --checkpoint rendernet_tiny_amd_patch8_ft.pth --dataset-path minitrain3 --output-path quantization_results_512 --img-size 256
```