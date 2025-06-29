# STYLO-Pipeline: Semantic Transformation of your looks and outfits
Application of Vision Foundation Models for image editing and Virtual Try-On (VITON).


## Pipeline Architecture
![Image of Pipeline Architecture](./assets/pipeline_architecture.svg "Illustration of pipeline architecture")


## Pipeline Components
We utilize several publicly available image editing models like:
- [x] Background Manipulation: Yahoo's diffusion model [photo-background-generation](https://github.com/yahoo/photo-background-generation.git) 
- [x] Fix Lighting: High-Resolution Image [Harmonizer](https://github.com/ZHKKKe/Harmonizer/) 
- [x] Person Selection: Meta's semantic segmentation model [sam2](https://github.com/facebookresearch/sam2), Ultralytics: [YOLO-11](https://github.com/ultralytics/ultralytics)
- [x] Garment Generator: Stable Diffusion 3.5: [sd3.5](https://github.com/Stability-AI/sd3.5)
- [x] Garment Prompt Generation: LLaVA combines language understanding with vision capabilities [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf) 
- [x] Fit Garment: KAIST Research Group, South Korea VITON model [StableVITON](https://github.com/rlawjdghek/StableVITON)


## Setup
### Install environment
```sh
# setup environment and install local dependencies
./let_it_coock.sh

conda activate stylo2
conda env update -f environment.yaml
```

### Known Issues
- **Fix Foreground BF16 Error**
<br>Replace line 163 in Remover: pred = pred.float().numpy().squeeze()


### Add Foundation Model Repositories to this Codebase
1. Fork repository of foundation model
2. Clone forked repository under `building_blocks`
2. Add original repository URL to `.gitsubmodules` file

```
[submodule "building_blocks/StableVITON"]
	path = building_blocks/StableVITON
	url = https://github.com/rlawjdghek/StableVITON.git
```

### Add Model Files to for each Foundation Model
Most of the model weights are available on huggingface.


#### Stable-Diffusion 3.5
```sh
huggingface-cli login
huggingface-cli download stabilityai/stable-diffusion-3.5-medium --local-dir building_blocks/sd3_5/models/3_5medium
```

### Python Environent
```
conda env update -f environment.yaml
conda activate stylo-pipeline
```

# Slurm Usage
1. go in stylo-pipeline root dir
2. allocate hardware. Check availability at https://adm9.in.tum.de/slurm with I=Idle T=total and U=used
```
srun --gpus=1 --mem=64G --gres=gpu:1,VRAM:24G --cpus-per-task=8 --time=5:00:00 --pty zsh
```
3. run startup script 
```
./run_server.sh <username> <forward_machine> <ssh key file>
```
Example:
```
./run_server.sh tom computer1.tum.de ~/.ssh/key_file
```
In case you have no ssh key file put a none existent file down and type your password
(4. You may have to fix LangSam cuda device selcection or conda init)
