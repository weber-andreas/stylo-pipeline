# STYLO-Pipeline: Semantic Transformation of your looks and outfits
Application of Vision Foundation Models for image editing and Virtual Try-On (VITON).


## Pipeline Architecture
![Image of Pipeline Architecture](./assets/pipeline_architecture.svg "Illustration of pipeline architecture")


## Pipeline Components
We utilize several publicly available image editing models like:
- [x] Background Manipulation: Yahoo's diffusion model [photo-background-generation](https://github.com/yahoo/photo-background-generation.git) 
- [ ] Fix Lighting:
- [ ] Person Selection: Meta's semantic segmentation model [sam2](https://github.com/facebookresearch/sam2), Ultralytics: [YOLO-11](https://github.com/ultralytics/ultralytics)
- [ ] Garment Generator: Stable Diffusion 3.5: [sd3.5](https://github.com/Stability-AI/sd3.5)
- [x] Fit Garment: KAIST Research Group, South Korea VITON model [StableVITON](https://github.com/rlawjdghek/StableVITON)


## Setup

### Additional Foundation Models
1. Fork repository of foundation model
2. Clone forked repository under `building_blocks`
2. Add original repository URL to `.gitsubmodules` file

```
[submodule "building_blocks/StableVITON"]
	path = building_blocks/StableVITON
	url = https://github.com/rlawjdghek/StableVITON.git
```


### Python Environent
```
conda env create -f environment.yaml
conda activate stylo-pipeline
```