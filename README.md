# STYLO-Pipeline: Semantic Transformation of your looks and outfits
Application of Vision Foundation Models for image editing and Virtual Try-On (VITON).


## Pipeline Architecture
![Image of Pipeline Architecture](./assets/pipeline_architecture.svg "Illustration of pipeline architecture")


## Pipeline Components
We utilize several publicly available image editing models like:
- [x] Background Manipulation: Yahoo's [photo-background-generation](https://github.com/yahoo/photo-background-generation.git) diffusion model
- [ ] Fix Lighting:
- [ ] Person Selection: Meta's [https://github.com/facebookresearch/sam2](sam2) for semantic segmentation
- [x] Fit Garment: KAIST Research Group, South Korea [StableVITON](https://github.com/rlawjdghek/StableVITON)
