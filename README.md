# ✏️ Edit One for All: Interactive Batch Image Editing

[![arXiv](https://img.shields.io/badge/arXiv-2104.01867-red.svg)](https://arxiv.org/abs/2401.10219) - [![project page](https://img.shields.io/badge/ProjectPage-up-green.svg)](https://thaoshibe.github.io/edit-one-for-all)

Given an edit specified by users in an example image (e.g., dog pose),
Our method can automatically transfer that edit to other test images (e.g., all dog same pose).

![](./images/teaser-gif.gif "teaser")

✏️ [Edit One for All: Interactive Batch Image Editing](https://thaoshibe.github.io/edit-one-for-all/) (CVPR 2024)<br>
By [Thao Nguyen](https://thaoshibe.github.io/), [Utkarsh Ojha](https://utkarshojha.github.io/), [Yuheng Li](https://yuheng-li.github.io/), [Haotian Liu](https://hliu.cc/), [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/) <br>
🦡 University of Wisconsin-Madison<br>

---

### Getting Started

This repo is heavily built upon [DragGAN](https://github.com/XingangPan/DragGAN). Please refer to the original repo for more details about installation/download checkpoints.

```
#- clone this repo
git clone https://github.com/WisconsinAIVision/edit-one-for-all.git
cd edit-one-for-all

#- install packes
conda env create -f environment.yml
conda activate stylegan3
pip install -r requirements.txt

#- download checkpoints
python scripts/download_model.py

```

If you want to try other pretrained model, put them under `./checkpoints/` folder.


### Usage

Launch the gradio demo by:

```
python my_gradio.py --port 7681
```

Then, open your browser and go to `http://localhost:7681/` to interact with the demo.

### Galleries

As users adjust the editing strength in the example image (top row), all test images will be automatically updated. (Red bounding boxes indicate the edit according to the drag points).

- Interactive Dog Pose:

![](./images/interactive/Slide1.png "teaser")

- Interactive Anime Hair Length:

![](./images/interactive/Slide2.png "teaser")

- Interactive Mountain Height:

![](./images/interactive/Slide3.png "teaser")

- Interactive Human Pose:

![](./images/interactive/Slide4.png "teaser")

- Interactive Face Slimming:

![](./images/interactive/Slide5.png "teaser")

- Interactive Tiger Roar:

![](./images/interactive/Slide7.png "teaser")


### Related Works

- [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://vcai.mpi-inf.mpg.de/projects/DragGAN/) (SIGGRAPH 2023)
- [Analyzing and Improving the Image Quality of StyleGAN](https://openaccess.thecvf.com/content_CVPR_2020/papers/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.pdf) (CVPR 2020)
- [LARGE: Latent-Based Regression through GAN Semantics](https://yotamnitzan.github.io/LARGE/) (CVPR 2022)

Special thanks to DragGAN for making the code available!

### BibTeX


```
@inproceedings{nguyen2024edit,
      title={Edit One for All: Interactive Batch Image Editing},
      author={Thao Nguyen and Utkarsh Ojha and Yuheng Li and Haotian Liu and Yong Jae Lee},
      year={2024},
      eprint={2401.10219},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
   }
```