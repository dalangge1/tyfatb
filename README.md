# Swap-Mukham
[![Open In Colab](https://img.shields.io/badge/Open%20In-Colab-blue?logo=google-colab&logoColor=white)](https://colab.research.google.com/github/harisreedhar/Swap-Mukham/blob/main/swap_mukham_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/bluefoxcreation/SwapMukham)
## 描述

一个简单的面部交换器，基于insightface inswapper，很大程度上受到roop的启发。


## [使用教程 https://www.bilibili.com/video/BV1pN411p7vh/?vd_source=faa4615f3c71b2b526ed2b1f48a70b2c](https://www.bilibili.com/video/BV1pN411p7vh/?vd_source=faa4615f3c71b2b526ed2b1f48a70b2c)

## 特征
- 易于使用的渐变图形用户界面
- 支持图片、视频、目录输入
- 达成场景特定（人脸识别）
- 视频工具修剪
- 人脸增强器（GFPGAN、Real-ESRGAN）
- 人脸解析地址
- 合作实验室支持

## 基础环境 ： git 和 pyrhon3.10 +   venv或者conda管理Python环境的工具

### CPU安装
````
git clone https://github.com/douhaohaode/Swap-Mukham.git
cd Swap-Mukham

conda create -n swapmukham python=3.10 -y  
conda activate swapmukham
pip install torch==2.0.0+cpu torchvision==0.15.1+cpu torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_cpu.txt
python app.py
````
### GPU 安装 (CUDA)
````
git clone https://github.com/douhaohaode/Swap-Mukham.git
cd Swap-Mukham
conda create -n swapmukham python=3.10 -y
conda activate swapmukham
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
python app.py --cuda --batch_size 32


### MAC venv 安装 

```python

git clone https://github.com/douhaohaode/Swap-Mukham.git
cd Swap-Mukham
python3 -m venv venv
source venv/bin/activate
pip install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0
pip install -r requirements.txt
python3 app.py

```

````
## Download Models
- [inswapper_128.onnx](https://huggingface.co/thebiglaskowski/inswapper_128.onnx/resolve/main/inswapper_128.onnx)
- [GFPGANv1.4.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth)
- [79999_iter.pth](https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812)
- [RealESRGAN_x2.pth](https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth)
- [RealESRGAN_x4.pth](https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth)
- [RealESRGAN_x8.pth](https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x8.pth)
- [codeformer.onnx](https://huggingface.co/bluefoxcreation/Codeformer-ONNX/resolve/main/codeformer.onnx)
- [open-nsfw.onnx](https://huggingface.co/bluefoxcreation/open-nsfw/resolve/main/open-nsfw.onnx)
 将这些模型放入里面 ``/assets/pretrained_models/``




## 免责声明

我们想强调的是，我们的深度创造了一个软件，以谋求并合乎道德的使用。我们必须强调，用户对使用我们软件时的行为承担全部责任。

预期用途：我们的深度伪造软件旨在帮助用户创建逼真且有趣的内容，例如电影、视觉效果、虚拟现实体验和其他创意应用程序。我们鼓励用户在合法性、道德考虑和尊重他人隐私的范围内探索这些可能性。

道德准则：用户在使用我们的软件时应遵守一套道德准则。这些指南包括但不限于：

不创建或分享可能伤害、诽谤或骚扰个人的深度虚假内容。在使用内容中的个人肖像之前，获得其适当的同意和许可。避免将深度造假技术用于欺骗目的，包括错误信息或恶意意图。尊重并遵守适用的法律、法规和版权限制。

隐私和同意：用户有责任确保他们获得了他们打算在深度伪造作品中使用其肖像的个人的必要许可和同意。我们强烈反对在未经明确同意的情况下创建深度虚假内容，特别是涉及未经同意或私人内容的情况。尊重所有相关个人的隐私和尊严至关重要。

法律注意事项：用户必须了解并遵守与深度造假技术有关的所有相关本地、区域和国际法律。这包括与隐私、诽谤、知识产权相关的法律以及其他相关立法。如果用户对其深度造假作品的法律影响有任何疑问，应咨询法律专业人士。

责任与义务：我们作为深度造假软件的创建者和提供者，不对因使用我们的软件而导致的行为或后果承担责任。用户对与其创建的深度虚假内容相关的任何误用、意外影响或滥用行为承担全部责任。

通过使用我们的 Deepfake 软件，用户承认他们已阅读、理解并同意遵守上述指南和免责声明。我们强烈鼓励用户严格、诚信并尊重他人的福祉和权利来使用深度造假技术。

请记住，技术应该用于赋予权力和激励，而不是用于伤害或欺骗。让我们努力以道德和税收的方式利用深度造假技术，改善社会。

## 致谢

- [Roop](https://github.com/s0md3v/roop)
- [Insightface](https://github.com/deepinsight)
- [Ffmpeg](https://ffmpeg.org/)
- [Gradio](https://gradio.app/)
- [Wav2lip HQ](https://github.com/Markfryazino/wav2lip-hq)
- [Face Parsing](https://github.com/zllrunning/face-parsing.PyTorch)
- [Real-ESRGAN (ai-forever)](https://github.com/ai-forever/Real-ESRGAN)
- [Open-NSFW](https://github.com/yahoo/open_nsfw)
- [Code-Former](https://github.com/sczhou/CodeFormer)

## 喜欢我的工作吗？
[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/harisreedhar)

## License

[MIT](https://choosealicense.com/licenses/mit/)
