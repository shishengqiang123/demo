#  #Model share 🤞

# 一、MULTIMODEL MODELS

## 1.ALIGN

![](https://github.com/shishengqiang123/demo/blob/main/ALIGN.png)

### Overview

The ALIGN model was proposed in [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918) by Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig. ALIGN is a multi-modal vision and language model. It can be used for image-text similarity and for zero-shot image classification. ALIGN features a dual-encoder architecture with [EfficientNet](https://huggingface.co/docs/transformers/model_doc/efficientnet) as its vision encoder and [BERT](https://huggingface.co/docs/transformers/model_doc/bert) as its text encoder, and learns to align visual and text representations with contrastive learning. Unlike previous work, ALIGN leverages a massive noisy dataset and shows that the scale of the corpus can be used to achieve SOTA representations with a simple recipe.

The abstract from the paper is the following:

*Pre-trained representations are becoming crucial for many NLP and perception tasks. While representation learning in NLP has transitioned to training on raw text without human annotations, visual and vision-language representations still rely heavily on curated training datasets that are expensive or require expert knowledge. For vision applications, representations are mostly learned using datasets with explicit class labels such as ImageNet or OpenImages. For vision-language, popular datasets like Conceptual Captions, MSCOCO, or CLIP all involve a non-trivial data collection (and cleaning) process. This costly curation process limits the size of datasets and hence hinders the scaling of trained models. In this paper, we leverage a noisy dataset of over one billion image alt-text pairs, obtained without expensive filtering or post-processing steps in the Conceptual Captions dataset. A simple dual-encoder architecture learns to align visual and language representations of the image and text pairs using a contrastive loss. We show that the scale of our corpus can make up for its noise and leads to state-of-the-art representations even with such a simple learning scheme. Our visual representation achieves strong performance when transferred to classification tasks such as ImageNet and VTAB. The aligned visual and language representations enables zero-shot image classification and also set new state-of-the-art results on Flickr30K and MSCOCO image-text retrieval benchmarks, even when compared with more sophisticated cross-attention models. The representations also enable cross-modality search with complex text and text + image queries.*

This model was contributed by [Alara Dirik](https://huggingface.co/adirik). The original code is not released, this implementation is based on the Kakao Brain implementation based on the original paper.

## 2.SAM

![sam](https://github.com/shishengqiang123/demo/blob/main/sam.png)

### Segment Anything Model (SAM): a new AI model from Meta AI that can "cut out" any object, in any image, with a single click [[video]](https://segment-anything.com/assets/section-1.4a.mp4) [[demo]](https://segment-anything.com/demo) [[code]](https://github.com/facebookresearch/segment-anything)[[paper]](https://arxiv.org/pdf/2304.02643v1.pdf)

The abstract from the paper is the following:

*We introduce the Segment Anything (SA) project: a new task, model, and dataset for image segmentation. Using our efficient model in a data collection loop, we built the largest segmentation dataset to date (by far), with over 1 billion masks on 11M licensed and privacy respecting images. The model is designed and trained to be promptable, so it can transfer zero-shot to new image distributions and tasks. We evaluate its capabilities on numerous tasks and find that its zero-shot performance is impressive — often competitive with or even superior to prior fully supervised results. We are releasing the Segment Anything Model (SAM) and corresponding dataset (SA-1B) of 1B masks and 11M images at [https://segment-anything.com](https://segment-anything.com/) to foster research into foundation models for computer vision.*

Tips:

- The model predicts binary masks that states the presence or not of the object of interest given an image.
- The model predicts much better results if input 2D points and/or input bounding boxes are provided
- You can prompt multiple points for the same image, and predict a single mask.
- Fine-tuning the model is not supported yet
- According to the paper, textual input should be also supported. However, at this time of writing this seems to be not supported according to [the official repository](https://github.com/facebookresearch/segment-anything/issues/4#issuecomment-1497626844).

## 3.BLIP

![](https://github.com/shishengqiang123/demo/blob/main/BLIP.gif)

### Overview

The BLIP model was proposed in [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) by Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi. [[code]](https://github.com/salesforce/BLIP)

BLIP is a model that is able to perform various multi-modal tasks including:

- Visual Question Answering
- Image-Text retrieval (Image-text matching)
- Image Captioning

The abstract from the paper is the following:

*Vision-Language Pre-training (VLP) has advanced the performance for many vision-language tasks. However, most existing pre-trained models only excel in either understanding-based tasks or generation-based tasks. Furthermore, performance improvement has been largely achieved by scaling up the dataset with noisy image-text pairs collected from the web, which is a suboptimal source of supervision. In this paper, we propose BLIP, a new VLP framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones. We achieve state-of-the-art results on a wide range of vision-language tasks, such as image-text retrieval (+2.7% in average recall@1), image captioning (+2.8% in CIDEr), and VQA (+1.6% in VQA score). BLIP also demonstrates strong generalization ability when directly transferred to videolanguage tasks in a zero-shot manner. Code, models, and datasets are released.*

## 4.CLIP

![](https://github.com/shishengqiang123/demo/blob/main/CLIP.png)

### Overview

The CLIP model was proposed in [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) by Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. [[code]](https://github.com/openai/CLIP)

The abstract from the paper is the following:

*State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on. We release our code and pre-trained model weights at this https URL.*

## 5.Stable Diffusion

![](https://github.com/shishengqiang123/demo/blob/main/stable%20diffusion.png)

### Owerview

Stable Diffusion is a state-of-the-art machine learning model designed for generating and modifying images based on textual descriptions. Developed by Stability AI, it represents a significant advancement in the field of AI-generated art and content creation. The model operates by understanding and interpreting text inputs, then translating these inputs into complex visual outputs, effectively "imagining" visual representations of the described scenes or objects. 