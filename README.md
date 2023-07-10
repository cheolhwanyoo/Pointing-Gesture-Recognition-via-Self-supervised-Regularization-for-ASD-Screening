# PointDetectNet: Offcial Pytorch Implementation

## Updates
07/10/2023: Project page built

This repository provides the official PyTorch implementation of the following paper:
<img src="fig_architecture.png" width="800">

> Pointing Gesture Recognition via Self-supervised Regularization for ASD
Screening \
>
> Abstract: The ability to point to objects for sharing social purpose or attention is known as one of the key indicators in distinguishing children with typical development (TD) from those with autism spectrum disorder (ASD). However, there is a lack of datasets specifically tailored for children’s pointing gestures. This lack of training data from the target domain becomes a major factor in the performance degradation of conventional supervised CNNs due to domain shift. Toward an effective and practical solution, we propose an end-to-end learning scheme for domain generalized pointing gesture recognition adopting self-supervised regularization (SSR). To prove the effectiveness of our method in real-world situations, we designed a Social Interaction-Inducing Content (SIIC)-based ASD diagnostic system and collected an ASD-Pointing dataset consisting of 40 TD and ASD children. Through extensive experiments on our collected datasets, we achieved an ASD screening accuracy of 72.5%, showing that pointing ability can play a vital role as an indicator in distinguishing between ASD and TD.

---

## Test
- To test code, run the command below:
```python
python demo_livinglab.py
```

## Training
- To train code, run the command below:
```python
python main.py --mode "train"
```


## Experimental Results

Examples of result images on the *ASD-Pointing* dataset.

<img src="fig_results.png" width="1000">
