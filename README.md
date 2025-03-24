# Transformer-UNet

I used a pretrained Swin Transformer as the encoder for a UNet architecture and fine-tuned the model on the BBBC018 dataset, which features human HT29 colon cancer cells. The provided masks in BBBC018 are cell outlines, resulting in unbalanced and sparse labels.

For the baseline UNet, I implemented a customized loss function that rewards predictions on the outlines while penalizing predictions in the cell bodies. This approach significantly improved the baseline model’s ability to predict accurate outlines. Image preprocessing, data augmentation, baseline model design, and loss function details are included in the accompanying Jupyter Notebook file for UNet.

In the second notebook, I replaced the baseline encoder with a Swin Transformer to test whether integrating attention mechanisms from Vision Transformers (ViT) could enhance feature extraction by capturing global context. Based on the pretrained ViT weights the encoder params will be fine-tuned, decoder and segmentation head’s params will be trained from scratch on BBBC018. 
Below is the modified model architecture:

```
Input Image
       │
┌───────────────┐
│  Transformer  │
└──────┬────────┘
       │   (features [0], [1], [2], [3])
       ▼
Decoder4 (features[3])
       ▼
Decoder3 (upsample d4 + features[2])
       ▼
Decoder2 (upsample d3 + features[1])
       ▼
Decoder1 (upsample d2 + features[0])
       ▼
Segmentation Head (Conv2d 1x1)
       ▼
Segmentation Mask (Output)
```

The goal is to evaluate whether attention-based feature encoding improves segmentation performance on sparse, outline-based masks.
