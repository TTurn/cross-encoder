## Cross-Encoder

    This repository contains the code for cross-encoder

### Dependencies: 
    sentence-transformers==2.0.0
    torch==1.8.1+cu101


### Training Data
    train data is similarity sentence data from E-commerce dialogue, about 50w sentence pairs.

### Run 
```bash
python3 train.py 
```

### Model
    [roberta-wwm-ext](https://huggingface.co/tuhailong/cross_encoder_roberta-wwm-ext_v0)
    [roberta-wwm-ext](https://huggingface.co/tuhailong/cross_encoder_roberta-wwm-ext_v1)
    [roberta-wwm-ext](https://huggingface.co/tuhailong/cross_encoder_roberta-wwm-ext_v2)
    [electra-180g-large-discriminator](https://huggingface.co/tuhailong/cross_encoder_electra-180g-large-discriminator)
    [roberta-wwm-ext-large](https://huggingface.co/tuhailong/cross_encoder_roberta-wwm-ext-large)

### Language
    Chinese
