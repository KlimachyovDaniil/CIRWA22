# CIRWA
Card Information Recognition Web Application

## Data generation

For data generation see proccesor\dataset\datagen.ipynb

## Card cropping

*card_processor.py* implements basic API *get_card_img(img)* that allows one to extract bank card from an image. 

The image restricions are:

1. The image is illuminated evenly
2. The bank card contrasts with the background 

One can run *process.py* to see the working pipeline:

```bash
python proccesor/sample.py --input-dir input --output-dir output 
```
