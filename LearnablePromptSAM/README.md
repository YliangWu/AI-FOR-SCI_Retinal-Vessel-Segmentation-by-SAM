# LearnablePromptSAM
Try to use the SAM-ViT as the backbone to create the visual prompt tuning model for semantic segmentation. 
download all checkpoint [here](https://drive.google.com/drive/folders/1-kzNpA_vdlIzaGZEURr5DJvVMDCugaQc?usp=drive_link)

## 1. download checkpoint and dataset
put chasedb1 dataset to:
```
dataset
| __train
| |__img
| |__mask
| __test
| |__img
| |__mask
| __valid
| |__img
| |__mask
```

put checkpoint to ```./weights/sam_vit_b_01ec64.pth```
	
## 2.train

  ```
  python imporved_sam.py 
  ```

## 3.predict

download pretrained checkpoint to
```"./weights/sam_vit_b_prompt.pth```

 run gui.py in pycharm

