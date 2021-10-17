# Dataset Extension for Mask RCNN training
Extension dataset for training Mask RCNN model

Data for training stored in Google Drive
https://drive.google.com/drive/folders/1-RxOIuePdoR9iBdt3hYbAtZACkFlKZZk

## Extension

For running extension script on images directory need to run `extend_dataset.py` file
Parameters:
- `-i --images`: required, relative path to images directory
- `-a --annotations`: required, relative path to annotations file directory
- `-v --verbose`: optional, flag for displaying all images before saving

Example:
```shell script
python extend_dataset.py -i "test" -a "test" -v
```

## Displaying:

For running displaying script on images directory need to run `display_dataset.py` file
Parameters:
- `-i --images`: required, relative path to images directory
- `-a --annotations`: required, relative path to annotations file

Example:
```shell script
python display_dataset.py -i "test" -a "test/annotations.json"
```