# cnn_orange
Use Deep Learning technology to improve classification for fruit.

## Source Dataset
We collect orange by myself and according to orange sweet level.<br>
Label 0(Level C) --> color:green , sweetness:9 brix down.<br>
Label 1(Level B) --> color:green_yellow , sweetness:9 ~ 12 brix.<br>
Label 2(Level A) --> color:yellow , sweetness:12 brix up.<br>
https://drive.google.com/drive/folders/1-2XOwcFmqSTXZCXeHRnXRXySk01tyhGS?usp=sharing

## Step
Step 1: python process_dataset.py<br>
Process dataset and convert to npz dataset.

Step 2: python train.py<br>
Training model.

Step 3: python test.py<br>
Test test data and evaluation metrics.

## Requirements
python version 3.8<br>
tensorflow-gpu version 2.5

## Join Taiwan Automation Intelligence and Robot Show
https://www.youtube.com/watch?v=TlBJyNHABJw

## Best paper award
https://github.com/weihung0831/CNN-classification-from-orange/blob/master/DLT2022-SS27-03_%E7%8D%8E%E7%8B%80.pdf

## My paper
https://github.com/weihung0831/CNN-classification-from-orange/blob/master/DLT2022-SS27-03-%E6%87%89%E7%94%A8%E5%BD%B1%E5%83%8F%E7%B5%90%E5%90%88%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF%E6%96%BC%E6%9F%B3%E4%B8%81%E5%88%86%E7%B4%9A%E4%B9%8B%E7%A0%94%E7%A9%B6-%E8%AB%96%E6%96%87.pdf