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
https://github.com/weihung0831/AI-Classification-From-Orange/blob/master/pdf/DLT2022-SS27-03_獎狀.pdf

## My paper
https://github.com/weihung0831/AI-Classification-From-Orange/blob/master/pdf/DLT2022-SS27-03-應用影像結合卷積神經網路於柳丁分級之研究-論文.pdf