#!/bin/bash

cd Segmentation
python segNet_03_predict.py
cp gen/Outputs/test_out.png ../Captioning/gen/Inputs/test.png
cd ../Captioning
python caption_05_predict.py
cp gen/Outputs/test.txt ../Tacotron2-Wavenet-Korean-TTS/gen/Inputs/test.txt
cd ../Tacotron2-Wavenet-Korean-TTS
python tacotron2_01_sythesizer.py
