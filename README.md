# Deep Learning with ML.NET (Rome .NET Conference 2022)

In this repository you can find the slides and demo for **Deep Learning with ML.NET** session, presented (in Italian) at [Rome .NET Conference 2022](https://dotnetconf.it/) on March 18th, 2022.

Abstract:

Machine Learning and Deep Learning are more and more utilized at all levels, from embedded devices to web browsers.  
In this session, we will see how we can leverage our .NET expertise and tools to develop applications that utilize AI models: with a practical approach, we explore a different way to use the most common machine learning/deep learning frameworks to train and score models in .NET.

Speakers:

- [Clemente Giorio](https://www.linkedin.com/in/clemente-giorio-03a61811/) (Deltatre, Microsoft MVP)
- [Gianni Rosa Gallina](https://www.linkedin.com/in/gianni-rosa-gallina-b206a821/) (Deltatre, Microsoft MVP)

---

## Setup local environment

### Export yolov5 in ONNX model format

https://github.com/ultralytics/yolov5/releases

git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install wheel
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
#Comment torch, torchvision within requirements.txt file. 
pip install -r requirements.txt
pip install onnx

#Example on how to convert yolo5x6 model in onnx model format.
#Note: Take aware in specify the right model image size.
#python export.py --weights yolov5x6.pt --device 1 --imgsz 1280 1280 --include onnx

### Model Optimization
TBD
https://github.com/daquexian/onnx-simplifier
pip install onnx-simplifier
onnxsim input_onnx_model output_onnx_model

## How to run demo on your content

TBD

## License
---

Copyright (C) 2022 Deltatre.  
Licensed under [CC BY-NC-SA 4.0](./LICENSE).
