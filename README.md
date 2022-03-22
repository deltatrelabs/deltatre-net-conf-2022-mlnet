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

Software requirements:

- Windows 10 21H2 or Windows 11
- Visual Studio 2022
- .NET 6 SDK

To setup a local copy, just clone the repository and open the solution for the demo you want to run:

- `src/BallDetectorOnnxDemo/BallDetectorOnnxDemo.sln`
- `src/ModelFineTuningDemo/ModelFineTuningDemo.sln`

### Ball Detector Onnx Demo

Before executing the application, you need to:

- place all the images you want to process in the `src\BallDetectorOnnxDemo\Deltatre.BallDetector.Onnx.Demo.CLI\SampleData` folder
- download pre-trained Yolov5 ONNX models from [Ultralytics GitHub repository](https://github.com/ultralytics/yolov5/releases) and place them in the `src\BallDetectorOnnxDemo\Deltatre.BallDetector.Onnx.Demo.YoloModel\Assets\ModelWeights` folder

To start the scoring application, set the `Deltatre.BallDetector.Onnx.Demo.CLI` project as *Startup project*, and launch a debug session. It will load the configured YOLOv5 pre-trained model and score all the images in the `SampleData` folder. All results will be saved in the `Outputs` folder.

If you want to customize the folder where to look for images to process and where to store the results, you can edit the `Program.cs` file and change the following lines:

```csharp
var datasetRelativePath = @"../../../";
string datasetPath = GetAbsolutePath(datasetRelativePath);
var imagesFolder = Path.Combine(datasetPath, "SampleData");
var outputFolder = Path.Combine(datasetPath, "SampleData", "Outputs");
```

### Model Fine Tuning Demo

Before executing the application, you need to place your set of images you want to use in the `src\ModelFineTuningDemo\SampleData` folder. There you have a `Training` folder and a `Test` folder. Within each folder, you may put images in different sub-folders, representing the "label" of the contained images.

An example of the dataset folder structure is:

```bash
└── src
    └── ModelFineTuningDemo
        └── SampleData
            ├── MLModels
            ├── Training
            │   ├── Class1
            │   │   ├── image1.png
            │   │   ├── image2.png
            │   │   ├── ...
            │   │   └── imageN.png
            │   ├── Class2
            │   │   ├── image1.png
            │   │   ├── image2.png
            │   │   ├── ...
            │   │   └── imageN.png
            │   ├── ...
            │   └── ClassN
            │       ├── image1.png
            │       ├── image2.png
            │       ├── ...
            │       └── imageN.png
            └── Test
                ├── Class1
                │   ├── image1.png
                │   ├── image2.png
                │   ├── ...
                │   └── imageN.png
                ├── Class2
                │   ├── image1.png
                │   ├── image2.png
                │   ├── ...
                │   └── imageN.png
                ├── ...
                └── ClassN
                    ├── image1.png
                    ├── image2.png
                    ├── ...
                    └── imageN.png
```

For the best results, each class should contain almost the same number of representative samples, and usually, the more samples you have, the more the quality of the fine-tuned model improves. But you need to test and verify on your own dataset the metrics and decide the proper actions to take to fulfill your requirements.

If you want to customize the folder where to look for images to process and where to store the results, you can edit the `Program.cs` file and change the following lines:

```csharp
var datasetRelativePath = @"../../../../";
string datasetPath = GetAbsolutePath(datasetRelativePath);

var datasetFolder = Path.Combine(datasetPath, "SampleData");
var outputFolder = Path.Combine(datasetPath, "SampleData", "Outputs");
var trainingDatasetFolder = Path.Combine(datasetFolder, "Training");
var testDatasetFolder = Path.Combine(datasetFolder, "Test");

var modelFilePath = Path.Combine(datasetPath, "SampleData", "MLModels", "TF_Sport_Classification.zip");
```

To start the training application, set the `Deltatre.ModelFineTuningDemo.Train.CLI` project as *Startup project*, and launch a debug session: this will train/fine-tune a Tensorflow model on your own dataset. Once the model has been trained, you can use it for inference.

To start the scoring application, set the `Deltatre.ModelFineTuningDemo.Score.CLI` project as *Startup project*, and launch a debug session. It will load the previously trained model and score all the images in the `Test` folder.

> If you want to use GPU for training/inference, you need to replace the `SciSharp.Tensorflow` package with the proper GPU-enabled version. Please keep in mind that, as of ML.NET version 1.7.1, the latest supported version is v2.3.1. More recent versions do not work.

### Export yolov5 in ONNX model format

**TBD**

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

**TBD**

https://github.com/daquexian/onnx-simplifier
pip install onnx-simplifier
onnxsim input_onnx_model output_onnx_model


## References and other useful links

### Ball Detector ONNX Demo

- <https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DeepLearning_ObjectDetection_Onnx>
- <https://docs.microsoft.com/en-us/dotnet/machine-learning/resources/transforms>
- <https://towardsdatascience.com/mask-detection-using-yolov5-ae40979227a6>
- <https://dev.to/azure/onnx-no-it-s-not-a-pokemon-deploy-your-onnx-model-with-c-and-azure-functions-28f>
- <https://stackoverflow.com/questions/57264865/cant-get-input-column-name-of-onnx-model-to-work>
- <https://docs.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/inspect-intermediate-data-ml-net>
- <https://github.com/dotnet/machinelearning/blob/main/docs/code/VBufferCareFeeding.md>
- <https://github.com/dotnet/machinelearning/blob/main/src/Microsoft.ML.OnnxTransformer/OnnxTransform.cs>
- <https://stackoverflow.com/questions/64357642/how-to-load-image-from-memory-with-bitmap-or-byte-array-for-image-processing-in>
- <https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.imageestimatorscatalog.extractpixels?view=ml-dotnet>
- <https://stackoverflow.com/questions/70880362/transform-densetensor-in-microsoft-ml-onnxruntime>

### TensorFlow fine-tuning Demo

- <https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DeepLearning_ImageClassification_Training>
- <https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/image-classification>
- <https://levelup.gitconnected.com/training-an-ml-net-image-classification-model-on-gpus-using-google-colab-ee40b38af7e5>

## License

---

Copyright (C) 2022 Deltatre.  
Licensed under [CC BY-NC-SA 4.0](./LICENSE).
