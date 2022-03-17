// Based on: https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DeepLearning_ImageClassification_Training

// References:
// https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/image-classification
// https://levelup.gitconnected.com/training-an-ml-net-image-classification-model-on-gpus-using-google-colab-ee40b38af7e5
//
// For GPU -- SciSharp.Tensorflow --> GPU v2.3.1
// See:
// - https://stackoverflow.com/questions/65542317/how-to-speed-up-the-adding-visible-gpu-devices-process-in-tensorflow-with-a-30
// - https://stackoverflow.com/questions/39649102/how-do-i-select-which-gpu-to-run-a-job-on
using Deltatre.ModelFineTuningDemo.Common;
using Deltatre.ModelFineTuningDemo.Train.Model;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Diagnostics;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;

var datasetRelativePath = @"../../../../";
string datasetPath = GetAbsolutePath(datasetRelativePath);

var datasetFolder = Path.Combine(datasetPath, "SampleData");
var outputFolder = Path.Combine(datasetPath, "SampleData", "Outputs");
var trainingDatasetFolder = Path.Combine(datasetFolder, "Training");
var testDatasetFolder = Path.Combine(datasetFolder, "Test");

var modelFilePath = Path.Combine(datasetPath, "SampleData", "MLModels", "TF_Sport_Classification.zip");

var mlContext = new MLContext(seed: 678);

// Specify MLContext Filter to only show feedback log/traces about ImageClassification
// This is not needed for feedback output if using the explicit MetricsCallback parameter
mlContext.Log += FilterMLContextLog;

// 2. Load the initial full image-set into an IDataView and shuffle so it'll be better balanced
IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: trainingDatasetFolder, useFolderNameAsLabel: true);
IDataView trainingDataset = mlContext.Data.LoadFromEnumerable(images);
IDataView shuffledDataset = mlContext.Data.ShuffleRows(trainingDataset);

// 3. Load Images with in-memory type within the IDataView and Transform Labels to Keys (Categorical)
IDataView shuffledImagesDataset = mlContext.Transforms.Conversion.
        MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label", keyOrdinality: KeyOrdinality.ByValue)
    .Append(mlContext.Transforms.LoadRawImageBytes(outputColumnName: "Image", imageFolder: trainingDatasetFolder, inputColumnName: "ImagePath"))
    .Fit(shuffledDataset)
    .Transform(shuffledDataset);

// 4. Split the data 80:20 into train and test sets, train and evaluate.
var trainTestData = mlContext.Data.TrainTestSplit(shuffledImagesDataset, testFraction: 0.2);
IDataView trainingDataView = trainTestData.TrainSet;
IDataView validationDataView = trainTestData.TestSet;

// 5. Define the model's training pipeline using DNN default values
//
var pipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(featureColumnName: "Image", labelColumnName: "LabelAsKey", validationSet: validationDataView)
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));

// 5.1 (OPTIONAL) Define the model's training pipeline by using explicit hyper-parameters
//
//var options = new ImageClassificationTrainer.Options()
//{
//    FeatureColumnName = "Image",
//    LabelColumnName = "LabelAsKey",
//    // Just by changing/selecting InceptionV3/MobilenetV2/ResnetV250  
//    // you can try a different DNN architecture (TensorFlow pre-trained model). 
//    Arch = ImageClassificationTrainer.Architecture.MobilenetV2,
//    Epoch = 50,       //100
//    BatchSize = 10,
//    LearningRate = 0.01f,
//    MetricsCallback = (metrics) => Console.WriteLine(metrics),
//    ValidationSet = testDataView
//};

//var pipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(options)
//        .Append(mlContext.Transforms.Conversion.MapKeyToValue(
//            outputColumnName: "PredictedLabel",
//            inputColumnName: "PredictedLabel"));

// 6. Train/create the ML model
Console.WriteLine("*** Training the image classification model with DNN Transfer Learning on top of the selected pre-trained model/architecture ***");

// Measuring training time
var watch = Stopwatch.StartNew();

//Train
ITransformer trainedModel = pipeline.Fit(trainingDataView);

watch.Stop();
var elapsedMs = watch.ElapsedMilliseconds;

Console.WriteLine($"Training with transfer learning took: {elapsedMs / 1000} seconds");

// 7. Get the quality metrics (accuracy, etc.) on a test dataset
IEnumerable<ImageData> testImages = LoadImagesFromDirectory(folder: testDatasetFolder, useFolderNameAsLabel: true);
IDataView testDataset = mlContext.Data.LoadFromEnumerable(images);
IDataView testImagesDataset = mlContext.Transforms.Conversion.
        MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label", keyOrdinality: KeyOrdinality.ByValue)
    .Append(mlContext.Transforms.LoadRawImageBytes(outputColumnName: "Image", imageFolder: trainingDatasetFolder, inputColumnName: "ImagePath"))
    .Fit(testDataset)
    .Transform(testDataset);

EvaluateModel(mlContext, testImagesDataset, trainedModel);

// 8. Save the model to assets/outputs (You get ML.NET .zip model file and TensorFlow .pb model file)
mlContext.Model.Save(trainedModel, trainingDataView.Schema, modelFilePath);
Console.WriteLine($"Model saved to: {modelFilePath}");

Console.WriteLine("Press any key to finish");
Console.ReadKey();

string GetAbsolutePath(string relativePath)
{
    FileInfo _dataRoot = new(typeof(Program).Assembly.Location);
    string? assemblyFolderPath = _dataRoot?.Directory?.FullName;

    if (!string.IsNullOrWhiteSpace(assemblyFolderPath))
    {
        return Path.Combine(assemblyFolderPath, relativePath);
    }

    return relativePath;
}

void FilterMLContextLog(object? sender, LoggingEventArgs e)
{
    if (e.Message.StartsWith("[Source=ImageClassificationTrainer;"))
    {
        Console.WriteLine(e.Message);
    }
}

IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true) => FileUtils.LoadImagesFromDirectory(folder, useFolderNameAsLabel).Select(x => new ImageData(x.imagePath, x.label));

void EvaluateModel(MLContext mlContext, IDataView testDataset, ITransformer trainedModel)
{
    Console.WriteLine("Making predictions in bulk for evaluating model's quality...");

    // Measuring time
    var watch = Stopwatch.StartNew();

    var predictionsDataView = trainedModel.Transform(testDataset);

    var metrics = mlContext.MulticlassClassification.Evaluate(predictionsDataView, labelColumnName: "LabelAsKey", predictedLabelColumnName: "PredictedLabel");
    PrintMultiClassClassificationMetrics("TensorFlow DNN Transfer Learning", metrics);

    watch.Stop();
    var elapsed2Ms = watch.ElapsedMilliseconds;

    Console.WriteLine($"Predicting and Evaluation took: {elapsed2Ms / 1000} seconds");
}

void PrintMultiClassClassificationMetrics(string name, MulticlassClassificationMetrics metrics)
{
    Console.WriteLine($"************************************************************");
    Console.WriteLine($"*    Metrics for {name} multi-class classification model   ");
    Console.WriteLine($"*-----------------------------------------------------------");
    Console.WriteLine($"    AccuracyMacro = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
    Console.WriteLine($"    AccuracyMicro = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
    Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");

    int i = 0;
    foreach (var classLogLoss in metrics.PerClassLogLoss)
    {
        i++;
        Console.WriteLine($"    LogLoss for class {i} = {classLogLoss:0.####}, the closer to 0, the better");
    }
    Console.WriteLine($"************************************************************");
}