// Based on: https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DeepLearning_ImageClassification_Training

using Deltatre.ModelFineTuningDemo.Common;
using Deltatre.ModelFineTuningDemo.Common.Model;
using Microsoft.ML;

var datasetRelativePath = @"../../../../";
string datasetPath = GetAbsolutePath(datasetRelativePath);

var datasetFolder = Path.Combine(datasetPath, "SampleData");
var outputFolder = Path.Combine(datasetPath, "SampleData", "Outputs");
var trainingDatasetFolder = Path.Combine(datasetFolder, "Training");
var validationDatasetFolder = Path.Combine(datasetFolder, "Validation");
var testDatasetFolder = Path.Combine(datasetFolder, "Test");

var modelFilePath = Path.Combine(datasetPath, "SampleData", "MLModels", "Tensorflow.zip");

try
{
    var mlContext = new MLContext(seed: 678);

    Console.WriteLine($"Loading model from: {modelFilePath}");

    // Load the model
    var loadedModel = mlContext.Model.Load(modelFilePath, out var modelInputSchema);

    // Create prediction engine
    var predictionEngine = mlContext.Model.CreatePredictionEngine<InMemoryImageData, ImagePrediction>(loadedModel);

    // Predict images in the folder
    var imagesToPredict = FileUtils.LoadInMemoryImagesFromDirectory(testDatasetFolder, true);

    // Measure prediction execution time
    var watch = System.Diagnostics.Stopwatch.StartNew();
    
    // Predict all images in the folder
    
    Console.WriteLine("");
    Console.WriteLine($"Predicting images from '{testDatasetFolder}'");

    int counter = 0;
    foreach (var currentImageToPredict in imagesToPredict)
    {
        var currentPrediction = predictionEngine.Predict(currentImageToPredict);
        Console.WriteLine($"Image Filename : [{currentImageToPredict.ImageFileName}], Predicted Label : [{currentPrediction.PredictedLabel}], Probability : [{currentPrediction.Score.Max()}]");
        counter++;
    }

    // Stop measuring time
    watch.Stop();

    Console.WriteLine($"Predictions took {watch.ElapsedMilliseconds}ms ({watch.ElapsedMilliseconds / counter}ms per prediction)");
}
catch (Exception ex)
{
    Console.WriteLine(ex.ToString());
}

Console.WriteLine("Press any key to end the app...");
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
