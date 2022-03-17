using System.Drawing;
using System.Drawing.Drawing2D;
using Deltatre.BallDetector.Onnx.Demo;
using Deltatre.BallDetector.Onnx.Demo.MLModels;
using Deltatre.BallDetector.Onnx.Demo.Model;
using Microsoft.ML;

var datasetRelativePath = @"../../../";
string datasetPath = GetAbsolutePath(datasetRelativePath);
var imagesFolder = Path.Combine(datasetPath, "SampleData");
var outputFolder = Path.Combine(datasetPath, "SampleData", "Outputs");

// Initialize MLContext
MLContext mlContext = new();

try
{
    // Load Data
    Console.WriteLine($"Images location: {imagesFolder}");
    Console.WriteLine("");

    IEnumerable<ImageData> images = ImageData.ReadFromFile(imagesFolder);
    IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);

    Console.WriteLine("===== Identify objects in the images =====");
    Console.WriteLine("");

    //// Create instance of model scorer (using ML.NET OnnxTransform)
    //var modelScorer1 = new OnnxTransformModelScorer<Yolov5x6Model>(mlContext);

    // Create instance of model scorer (using OnnxRuntime)
    var modelScorer2 = new OnnxRuntimeModelScorer<Yolov5x6Model>(mlContext);

    //// Use model to score data
    //var results1 = modelScorer1.Score(imageDataView);
    //LogDetectedObjects(results1);

    var results2 = modelScorer2.Score(imageDataView);
    LogDetectedObjects(results2);

}
catch (Exception ex)
{
    Console.WriteLine(ex.ToString());
}

Console.WriteLine("========= End of Process..Hit any Key ========");

string GetAbsolutePath(string relativePath)
{
    FileInfo _dataRoot = new(typeof(Program).Assembly.Location);
    string? assemblyFolderPath = _dataRoot?.Directory?.FullName;

    if (!string.IsNullOrWhiteSpace(assemblyFolderPath))
        return Path.Combine(assemblyFolderPath, relativePath);

    return relativePath;
}

void DrawBoundingBox(string outputImageLocation, ImagePrediction prediction)
{
    Image image = Image.FromFile(prediction.ImagePath);

    var originalImageHeight = image.Height;
    var originalImageWidth = image.Width;

    // Define Text Options
    Font drawFont = new("Arial", 12, FontStyle.Bold);
    SolidBrush fontBrush = new(Color.Black);

    foreach (var box in prediction.DetectedObjects)
    {
        // Get Bounding Box Dimensions
        var x = (uint)Math.Max(box.Rectangle.X, 0);
        var y = (uint)Math.Max(box.Rectangle.Y, 0);
        var width = (uint)Math.Min(originalImageWidth - x, box.Rectangle.Width);
        var height = (uint)Math.Min(originalImageHeight - y, box.Rectangle.Height);

        // Resize To Image
        //x = (uint)originalImageWidth * x / (uint)prediction.ModelInputWidth;
        //y = (uint)originalImageHeight * y / (uint)prediction.ModelInputHeight;
        //width = (uint)originalImageWidth * width / (uint)prediction.ModelInputWidth;
        //height = (uint)originalImageHeight * height / (uint)prediction.ModelInputHeight;

        // Bounding Box Text
        string text = $"{box.Label.Name} ({box.Score * 100:0}%)";

        using (Graphics thumbnailGraphic = Graphics.FromImage(image))
        {
            thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
            thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
            thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

            // Define Text Options
            SizeF size = thumbnailGraphic.MeasureString(text, drawFont);
            Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

            // Define BoundingBox options
            Pen pen = new Pen(box.Label.Color, 3.2f);
            SolidBrush colorBrush = new SolidBrush(box.Label.Color);

            // Draw text on image 
            thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
            thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);

            // Draw bounding box on image
            thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
        }
    }

    if (!Directory.Exists(outputImageLocation))
    {
        Directory.CreateDirectory(outputImageLocation);
    }

    image.Save(Path.Combine(outputImageLocation, prediction.ImageName));
}

void LogDetectedObjects(IEnumerable<ImagePrediction> predictions)
{
    foreach (var imagePrediction in predictions)
    {
        var imageName = imagePrediction.ImageName;
        var boundingBoxes = imagePrediction.DetectedObjects;
        Console.WriteLine($"Detected objects in image '{imageName}':");

        DrawBoundingBox(outputFolder, imagePrediction);

        foreach (var box in boundingBoxes)
        {
            Console.WriteLine($"- {box.Label.Name} [{box.Score}]");
        }
    }
    Console.WriteLine("");
}
