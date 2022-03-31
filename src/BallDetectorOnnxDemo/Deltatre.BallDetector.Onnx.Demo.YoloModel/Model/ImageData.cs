// Based on: https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DeepLearning_ObjectDetection_Onnx

namespace Deltatre.BallDetector.Onnx.Demo.Model
{
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using Microsoft.ML.Data;

    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;

        public static IEnumerable<ImageData> ReadFromFile(string imageFolder)
        {
            return Directory
                .GetFiles(imageFolder)
                .Where(filePath => !Path.GetFileNameWithoutExtension(filePath).StartsWith("ADD_HERE_") && Path.GetExtension(filePath) != ".md")
                .Select(filePath => new ImageData { ImagePath = filePath, Label = Path.GetFileName(filePath) });
        }
    }
}