// Based on: https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DeepLearning_ObjectDetection_Onnx

namespace Deltatre.BallDetector.Onnx.Demo.Model
{
    public class ImagePrediction
    {
        public int ModelInputWidth { get; set; }
        public int ModelInputHeight { get; set; }
        public bool ResizeDetections { get; set; }
        public string ImagePath { get; set; }
        public string ImageName { get; set; }
        public IEnumerable<YoloPrediction> DetectedObjects { get; set; }
    }
}