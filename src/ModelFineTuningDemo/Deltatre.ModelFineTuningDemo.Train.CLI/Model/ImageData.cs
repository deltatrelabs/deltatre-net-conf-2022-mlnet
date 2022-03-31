// Based on: https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DeepLearning_ImageClassification_Training

namespace Deltatre.ModelFineTuningDemo.Train.Model
{
    public class ImageData
    {
        public ImageData(string imagePath, string label)
        {
            ImagePath = imagePath;
            Label = label;
        }

        public readonly string ImagePath;

        public readonly string Label;
    }

}
