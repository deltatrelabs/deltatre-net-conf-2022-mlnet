// Based on: https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DeepLearning_ImageClassification_Training

namespace Deltatre.ModelFineTuningDemo.Common.Model
{
    public class InMemoryImageData
    {
        public InMemoryImageData(byte[] image, string label, string imageFileName)
        {
            Image = image;
            Label = label;
            ImageFileName = imageFileName;
        }

        public readonly byte[] Image;

        public readonly string Label;

        public readonly string ImageFileName;
    }
}
