// Based on: https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DeepLearning_ImageClassification_Training

namespace Deltatre.ModelFineTuningDemo.Common
{
    using Deltatre.ModelFineTuningDemo.Common.Model;

    public class FileUtils
    {
        public static IEnumerable<(string imagePath, string label)> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel)
        {
            var imagesPath = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories)
                .Where(x => Path.GetExtension(x) == ".jpg" || Path.GetExtension(x) == ".png");

            return useFolderNameAsLabel
                ? imagesPath.Select(imagePath => (imagePath, Directory.GetParent(imagePath).Name))
                : imagesPath.Select(imagePath =>
                {
                    var label = Path.GetFileName(imagePath);
                    for (var index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label[..index];
                            break;
                        }
                    }
                    return (imagePath, label);
                });
        }

        public static IEnumerable<InMemoryImageData> LoadInMemoryImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            return LoadImagesFromDirectory(folder, useFolderNameAsLabel)
                  .Select(x => new InMemoryImageData(
                      image: File.ReadAllBytes(x.imagePath),
                      label: x.label,
                      imageFileName: Path.GetFileName(x.imagePath)));
        }
    }
}
