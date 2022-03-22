// Based on: https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DeepLearning_ObjectDetection_Onnx

namespace Deltatre.BallDetector.Onnx.Demo
{
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Linq;
    using Deltatre.BallDetector.Onnx.Demo.MLModels.Abstract;
    using Deltatre.BallDetector.Onnx.Demo.Model;
    using Microsoft.ML;
    using Microsoft.ML.Data;
    using Microsoft.ML.OnnxRuntime.Tensors;

    public class OnnxTransformModelScorer<T> : IDisposable where T : YoloModel
    {
        #region Private fields
        private readonly MLContext m_mlContext;
        private readonly T m_model;
        private readonly YoloParser<T> m_outputParser;
        private bool m_disposedValue;
        #endregion

        #region Constructor
        public OnnxTransformModelScorer(MLContext mlContext)
        {
            m_mlContext = mlContext;
            m_model = Activator.CreateInstance<T>();
            m_outputParser = new YoloParser<T>(m_model);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!m_disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects)
                }

                // TODO: free unmanaged resources (unmanaged objects) and override finalizer
                // TODO: set large fields to null
                m_disposedValue = true;
            }
        }

        // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        // ~OnnxTransformModelScorer()
        // {
        //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        //     Dispose(disposing: false);
        // }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
        #endregion

        #region Methods
        /// <summary>
        /// Runs object detection
        /// </summary>
        public IEnumerable<ImagePrediction> Score(IDataView data)
        {
            var model = LoadModel(m_model.ModelWeightsFilePath);

            return PredictDataUsingModel(data, model);
        }
        #endregion

        #region Private methods
        private ITransformer LoadModel(string modelLocation)
        {
            Console.WriteLine("Read model");
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Default parameters: image size=({m_model.Width},{m_model.Height})");

            // Create IDataView from empty list to obtain input data schema
            var data = m_mlContext.Data.LoadFromEnumerable(new List<ImageData>());

            // Define scoring pipeline
            var pipeline = m_mlContext.Transforms.LoadImages(outputColumnName: "images", imageFolder: string.Empty, inputColumnName: nameof(ImageData.ImagePath))
                            .Append(m_mlContext.Transforms.ResizeImages(outputColumnName: "images", imageWidth: m_model.Width, imageHeight: m_model.Height, inputColumnName: "images"))
                            .Append(m_mlContext.Transforms.ExtractPixels(outputColumnName: "images", colorsToExtract: Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator.ColorBits.Rgb, orderOfExtraction: Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator.ColorsOrder.ABGR))
                            .Append(m_mlContext.Transforms.ApplyOnnxModel(modelFile: modelLocation, outputColumnNames: m_model.Outputs, inputColumnNames: new[] { "images" })); //, gpuDeviceId: 0, fallbackToCpu: false));

            // Fit scoring pipeline
            var model = pipeline.Fit(data);

            return model;
        }

        // Needed for the alternative with CreateEnumerable
        //internal class ModelRawPrediction
        //{
        //    public string ImagePath;
        //    public string Label;
        //    public float[] output;
        //}

        private IEnumerable<ImagePrediction> PredictDataUsingModel(IDataView testData, ITransformer model)
        {
            var results = new List<ImagePrediction>();

            // Measure prediction execution time
            var watch = System.Diagnostics.Stopwatch.StartNew();

            // Score data
            IDataView scoredData = model.Transform(testData);

            // Get DataViewSchema of IDataView
            DataViewSchema columns = scoredData.Schema;

            // Create DataViewCursor
            int counter = 0;
            using (DataViewRowCursor cursor = scoredData.GetRowCursor(columns))
            {
                // Define variables where extracted values will be stored to
                ReadOnlyMemory<char> imagePath = default;
                ReadOnlyMemory<char> imageLabel = default;
                VBuffer<float> probabilities = default;

                // Define delegates for extracting values from columns
                ValueGetter<ReadOnlyMemory<char>> imagePathDelegate = cursor.GetGetter<ReadOnlyMemory<char>>(columns[0]);
                ValueGetter<ReadOnlyMemory<char>> imageLabelDelegate = cursor.GetGetter<ReadOnlyMemory<char>>(columns[1]);
                ValueGetter<VBuffer<float>> probabilitiesDelegate = cursor.GetGetter<VBuffer<float>>(columns[5]);

                // Iterate over each row
                while (cursor.MoveNext())
                {
                    // Get values from respective columns
                    imagePathDelegate.Invoke(ref imagePath);
                    imageLabelDelegate.Invoke(ref imageLabel);
                    probabilitiesDelegate.Invoke(ref probabilities);
                    var tensorDimensions = new int[] { 1, probabilities.Length / m_model.Dimensions, m_model.Dimensions };
                    var output = new DenseTensor<float>(new Memory<float>(probabilities.GetValues().ToArray()), tensorDimensions);
                    using var image = Image.FromFile(imagePath.ToString());
                    results.Add(new ImagePrediction { ImagePath = imagePath.ToString(), DetectedObjects = m_outputParser.ParseOutput(new[] { output }, image.Width, image.Height), ImageName = imageLabel.ToString(), ModelInputWidth = m_model.Width, ModelInputHeight = m_model.Height, ResizeDetections = false });
                    counter++;
                }
            }

            // Alternative way (more memory-hungry)
            // Uncomment also the helper DTO ModelRawPrediction define above

            //var data = m_mlContext.Data.CreateEnumerable<ModelRawPrediction>(scoredData, reuseRowObject: true);
            //foreach (var imageData in data)
            //{
            //    using var image = Image.FromFile(imageData.ImagePath);
            //    var output = new DenseTensor<float>(new Memory<float>(imageData.output), new int[] { 1, imageData.output.Length/ m_model.Dimensions, m_model.Dimensions });
            //    results.Add(new ImagePrediction { ImagePath = imageData.ImagePath, DetectedObjects = m_outputParser.ParseOutput(new[] { output }, image.Width, image.Height), ImageName = imageData.Label, ModelInputWidth = m_model.Width, ModelInputHeight = m_model.Height, ResizeDetections = false });
            //}

            // Stop measuring time
            watch.Stop();
            Console.WriteLine($"Predictions took {watch.ElapsedMilliseconds}ms ({watch.ElapsedMilliseconds / counter}ms per prediction)");

            return results;
        }
        #endregion
    }
}

