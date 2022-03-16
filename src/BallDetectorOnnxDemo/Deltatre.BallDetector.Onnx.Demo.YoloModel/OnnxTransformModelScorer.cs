// Based on: https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/DeepLearning_ObjectDetection_Onnx

namespace Deltatre.BallDetector.Onnx.Demo
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using Deltatre.BallDetector.Onnx.Demo.MLModels.Abstract;
    using Deltatre.BallDetector.Onnx.Demo.Model;
    using Microsoft.ML;
    using Microsoft.ML.Data;

    public class OnnxTransformModelScorer<T> : IDisposable where T : YoloModel
    {
        #region Private fields
        private readonly MLContext m_mlContext;
        private readonly T m_model;
        private bool m_disposedValue;
        #endregion

        #region Constructor
        public OnnxTransformModelScorer(MLContext mlContext)
        {
            m_mlContext = mlContext;
            m_model = Activator.CreateInstance<T>();
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
            var results = new List<ImagePrediction>();

            var model = LoadModel(m_model.ModelWeightsFilePath);

            var modelOutputs = PredictDataUsingModel(data, model);

            //public IList<YoloBoundingBox> ParseOutputs(float[] yoloModelOutputs, float threshold = .3F)

            //// Post-process model output
            //YoloOutputParser parser = new YoloOutputParser();

            //var boundingBoxes =
            //    probabilities
            //    .Select(probability => parser.ParseOutputs(probability))
            //    .Select(boxes => parser.FilterBoundingBoxes(boxes, 5, .5F));

            //// Draw bounding boxes for detected objects in each of the images
            //for (var i = 0; i < images.Count(); i++)
            //{
            //    string imageFileName = images.ElementAt(i).Label;
            //    IList<YoloBoundingBox> detectedObjects = boundingBoxes.ElementAt(i);

            //    DrawBoundingBox(imagesFolder, outputFolder, imageFileName, detectedObjects);

            //    LogDetectedObjects(imageFileName, detectedObjects);
            //}
            return results;
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

            // PLEASE VERIFY https://stackoverflow.com/questions/57264865/cant-get-input-column-name-of-onnx-model-to-work

            // Define scoring pipeline
            var pipeline = m_mlContext.Transforms.LoadImages(outputColumnName: "images", imageFolder: string.Empty, inputColumnName: nameof(ImageData.ImagePath))
                            .Append(m_mlContext.Transforms.ResizeImages(outputColumnName: "images", imageWidth: m_model.Width, imageHeight: m_model.Height, inputColumnName: "images"))
                            .Append(m_mlContext.Transforms.ExtractPixels(outputColumnName: "images"))
                            .Append(m_mlContext.Transforms.ApplyOnnxModel(modelFile: modelLocation, outputColumnNames: m_model.Outputs, inputColumnNames: new[] { "images" }));

            // Fit scoring pipeline
            var model = pipeline.Fit(data);

            return model;
        }

        private IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
        {
            IDataView scoredData = model.Transform(testData);

            IEnumerable<float[]> probabilities = scoredData.GetColumn<float[]>(m_model.Outputs[0]);

            return probabilities;
        } 
        #endregion
    }
}

