// Based on: https://github.com/mentalstack/yolov5-net

namespace Deltatre.BallDetector.Onnx.Demo
{
    using Deltatre.BallDetector.Onnx.Demo.MLModels.Abstract;
    using Deltatre.BallDetector.Onnx.Demo.Model;
    using Microsoft.ML;
    using Microsoft.ML.OnnxRuntime;
    using Microsoft.ML.OnnxRuntime.Tensors;
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Drawing.Drawing2D;
    using System.Drawing.Imaging;
    using System.IO;

    /// <summary>
    /// Yolov5 scorer.
    /// </summary>
    public class OnnxRuntimeModelScorer<T> : IDisposable where T : YoloModel
    {
        #region Private fields
        private readonly MLContext m_mlContext;
        private readonly T m_model;
        private readonly YoloParser<T> m_outputParser;
        private readonly InferenceSession m_inferenceSession;
        private bool m_disposedValue;
        #endregion

        #region Constructor
        public OnnxRuntimeModelScorer(MLContext mlContext, SessionOptions? opts = null)
        {
            m_mlContext = mlContext;

            m_model = Activator.CreateInstance<T>();
            m_outputParser = new YoloParser<T>(m_model);
            m_inferenceSession = new InferenceSession(File.ReadAllBytes(m_model.ModelWeightsFilePath), opts ?? new SessionOptions());
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!m_disposedValue)
            {
                if (disposing)
                {
                    // Dispose managed state (managed objects)
                    m_inferenceSession.Dispose();
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

        #region Public Methods
        /// <summary>
        /// Runs object detection
        /// </summary>
        public IEnumerable<ImagePrediction> Score(IDataView testData)
        {
            return PredictDataUsingModel(testData);
        }
        #endregion

        #region Private methods
        private IEnumerable<ImagePrediction> PredictDataUsingModel(IDataView testData)
        {
            var results = new List<ImagePrediction>();
            var data = m_mlContext.Data.CreateEnumerable<ImageData>(testData, reuseRowObject: false);
            
            foreach (var imageData in data)
            {
                using var image = Image.FromFile(imageData.ImagePath);
                results.Add(new ImagePrediction { ImagePath = imageData.ImagePath, DetectedObjects = m_outputParser.ParseOutput(Inference(image), image), ImageName = imageData.Label, ModelInputWidth = m_model.Width, ModelInputHeight = m_model.Height });
            }

            return results;
        }

        /// <summary>
        /// Resizes image keeping ratio to fit model input size
        /// </summary>
        private Bitmap ResizeImage(Image image)
        {
            PixelFormat format = image.PixelFormat;

            var output = new Bitmap(m_model.Width, m_model.Height, format);

            var (w, h) = (image.Width, image.Height); // image width and height
            var (xRatio, yRatio) = (m_model.Width / (float)w, m_model.Height / (float)h); // x, y ratios
            var ratio = Math.Min(xRatio, yRatio); // ratio = resized / original
            var (width, height) = ((int)(w * ratio), (int)(h * ratio)); // roi width and height
            var (x, y) = ((m_model.Width / 2) - (width / 2), (m_model.Height / 2) - (height / 2)); // roi x and y coordinates
            var roi = new Rectangle(x, y, width, height); // region of interest

            using (var graphics = Graphics.FromImage(output))
            {
                graphics.Clear(Color.FromArgb(0, 0, 0, 0)); // clear canvas

                graphics.SmoothingMode = SmoothingMode.None; // no smoothing
                graphics.InterpolationMode = InterpolationMode.Bilinear; // bilinear interpolation
                graphics.PixelOffsetMode = PixelOffsetMode.Half; // half pixel offset

                graphics.DrawImage(image, roi); // draw scaled
            }

            return output;
        }

        /// <summary>
        /// Extracts pixels into tensor for network input
        /// </summary>
        private Tensor<float> ExtractPixels(Image image)
        {
            var bitmap = (Bitmap)image;

            var rectangle = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            BitmapData bitmapData = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, bitmap.PixelFormat);
            int bytesPerPixel = Image.GetPixelFormatSize(bitmap.PixelFormat) / 8;

            var tensor = new DenseTensor<float>(new[] { 1, 3, m_model.Height, m_model.Width });

            unsafe // Speed up conversion by direct work with memory
            {
                Parallel.For(0, bitmapData.Height, (y) =>
                {
                    byte* row = (byte*)bitmapData.Scan0 + (y * bitmapData.Stride);

                    Parallel.For(0, bitmapData.Width, (x) =>
                    {
                        tensor[0, 0, y, x] = row[x * bytesPerPixel + 2] / 255.0F; // r
                        tensor[0, 1, y, x] = row[x * bytesPerPixel + 1] / 255.0F; // g
                        tensor[0, 2, y, x] = row[x * bytesPerPixel + 0] / 255.0F; // b
                    });
                });

                bitmap.UnlockBits(bitmapData);
            }

            return tensor;
        }

        /// <summary>
        /// Runs inference session
        /// </summary>
        private DenseTensor<float>[] Inference(Image image)
        {
            Bitmap? resized = null;

            if (image.Width != m_model.Width || image.Height != m_model.Height)
            {
                resized = ResizeImage(image); // Fit image size to specified input size
            }

            var inputs = new List<NamedOnnxValue> // Add image as ONNX input
            {
                NamedOnnxValue.CreateFromTensor(m_model.Inputs[0], ExtractPixels(resized ?? image))
            };

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result = m_inferenceSession.Run(inputs); // run inference

            var output = new List<DenseTensor<float>>();

            foreach (var item in m_model.Outputs) // Add outputs for processing
            {
                output.Add((DenseTensor<float>)result.First(x => x.Name == item).Value);
            };

            resized?.Dispose();

            return output.ToArray();
        }
        #endregion
    }
}
