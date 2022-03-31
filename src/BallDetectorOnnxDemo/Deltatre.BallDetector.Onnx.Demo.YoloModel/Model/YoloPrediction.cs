// Based on: https://github.com/mentalstack/yolov5-net

namespace Deltatre.BallDetector.Onnx.Demo.Model
{
    using System.Drawing;

    /// <summary>
    /// Object prediction
    /// </summary>
    public class YoloPrediction
    {
        public YoloLabel Label { get; set; }
        public RectangleF Rectangle { get; set; }
        public float Score { get; set; }

        public YoloPrediction(YoloLabel label, float confidence) : this(label)
        {
            Score = confidence;
        }

        public YoloPrediction(YoloLabel label)
        {
            Label = label;
        }
    }
}
