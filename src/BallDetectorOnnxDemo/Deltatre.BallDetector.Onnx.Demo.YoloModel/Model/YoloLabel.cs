// Based on: https://github.com/mentalstack/yolov5-net

namespace Deltatre.BallDetector.Onnx.Demo.Model
{
    using System.Drawing;
    
    /// <summary>
    /// Label of detected object.
    /// </summary>
    public class YoloLabel
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public YoloLabelKind Kind { get; set; }
        public Color Color { get; set; }

        public YoloLabel()
        {
            Color = Color.Yellow;
        }
    }
}
