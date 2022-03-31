namespace Deltatre.BallDetector.Onnx.Demo.Extensions
{
    using System.Drawing;

    public static class RectangleExtensions
    {
        public static float Area(this RectangleF source)
        {
            return source.Width * source.Height;
        }
    }
}
