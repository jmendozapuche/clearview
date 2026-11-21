using System;

namespace ClearView.Core
{
    /// <summary>
    /// Axis-aligned rectangle in normalized image coordinates.
    /// Origin is the TOP-LEFT corner of the camera image, x grows right, y grows DOWN,
    /// and all values are expressed as a fraction of the image size (0..1).
    /// </summary>
    [Serializable]
    public struct RectN
    {
        public float X;
        public float Y;
        public float W;
        public float H;

        public RectN(float x, float y, float w, float h)
        {
            X = x;
            Y = y;
            W = w < 0f ? 0f : w;
            H = h < 0f ? 0f : h;
        }

        public static RectN FromCenter(float cx, float cy, float w, float h)
        {
            return new RectN(cx - w * 0.5f, cy - h * 0.5f, w, h);
        }

        public float Right => X + W;
        public float Bottom => Y + H;
        public float CenterX => X + W * 0.5f;
        public float CenterY => Y + H * 0.5f;
        public float Area => W * H;

        public static float IntersectionArea(RectN a, RectN b)
        {
            float w = Math.Min(a.Right, b.Right) - Math.Max(a.X, b.X);
            float h = Math.Min(a.Bottom, b.Bottom) - Math.Max(a.Y, b.Y);
            return (w <= 0f || h <= 0f) ? 0f : w * h;
        }

        public static float IoU(RectN a, RectN b)
        {
            float inter = IntersectionArea(a, b);
            float union = a.Area + b.Area - inter;
            return union <= 0f ? 0f : inter / union;
        }

        /// <summary>Shrinks the rectangle by <paramref name="fraction"/> of its size on every side.</summary>
        public RectN Inset(float fraction)
        {
            float dx = W * fraction;
            float dy = H * fraction;
            return new RectN(X + dx, Y + dy, W - 2f * dx, H - 2f * dy);
        }

        /// <summary>Grows the rectangle by <paramref name="fraction"/> of its size on every side.</summary>
        public RectN Expand(float fraction)
        {
            return Inset(-fraction);
        }

        public RectN ClampToUnit()
        {
            float x0 = Clamp01(X), y0 = Clamp01(Y);
            float x1 = Clamp01(Right), y1 = Clamp01(Bottom);
            return new RectN(x0, y0, x1 - x0, y1 - y0);
        }

        public static RectN Lerp(RectN a, RectN b, float t)
        {
            return new RectN(
                a.X + (b.X - a.X) * t,
                a.Y + (b.Y - a.Y) * t,
                a.W + (b.W - a.W) * t,
                a.H + (b.H - a.H) * t);
        }

        internal static float Clamp01(float v) => v < 0f ? 0f : (v > 1f ? 1f : v);

        public override string ToString() => $"RectN(x={X:F3}, y={Y:F3}, w={W:F3}, h={H:F3})";
    }
}
