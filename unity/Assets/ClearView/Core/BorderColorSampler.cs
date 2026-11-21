using System;

namespace ClearView.Core
{
    public struct Rgb
    {
        public float R, G, B;

        public Rgb(float r, float g, float b)
        {
            R = r;
            G = g;
            B = b;
        }

        public static readonly Rgb NeutralGray = new Rgb(0.5f, 0.5f, 0.5f);
    }

    /// <summary>Average colour just outside each edge of a box. The shader blends them so the patch looks like the surrounding wall.</summary>
    public struct EdgeColors
    {
        public Rgb Left, Right, Top, Bottom;

        public static EdgeColors Uniform(Rgb c) => new EdgeColors { Left = c, Right = c, Top = c, Bottom = c };
    }

    public static class BorderColorSampler
    {
        /// <param name="rgba">Tightly packed RGBA32 pixels.</param>
        /// <param name="box">Normalized, y-down box (the same space as detections).</param>
        /// <param name="ringFraction">Thickness of the sampled band, as a fraction of the box size.</param>
        /// <param name="rowsBottomUp">True if row 0 of <paramref name="rgba"/> is the BOTTOM of the image (Unity GPU readback default).</param>
        public static EdgeColors Sample(byte[] rgba, int width, int height, RectN box, float ringFraction, bool rowsBottomUp)
        {
            if (rgba == null || width <= 0 || height <= 0 || rgba.Length < width * height * 4)
                return EdgeColors.Uniform(Rgb.NeutralGray);

            int x0 = ToPx(box.X, width), x1 = ToPx(box.Right, width);
            int y0 = ToPx(box.Y, height), y1 = ToPx(box.Bottom, height);
            if (x1 <= x0) x1 = Math.Min(width, x0 + 1);
            if (y1 <= y0) y1 = Math.Min(height, y0 + 1);
            int bandX = Math.Max(1, (int)Math.Round((x1 - x0) * ringFraction));
            int bandY = Math.Max(1, (int)Math.Round((y1 - y0) * ringFraction));

            var c = new EdgeColors
            {
                // Prefer the band just outside the edge; if that falls off-image use the band just inside.
                Left = Band(rgba, width, height, rowsBottomUp, x0 - bandX, x0, y0, y1, x0, x0 + bandX),
                Right = Band(rgba, width, height, rowsBottomUp, x1, x1 + bandX, y0, y1, x1 - bandX, x1),
                Top = BandY(rgba, width, height, rowsBottomUp, x0, x1, y0 - bandY, y0, y0, y0 + bandY),
                Bottom = BandY(rgba, width, height, rowsBottomUp, x0, x1, y1, y1 + bandY, y1 - bandY, y1),
            };
            return c;
        }

        private static Rgb Band(byte[] px, int w, int h, bool bottomUp, int bx0, int bx1, int y0, int y1, int fx0, int fx1)
        {
            if (bx0 < 0 || bx1 > w) { bx0 = fx0; bx1 = fx1; }
            return Average(px, w, h, bottomUp, bx0, bx1, y0, y1);
        }

        private static Rgb BandY(byte[] px, int w, int h, bool bottomUp, int x0, int x1, int by0, int by1, int fy0, int fy1)
        {
            if (by0 < 0 || by1 > h) { by0 = fy0; by1 = fy1; }
            return Average(px, w, h, bottomUp, x0, x1, by0, by1);
        }

        private static Rgb Average(byte[] px, int w, int h, bool bottomUp, int x0, int x1, int y0, int y1)
        {
            x0 = Math.Max(0, x0); x1 = Math.Min(w, x1);
            y0 = Math.Max(0, y0); y1 = Math.Min(h, y1);
            long r = 0, g = 0, b = 0, n = 0;
            for (int y = y0; y < y1; y++)
            {
                int row = bottomUp ? (h - 1 - y) : y;
                int idx = (row * w + x0) * 4;
                for (int x = x0; x < x1; x++, idx += 4)
                {
                    r += px[idx];
                    g += px[idx + 1];
                    b += px[idx + 2];
                    n++;
                }
            }
            if (n == 0) return Rgb.NeutralGray;
            return new Rgb(r / (255f * n), g / (255f * n), b / (255f * n));
        }

        private static int ToPx(float normalized, int size)
        {
            int v = (int)Math.Round(normalized * size);
            return v < 0 ? 0 : (v > size ? size : v);
        }
    }
}
