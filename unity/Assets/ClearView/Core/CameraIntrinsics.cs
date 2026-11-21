using System;

namespace ClearView.Core
{
    /// <summary>
    /// Pinhole intrinsics in pixels, using the same convention as Meta's PassthroughCameraUtils:
    /// the principal point and image rows are measured with y growing UP.
    /// </summary>
    [Serializable]
    public struct CameraIntrinsics
    {
        public float Fx, Fy, Cx, Cy;
        public int Width, Height;

        public CameraIntrinsics(float fx, float fy, float cx, float cy, int width, int height)
        {
            Fx = fx; Fy = fy; Cx = cx; Cy = cy; Width = width; Height = height;
        }

        /// <summary>Intrinsics for an ideal camera with the given horizontal field of view.</summary>
        public static CameraIntrinsics FromHorizontalFov(float hfovDegrees, int width, int height)
        {
            float f = (float)(width * 0.5 / Math.Tan(hfovDegrees * Math.PI / 360.0));
            return new CameraIntrinsics(f, f, width * 0.5f, height * 0.5f, width, height);
        }

        public bool IsValid => Fx > 0f && Fy > 0f && Width > 0 && Height > 0;

        /// <summary>
        /// Converts a normalized, y-down image point into a direction in camera space
        /// (x right, y up, z forward). The result is not normalized; z is always 1.
        /// </summary>
        public void Unproject(float nx, float nyDown, out float dx, out float dy, out float dz)
        {
            float u = nx * Width;
            float vUp = (1f - nyDown) * Height;
            dx = (u - Cx) / Fx;
            dy = (vUp - Cy) / Fy;
            dz = 1f;
        }
    }
}
