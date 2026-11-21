using ClearView.Core;
using UnityEngine;

namespace ClearView
{
    /// <summary>Turns a 2D box from the camera image into 4 world-space corners lying on the real surface.</summary>
    public static class MaskProjector
    {
        // If the surface normal is more than ~70 degrees from facing the camera, distrust it.
        private const float MinFacingDot = 0.34f;

        /// <param name="corners">Output, length 4: bottom-left, bottom-right, top-left, top-right.</param>
        /// <returns>False if the surface could not be found: the caller must then leave the ad visible.</returns>
        public static bool TryProject(
            RectN box, CameraIntrinsics k, Pose camera, IDepthProvider depth, float maxDistance,
            Vector3[] corners, out float distance)
        {
            distance = float.NaN;
            if (depth == null || !k.IsValid) return false;

            Ray center = RayFor(k, camera, box.CenterX, box.CenterY);
            if (!depth.TryRaycast(center, maxDistance, out Vector3 hitPoint, out Vector3 normal)) return false;

            distance = Vector3.Distance(camera.position, hitPoint);
            float alongRay = Vector3.Dot(hitPoint - center.origin, center.direction);

            if (normal.sqrMagnitude < 1e-6f || Vector3.Dot(normal.normalized, -center.direction) < MinFacingDot)
                normal = -center.direction;
            var plane = new Plane(normal.normalized, hitPoint);

            corners[0] = CornerOnPlane(k, camera, plane, box.X, box.Bottom, alongRay, maxDistance);
            corners[1] = CornerOnPlane(k, camera, plane, box.Right, box.Bottom, alongRay, maxDistance);
            corners[2] = CornerOnPlane(k, camera, plane, box.X, box.Y, alongRay, maxDistance);
            corners[3] = CornerOnPlane(k, camera, plane, box.Right, box.Y, alongRay, maxDistance);
            return true;
        }

        public static Ray RayFor(CameraIntrinsics k, Pose camera, float nx, float nyDown)
        {
            k.Unproject(nx, nyDown, out float dx, out float dy, out float dz);
            Vector3 dir = camera.rotation * new Vector3(dx, dy, dz).normalized;
            return new Ray(camera.position, dir);
        }

        private static Vector3 CornerOnPlane(CameraIntrinsics k, Pose camera, Plane plane, float nx, float ny, float fallbackDistance, float maxDistance)
        {
            Ray r = RayFor(k, camera, nx, ny);
            if (plane.Raycast(r, out float t) && t > 0f && t < maxDistance * 2f) return r.GetPoint(t);
            return r.GetPoint(fallbackDistance);
        }
    }
}
