using ClearView.Core;
using UnityEngine;

namespace ClearView
{
    /// <summary>Supplies camera frames plus the calibration needed to map pixels back into the world.</summary>
    public interface IFrameSource
    {
        bool IsReady { get; }
        Texture Texture { get; }
        CameraIntrinsics Intrinsics { get; }
        /// <summary>Pose of the physical camera in world space, sampled as close as possible to the frame capture.</summary>
        Pose CameraPoseInWorld { get; }
    }

    /// <summary>Finds where a ray hits a real surface.</summary>
    public interface IDepthProvider
    {
        bool TryRaycast(Ray ray, float maxDistance, out Vector3 point, out Vector3 normal);
    }
}
