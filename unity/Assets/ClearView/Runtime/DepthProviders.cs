using UnityEngine;
#if CLEARVIEW_MRUK
using Meta.XR;
#endif

namespace ClearView
{
    /// <summary>
    /// Fallback when no environment depth is available: assumes every ad sits on a plane facing the
    /// camera at a fixed distance. Masks still line up from the capture viewpoint but drift with parallax.
    /// The distance rule (R6) cannot be evaluated for real, so keep this ONLY for desk testing.
    /// </summary>
    public sealed class FixedDistanceDepthProvider : MonoBehaviour, IDepthProvider
    {
        [SerializeField] private float distanceMeters = 4f;

        public bool TryRaycast(Ray ray, float maxDistance, out Vector3 point, out Vector3 normal)
        {
            float d = Mathf.Min(distanceMeters, maxDistance);
            point = ray.GetPoint(d);
            normal = -ray.direction;
            return true;
        }
    }

#if CLEARVIEW_MRUK
    /// <summary>
    /// Uses MR Utility Kit's EnvironmentRaycastManager (Depth API) to find the real surface the ad is on.
    /// Requires the USE_SCENE permission and Quest 3 / 3S.
    /// </summary>
    public sealed class MrukDepthProvider : MonoBehaviour, IDepthProvider
    {
        [SerializeField] private EnvironmentRaycastManager raycastManager;

        public bool TryRaycast(Ray ray, float maxDistance, out Vector3 point, out Vector3 normal)
        {
            if (raycastManager != null && raycastManager.Raycast(ray, out EnvironmentRaycastHit hit, maxDistance))
            {
                point = hit.point;
                normal = hit.normal;
                return true;
            }
            point = default;
            normal = default;
            return false;
        }
    }
#endif
}
