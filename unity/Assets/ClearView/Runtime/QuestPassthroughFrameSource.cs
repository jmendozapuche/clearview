#if CLEARVIEW_META
using ClearView.Core;
using PassthroughCameraSamples;
using UnityEngine;

namespace ClearView
{
    /// <summary>
    /// Quest 3 / 3S source built on Meta's Passthrough Camera API samples
    /// (github.com/oculus-samples/Unity-PassthroughCameraApiSamples): WebCamTextureManager handles the
    /// HEADSET_CAMERA permission and the WebCamTexture, PassthroughCameraUtils provides calibration.
    ///
    /// THIS IS THE ONLY FILE THAT TOUCHES THE CAMERA API. If your Meta SDK version exposes the camera
    /// differently (e.g. a newer MRUK component), adapt this adapter and nothing else.
    /// </summary>
    public sealed class QuestPassthroughFrameSource : MonoBehaviour, IFrameSource
    {
        [SerializeField] private WebCamTextureManager webCamTextureManager;
        [SerializeField] private PassthroughCameraEye eye = PassthroughCameraEye.Left;

        private bool _hasIntrinsics;
        private CameraIntrinsics _intrinsics;

        public bool IsReady
        {
            get
            {
                WebCamTexture t = webCamTextureManager != null ? webCamTextureManager.WebCamTexture : null;
                return t != null && t.isPlaying && t.width > 16;
            }
        }

        public Texture Texture => webCamTextureManager.WebCamTexture;

        public CameraIntrinsics Intrinsics
        {
            get
            {
                if (!_hasIntrinsics)
                {
                    var k = PassthroughCameraUtils.GetCameraIntrinsics(eye);
                    // Detections use normalized coordinates, so using the calibration resolution is fine
                    // as long as the WebCamTexture keeps the same aspect ratio (no crop).
                    _intrinsics = new CameraIntrinsics(
                        k.FocalLength.x, k.FocalLength.y,
                        k.PrincipalPoint.x, k.PrincipalPoint.y,
                        k.Resolution.x, k.Resolution.y);
                    _hasIntrinsics = _intrinsics.IsValid;
                }
                return _intrinsics;
            }
        }

        public Pose CameraPoseInWorld => PassthroughCameraUtils.GetCameraPoseInWorld(eye);
    }
}
#endif
