using ClearView.Core;
using UnityEngine;

namespace ClearView
{
    /// <summary>
    /// Development source: uses a PC webcam and an assumed field of view so the detection
    /// pipeline can be exercised in the Editor without a headset.
    /// </summary>
    public sealed class EditorWebcamFrameSource : MonoBehaviour, IFrameSource
    {
        [SerializeField] private int requestedWidth = 1280;
        [SerializeField] private int requestedHeight = 960;
        [SerializeField] private float horizontalFovDegrees = 70f;
        [Tooltip("Defaults to the main camera.")]
        [SerializeField] private Transform cameraTransform;

        private WebCamTexture _texture;

        public bool IsReady => _texture != null && _texture.isPlaying && _texture.width > 16;
        public Texture Texture => _texture;
        public CameraIntrinsics Intrinsics => CameraIntrinsics.FromHorizontalFov(horizontalFovDegrees, _texture.width, _texture.height);

        public Pose CameraPoseInWorld
        {
            get
            {
                Transform t = cameraTransform != null ? cameraTransform : (Camera.main != null ? Camera.main.transform : transform);
                return new Pose(t.position, t.rotation);
            }
        }

        private void Start()
        {
            if (WebCamTexture.devices.Length == 0)
            {
                Debug.LogWarning("[ClearView] No webcam found; EditorWebcamFrameSource disabled.");
                enabled = false;
                return;
            }
            _texture = new WebCamTexture(WebCamTexture.devices[0].name, requestedWidth, requestedHeight, 30);
            _texture.Play();
        }

        private void OnDestroy()
        {
            if (_texture != null) _texture.Stop();
        }
    }
}
