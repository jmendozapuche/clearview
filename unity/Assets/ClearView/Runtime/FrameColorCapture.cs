using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace ClearView
{
    /// <summary>Keeps a tiny CPU copy of the camera frame to pick the colour around each ad.</summary>
    public sealed class FrameColorCapture : IDisposable
    {
        private readonly RenderTexture _rt;
        private byte[] _pixels;
        private bool _pending;

        public FrameColorCapture(int width, int height)
        {
            Width = width;
            Height = height;
            _rt = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32);
            _rt.Create();
        }

        public int Width { get; }
        public int Height { get; }
        public bool HasData { get; private set; }
        public byte[] Pixels => _pixels;

        public void Capture(Texture source)
        {
            if (_pending || source == null) return;
            Graphics.Blit(source, _rt);
            _pending = true;
            AsyncGPUReadback.Request(_rt, 0, TextureFormat.RGBA32, OnReadback);
        }

        private void OnReadback(AsyncGPUReadbackRequest req)
        {
            _pending = false;
            if (req.hasError)
            {
                HasData = false;
                return;
            }
            var data = req.GetData<byte>();
            if (_pixels == null || _pixels.Length != data.Length) _pixels = new byte[data.Length];
            data.CopyTo(_pixels);
            HasData = true;
        }

        public void Dispose()
        {
            if (_rt != null)
            {
                _rt.Release();
                UnityEngine.Object.Destroy(_rt);
            }
        }
    }
}
