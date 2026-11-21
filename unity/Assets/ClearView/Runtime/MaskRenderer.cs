using System.Collections.Generic;
using ClearView.Core;
using UnityEngine;

namespace ClearView
{
    /// <summary>
    /// Draws one world-locked, soft-edged patch per hidden ad. Patches fade in slowly and fade out fast.
    /// </summary>
    public sealed class MaskRenderer : MonoBehaviour
    {
        [SerializeField] private Material maskMaterial;
        [SerializeField] private float fadeInSeconds = 0.3f;
        [SerializeField] private float fadeOutSeconds = 0.1f;
        [Tooltip("Higher = corners follow new detections faster.")]
        [SerializeField] private float positionFollowRate = 10f;
        [SerializeField] private float colorFollowRate = 4f;
        [SerializeField, Range(0.01f, 0.5f)] private float feather = 0.12f;
        [Tooltip("Paint a magenta outline to check alignment while developing.")]
        [SerializeField] private bool debugOutline;

        private static readonly int ColorL = Shader.PropertyToID("_ColorL");
        private static readonly int ColorR = Shader.PropertyToID("_ColorR");
        private static readonly int ColorT = Shader.PropertyToID("_ColorT");
        private static readonly int ColorB = Shader.PropertyToID("_ColorB");
        private static readonly int Opacity = Shader.PropertyToID("_Opacity");
        private static readonly int Feather = Shader.PropertyToID("_Feather");
        private static readonly int DebugOutline = Shader.PropertyToID("_DebugOutline");

        private sealed class Visual
        {
            public GameObject Go;
            public Mesh Mesh;
            public MeshRenderer Renderer;
            public readonly Vector3[] Current = new Vector3[4];
            public readonly Vector3[] Target = new Vector3[4];
            public readonly Color[] CurrentColors = new Color[4];
            public readonly Color[] TargetColors = new Color[4];
            public float Opacity;
            public float TargetOpacity;
            public bool SeenThisCycle;
        }

        private readonly Dictionary<int, Visual> _visuals = new Dictionary<int, Visual>();
        private readonly List<int> _toRemove = new List<int>();
        private MaterialPropertyBlock _mpb;

        public int VisibleCount { get; private set; }

        private void Awake()
        {
            _mpb = new MaterialPropertyBlock();
        }

        public void BeginCycle()
        {
            foreach (var v in _visuals.Values) v.SeenThisCycle = false;
        }

        public void SetTarget(int id, Vector3[] corners, EdgeColors colors, float opacity)
        {
            bool isNew = !_visuals.TryGetValue(id, out Visual v);
            if (isNew)
            {
                v = Create(id);
                _visuals.Add(id, v);
            }
            for (int i = 0; i < 4; i++) v.Target[i] = corners[i];
            v.TargetColors[0] = ToColor(colors.Left);
            v.TargetColors[1] = ToColor(colors.Right);
            v.TargetColors[2] = ToColor(colors.Top);
            v.TargetColors[3] = ToColor(colors.Bottom);
            if (isNew)
            {
                for (int i = 0; i < 4; i++)
                {
                    v.Current[i] = v.Target[i];
                    v.CurrentColors[i] = v.TargetColors[i];
                }
            }
            v.TargetOpacity = opacity;
            v.SeenThisCycle = true;
        }

        /// <summary>Anything not refreshed during this detection cycle starts fading out.</summary>
        public void EndCycle()
        {
            foreach (var v in _visuals.Values)
                if (!v.SeenThisCycle) v.TargetOpacity = 0f;
        }

        /// <param name="globalFactor">0..1 multiplier from motion guard and health.</param>
        /// <param name="hardOff">Kill switch: remove everything this frame, no fade.</param>
        public void Tick(float dt, float globalFactor, bool hardOff)
        {
            VisibleCount = 0;
            float posK = 1f - Mathf.Exp(-positionFollowRate * dt);
            float colK = 1f - Mathf.Exp(-colorFollowRate * dt);

            foreach (var kv in _visuals)
            {
                Visual v = kv.Value;
                float goal = hardOff ? 0f : v.TargetOpacity * Mathf.Clamp01(globalFactor);
                if (hardOff) v.Opacity = 0f;
                else
                {
                    float seconds = goal > v.Opacity ? fadeInSeconds : fadeOutSeconds;
                    v.Opacity = Mathf.MoveTowards(v.Opacity, goal, dt / Mathf.Max(0.01f, seconds));
                }

                for (int i = 0; i < 4; i++)
                {
                    v.Current[i] = Vector3.Lerp(v.Current[i], v.Target[i], posK);
                    v.CurrentColors[i] = Color.Lerp(v.CurrentColors[i], v.TargetColors[i], colK);
                }

                bool visible = v.Opacity > 0.001f;
                v.Go.SetActive(visible);
                if (visible)
                {
                    VisibleCount++;
                    v.Mesh.vertices = v.Current;
                    v.Mesh.RecalculateBounds();
                    _mpb.SetColor(ColorL, v.CurrentColors[0]);
                    _mpb.SetColor(ColorR, v.CurrentColors[1]);
                    _mpb.SetColor(ColorT, v.CurrentColors[2]);
                    _mpb.SetColor(ColorB, v.CurrentColors[3]);
                    _mpb.SetFloat(Opacity, v.Opacity);
                    _mpb.SetFloat(Feather, feather);
                    _mpb.SetFloat(DebugOutline, debugOutline ? 1f : 0f);
                    v.Renderer.SetPropertyBlock(_mpb);
                }
                else if (v.TargetOpacity <= 0f)
                {
                    _toRemove.Add(kv.Key);
                }
            }

            foreach (int id in _toRemove) RemoveVisual(id);
            _toRemove.Clear();
        }

        public void ClearAll()
        {
            foreach (var v in _visuals.Values) DestroyVisual(v);
            _visuals.Clear();
            VisibleCount = 0;
        }

        private Visual Create(int id)
        {
            var go = new GameObject($"ClearViewMask_{id}");
            // Vertices are written in world space, so the mask lives at the scene root with an identity transform.

            var mesh = new Mesh { name = go.name };
            mesh.MarkDynamic();
            mesh.vertices = new Vector3[4];
            mesh.uv = new[] { new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1), new Vector2(1, 1) };
            mesh.triangles = new[] { 0, 2, 1, 2, 3, 1 };

            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            var mr = go.AddComponent<MeshRenderer>();
            mr.sharedMaterial = maskMaterial;
            mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            mr.receiveShadows = false;
            go.SetActive(false);
            return new Visual { Go = go, Mesh = mesh, Renderer = mr };
        }

        private void RemoveVisual(int id)
        {
            if (_visuals.TryGetValue(id, out Visual v))
            {
                DestroyVisual(v);
                _visuals.Remove(id);
            }
        }

        private static void DestroyVisual(Visual v)
        {
            Object.Destroy(v.Mesh);
            Object.Destroy(v.Go);
        }

        // Camera bytes are sRGB; SetColor converts sRGB to linear when the project uses linear colour space.
        private static Color ToColor(Rgb c) => new Color(c.R, c.G, c.B, 1f);

        private void OnDestroy() => ClearAll();
    }
}
