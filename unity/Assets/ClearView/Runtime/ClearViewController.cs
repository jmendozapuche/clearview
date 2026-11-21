using System;
using System.Collections.Generic;
using System.Text;
using ClearView.Core;
using Unity.InferenceEngine; // On Sentis 2.1 use: using Unity.Sentis;
using UnityEngine;

namespace ClearView
{
    /// <summary>
    /// Orchestrates the pipeline: camera frame -> YOLO -> tracker -> safety policy -> world-locked masks.
    /// Every failure path removes the masks and leaves plain passthrough (fail-open).
    /// </summary>
    public sealed class ClearViewController : MonoBehaviour
    {
        [Header("Components")]
        [Tooltip("A component implementing IFrameSource (QuestPassthroughFrameSource or EditorWebcamFrameSource).")]
        [SerializeField] private MonoBehaviour frameSource;
        [Tooltip("A component implementing IDepthProvider (MrukDepthProvider on device).")]
        [SerializeField] private MonoBehaviour depthProvider;
        [SerializeField] private MaskRenderer maskRenderer;
        [Tooltip("The user's head (CenterEyeAnchor). Defaults to the main camera.")]
        [SerializeField] private Transform head;
        [Tooltip("Small head-locked object that shows the state: green = hiding ads, amber = paused, hidden = off.")]
        [SerializeField] private Renderer statusIndicator;

        [Header("Model")]
        [SerializeField] private ModelAsset model;
        [Tooltip("clearview_labels.txt produced by ml/scripts/export.py")]
        [SerializeField] private TextAsset labels;
        [SerializeField] private int inputWidth = 416;
        [SerializeField] private int inputHeight = 320;
        [SerializeField] private BackendType backend = BackendType.GPUCompute;
        [Tooltip("Network layers scheduled per rendered frame. Lower = smoother frame rate, slower detections.")]
        [SerializeField] private int layersPerFrame = 20;

        [Header("Placement and colour")]
        [SerializeField] private float maxRaycastDistance = 30f;
        [SerializeField] private int colorSampleWidth = 96;
        [SerializeField] private int colorSampleHeight = 72;
        [SerializeField] private bool colorRowsBottomUp = true;
        [Tooltip("Multiplies sampled colours to match how the passthrough looks.")]
        [SerializeField, Range(0.5f, 1.5f)] private float colorGain = 1f;
        [SerializeField, Range(0.02f, 0.5f)] private float borderRingFraction = 0.15f;
        [Tooltip("Never run with the fixed-distance fallback on a headset: the distance rule would be fake.")]
        [SerializeField] private bool allowFixedDepthOnDevice;

        [Header("Safety (see docs/SAFETY.md)")]
        public DecoderConfig decoder = new DecoderConfig();
        public TrackerConfig tracker = new TrackerConfig();
        public SafetyConfig safety = new SafetyConfig();
        public MotionConfig motion = new MotionConfig();
        public FailSafeConfig failSafe = new FailSafeConfig();

        [Header("Kill switch")]
        [Tooltip("Off by default: the user turns ClearView on consciously.")]
        [SerializeField] private bool startEnabled;
#if ENABLE_LEGACY_INPUT_MANAGER
        [SerializeField] private KeyCode editorToggleKey = KeyCode.Space;
#endif

        [Header("Debug")]
        [SerializeField] private bool logStats = true;
        [SerializeField] private float logIntervalSeconds = 5f;

        private IFrameSource _source;
        private IDepthProvider _depth;
        private LabelMap _labels;
        private AdDetector _detector;
        private FrameColorCapture _colors;
        private AdTracker _tracker;
        private SafetyPolicy _policy;
        private MotionGuard _motion;
        private FailSafe _failSafe;
        private bool _broken;

        private Pose _jobPose;
        private CameraIntrinsics _jobIntrinsics;
        private double _jobStart;

        private Vector3 _prevHeadPos;
        private Quaternion _prevHeadRot;
        private bool _hasPrevHead;

        private readonly List<Detection> _ads = new List<Detection>();
        private readonly List<Detection> _safetyObjects = new List<Detection>();
        private readonly Vector3[] _corners = new Vector3[4];
        private MaterialPropertyBlock _statusMpb;

        // Stats
        private double _nextLog;
        private float _lastInferenceMs;
        private int _lastAdCount, _lastSafetyCount, _lastMaskCount, _noSurface, _tooClose;
        private PolicyResult _lastPolicy;

        /// <summary>True when the user has switched ClearView on.</summary>
        public bool IsOn => _failSafe != null && _failSafe.UserEnabled;

        public void SetOn(bool on)
        {
            if (_failSafe == null) return;
            _failSafe.UserEnabled = on && !_broken;
            if (!_failSafe.UserEnabled)
            {
                _tracker?.Clear();
                if (maskRenderer != null) maskRenderer.ClearAll();
            }
            Debug.Log($"[ClearView] {(on ? "ON" : "OFF")}");
        }

        private void Start()
        {
            _source = frameSource as IFrameSource;
            _depth = depthProvider as IDepthProvider;
            _tracker = new AdTracker(tracker);
            _policy = new SafetyPolicy(safety, tracker);
            _motion = new MotionGuard(motion);
            _failSafe = new FailSafe(failSafe);
            _statusMpb = new MaterialPropertyBlock();

            try
            {
                if (_source == null) throw new InvalidOperationException("frameSource does not implement IFrameSource");
                if (_depth == null) throw new InvalidOperationException("depthProvider does not implement IDepthProvider");
                if (maskRenderer == null) throw new InvalidOperationException("maskRenderer is not assigned");
                if (labels == null) throw new InvalidOperationException("labels file is not assigned");
                if (!Application.isEditor && depthProvider is FixedDistanceDepthProvider && !allowFixedDepthOnDevice)
                    throw new InvalidOperationException("FixedDistanceDepthProvider is for desk testing only; use MrukDepthProvider on device");

                _labels = LabelMap.Parse(labels.text);
                _detector = new AdDetector(model, inputWidth, inputHeight, backend, layersPerFrame);
                _colors = new FrameColorCapture(colorSampleWidth, colorSampleHeight);
            }
            catch (Exception e)
            {
                Fail(e);
                return;
            }

            SetOn(startEnabled);
        }

        private void Update()
        {
            double now = Time.realtimeSinceStartupAsDouble;
            float dt = Time.unscaledDeltaTime;

            HandleKillSwitch();
            float motionFactor = UpdateMotion(dt);
            _failSafe?.ReportFrame(now, dt);

            if (!_broken && IsOn)
            {
                try
                {
                    if (_detector.IsIdle && _source.IsReady) StartDetection(now);
                    if (_detector.Tick(out float[] output, out int channels, out int anchors))
                        ProcessResult(output, channels, anchors, now);
                }
                catch (Exception e)
                {
                    Fail(e);
                }
            }

            bool healthy = _failSafe != null && _failSafe.IsHealthy(now);
            if (maskRenderer != null) maskRenderer.Tick(dt, healthy ? motionFactor : 0f, hardOff: !IsOn);
            UpdateStatusIndicator(healthy && motionFactor > 0.5f);
            LogStats(now, motionFactor);
        }

        private void StartDetection(double now)
        {
            // Pose and calibration are captured together with the frame, so masks are placed
            // where the ad WAS when photographed, not where the head is when inference finishes.
            _jobPose = _source.CameraPoseInWorld;
            _jobIntrinsics = _source.Intrinsics;
            _jobStart = now;
            _colors.Capture(_source.Texture);
            _detector.Begin(_source.Texture);
        }

        private void ProcessResult(float[] output, int channels, int anchors, double now)
        {
            _lastInferenceMs = (float)((now - _jobStart) * 1000.0);
            int numClasses = channels - 4;
            List<Detection> detections = YoloDecoder.Decode(output, numClasses, anchors, inputWidth, inputHeight, _labels, decoder);

            _ads.Clear();
            _safetyObjects.Clear();
            foreach (Detection d in detections)
            {
                if (d.Category == LabelCategory.Ad) _ads.Add(d);
                else if (d.Category == LabelCategory.Safety) _safetyObjects.Add(d);
            }

            _tracker.Update(_ads, now);
            PolicyResult result = _policy.Evaluate(_tracker.Tracks, _safetyObjects, now);

            maskRenderer.BeginCycle();
            int shown = 0;
            foreach (MaskDecision m in result.Masks)
            {
                if (!MaskProjector.TryProject(m.Box, _jobIntrinsics, _jobPose, _depth, maxRaycastDistance, _corners, out float distance))
                {
                    _noSurface++; // R6 cannot be checked without a surface: leave the ad visible.
                    continue;
                }
                if (!_policy.AllowsDistance(distance))
                {
                    _tooClose++;
                    continue;
                }

                EdgeColors colors = _colors.HasData
                    ? BorderColorSampler.Sample(_colors.Pixels, _colors.Width, _colors.Height, m.SourceBox, borderRingFraction, colorRowsBottomUp)
                    : EdgeColors.Uniform(Rgb.NeutralGray);
                maskRenderer.SetTarget(m.TrackId, _corners, Gain(colors), m.Opacity);
                shown++;
            }
            maskRenderer.EndCycle();

            _failSafe.ReportResult(now);
            _lastPolicy = result;
            _lastAdCount = _ads.Count;
            _lastSafetyCount = _safetyObjects.Count;
            _lastMaskCount = shown;
        }

        private float UpdateMotion(float dt)
        {
            Transform t = head != null ? head : (Camera.main != null ? Camera.main.transform : null);
            if (t == null || _motion == null) return 0f;
            if (!_hasPrevHead || dt <= 0f)
            {
                _prevHeadPos = t.position;
                _prevHeadRot = t.rotation;
                _hasPrevHead = true;
                return _motion.Update(0f, 0f, Mathf.Max(dt, 1e-3f));
            }
            float ang = Quaternion.Angle(_prevHeadRot, t.rotation) / dt;
            float lin = (t.position - _prevHeadPos).magnitude / dt;
            _prevHeadPos = t.position;
            _prevHeadRot = t.rotation;
            return _motion.Update(ang, lin, dt);
        }

        private void HandleKillSwitch()
        {
            bool toggle = false;
#if CLEARVIEW_META
            // A / X on the controllers, or the menu (Start) button / left-hand menu gesture.
            toggle |= OVRInput.GetDown(OVRInput.Button.One) || OVRInput.GetDown(OVRInput.Button.Three) || OVRInput.GetDown(OVRInput.Button.Start);
#endif
#if ENABLE_LEGACY_INPUT_MANAGER
            toggle |= Input.GetKeyDown(editorToggleKey);
#endif
            if (toggle) SetOn(!IsOn);
        }

        private void Fail(Exception e)
        {
            _broken = true;
            if (_failSafe != null) _failSafe.UserEnabled = false;
            _tracker?.Clear();
            if (maskRenderer != null) maskRenderer.ClearAll();
            Debug.LogError($"[ClearView] Disabled (showing plain passthrough): {e}");
        }

        private EdgeColors Gain(EdgeColors c)
        {
            if (Mathf.Approximately(colorGain, 1f)) return c;
            c.Left = Gain(c.Left);
            c.Right = Gain(c.Right);
            c.Top = Gain(c.Top);
            c.Bottom = Gain(c.Bottom);
            return c;
        }

        private Rgb Gain(Rgb c) => new Rgb(Mathf.Clamp01(c.R * colorGain), Mathf.Clamp01(c.G * colorGain), Mathf.Clamp01(c.B * colorGain));

        private void UpdateStatusIndicator(bool active)
        {
            if (statusIndicator == null) return;
            statusIndicator.enabled = IsOn;
            if (!IsOn) return;
            Color c = active ? new Color(0.2f, 0.85f, 0.4f) : new Color(1f, 0.7f, 0.1f);
            statusIndicator.GetPropertyBlock(_statusMpb);
            _statusMpb.SetColor("_Color", c);      // Built-in
            _statusMpb.SetColor("_BaseColor", c);  // URP
            statusIndicator.SetPropertyBlock(_statusMpb);
        }

        private void LogStats(double now, float motionFactor)
        {
            if (!logStats || now < _nextLog) return;
            _nextLog = now + logIntervalSeconds;
            var sb = new StringBuilder("[ClearView] ");
            sb.Append(IsOn ? "on" : "off");
            if (_failSafe != null) sb.Append(" health=").Append(_failSafe.Reason);
            sb.Append($" fps={1f / Mathf.Max(Time.unscaledDeltaTime, 1e-4f):F0} inference={_lastInferenceMs:F0}ms motion={motionFactor:F2}");
            sb.Append($" ads={_lastAdCount} safetyObjs={_lastSafetyCount} masks={_lastMaskCount} visible={(maskRenderer != null ? maskRenderer.VisibleCount : 0)}");
            if (_lastPolicy != null)
                foreach (var kv in _lastPolicy.Skipped) sb.Append($" skip.{kv.Key}={kv.Value}");
            sb.Append($" noSurface={_noSurface} tooClose={_tooClose}");
            Debug.Log(sb.ToString());
        }

        private void OnDestroy()
        {
            _detector?.Dispose();
            _colors?.Dispose();
        }
    }
}
