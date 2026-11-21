using System;

namespace ClearView.Core
{
    [Serializable]
    public sealed class FailSafeConfig
    {
        /// <summary>R1: if no detection result arrives within this time, every mask is removed.</summary>
        public float ResultTimeoutSeconds = 1.0f;
        /// <summary>R10: frame rate under which the app is considered unhealthy.</summary>
        public float MinFps = 60f;
        /// <summary>How long the frame rate may stay below MinFps before masks are removed.</summary>
        public float LowFpsGraceSeconds = 1.0f;
    }

    /// <summary>
    /// Watchdog. Anything abnormal turns the system off and shows plain passthrough (fail-open).
    /// </summary>
    public sealed class FailSafe
    {
        private readonly FailSafeConfig _cfg;
        private double _lastResult = double.NegativeInfinity;
        private double _lowFpsSince = double.NaN;
        private bool _lowFps;

        public FailSafe(FailSafeConfig cfg)
        {
            _cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
        }

        /// <summary>Set by the user's kill switch. Starts disabled so the user opts in consciously.</summary>
        public bool UserEnabled { get; set; }

        public string Reason { get; private set; } = "disabled by user";

        public void ReportResult(double now) => _lastResult = now;

        public void ReportFrame(double now, float dt)
        {
            if (dt <= 0f) return;
            float fps = 1f / dt;
            if (fps < _cfg.MinFps)
            {
                if (double.IsNaN(_lowFpsSince)) _lowFpsSince = now;
                _lowFps = now - _lowFpsSince >= _cfg.LowFpsGraceSeconds;
            }
            else
            {
                _lowFpsSince = double.NaN;
                _lowFps = false;
            }
        }

        public bool IsHealthy(double now)
        {
            if (!UserEnabled) { Reason = "disabled by user"; return false; }
            if (now - _lastResult > _cfg.ResultTimeoutSeconds) { Reason = "no recent detections"; return false; }
            if (_lowFps) { Reason = "frame rate too low"; return false; }
            Reason = "ok";
            return true;
        }
    }
}
