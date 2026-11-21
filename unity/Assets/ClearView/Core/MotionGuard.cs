using System;

namespace ClearView.Core
{
    [Serializable]
    public sealed class MotionConfig
    {
        /// <summary>Head rotation (deg/s) where masks start to fade.</summary>
        public float AngularSoftDegPerSec = 60f;
        /// <summary>Head rotation (deg/s) where masks are fully gone (e.g. turning to check traffic).</summary>
        public float AngularHardDegPerSec = 120f;
        /// <summary>Head translation (m/s) where masks start to fade. ~1.4 m/s is a normal walk.</summary>
        public float LinearSoftMps = 1.6f;
        /// <summary>Head translation (m/s) where masks are fully gone (jogging, riding a vehicle).</summary>
        public float LinearHardMps = 2.3f;
        /// <summary>How fast (per second) the factor may climb back to 1 once motion calms down.</summary>
        public float RecoverPerSecond = 1.5f;
        /// <summary>Smoothing for the raw speed inputs (0..1, higher = less smoothing).</summary>
        public float InputSmoothing = 0.3f;
    }

    /// <summary>
    /// R8: converts head motion into a 0..1 multiplier for mask opacity.
    /// Drops instantly when motion increases, recovers slowly.
    /// </summary>
    public sealed class MotionGuard
    {
        private readonly MotionConfig _cfg;
        private float _ang;
        private float _lin;
        private bool _hasInput;

        public MotionGuard(MotionConfig cfg)
        {
            _cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
            Factor = 1f;
        }

        public float Factor { get; private set; }

        public float Update(float angularDegPerSec, float linearMps, float dt)
        {
            if (float.IsNaN(angularDegPerSec) || float.IsNaN(linearMps) || dt <= 0f)
            {
                // Unknown motion: fail open.
                Factor = 0f;
                return Factor;
            }

            if (!_hasInput)
            {
                _ang = angularDegPerSec;
                _lin = linearMps;
                _hasInput = true;
            }
            else
            {
                float k = _cfg.InputSmoothing;
                _ang += (angularDegPerSec - _ang) * k;
                _lin += (linearMps - _lin) * k;
            }

            // Use the max of smoothed and raw so a sudden head turn is never smoothed away.
            float ang = Math.Max(_ang, angularDegPerSec);
            float lin = Math.Max(_lin, linearMps);

            float target = Math.Min(
                Ramp(ang, _cfg.AngularSoftDegPerSec, _cfg.AngularHardDegPerSec),
                Ramp(lin, _cfg.LinearSoftMps, _cfg.LinearHardMps));

            if (target < Factor) Factor = target;
            else Factor = Math.Min(target, Factor + _cfg.RecoverPerSecond * dt);
            return Factor;
        }

        private static float Ramp(float v, float soft, float hard)
        {
            if (v <= soft) return 1f;
            if (v >= hard || hard <= soft) return 0f;
            return 1f - (v - soft) / (hard - soft);
        }
    }
}
