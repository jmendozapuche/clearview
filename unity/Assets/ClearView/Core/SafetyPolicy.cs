using System;
using System.Collections.Generic;

namespace ClearView.Core
{
    /// <summary>
    /// Every knob that decides WHAT may be hidden. Defaults are conservative on purpose;
    /// see docs/SAFETY.md for the reasoning behind each rule.
    /// </summary>
    [Serializable]
    public sealed class SafetyConfig
    {
        /// <summary>R3: the mask covers only the interior of the ad, so the physical object's outline stays visible.</summary>
        public float InsetFraction = 0.06f;
        /// <summary>R2: safety boxes are grown by this fraction before the overlap test.</summary>
        public float SafetyMargin = 0.10f;
        /// <summary>R2: minimum margin around safety boxes, as a fraction of the image, so small (distant) people still get a buffer.</summary>
        public float SafetyMarginMin = 0.02f;
        /// <summary>R4: a single mask may not cover more than this fraction of the camera image.</summary>
        public float MaxSingleMaskArea = 0.12f;
        /// <summary>R4: all masks together may not cover more than this fraction of the camera image.</summary>
        public float MaxTotalMaskArea = 0.25f;
        /// <summary>R5: ads whose bottom edge is below this line (0 = top, 1 = bottom) are left visible (ground-level obstacles).</summary>
        public float GroundGuardY = 0.80f;
        /// <summary>R6: ads closer than this distance are left visible (things you could walk into).</summary>
        public float MinDistanceMeters = 2.0f;
        /// <summary>R9: maximum opacity of a mask. Below 1 so a misclassified object is still faintly visible.</summary>
        public float MaxOpacity = 0.90f;
    }

    public enum SkipReason
    {
        None = 0,
        Unconfirmed,
        Stale,
        SafetyOverlap,
        TooLarge,
        GroundLevel,
        AreaBudget,
        TooClose,
    }

    public struct MaskDecision
    {
        public int TrackId;
        /// <summary>The full detected ad region (used to sample the surrounding colour).</summary>
        public RectN SourceBox;
        /// <summary>Region to paint, already inset (normalized, y-down).</summary>
        public RectN Box;
        public float Opacity;
    }

    public sealed class PolicyResult
    {
        public readonly List<MaskDecision> Masks = new List<MaskDecision>();
        public readonly Dictionary<SkipReason, int> Skipped = new Dictionary<SkipReason, int>();

        public int SkippedCount(SkipReason r) => Skipped.TryGetValue(r, out int n) ? n : 0;

        internal void Skip(SkipReason r) => Skipped[r] = SkippedCount(r) + 1;
    }

    /// <summary>
    /// Decides which tracked ads are hidden. Pure function of its inputs: no Unity, no state.
    /// The guiding principle is FAIL-OPEN: whenever a rule is in doubt, the ad stays visible.
    /// </summary>
    public sealed class SafetyPolicy
    {
        private readonly SafetyConfig _cfg;
        private readonly TrackerConfig _trackerCfg;

        public SafetyPolicy(SafetyConfig cfg, TrackerConfig trackerCfg)
        {
            _cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
            _trackerCfg = trackerCfg ?? throw new ArgumentNullException(nameof(trackerCfg));
        }

        public SafetyConfig Config => _cfg;

        public PolicyResult Evaluate(IReadOnlyList<Track> tracks, IReadOnlyList<Detection> safetyObjects, double now)
        {
            var result = new PolicyResult();
            var candidates = new List<(Track track, RectN box)>();

            foreach (Track t in tracks)
            {
                if (t.Hits < _trackerCfg.MinHitsToConfirm) { result.Skip(SkipReason.Unconfirmed); continue; }
                if (now - t.LastSeen > _trackerCfg.MaxAgeSeconds) { result.Skip(SkipReason.Stale); continue; }

                RectN box = t.Box.Inset(_cfg.InsetFraction);
                if (box.Area <= 0f) { result.Skip(SkipReason.TooLarge); continue; }
                if (box.Area > _cfg.MaxSingleMaskArea) { result.Skip(SkipReason.TooLarge); continue; }
                if (t.Box.Bottom > _cfg.GroundGuardY) { result.Skip(SkipReason.GroundLevel); continue; }
                if (OverlapsSafety(box, safetyObjects)) { result.Skip(SkipReason.SafetyOverlap); continue; }

                candidates.Add((t, box));
            }

            // Spend the area budget on the most confident detections first.
            candidates.Sort((a, b) => b.track.Score.CompareTo(a.track.Score));
            float used = 0f;
            foreach (var c in candidates)
            {
                // Overlapping masks are counted twice: conservative.
                if (used + c.box.Area > _cfg.MaxTotalMaskArea) { result.Skip(SkipReason.AreaBudget); continue; }
                used += c.box.Area;
                result.Masks.Add(new MaskDecision
                {
                    TrackId = c.track.Id,
                    SourceBox = c.track.Box,
                    Box = c.box,
                    Opacity = Clamp01(_cfg.MaxOpacity),
                });
            }
            return result;
        }

        /// <summary>R6, evaluated once the runtime knows the real distance to the surface.</summary>
        public bool AllowsDistance(float meters)
        {
            return !float.IsNaN(meters) && meters >= _cfg.MinDistanceMeters;
        }

        private bool OverlapsSafety(RectN mask, IReadOnlyList<Detection> safetyObjects)
        {
            if (safetyObjects == null) return false;
            for (int i = 0; i < safetyObjects.Count; i++)
            {
                if (safetyObjects[i].Category != LabelCategory.Safety) continue;
                RectN b = safetyObjects[i].Box;
                float mx = Math.Max(b.W * _cfg.SafetyMargin, _cfg.SafetyMarginMin);
                float my = Math.Max(b.H * _cfg.SafetyMargin, _cfg.SafetyMarginMin);
                var guard = new RectN(b.X - mx, b.Y - my, b.W + 2f * mx, b.H + 2f * my);
                if (RectN.IntersectionArea(mask, guard) > 0f) return true;
            }
            return false;
        }

        private static float Clamp01(float v) => v < 0f ? 0f : (v > 1f ? 1f : v);
    }
}
