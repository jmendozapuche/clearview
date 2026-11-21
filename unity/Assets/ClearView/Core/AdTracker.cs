using System;
using System.Collections.Generic;

namespace ClearView.Core
{
    [Serializable]
    public sealed class TrackerConfig
    {
        /// <summary>Minimum IoU to associate a detection with an existing track.</summary>
        public float MatchIoU = 0.3f;
        /// <summary>Detections needed before a track may be masked (hysteresis against flicker / one-off false positives).</summary>
        public int MinHitsToConfirm = 3;
        /// <summary>Consecutive inference cycles without a match before the track is dropped.</summary>
        public int MaxMisses = 2;
        /// <summary>Seconds without a match before the track is dropped, whatever the cycle count.</summary>
        public float MaxAgeSeconds = 1.0f;
        /// <summary>Exponential smoothing applied to the box (0 = frozen, 1 = no smoothing).</summary>
        public float BoxSmoothing = 0.5f;
    }

    public sealed class Track
    {
        public int Id;
        public RectN Box;
        public int ClassId;
        public float Score;
        public int Hits;
        public int Misses;
        public double LastSeen;
    }

    /// <summary>Greedy IoU tracker for advertising detections.</summary>
    public sealed class AdTracker
    {
        private readonly TrackerConfig _cfg;
        private readonly List<Track> _tracks = new List<Track>();
        private int _nextId = 1;

        public AdTracker(TrackerConfig cfg)
        {
            _cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
        }

        public IReadOnlyList<Track> Tracks => _tracks;

        public bool IsConfirmed(Track t) => t.Hits >= _cfg.MinHitsToConfirm;

        public void Clear() => _tracks.Clear();

        public void Update(IReadOnlyList<Detection> ads, double now)
        {
            var pairs = new List<(float iou, int track, int det)>();
            for (int t = 0; t < _tracks.Count; t++)
            {
                for (int d = 0; d < ads.Count; d++)
                {
                    float iou = RectN.IoU(_tracks[t].Box, ads[d].Box);
                    if (iou >= _cfg.MatchIoU) pairs.Add((iou, t, d));
                }
            }
            pairs.Sort((a, b) => b.iou.CompareTo(a.iou));

            var trackUsed = new bool[_tracks.Count];
            var detUsed = new bool[ads.Count];
            foreach (var p in pairs)
            {
                if (trackUsed[p.track] || detUsed[p.det]) continue;
                trackUsed[p.track] = true;
                detUsed[p.det] = true;
                Track tr = _tracks[p.track];
                Detection det = ads[p.det];
                tr.Box = RectN.Lerp(tr.Box, det.Box, _cfg.BoxSmoothing);
                tr.Score = det.Score;
                tr.ClassId = det.ClassId;
                tr.Hits++;
                tr.Misses = 0;
                tr.LastSeen = now;
            }

            for (int t = _tracks.Count - 1; t >= 0; t--)
            {
                if (trackUsed[t]) continue;
                Track tr = _tracks[t];
                tr.Misses++;
                if (tr.Misses > _cfg.MaxMisses || now - tr.LastSeen > _cfg.MaxAgeSeconds)
                    _tracks.RemoveAt(t);
            }

            for (int d = 0; d < ads.Count; d++)
            {
                if (detUsed[d]) continue;
                _tracks.Add(new Track
                {
                    Id = _nextId++,
                    Box = ads[d].Box,
                    ClassId = ads[d].ClassId,
                    Score = ads[d].Score,
                    Hits = 1,
                    Misses = 0,
                    LastSeen = now,
                });
            }
        }
    }
}
