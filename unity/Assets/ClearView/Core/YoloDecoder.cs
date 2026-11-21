using System;
using System.Collections.Generic;

namespace ClearView.Core
{
    public struct Detection
    {
        public RectN Box;
        public int ClassId;
        public float Score;
        public LabelCategory Category;

        public Detection(RectN box, int classId, float score, LabelCategory category)
        {
            Box = box;
            ClassId = classId;
            Score = score;
            Category = category;
        }
    }

    [Serializable]
    public sealed class DecoderConfig
    {
        /// <summary>Minimum score for an advertising detection.</summary>
        public float AdScoreThreshold = 0.45f;
        /// <summary>
        /// Minimum score for a safety detection. Deliberately LOWER than the ad threshold:
        /// a false "person" only costs a missed mask, a missed person could be dangerous.
        /// </summary>
        public float SafetyScoreThreshold = 0.20f;
        public float NmsIoU = 0.45f;
        /// <summary>Set if the tensor's rows come out vertically flipped on your device.</summary>
        public bool FlipY = false;
        public int MaxDetections = 100;
    }

    /// <summary>
    /// Decodes the raw output of an Ultralytics YOLOv8 / YOLO11 detection model exported to ONNX
    /// without embedded NMS. Expected layout: [1, 4 + numClasses, numAnchors], row-major,
    /// boxes as (cx, cy, w, h) in input-image pixels and class scores already sigmoid-activated.
    /// </summary>
    public static class YoloDecoder
    {
        public static List<Detection> Decode(
            float[] output, int numClasses, int numAnchors,
            int inputWidth, int inputHeight, LabelMap labels, DecoderConfig cfg)
        {
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (labels == null) throw new ArgumentNullException(nameof(labels));
            if (cfg == null) throw new ArgumentNullException(nameof(cfg));
            if (labels.Count != numClasses)
                throw new ArgumentException($"Label map has {labels.Count} classes but model has {numClasses}");
            int expected = (4 + numClasses) * numAnchors;
            if (output.Length < expected)
                throw new ArgumentException($"Output has {output.Length} values, expected {expected}");

            float invW = 1f / inputWidth;
            float invH = 1f / inputHeight;
            var candidates = new List<Detection>();

            for (int a = 0; a < numAnchors; a++)
            {
                int best = -1;
                float bestScore = 0f;
                for (int c = 0; c < numClasses; c++)
                {
                    float s = output[(4 + c) * numAnchors + a];
                    if (s > bestScore)
                    {
                        bestScore = s;
                        best = c;
                    }
                }
                if (best < 0) continue;

                LabelCategory cat = labels.Category(best);
                float threshold = cat == LabelCategory.Safety ? cfg.SafetyScoreThreshold : cfg.AdScoreThreshold;
                if (bestScore < threshold) continue;

                float cx = output[a] * invW;
                float cy = output[numAnchors + a] * invH;
                float w = output[2 * numAnchors + a] * invW;
                float h = output[3 * numAnchors + a] * invH;
                if (cfg.FlipY) cy = 1f - cy;

                RectN box = RectN.FromCenter(cx, cy, w, h).ClampToUnit();
                if (box.Area <= 0f) continue;
                candidates.Add(new Detection(box, best, bestScore, cat));
            }

            return Nms.PerClass(candidates, cfg.NmsIoU, cfg.MaxDetections);
        }
    }

    public static class Nms
    {
        public static List<Detection> PerClass(List<Detection> detections, float iouThreshold, int maxDetections)
        {
            detections.Sort((x, y) => y.Score.CompareTo(x.Score));
            var kept = new List<Detection>();
            for (int i = 0; i < detections.Count && kept.Count < maxDetections; i++)
            {
                Detection d = detections[i];
                bool suppressed = false;
                for (int k = 0; k < kept.Count; k++)
                {
                    if (kept[k].ClassId == d.ClassId && RectN.IoU(kept[k].Box, d.Box) > iouThreshold)
                    {
                        suppressed = true;
                        break;
                    }
                }
                if (!suppressed) kept.Add(d);
            }
            return kept;
        }
    }
}
