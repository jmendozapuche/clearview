using System;
using System.Collections.Generic;
using ClearView.Core;
using Xunit;

namespace ClearView.Core.Tests
{
    public class RectNTests
    {
        [Fact]
        public void IoU_of_identical_boxes_is_one()
        {
            var a = new RectN(0.1f, 0.1f, 0.2f, 0.2f);
            Assert.Equal(1f, RectN.IoU(a, a), 4);
        }

        [Fact]
        public void IoU_of_disjoint_boxes_is_zero()
        {
            Assert.Equal(0f, RectN.IoU(new RectN(0, 0, 0.1f, 0.1f), new RectN(0.5f, 0.5f, 0.1f, 0.1f)));
        }

        [Fact]
        public void Inset_shrinks_every_side()
        {
            var r = new RectN(0.2f, 0.2f, 0.5f, 0.5f).Inset(0.1f);
            Assert.Equal(0.25f, r.X, 4);
            Assert.Equal(0.4f, r.W, 4);
        }

        [Fact]
        public void ClampToUnit_clips_out_of_frame_boxes()
        {
            var r = new RectN(-0.1f, 0.9f, 0.3f, 0.3f).ClampToUnit();
            Assert.Equal(0f, r.X, 4);
            Assert.Equal(0.2f, r.W, 4);
            Assert.Equal(0.1f, r.H, 4);
        }
    }

    public class LabelMapTests
    {
        [Fact]
        public void Parses_valid_file()
        {
            var m = LabelMap.Parse("# comment\n0,billboard,ad\n1,person,safety\r\n2,bench,ignore\n");
            Assert.Equal(3, m.Count);
            Assert.Equal(LabelCategory.Ad, m.Category(0));
            Assert.Equal(LabelCategory.Safety, m.Category(1));
            Assert.Equal("bench", m.Name(2));
        }

        [Theory]
        [InlineData("0,a,ad\n2,b,ad")]
        [InlineData("0,a,banana")]
        [InlineData("0,a")]
        [InlineData("")]
        [InlineData("0,a,ad\n0,b,ad")]
        public void Rejects_invalid_files(string text)
        {
            Assert.Throws<FormatException>(() => LabelMap.Parse(text));
        }
    }

    public class YoloDecoderTests
    {
        private static readonly LabelMap Labels = LabelMap.Parse("0,billboard,ad\n1,person,safety");

        // Builds a [1, 4+nc, N] tensor from (cx, cy, w, h, score0, score1) anchors.
        private static float[] Tensor(params float[][] anchors)
        {
            int n = anchors.Length, ch = anchors[0].Length;
            var t = new float[ch * n];
            for (int a = 0; a < n; a++)
                for (int c = 0; c < ch; c++)
                    t[c * n + a] = anchors[a][c];
            return t;
        }

        [Fact]
        public void Decodes_and_normalizes_boxes()
        {
            float[] t = Tensor(new[] { 200f, 100f, 40f, 20f, 0.9f, 0.0f });
            var dets = YoloDecoder.Decode(t, 2, 1, 400, 200, Labels, new DecoderConfig());
            Assert.Single(dets);
            Assert.Equal(0.45f, dets[0].Box.X, 4);
            Assert.Equal(0.45f, dets[0].Box.Y, 4);
            Assert.Equal(0.1f, dets[0].Box.W, 4);
            Assert.Equal(LabelCategory.Ad, dets[0].Category);
        }

        [Fact]
        public void Safety_threshold_is_lower_than_ad_threshold()
        {
            float[] t = Tensor(
                new[] { 100f, 100f, 20f, 20f, 0.30f, 0.0f },   // weak ad -> dropped
                new[] { 300f, 100f, 20f, 20f, 0.0f, 0.30f });  // weak person -> kept
            var dets = YoloDecoder.Decode(t, 2, 2, 400, 200, Labels, new DecoderConfig());
            Assert.Single(dets);
            Assert.Equal(LabelCategory.Safety, dets[0].Category);
        }

        [Fact]
        public void Nms_removes_duplicates_of_same_class_only()
        {
            float[] t = Tensor(
                new[] { 100f, 100f, 40f, 40f, 0.9f, 0.0f },
                new[] { 102f, 101f, 40f, 40f, 0.8f, 0.0f },
                new[] { 101f, 100f, 40f, 40f, 0.0f, 0.7f });
            var dets = YoloDecoder.Decode(t, 2, 3, 400, 400, Labels, new DecoderConfig());
            Assert.Equal(2, dets.Count);
            Assert.Equal(0.9f, dets[0].Score, 4);
        }

        [Fact]
        public void Throws_on_label_count_mismatch()
        {
            Assert.Throws<ArgumentException>(() =>
                YoloDecoder.Decode(new float[7], 3, 1, 10, 10, Labels, new DecoderConfig()));
        }
    }

    public class TrackerTests
    {
        private static Detection Ad(float x, float y, float s = 0.9f) =>
            new Detection(new RectN(x, y, 0.1f, 0.1f), 0, s, LabelCategory.Ad);

        [Fact]
        public void Track_is_confirmed_after_min_hits_and_keeps_id()
        {
            var cfg = new TrackerConfig();
            var tr = new AdTracker(cfg);
            tr.Update(new[] { Ad(0.2f, 0.2f) }, 0.0);
            int id = tr.Tracks[0].Id;
            Assert.False(tr.IsConfirmed(tr.Tracks[0]));
            tr.Update(new[] { Ad(0.201f, 0.2f) }, 0.1);
            tr.Update(new[] { Ad(0.202f, 0.2f) }, 0.2);
            Assert.Single(tr.Tracks);
            Assert.Equal(id, tr.Tracks[0].Id);
            Assert.True(tr.IsConfirmed(tr.Tracks[0]));
        }

        [Fact]
        public void Track_is_dropped_after_max_misses()
        {
            var tr = new AdTracker(new TrackerConfig { MaxMisses = 2, MaxAgeSeconds = 100 });
            tr.Update(new[] { Ad(0.2f, 0.2f) }, 0);
            tr.Update(Array.Empty<Detection>(), 1);
            tr.Update(Array.Empty<Detection>(), 2);
            Assert.Single(tr.Tracks);
            tr.Update(Array.Empty<Detection>(), 3);
            Assert.Empty(tr.Tracks);
        }

        [Fact]
        public void Track_is_dropped_when_too_old()
        {
            var tr = new AdTracker(new TrackerConfig { MaxMisses = 100, MaxAgeSeconds = 1 });
            tr.Update(new[] { Ad(0.2f, 0.2f) }, 0);
            tr.Update(Array.Empty<Detection>(), 1.5);
            Assert.Empty(tr.Tracks);
        }

        [Fact]
        public void Far_detection_creates_new_track()
        {
            var tr = new AdTracker(new TrackerConfig());
            tr.Update(new[] { Ad(0.1f, 0.1f) }, 0);
            tr.Update(new[] { Ad(0.1f, 0.1f), Ad(0.7f, 0.1f) }, 0.1);
            Assert.Equal(2, tr.Tracks.Count);
        }
    }

    public class SafetyPolicyTests
    {
        private static readonly TrackerConfig TCfg = new TrackerConfig();

        private static Track T(int id, RectN box, float score = 0.9f, int hits = 5, double lastSeen = 10) =>
            new Track { Id = id, Box = box, Score = score, Hits = hits, LastSeen = lastSeen };

        private static Detection Person(RectN box) => new Detection(box, 1, 0.5f, LabelCategory.Safety);

        private static PolicyResult Eval(IReadOnlyList<Track> tracks, IReadOnlyList<Detection> safety = null, SafetyConfig cfg = null) =>
            new SafetyPolicy(cfg ?? new SafetyConfig(), TCfg).Evaluate(tracks, safety ?? Array.Empty<Detection>(), 10);

        [Fact]
        public void Hides_a_normal_confirmed_ad_with_inset_and_capped_opacity()
        {
            var r = Eval(new[] { T(1, new RectN(0.4f, 0.2f, 0.2f, 0.2f)) });
            Assert.Single(r.Masks);
            Assert.True(r.Masks[0].Box.W < 0.2f, "mask must be inset so the object's outline remains visible");
            Assert.True(r.Masks[0].Opacity < 1f);
        }

        [Fact]
        public void Never_hides_unconfirmed_or_stale_tracks()
        {
            var r = Eval(new[]
            {
                T(1, new RectN(0.1f, 0.1f, 0.1f, 0.1f), hits: 1),
                T(2, new RectN(0.5f, 0.1f, 0.1f, 0.1f), lastSeen: 5),
            });
            Assert.Empty(r.Masks);
            Assert.Equal(1, r.SkippedCount(SkipReason.Unconfirmed));
            Assert.Equal(1, r.SkippedCount(SkipReason.Stale));
        }

        [Fact]
        public void Never_covers_a_person_even_partially()
        {
            var ad = new RectN(0.3f, 0.2f, 0.3f, 0.3f);
            var person = new RectN(0.55f, 0.1f, 0.1f, 0.6f); // overlaps right edge of ad
            var r = Eval(new[] { T(1, ad) }, new[] { Person(person) });
            Assert.Empty(r.Masks);
            Assert.Equal(1, r.SkippedCount(SkipReason.SafetyOverlap));
        }

        [Fact]
        public void Safety_margin_protects_objects_just_outside_the_ad()
        {
            var ad = new RectN(0.3f, 0.2f, 0.2f, 0.2f);
            var person = new RectN(0.50f, 0.2f, 0.1f, 0.4f); // touches ad edge; inset alone would separate them
            Assert.Empty(Eval(new[] { T(1, ad) }, new[] { Person(person) }).Masks);
            Assert.Single(Eval(new[] { T(1, ad) }, new[] { Person(person) }, new SafetyConfig { SafetyMargin = 0f, SafetyMarginMin = 0f }).Masks);
        }

        [Fact]
        public void Distant_small_person_still_gets_absolute_margin()
        {
            var ad = new RectN(0.3f, 0.2f, 0.2f, 0.2f);
            var tinyPerson = new RectN(0.505f, 0.3f, 0.01f, 0.03f); // 0.5% to the right of the ad
            Assert.Empty(Eval(new[] { T(1, ad) }, new[] { Person(tinyPerson) }).Masks);
        }

        [Fact]
        public void Ignores_non_safety_categories_in_safety_list()
        {
            var ad = new RectN(0.3f, 0.2f, 0.2f, 0.2f);
            var other = new Detection(ad, 0, 0.9f, LabelCategory.Ad);
            Assert.Single(Eval(new[] { T(1, ad) }, new[] { other }).Masks);
        }

        [Fact]
        public void Large_ads_are_left_visible()
        {
            var r = Eval(new[] { T(1, new RectN(0.1f, 0.1f, 0.6f, 0.6f)) });
            Assert.Empty(r.Masks);
            Assert.Equal(1, r.SkippedCount(SkipReason.TooLarge));
        }

        [Fact]
        public void Ground_level_ads_are_left_visible()
        {
            var r = Eval(new[] { T(1, new RectN(0.4f, 0.75f, 0.1f, 0.15f)) });
            Assert.Empty(r.Masks);
            Assert.Equal(1, r.SkippedCount(SkipReason.GroundLevel));
        }

        [Fact]
        public void Total_area_budget_prefers_most_confident()
        {
            var cfg = new SafetyConfig { MaxTotalMaskArea = 0.05f, InsetFraction = 0f };
            var r = Eval(new[]
            {
                T(1, new RectN(0.0f, 0.1f, 0.2f, 0.2f), score: 0.6f),  // 0.04
                T(2, new RectN(0.5f, 0.1f, 0.2f, 0.2f), score: 0.95f), // 0.04
            }, cfg: cfg);
            Assert.Single(r.Masks);
            Assert.Equal(2, r.Masks[0].TrackId);
            Assert.Equal(1, r.SkippedCount(SkipReason.AreaBudget));
        }

        [Theory]
        [InlineData(1.0f, false)]
        [InlineData(1.99f, false)]
        [InlineData(2.0f, true)]
        [InlineData(15f, true)]
        [InlineData(float.NaN, false)]
        public void Distance_rule(float meters, bool allowed)
        {
            Assert.Equal(allowed, new SafetyPolicy(new SafetyConfig(), TCfg).AllowsDistance(meters));
        }
    }

    public class MotionGuardTests
    {
        [Fact]
        public void Still_head_gives_full_factor()
        {
            var g = new MotionGuard(new MotionConfig());
            Assert.Equal(1f, g.Update(5f, 0.2f, 0.016f));
        }

        [Fact]
        public void Fast_head_turn_drops_to_zero_immediately()
        {
            var g = new MotionGuard(new MotionConfig());
            g.Update(0f, 0f, 0.016f);
            Assert.Equal(0f, g.Update(200f, 0f, 0.016f));
        }

        [Fact]
        public void Recovers_gradually()
        {
            var cfg = new MotionConfig { RecoverPerSecond = 1f, InputSmoothing = 1f };
            var g = new MotionGuard(cfg);
            g.Update(200f, 0f, 0.016f);
            float f = g.Update(0f, 0f, 0.1f);
            Assert.InRange(f, 0.09f, 0.11f);
        }

        [Fact]
        public void Running_disables_masks()
        {
            var g = new MotionGuard(new MotionConfig());
            Assert.Equal(0f, g.Update(0f, 3f, 0.016f));
        }

        [Fact]
        public void Normal_walking_keeps_masks()
        {
            var g = new MotionGuard(new MotionConfig());
            Assert.Equal(1f, g.Update(10f, 1.3f, 0.016f));
        }

        [Fact]
        public void Invalid_input_fails_open()
        {
            var g = new MotionGuard(new MotionConfig());
            Assert.Equal(0f, g.Update(float.NaN, 0f, 0.016f));
        }
    }

    public class FailSafeTests
    {
        [Fact]
        public void Starts_disabled_until_user_opts_in()
        {
            var f = new FailSafe(new FailSafeConfig());
            f.ReportResult(0);
            Assert.False(f.IsHealthy(0.1));
            f.UserEnabled = true;
            Assert.True(f.IsHealthy(0.1));
        }

        [Fact]
        public void Times_out_without_results()
        {
            var f = new FailSafe(new FailSafeConfig()) { UserEnabled = true };
            f.ReportResult(0);
            Assert.False(f.IsHealthy(1.5));
            Assert.Equal("no recent detections", f.Reason);
        }

        [Fact]
        public void Sustained_low_fps_disables()
        {
            var f = new FailSafe(new FailSafeConfig()) { UserEnabled = true };
            for (int i = 0; i <= 40; i++)
            {
                f.ReportFrame(i * 0.033, 0.033f);
                f.ReportResult(i * 0.033);
            }
            Assert.False(f.IsHealthy(40 * 0.033));
        }

        [Fact]
        public void Short_fps_hiccup_is_tolerated()
        {
            var f = new FailSafe(new FailSafeConfig()) { UserEnabled = true };
            f.ReportFrame(0.0, 0.05f);
            f.ReportFrame(0.1, 0.011f);
            f.ReportResult(0.1);
            Assert.True(f.IsHealthy(0.1));
        }
    }

    public class BorderColorSamplerTests
    {
        // 10x10 image: left half red, right half blue; top-down rows.
        private static byte[] Image(bool bottomUp)
        {
            var px = new byte[10 * 10 * 4];
            for (int y = 0; y < 10; y++)
                for (int x = 0; x < 10; x++)
                {
                    int i = (y * 10 + x) * 4;
                    bool topHalf = (bottomUp ? 9 - y : y) < 5;
                    px[i] = (byte)(x < 5 ? 255 : 0);
                    px[i + 1] = (byte)(topHalf ? 255 : 0);
                    px[i + 2] = (byte)(x >= 5 ? 255 : 0);
                    px[i + 3] = 255;
                }
            return px;
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void Samples_outside_each_edge(bool bottomUp)
        {
            var c = BorderColorSampler.Sample(Image(bottomUp), 10, 10, new RectN(0.3f, 0.3f, 0.4f, 0.4f), 0.25f, bottomUp);
            Assert.Equal(1f, c.Left.R, 3);
            Assert.Equal(0f, c.Left.B, 3);
            Assert.Equal(1f, c.Right.B, 3);
            Assert.Equal(1f, c.Top.G, 3);    // above the box is the "top half"
            Assert.Equal(0f, c.Bottom.G, 3);
        }

        [Fact]
        public void Falls_back_inside_at_image_border()
        {
            var c = BorderColorSampler.Sample(Image(false), 10, 10, new RectN(0f, 0.3f, 0.3f, 0.4f), 0.34f, false);
            Assert.Equal(1f, c.Left.R, 3);
        }

        [Fact]
        public void Invalid_buffer_returns_gray()
        {
            var c = BorderColorSampler.Sample(new byte[3], 10, 10, new RectN(0, 0, 1, 1), 0.1f, false);
            Assert.Equal(0.5f, c.Top.R, 3);
        }
    }

    public class IntrinsicsTests
    {
        [Fact]
        public void Center_pixel_maps_to_forward()
        {
            var k = CameraIntrinsics.FromHorizontalFov(90f, 400, 300);
            k.Unproject(0.5f, 0.5f, out float x, out float y, out float z);
            Assert.Equal(0f, x, 4);
            Assert.Equal(0f, y, 4);
            Assert.Equal(1f, z);
        }

        [Fact]
        public void Top_right_maps_up_and_right()
        {
            var k = CameraIntrinsics.FromHorizontalFov(90f, 400, 300);
            k.Unproject(1f, 0f, out float x, out float y, out _);
            Assert.Equal(1f, x, 3);   // tan(45deg)
            Assert.True(y > 0f);
        }
    }
}
