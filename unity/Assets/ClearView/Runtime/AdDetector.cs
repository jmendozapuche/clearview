using System;
using System.Collections;
using Unity.InferenceEngine; // Unity Inference Engine 2.2+. On Sentis 2.1 use: using Unity.Sentis;
using UnityEngine;

namespace ClearView
{
    /// <summary>
    /// Runs the YOLO model with Unity Inference Engine. The network is scheduled a few layers per
    /// frame (ScheduleIterable) so inference never causes a dropped frame, then read back asynchronously.
    /// </summary>
    public sealed class AdDetector : IDisposable
    {
        private enum State { Idle, Scheduling, Readback }

        private readonly Worker _worker;
        private readonly Tensor<float> _input;
        private readonly int _layersPerFrame;
        private IEnumerator _schedule;
        private Tensor<float> _output;
        private State _state = State.Idle;

        public AdDetector(ModelAsset modelAsset, int inputWidth, int inputHeight, BackendType backend, int layersPerFrame)
        {
            if (modelAsset == null) throw new ArgumentNullException(nameof(modelAsset));
            Model model = ModelLoader.Load(modelAsset);
            _worker = new Worker(model, backend);
            _input = new Tensor<float>(new TensorShape(1, 3, inputHeight, inputWidth));
            _layersPerFrame = Mathf.Max(1, layersPerFrame);
        }

        public bool IsIdle => _state == State.Idle;

        /// <summary>Copies (and resizes) the frame into the input tensor and starts inference.</summary>
        public void Begin(Texture frame)
        {
            if (_state != State.Idle) throw new InvalidOperationException("Detector is busy");
            TextureConverter.ToTensor(frame, _input, new TextureTransform());
            _schedule = _worker.ScheduleIterable(_input);
            _state = State.Scheduling;
        }

        /// <summary>Advances inference. Returns true exactly once per Begin, when the output is on the CPU.</summary>
        public bool Tick(out float[] output, out int channels, out int anchors)
        {
            output = null;
            channels = 0;
            anchors = 0;

            if (_state == State.Scheduling)
            {
                for (int i = 0; i < _layersPerFrame; i++)
                {
                    if (_schedule.MoveNext()) continue;
                    _output = _worker.PeekOutput() as Tensor<float>;
                    if (_output == null) throw new InvalidOperationException("Model output is not a float tensor");
                    _output.ReadbackRequest();
                    _state = State.Readback;
                    break;
                }
            }
            else if (_state == State.Readback && _output.IsReadbackRequestDone())
            {
                if (_output.shape.rank != 3)
                    throw new InvalidOperationException($"Expected output [1, 4+classes, anchors], got {_output.shape}");
                channels = _output.shape[1];
                anchors = _output.shape[2];
                output = _output.DownloadToArray();
                _state = State.Idle;
                return true;
            }
            return false;
        }

        public void Dispose()
        {
            _input?.Dispose();
            _worker?.Dispose();
        }
    }
}
