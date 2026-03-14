# OpenVINO Latency Results

Measured with `benchmark_app -hint latency -niter 100 -api sync` inside the
`openvino-util` container (OpenVINO 2026.0.0).  All models are the preprocessing-baked
sweep-export variants (`_denorm_nhwc_uint8input.onnx`).

## Hardware

| Device | Model |
|---|---|
| CPU | Intel Core i7-8700K @ 3.70 GHz (Coffee Lake, 6 cores) |
| GPU | Intel UHD Graphics 630 (CoffeeLake-S GT2, Gen9.5, 24 EUs) |

## Results

| Model variant | Input size | Precision | CPU median (ms) | CPU FPS | GPU median (ms) | GPU FPS |
|---|---|---|---:|---:|---:|---:|
| MDV6-yolov10-e | 640 | float16 | 327 | 3.0 | 361 | 2.8 |
| MDV6-yolov10-e | 640 | int8 | **173** | **5.7** | 507 | 2.0 |
| MDV6-yolov10-e | 320 | float16 | 72 | 13.7 | 105 | 9.5 |
| MDV6-yolov10-e | 320 | int8 | **43** | **22.8** | 151 | 6.6 |
| MDV6-yolov10-c | 640 | float16 | 19 | 51 | 27 | 36.6 |
| MDV6-yolov10-c | 640 | int8 | **13** | **75** | 46 | 21.5 |
| MDV6-yolov10-c | 320 | float16 | 5.5 | 172 | 10 | 97 |
| MDV6-yolov10-c | 320 | int8 | **3.6** | **258** | 18 | 55 |

## Observations

- **INT8 loads cleanly.** All four QDQ ONNX models are accepted by OpenVINO 2026.0
  without graph errors.
- **CPU beats GPU on every variant.** UHD 630 is slower than the i7-8700K for
  single-stream latency mode across all model sizes and precisions.
- **INT8 gives a real CPU speedup:** ~1.9× on yolov10-e, ~1.5× on yolov10-c.
  OpenVINO successfully lowers the QDQ nodes to AVX2 INT8 kernels on CPU.
- **INT8 is slower than float16 on GPU (~1.4–1.7× regression).** The UHD 630 GPU
  plugin (Gen9.5) does not appear to fuse QDQ nodes to INT8 EU kernels — it likely
  dequantizes and computes in float16, adding overhead. Newer iGPUs (Xe / Arc) may
  behave differently.
- **Practical recommendation for i7-8700K + UHD 630 systems:** use CPU inference with
  INT8. yolov10-c 320 at 3.6 ms / 258 FPS is more than sufficient for real-time NVR
  use; yolov10-c 640 at 13 ms / 75 FPS is also very usable.
