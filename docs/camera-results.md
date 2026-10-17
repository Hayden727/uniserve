# Camera result bundles

Camera-conditioned video requests return `application/zip` with download filename `camera-result.zip`. Set `camera.include_video` to `true` to include a decoded MP4 preview; the default is `false`. Both synchronous responses and completed job content use this contract. Ordinary video requests retain their MP4 response.

Provision `--max-video-seconds` for the complete concatenated camera output. For a fixed layout, its duration is `views × frames_per_view / fps`; a 32-view layout with 57 decoded frames per view at 8 fps requires 228 seconds of capacity. The request must match the declared grid and complete decoded frame count. Camera-only deployments may exceed 15 seconds; ordinary video deployments retain their 4–15 second duration range.

Each bundle contains `result.safetensors` and `metadata.json`, plus `video.mp4` when requested. Native latents retain their complete `[batch, channel, cell, height, width]` shape, including the batch dimension of one, and their original floating dtype. The `cell` axis follows the request's grid order. Clean cells contain the final fixed condition values.

| Tensor | Contract |
| --- | --- |
| `latents` | Complete native `[1,C,cells,H,W]` latent, unchanged dtype. |
| `camera_to_world` | FP32 `[cells,4,4]` canonical OpenCV poses, in latent-cell order, with the same reference frame and distance normalization used for inference. |
| `intrinsics` | FP32 pixel intrinsics, shared `[3,3]` or per-cell `[cells,3,3]`. |
| `input_world_to_camera` or `input_camera_to_world` | Exactly one original pose tensor, preserved without inverse conversion or frame selection. |
| `input_intrinsics` | Original pixel intrinsics, preserved before frame selection. |
| `scene_scale` | Optional FP32 `[1]` scene scale in metres. |
| `cell_time_index` | Optional canonical integer timestamp index per latent cell. |
| `input_cell_time_index` | Optional original timestamp indices. |

`metadata.json` has schema `uniserve.camera.result`, version `1`. It records the unsigned 64-bit `seed`, `grid` (`views`, `columns`, `order`), `clean_cells`, `input_pose_space`, and `camera_pose_space`. The canonical pose space is `prepared_camera_to_world`; the original is either `raw_world_to_camera` or `prepared_camera_to_world`. `camera_convention` is `opencv`, `intrinsics_units` is `pixels`, and `scene_scale_units` is `metres`.

The `latents` metadata object names its safetensors `file` and `tensor`, and records `shape`, `dtype`, and `axes`. The `video` object records `included`, `frames`, `height`, and `width`, even when preview decoding is omitted. Read the seed with an integer-preserving JSON implementation.

Downstream stages should use the canonical `camera_to_world` and `intrinsics` directly with `latents`; these cameras already include the input's normalization and cell selection. The `input_*` tensors retain the submitted representation for provenance and alternate preparation. Neither pose representation needs to be inferred from tensor names or reconstructed by inverting the other.

The worker gathers the final native tensor through its normal product storage and performs serialization on its host lane. Publication, cancellation, job retention, active-download ownership, and release use the same runtime-owned media storage as ordinary video output. Models expose numerical camera preparation only and do not own result files or publication.
