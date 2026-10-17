#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Public camera request admission follows the worker's declared geometry.

use std::sync::Arc;

use base64::Engine as _;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{AddedToken, Tokenizer};
use uniserve_core::{CameraOrder, Canvas};
use uniserve_engine::{CameraLayout, FrameClips, ServedCanvas, VideoDenoiserInfo};
use uniserve_server::openai::{ApiError, VideoGenerationRequest};
use uniserve_server::profile::tokenizer::HuggingFaceTokenizer;
use uniserve_server::serving::ServeRequestId;
use uniserve_server::serving::video::VideoService;
use uniserve_server::{Config, VideoMediaSettings};

fn service(camera_layout: Option<CameraLayout>, frames: FrameClips) -> VideoService {
    let mut vocabulary: Vocab = (1_u32..=127)
        .map(|id| (char::from_u32(id).unwrap().to_string(), id))
        .collect();
    vocabulary.insert("<unk>".into(), 0);
    vocabulary.insert("</s>".into(), 128);
    let model = BPE::builder()
        .vocab_and_merges(vocabulary, Vec::new())
        .unk_token("<unk>".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.add_special_tokens(&[AddedToken::from("</s>", true)]);
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("$A </s>")
            .unwrap()
            .special_tokens(vec![("</s>", 128)])
            .build()
            .unwrap(),
    ));
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("tokenizer.json");
    tokenizer.save(&path, false).unwrap();
    VideoService::new(
        VideoDenoiserInfo {
            tasks: vec!["camera2v".into()],
            schedule_points: 5,
            video_shift: 5.0,
            audio_shift: 0.0,
            frame_rate: 8,
            audio_sample_rate: 0,
            frames,
            camera_layout,
            canvases: vec![ServedCanvas {
                short_edge: 480,
                aspect_ratio: "26:15".into(),
                canvas: Canvas {
                    width: 832,
                    height: 480,
                },
            }],
            references: Default::default(),
        },
        None,
        240.0,
        1,
        &VideoMediaSettings::default(),
        Default::default(),
        Arc::new(HuggingFaceTokenizer::new(&path).unwrap()),
    )
    .unwrap()
}

fn layout() -> CameraLayout {
    CameraLayout {
        views: 32,
        columns: 16,
        order: CameraOrder::TimeMajor,
        frames_per_view: 57,
        latent_frames_per_view: 15,
    }
}

fn request(views: u32, columns: u32, order: &str, seconds: f64) -> VideoGenerationRequest {
    // Rust transports opaque tensor bytes. Numerical archive validation is
    // the worker media reader's responsibility; this is a valid empty
    // safetensors envelope used only to test publication and admission.
    let mut bytes = 2_u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    serde_json::from_value(serde_json::json!({
        "model": "camera-test", "prompt": "a camera", "task": "camera2v", "seed": 0,
        "target": {"short_edge": 480, "aspect_ratio": "26:15", "duration_seconds": seconds},
        "camera": {
            "uri": format!("data:application/octet-stream;base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes)),
            "grid": {"views": views, "columns": columns, "order": order},
            "clean_cells": [0],
        },
    }))
    .unwrap()
}

#[tokio::test]
async fn retained_camera_grid_is_admitted_with_its_distinct_decoded_frame_count() {
    // A real serve invocation validates the frontend before binding the model.
    let mut config = Config::default();
    config.engine.max_video_seconds = 228.0;
    config.validate().unwrap();
    let service = service(
        Some(layout()),
        FrameClips {
            min_frames: 57,
            clip_frames: 57,
            clip_latent_frames: 15,
            tail_latent_frames: 0,
        },
    );
    let mut input = request(32, 16, "time_major", 228.0);
    input.prompt = String::new();
    input.camera.as_mut().unwrap().clean_cells = vec![0, 511];
    let processed = service
        .process(&ServeRequestId::new("camera-grid"), &input, 512)
        .await
        .unwrap();
    assert_eq!(processed.request.sampling.num_frames, 1824);
    assert_eq!(processed.request.sampling.video_units, 32);
    assert_eq!(processed.request.prompt_token_ids, [128]);
    let camera = processed.request.camera.as_ref().unwrap();
    assert_eq!((camera.grid.views, camera.grid.columns), (32, 16));
    assert_eq!(camera.grid.order, CameraOrder::TimeMajor);
    assert_eq!(camera.clean_cells, [0, 511]);
    assert_eq!(camera.source.bytes, 10);
    assert!(processed.request.camera_media.is_some());
    let capabilities = service.capabilities(512);
    assert_eq!(capabilities["model_max_seconds"], 228.0);
    assert_eq!(
        capabilities["camera_layout"],
        serde_json::json!({
            "views": 32, "columns": 16, "order": "time_major",
            "frames_per_view": 57, "latent_frames_per_view": 15,
        })
    );
}

#[tokio::test]
async fn camera_admission_rejects_wrong_axes_order_or_decoded_duration() {
    let service = service(
        Some(layout()),
        FrameClips {
            min_frames: 57,
            clip_frames: 57,
            clip_latent_frames: 15,
            tail_latent_frames: 0,
        },
    );
    for (request, expected) in [
        (request(16, 32, "time_major", 228.0), "camera.grid"),
        (request(32, 16, "view_major", 228.0), "camera.grid"),
        (request(32, 15, "time_major", 228.0), "camera.grid"),
        (
            request(32, 16, "time_major", 7.125),
            "target.duration_seconds",
        ),
    ] {
        let error = service
            .process(&ServeRequestId::new("wrong-camera-grid"), &request, 512)
            .await
            .unwrap_err();
        assert!(
            matches!(error, ApiError::InvalidRequest { param: Some(name), .. }
                if name == expected),
            "{error:?}"
        );
    }
}

#[tokio::test]
async fn absent_camera_layout_preserves_independent_clip_admission() {
    let service = service(
        None,
        FrameClips {
            min_frames: 13,
            clip_frames: 13,
            clip_latent_frames: 4,
            tail_latent_frames: 0,
        },
    );
    let processed = service
        .process(
            &ServeRequestId::new("camera-clips"),
            &request(48, 1, "view_major", 19.5),
            512,
        )
        .await
        .unwrap();
    assert_eq!(processed.request.sampling.num_frames, 156);
    assert_eq!(processed.request.sampling.video_units, 12);
    assert!(service.capabilities(512)["camera_layout"].is_null());
    let rejected = service
        .process(
            &ServeRequestId::new("wrong-camera-clips"),
            &request(47, 1, "view_major", 19.5),
            512,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(rejected, ApiError::InvalidRequest { param: Some(name), .. }
            if name == "camera")
    );
}
