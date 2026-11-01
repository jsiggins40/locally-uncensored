//! Tauri command wrappers for the in-process MLX media engine (image + video).
//!
//! Thin pass-through to `commands::mlx` / `commands::video`, which were
//! ported verbatim from `uselu/apps/bridge` (an axum HTTP daemon) and take a
//! single `args: serde_json::Value` blob per call instead of individually
//! named parameters. That matches the existing `shell_task_*` convention in
//! this codebase (see `bg_tasks.rs`'s "Tauri-callable wrappers" section) —
//! every wrapper below takes exactly one `args: Value` parameter, and the
//! frontend MUST invoke with the payload nested as `{ args: {...} }` (see
//! `src/api/agents/bg-tasks.ts` for the precedent and its "missing required
//! key args" warning). `src/api/mlx-image.ts` / `src/api/mlx-video.ts` do
//! this via a small `invokeMedia` helper.
//!
//! No separate lu-bridge daemon: the app spawns its own MLX Python sidecar
//! (server.py on 127.0.0.1:47712) directly from `commands::mlx`, exactly
//! like the existing ComfyUI/whisper sidecars.

use crate::os_error;
use crate::state::AppState;
use base64::Engine;
use serde_json::Value;
use std::path::PathBuf;
use tauri::State;

// ── MLX image (MLX-Stable-Diffusion) ────────────────────────────────────

#[tauri::command]
pub async fn mlx_status(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::mlx::mlx_status(state.inner(), &args).await
}

#[tauri::command]
pub fn mlx_start(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::mlx::mlx_start(state.inner(), &args)
}

#[tauri::command]
pub async fn mlx_unload(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::mlx::mlx_unload(state.inner(), &args).await
}

#[tauri::command]
pub async fn mlx_generate(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::mlx::mlx_generate(state.inner(), &args).await
}

#[tauri::command]
pub fn mlx_image_models(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::mlx::mlx_image_models(state.inner(), &args)
}

#[tauri::command]
pub fn set_hf_token(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::mlx::set_hf_token(state.inner(), &args)
}

#[tauri::command]
pub fn hf_token_present(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::mlx::hf_token_present(state.inner(), &args)
}

#[tauri::command]
pub fn mlx_image_install_model(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::mlx::mlx_image_install_model(state.inner(), &args)
}

#[tauri::command]
pub fn mlx_image_install_status(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::mlx::mlx_image_install_status(state.inner(), &args)
}

#[tauri::command]
pub fn mlx_image_delete_model(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::mlx::mlx_image_delete_model(state.inner(), &args)
}

#[tauri::command]
pub fn install_mlx_diffusion(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::mlx::install_mlx_diffusion(state.inner(), &args)
}

#[tauri::command]
pub fn install_mlx_diffusion_status(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::mlx::install_mlx_diffusion_status(state.inner(), &args)
}

// ── MLX video ────────────────────────────────────────────────────────────

#[tauri::command]
pub fn video_status(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::video::video_status(state.inner(), &args)
}

#[tauri::command]
pub fn video_list_models(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::video::video_list_models(state.inner(), &args)
}

#[tauri::command]
pub fn video_install_mlx(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::video::video_install_mlx(state.inner(), &args)
}

#[tauri::command]
pub fn video_install_mlx_status(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::video::video_install_mlx_status(state.inner(), &args)
}

#[tauri::command]
pub fn video_install_model(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::video::video_install_model(state.inner(), &args)
}

#[tauri::command]
pub fn video_install_model_status(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::video::video_install_model_status(state.inner(), &args)
}

#[tauri::command]
pub fn video_delete_model(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::video::video_delete_model(state.inner(), &args)
}

#[tauri::command]
pub fn video_generate(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::video::video_generate(state.inner(), &args)
}

#[tauri::command]
pub fn video_progress(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::video::video_progress(state.inner(), &args)
}

#[tauri::command]
pub fn video_cancel(state: State<'_, AppState>, args: Value) -> Result<Value, String> {
    crate::commands::video::video_cancel(state.inner(), &args)
}

// ── Media playback (video → blob URL) ───────────────────────────────────
//
// MLX video never touches the sidecar HTTP surface the image path uses
// (`mlx_generate` proxies to :47712 and returns base64 straight from the
// response body) — `video_generate` runs `mlx_video.*.generate` as a bare
// subprocess and writes an mp4 to disk (`commands::video::outputs_root()`).
// With the lu-bridge daemon gone there is nothing serving that file at
// :47711/videos/<file> anymore, and the CSP's `media-src` has no `file:`
// grant (only `'self' blob: http://localhost:* http://127.0.0.1:*`), so a
// `<video src="file:///...">` would just be blocked. Read the bytes here
// and let the frontend build a `blob:` URL instead — that's an origin the
// CSP already allows.

/// Roots `read_media_file` is allowed to serve bytes from. Only MLX video
/// output today (MLX images are never written to disk — see above); add
/// more roots here if a future in-process media backend gains one.
fn allowed_media_roots() -> Vec<PathBuf> {
    vec![
        crate::commands::video::outputs_root(),
        // Stills from the MLX image lane. They land on disk so the gallery
        // survives a restart — the in-memory data URL does not (createStore's
        // partialize strips it), and without a file there was nothing left to
        // re-read.
        crate::commands::mlx::images_root(),
    ]
}

/// Read a file from one of the app's own media output directories and
/// return its bytes as base64, so the frontend can turn it into a
/// `Blob`/`URL.createObjectURL` blob: URL for `<video>` playback and
/// Download.
///
/// Path-traversal guard: canonicalizes both `path` and every allowed root
/// (resolving `..` and symlinks) and rejects anything whose canonical form
/// doesn't fall under an allowed root — this is NOT a general-purpose file
/// read, it only ever serves the app's own generated media.
#[tauri::command]
pub fn read_media_file(path: String) -> Result<String, String> {
    let candidate = PathBuf::from(&path);
    let canonical = candidate
        .canonicalize()
        .map_err(|e| format!("read_media_file: {}: {e}", candidate.display()))?;

    let allowed = allowed_media_roots().into_iter().any(|root| {
        root.canonicalize()
            .map(|r| canonical.starts_with(&r))
            .unwrap_or(false)
    });
    if !allowed {
        return Err(format!(
            "read_media_file: path escapes the allowed media directories: {}",
            canonical.display()
        ));
    }

    let bytes = std::fs::read(&canonical)
        .map_err(|e| format!("read_media_file: {}: {}", canonical.display(), os_error::english(&e)))?;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

// ── Remote stills (Atlas Cloud Seedream) ────────────────────────────────
//
// A hosted image arrives as a URL on the vendor's CDN. The CSP's img-src does
// not list that host, and vendor links expire, so the bytes are brought home
// once and kept next to the MLX stills: the gallery then shows them through
// `read_media_file` like any other local render, and a restart re-reads them
// from disk instead of chasing a dead link.

/// Upper bound for one downloaded still. A 4K PNG is a few tens of MB; past
/// this the "image" is not one.
const REMOTE_IMAGE_MAX_BYTES: usize = 64 * 1024 * 1024;

/// File extension for the image formats a render can come back in, read from
/// the magic bytes. `None` = not an image, which is refused rather than saved.
fn image_ext(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        Some("png")
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("jpg")
    } else if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some("webp")
    } else {
        None
    }
}

/// Download a finished remote render into the app's image folder and return
/// the file's path. Public hosts only (same SSRF guard as `fetch_external_bytes`).
#[tauri::command]
pub async fn save_remote_image(url: String) -> Result<String, String> {
    crate::commands::proxy::validate_public_url(&url)?;

    let client = reqwest::Client::builder()
        .user_agent("LocallyUncensored/2.0")
        .timeout(std::time::Duration::from_secs(120))
        .redirect(crate::commands::proxy::ssrf_safe_redirect_policy(10))
        .build()
        .map_err(|e| os_error::english(&e))?;
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("save_remote_image: {}", os_error::english(&e)))?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}: {}", resp.status().as_u16(), url));
    }
    let bytes = resp.bytes().await.map_err(|e| os_error::english(&e))?;
    if bytes.len() > REMOTE_IMAGE_MAX_BYTES {
        return Err(format!("save_remote_image: {} bytes is too large for an image", bytes.len()));
    }
    let ext = image_ext(&bytes).ok_or_else(|| "save_remote_image: the download is not a PNG, JPEG or WebP image".to_string())?;

    let dir = crate::commands::mlx::images_root();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir {}: {}", dir.display(), os_error::english(&e)))?;
    // Same naming rule as the MLX lane: millisecond stamp + process id, so two
    // renders landing in the same millisecond never overwrite each other.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = dir.join(format!("atlas-{stamp}-{}.{ext}", std::process::id()));
    std::fs::write(&path, &bytes).map_err(|e| format!("write {}: {}", path.display(), os_error::english(&e)))?;
    Ok(path.to_string_lossy().into_owned())
}

/// Atlas Cloud's upload endpoint. Fixed here, not passed in: this command
/// carries a user file and their key, so it must not be aimable elsewhere.
const ATLAS_UPLOAD_URL: &str = "https://api.atlascloud.ai/api/v1/model/uploadMedia";

/// Upper bound for one uploaded edit source.
const ATLAS_UPLOAD_MAX_BYTES: usize = 32 * 1024 * 1024;

/// Upload an edit source image to Atlas Cloud (multipart, field `file`) and
/// return Atlas's JSON reply, which carries the temporary URL the edit model
/// reads. The webview cannot send this itself: the CSP blocks the host, and
/// the generic proxy only forwards text bodies. Non-2xx answers come back as
/// `HTTP <status>: <body>`, the same shape `proxy_localhost` uses.
#[tauri::command]
pub async fn atlas_upload_media(
    api_key: String,
    filename: String,
    mime: String,
    data_b64: String,
) -> Result<String, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data_b64.as_bytes())
        .map_err(|e| format!("atlas_upload_media: decode: {e}"))?;
    if bytes.len() > ATLAS_UPLOAD_MAX_BYTES {
        return Err(format!("atlas_upload_media: {} bytes is too large to upload", bytes.len()));
    }
    if image_ext(&bytes).is_none() {
        return Err("atlas_upload_media: the source is not a PNG, JPEG or WebP image".to_string());
    }
    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name(filename)
        .mime_str(&mime)
        .map_err(|e| format!("atlas_upload_media: {}", os_error::english(&e)))?;
    let form = reqwest::multipart::Form::new().part("file", part);

    let client = reqwest::Client::builder()
        .user_agent("LocallyUncensored/2.0")
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| os_error::english(&e))?;
    let resp = client
        .post(ATLAS_UPLOAD_URL)
        .bearer_auth(api_key)
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("atlas_upload_media: {}", os_error::english(&e)))?;
    let status = resp.status();
    let text = resp.text().await.map_err(|e| os_error::english(&e))?;
    if !status.is_success() {
        return Err(format!("HTTP {}: {}", status.as_u16(), text));
    }
    Ok(text)
}

#[cfg(test)]
mod remote_image_tests {
    use super::image_ext;

    #[test]
    fn recognises_the_formats_a_render_comes_back_in() {
        assert_eq!(image_ext(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0]), Some("png"));
        assert_eq!(image_ext(&[0xFF, 0xD8, 0xFF, 0xE0]), Some("jpg"));
        assert_eq!(image_ext(b"RIFF\x00\x00\x00\x00WEBPVP8 "), Some("webp"));
    }

    #[test]
    fn refuses_anything_that_is_not_an_image() {
        assert_eq!(image_ext(b"<html>error</html>"), None);
        assert_eq!(image_ext(b"{\"error\":\"expired\"}"), None);
        assert_eq!(image_ext(b"RIFF\x00\x00\x00\x00WAVE"), None);
        assert_eq!(image_ext(&[]), None);
    }
}
