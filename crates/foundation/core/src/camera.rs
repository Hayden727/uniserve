//! Camera-grid conditions carried independently of encoded video media.

use serde::{Deserialize, Serialize};

use crate::MediaLocator;

/// Flattening order of camera views and temporal columns in latent storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraOrder {
    /// Every temporal column of a view precedes the next view.
    ViewMajor,
    /// Every view of a temporal column precedes the next column.
    TimeMajor,
}

/// The two semantic axes represented by a generated latent sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraGrid {
    /// Camera views in the grid.
    pub views: u32,
    /// Temporal columns per view.
    pub columns: u32,
    /// Order used by the input archive and generated latent.
    pub order: CameraOrder,
}

impl CameraGrid {
    /// Checks the grid extents and the distinct latent cells it anchors.
    pub fn validate_clean_cells(&self, clean_cells: &[u32]) -> Result<(), &'static str> {
        let Some(cells) = self
            .views
            .checked_mul(self.columns)
            .filter(|count| *count > 0)
        else {
            return Err("camera grid dimensions must have a positive u32 product");
        };
        let unique: std::collections::HashSet<_> = clean_cells.iter().collect();
        if unique.len() != clean_cells.len() || clean_cells.iter().any(|index| *index >= cells) {
            return Err("clean camera cells must be distinct and inside the grid");
        }
        Ok(())
    }
}

/// A camera request's immutable numerical input archive.
///
/// The safetensors archive contains raw world-to-camera or prepared
/// camera-to-world poses, intrinsics,
/// physical scale and either clean latent values or RGB input frames. The
/// numerical capability validates and prepares these tensors outside graph
/// capture. The engine keeps the publication alive until request retirement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CameraInput {
    /// Published archive bytes, read without executing serialized code.
    pub source: MediaLocator,
    /// Layout carried unchanged into output metadata.
    pub grid: CameraGrid,
    /// Flattened latent cells anchored to the clean input at every step.
    pub clean_cells: Vec<u32>,
    /// Include an MP4 preview beside the native latent and camera metadata.
    #[serde(default)]
    pub include_video: bool,
}

impl CameraInput {
    /// Checks dimensions, publication identity and unique in-range anchors.
    pub fn validate(&self) -> Result<(), &'static str> {
        self.grid.validate_clean_cells(&self.clean_cells)?;
        if self.source.name.is_empty() || self.source.name.contains('/') || self.source.bytes < 8 {
            return Err("camera input requires a published tensor archive");
        }
        Ok(())
    }
}
