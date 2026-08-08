//! Device acquisition and the headless path the tests render through.

use std::sync::Arc;

/// Why a GPU could not be acquired.
#[derive(Debug)]
pub enum GpuError {
    /// No adapter satisfied the request.
    ///
    /// On a machine with no GPU this is expected; install a software Vulkan
    /// implementation (Mesa's lavapipe) to render headlessly.
    NoAdapter(String),
    /// An adapter was found but would not produce a device.
    DeviceRequest(String),
    /// A GPU operation reported a failure.
    Operation(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter(detail) => write!(
                f,
                "no graphics adapter available: {detail}. On a headless machine, \
                 install mesa-vulkan-drivers for the lavapipe software renderer"
            ),
            GpuError::DeviceRequest(detail) => write!(f, "could not create a GPU device: {detail}"),
            GpuError::Operation(detail) => write!(f, "GPU operation failed: {detail}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// The device, queue and adapter the renderer draws with.
///
/// Cloning is cheap and shares the same underlying device, so subsystems can
/// each hold one without threading a reference through every call.
#[derive(Clone)]
pub struct GpuContext {
    /// The logical device.
    pub device: Arc<wgpu::Device>,
    /// The command queue.
    pub queue: Arc<wgpu::Queue>,
    /// A description of the adapter actually chosen.
    ///
    /// Recorded so golden-image failures can report which device produced them
    /// — a mismatch between a developer's GPU and CI's software renderer is the
    /// first thing to check.
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Acquires a device with no surface, for offscreen rendering and tests.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::NoAdapter`] when no backend is available and
    /// [`GpuError::DeviceRequest`] when the adapter refuses a device.
    pub fn headless() -> Result<GpuContext, GpuError> {
        pollster::block_on(GpuContext::headless_async())
    }

    /// Async form of [`GpuContext::headless`].
    ///
    /// # Errors
    ///
    /// See [`GpuContext::headless`].
    pub async fn headless_async() -> Result<GpuContext, GpuError> {
        // Honouring WGPU_BACKEND lets CI pin the software Vulkan driver rather
        // than silently picking a different device and invalidating the
        // golden images.
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::all());
        let instance = wgpu::Instance::new(descriptor);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                // Low power on purpose: a 2D sprite renderer is bandwidth-bound,
                // not compute-bound, and an integrated GPU keeps laptops cool.
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .map_err(|error| GpuError::NoAdapter(error.to_string()))?;

        let adapter_info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("verdant device"),
                required_features: wgpu::Features::empty(),
                // Downlevel defaults rather than the adapter's own limits: the
                // renderer must run identically on a software rasteriser, a
                // phone and a desktop GPU, and requesting only what is
                // guaranteed is what makes that true.
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .map_err(|error| GpuError::DeviceRequest(error.to_string()))?;

        Ok(GpuContext {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        })
    }

    /// Blocks until every submitted command has completed.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Operation`] if the device is lost while waiting.
    pub fn wait_for_idle(&self) -> Result<(), GpuError> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|error| GpuError::Operation(error.to_string()))
    }

    /// A one-line description of the chosen adapter, for logs and test output.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "{} ({:?}, {:?})",
            self.adapter_info.name, self.adapter_info.device_type, self.adapter_info.backend
        )
    }

    /// True when the chosen adapter is a software rasteriser.
    ///
    /// Golden-image comparisons need a looser tolerance on a software device:
    /// lavapipe's rasterisation rules differ from a hardware GPU's by a level
    /// of subpixel precision, which shows up as edge pixels being off by one.
    #[must_use]
    pub fn is_software(&self) -> bool {
        self.adapter_info.device_type == wgpu::DeviceType::Cpu
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("adapter", &self.describe())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_headless_device_can_be_acquired() {
        // Skipped rather than failed where no adapter exists at all, so the
        // suite still runs on a machine without even a software driver.
        let Ok(gpu) = GpuContext::headless() else {
            eprintln!("no graphics adapter; skipping");
            return;
        };
        assert!(!gpu.describe().is_empty());
        gpu.wait_for_idle()
            .expect("an idle device should not be lost");
    }

    #[test]
    fn the_error_message_points_at_the_fix() {
        let error = GpuError::NoAdapter("none found".to_string());
        assert!(error.to_string().contains("lavapipe"));
    }
}
