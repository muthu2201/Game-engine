//! The 2D camera and its projection.

use verdant_core_math::{Fx, Rect, Vec2};

/// A 2D camera looking at a rectangle of world space.
///
/// # Pixel snapping
///
/// The camera's position is snapped to whole pixels before it is turned into a
/// matrix. Without that, a camera at a fractional offset resamples every sprite
/// slightly differently each frame, and pixel art shimmers — rows of pixels
/// visibly swap width as the camera drifts. Snapping costs nothing and is the
/// single most important thing a pixel-art renderer does.
///
/// [`Camera2D::subpixel`] disables it for parallax layers, where the smoothness
/// is worth more than the crispness.
#[derive(Clone, Copy, Debug)]
pub struct Camera2D {
    /// The world point at the centre of the view.
    pub position: Vec2,
    /// Magnification. Kept an integer for pixel art, where a fractional zoom
    /// means some source pixels cover more screen pixels than others.
    pub zoom: u32,
    /// Viewport size in physical pixels.
    pub viewport: (u32, u32),
    /// World units per pixel at zoom 1.
    pub units_per_pixel: Fx,
    /// Whether to snap the camera to whole pixels.
    pub snap_to_pixel: bool,
}

impl Camera2D {
    /// A camera at the origin viewing a viewport of the given size.
    ///
    /// # Panics
    ///
    /// Panics if either viewport dimension is zero.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Camera2D {
        assert!(
            width > 0 && height > 0,
            "Camera2D: viewport must be non-empty"
        );
        Camera2D {
            position: Vec2::ZERO,
            zoom: 1,
            viewport: (width, height),
            units_per_pixel: Fx::ONE,
            snap_to_pixel: true,
        }
    }

    /// Returns this camera with pixel snapping disabled.
    #[must_use]
    pub const fn subpixel(mut self) -> Camera2D {
        self.snap_to_pixel = false;
        self
    }

    /// Returns this camera at the given zoom, clamped to at least 1.
    #[must_use]
    pub const fn with_zoom(mut self, zoom: u32) -> Camera2D {
        self.zoom = if zoom == 0 { 1 } else { zoom };
        self
    }

    /// The camera position actually used for rendering.
    #[must_use]
    pub fn effective_position(&self) -> Vec2 {
        if self.snap_to_pixel {
            // Snap in *pixel* space, not world space: at a zoom other than 1
            // those are different, and rounding in world space still leaves the
            // camera between screen pixels.
            let pixels_per_unit = self.pixels_per_unit();
            Vec2::new(
                (self.position.x * pixels_per_unit).round() / pixels_per_unit,
                (self.position.y * pixels_per_unit).round() / pixels_per_unit,
            )
        } else {
            self.position
        }
    }

    /// Screen pixels per world unit at the current zoom.
    #[must_use]
    pub fn pixels_per_unit(&self) -> Fx {
        Fx::from_num(i32::try_from(self.zoom.max(1)).unwrap_or(1)) / self.units_per_pixel
    }

    /// The rectangle of world space currently visible.
    ///
    /// Used to cull sprites and to decide which tiles to upload.
    #[must_use]
    pub fn visible_bounds(&self) -> Rect {
        let pixels_per_unit = self.pixels_per_unit();
        let size = Vec2::new(
            Fx::from_num(i32::try_from(self.viewport.0).unwrap_or(i32::MAX)) / pixels_per_unit,
            Fx::from_num(i32::try_from(self.viewport.1).unwrap_or(i32::MAX)) / pixels_per_unit,
        );
        Rect::from_centre(self.effective_position(), size)
    }

    /// The world-to-clip matrix, column-major, ready for the uniform buffer.
    ///
    /// Maps the visible rectangle onto clip space, with y pointing down in
    /// world space and up in clip space — which is why the y scale is negated.
    #[must_use]
    pub fn view_projection(&self) -> [[f32; 4]; 4] {
        let bounds = self.visible_bounds();
        let half_width = (bounds.size.x * Fx::HALF).to_f32().max(f32::EPSILON);
        let half_height = (bounds.size.y * Fx::HALF).to_f32().max(f32::EPSILON);
        let centre = self.effective_position();
        let centre_x = centre.x.to_f32();
        let centre_y = centre.y.to_f32();

        [
            [1.0 / half_width, 0.0, 0.0, 0.0],
            [0.0, -1.0 / half_height, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-centre_x / half_width, centre_y / half_height, 0.0, 1.0],
        ]
    }

    /// Converts a world position to a pixel position within the viewport.
    #[must_use]
    pub fn world_to_screen(&self, world: Vec2) -> Vec2 {
        let bounds = self.visible_bounds();
        let pixels_per_unit = self.pixels_per_unit();
        Vec2::new(
            (world.x - bounds.left()) * pixels_per_unit,
            (world.y - bounds.top()) * pixels_per_unit,
        )
    }

    /// Converts a pixel position within the viewport to a world position.
    #[must_use]
    pub fn screen_to_world(&self, screen: Vec2) -> Vec2 {
        let bounds = self.visible_bounds();
        let pixels_per_unit = self.pixels_per_unit();
        Vec2::new(
            bounds.left() + screen.x / pixels_per_unit,
            bounds.top() + screen.y / pixels_per_unit,
        )
    }

    /// Moves the camera toward `target` at a frame-rate independent rate.
    ///
    /// `half_life` is the time to close half the remaining distance, which is
    /// what makes the follow feel the same at 30 and 144 Hz. A naive
    /// `lerp(position, target, 0.1)` per frame would not.
    pub fn follow(&mut self, target: Vec2, half_life: Fx, dt: Fx) {
        if !half_life.is_positive() {
            self.position = target;
            return;
        }
        // 2^(-dt / half_life), approximated by repeated halving plus a linear
        // remainder so the whole camera stays in deterministic fixed point.
        let ratio = dt / half_life;
        let decay = exp2_negative(ratio);
        self.position = target + (self.position - target) * decay;
    }

    /// Constrains the camera so its view stays inside `bounds`.
    ///
    /// When the world is smaller than the viewport on an axis, the camera
    /// centres on it rather than clamping to an edge — otherwise a small room
    /// would sit against one side of the screen.
    pub fn clamp_to(&mut self, bounds: Rect) {
        let view = self.visible_bounds().size;
        let half = view * Fx::HALF;

        self.position.x = if view.x >= bounds.size.x {
            bounds.centre().x
        } else {
            self.position
                .x
                .clamp(bounds.left() + half.x, bounds.right() - half.x)
        };
        self.position.y = if view.y >= bounds.size.y {
            bounds.centre().y
        } else {
            self.position
                .y
                .clamp(bounds.top() + half.y, bounds.bottom() - half.y)
        };
    }
}

/// Computes `2^-x` for non-negative `x` in fixed point.
///
/// Exact halving for the integer part, then a quadratic fit for the fraction —
/// accurate to about `1e-3`, which is far below what a camera follow can show,
/// and deterministic, which `f64::exp2` is not.
fn exp2_negative(x: Fx) -> Fx {
    if !x.is_positive() {
        return Fx::ONE;
    }
    // Beyond this the result is smaller than the fixed-point epsilon.
    if x > Fx::from_num(40) {
        return Fx::ZERO;
    }
    let whole = x.floor_int();
    let fraction = x.fract();

    // 2^-f ≈ 1 - 0.6614*f + 0.1619*f^2 over f in [0, 1).
    let linear = Fx::from_ratio(6614, 10_000);
    let quadratic = Fx::from_ratio(1619, 10_000);
    let mut result = Fx::ONE - linear * fraction + quadratic * fraction * fraction;

    for _ in 0..whole {
        result *= Fx::HALF;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use verdant_core_math::fx;

    #[test]
    fn the_visible_area_is_centred_on_the_camera() {
        let mut camera = Camera2D::new(320, 240);
        camera.position = Vec2::from_ints(100, 50);
        let bounds = camera.visible_bounds();
        assert_eq!(bounds.centre(), Vec2::from_ints(100, 50));
        assert_eq!(bounds.size, Vec2::from_ints(320, 240));
    }

    #[test]
    fn zooming_in_shows_less_of_the_world() {
        let camera = Camera2D::new(320, 240);
        let wide = camera.visible_bounds().size;
        let close = camera.with_zoom(4).visible_bounds().size;
        assert_eq!(close.x, wide.x / fx(4));
        assert_eq!(close.y, wide.y / fx(4));
    }

    #[test]
    fn a_zero_zoom_is_treated_as_one() {
        assert_eq!(Camera2D::new(64, 64).with_zoom(0).zoom, 1);
    }

    #[test]
    fn pixel_snapping_removes_subpixel_camera_offsets() {
        let mut camera = Camera2D::new(320, 240);
        camera.position = Vec2::new(Fx::from_ratio(1003, 100), Fx::from_ratio(497, 100));

        let snapped = camera.effective_position();
        assert_eq!(snapped.x, fx(10), "10.03 snaps to the pixel at 10");
        assert_eq!(snapped.y, fx(5), "4.97 snaps to the pixel at 5");
    }

    #[test]
    fn subpixel_cameras_keep_their_exact_position() {
        let mut camera = Camera2D::new(320, 240).subpixel();
        camera.position = Vec2::new(Fx::from_ratio(1003, 100), Fx::ZERO);
        assert_eq!(camera.effective_position().x, Fx::from_ratio(1003, 100));
    }

    #[test]
    fn snapping_happens_in_pixel_space_not_world_space() {
        // At zoom 4 a world unit is four pixels, so quarter-unit positions are
        // exactly on a pixel and must survive snapping.
        let mut camera = Camera2D::new(320, 240).with_zoom(4);
        camera.position = Vec2::new(Fx::from_ratio(1, 4), Fx::ZERO);
        assert_eq!(camera.effective_position().x, Fx::from_ratio(1, 4));
    }

    #[test]
    fn screen_and_world_coordinates_round_trip() {
        let mut camera = Camera2D::new(320, 240);
        camera.position = Vec2::from_ints(1000, 500);

        for point in [Vec2::from_ints(1000, 500), Vec2::from_ints(900, 450)] {
            let screen = camera.world_to_screen(point);
            let back = camera.screen_to_world(screen);
            assert!(
                back.distance(point).to_f64() < 1e-4,
                "{point:?} did not round trip"
            );
        }
    }

    #[test]
    fn the_camera_centre_maps_to_the_viewport_centre() {
        let mut camera = Camera2D::new(320, 240);
        camera.position = Vec2::from_ints(50, 50);
        let screen = camera.world_to_screen(Vec2::from_ints(50, 50));
        assert_eq!(screen, Vec2::from_ints(160, 120));
    }

    #[test]
    fn the_projection_maps_the_view_onto_clip_space() {
        let mut camera = Camera2D::new(320, 240);
        camera.position = Vec2::from_ints(100, 100);
        let matrix = camera.view_projection();

        /// Applies the column-major matrix to a world point.
        fn project(matrix: [[f32; 4]; 4], x: f32, y: f32) -> (f32, f32) {
            (
                matrix[0][0] * x + matrix[1][0] * y + matrix[3][0],
                matrix[0][1] * x + matrix[1][1] * y + matrix[3][1],
            )
        }

        let centre = project(matrix, 100.0, 100.0);
        assert!(
            centre.0.abs() < 1e-5 && centre.1.abs() < 1e-5,
            "the centre maps to the origin"
        );

        // The left edge of the view maps to clip x = -1.
        let left = project(matrix, 100.0 - 160.0, 100.0);
        assert!(
            (left.0 + 1.0).abs() < 1e-4,
            "left edge mapped to {}",
            left.0
        );

        // World y grows downward, clip y grows upward, so the top maps to +1.
        let top = project(matrix, 100.0, 100.0 - 120.0);
        assert!((top.1 - 1.0).abs() < 1e-4, "top edge mapped to {}", top.1);
    }

    #[test]
    fn following_converges_on_the_target() {
        let mut camera = Camera2D::new(320, 240).subpixel();
        let target = Vec2::from_ints(500, 300);
        for _ in 0..600 {
            camera.follow(target, Fx::from_ratio(1, 4), Fx::from_ratio(1, 60));
        }
        assert!(camera.position.distance(target).to_f64() < 0.1);
    }

    #[test]
    fn following_is_frame_rate_independent() {
        let target = Vec2::from_ints(100, 0);
        let half_life = Fx::from_ratio(1, 2);

        let mut slow = Camera2D::new(320, 240).subpixel();
        for _ in 0..30 {
            slow.follow(target, half_life, Fx::from_ratio(1, 30));
        }
        let mut fast = Camera2D::new(320, 240).subpixel();
        for _ in 0..120 {
            fast.follow(target, half_life, Fx::from_ratio(1, 120));
        }
        // One second of following at either rate must land in the same place.
        assert!(
            (slow.position.x - fast.position.x).abs().to_f64() < 1.0,
            "30 Hz reached {} but 120 Hz reached {}",
            slow.position.x.to_f64(),
            fast.position.x.to_f64()
        );
    }

    #[test]
    fn a_zero_half_life_snaps_immediately() {
        let mut camera = Camera2D::new(320, 240).subpixel();
        camera.follow(Vec2::from_ints(9, 9), Fx::ZERO, Fx::from_ratio(1, 60));
        assert_eq!(camera.position, Vec2::from_ints(9, 9));
    }

    #[test]
    fn clamping_keeps_the_view_inside_the_world() {
        let mut camera = Camera2D::new(320, 240);
        let world = Rect::from_ints(0, 0, 1000, 1000);

        camera.position = Vec2::from_ints(-500, -500);
        camera.clamp_to(world);
        assert_eq!(camera.position, Vec2::from_ints(160, 120));

        camera.position = Vec2::from_ints(5000, 5000);
        camera.clamp_to(world);
        assert_eq!(camera.position, Vec2::from_ints(840, 880));
    }

    #[test]
    fn a_world_smaller_than_the_view_is_centred() {
        let mut camera = Camera2D::new(320, 240);
        let small_room = Rect::from_ints(0, 0, 100, 100);
        camera.position = Vec2::from_ints(-999, 999);
        camera.clamp_to(small_room);
        assert_eq!(camera.position, small_room.centre());
    }

    #[test]
    fn exp2_negative_matches_the_real_function() {
        for step in 0..80i32 {
            let x = Fx::from_ratio(step, 8);
            let expected = (-x.to_f64()).exp2();
            let actual = exp2_negative(x).to_f64();
            assert!(
                (actual - expected).abs() < 5e-3,
                "2^-{} gave {actual}, want {expected}",
                x.to_f64()
            );
        }
        assert_eq!(exp2_negative(Fx::ZERO), Fx::ONE);
        assert_eq!(exp2_negative(fx(-5)), Fx::ONE);
    }
}
