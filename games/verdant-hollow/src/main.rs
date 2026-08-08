//! Verdant Hollow's desktop executable.
//!
//! The game itself lives in the library, because there are two ways in — this
//! binary and an Android activity — and they differ only in how the platform
//! hands over control. See [`verdant_hollow::app`].

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("verdant_hollow=info,warn"),
    )
    .init();

    if let Err(error) = verdant_hollow::app::run_desktop() {
        // A machine with no display is a normal condition, not a crash: say
        // so plainly and point at the way to see the game without one.
        eprintln!("could not start Verdant Hollow: {error}");
        eprintln!(
            "The game needs a display. To render frames without one, run:\n  \
             cargo run -p verdant-hollow --example screenshot"
        );
        std::process::exit(1);
    }
}
