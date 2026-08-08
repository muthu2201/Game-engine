plugins {
    id("com.android.application")
}

android {
    namespace = "dev.verdant.hollow"
    compileSdk = 35

    defaultConfig {
        applicationId = "dev.verdant.hollow"
        // 26, and not lower, because that is where AAudio appears: cpal
        // reaches the speakers through Oboe, which links libaaudio, and the
        // NDK sysroot for an earlier level simply does not contain it. It is
        // also comfortably past 24, where Vulkan support becomes dependable.
        // Keep this in step with `--platform` in android/build-apk.sh.
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
    }

    // The .so files come from cargo-ndk, which writes them into this tree
    // before Gradle runs. Gradle is only a packager here — it never invokes
    // the Rust build, so a failure has one obvious owner.
    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }

    buildTypes {
        release {
            // Unsigned by default: a release key belongs to whoever publishes
            // the game, not to the repository.
            isMinifyEnabled = false
        }
        debug {
            // Signed with the standard debug key, so CI's artifact installs
            // on any device with `adb install`.
            isDebuggable = true
        }
    }

    // No Java or Kotlin source: android-activity's NativeActivity backend
    // means the whole game is Rust.
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    packaging {
        jniLibs {
            // The debug build keeps symbols so a native crash in CI has a
            // readable stack.
            keepDebugSymbols += "**/*.so"
        }
    }
}
