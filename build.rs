// Embed Info.plist into the binary's __TEXT,__info_plist section so TCC reads the
// NSMicrophoneUsageDescription / NSAudioCaptureUsageDescription strings and the
// stable CFBundleIdentifier. Without this a plain CLI binary has no usage strings
// and the "record system audio" / microphone prompts never fire.
fn main() {
    #[cfg(target_os = "macos")]
    {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let plist = format!("{manifest}/Info.plist");
        println!("cargo:rerun-if-changed=Info.plist");
        println!("cargo:rustc-link-arg=-Wl,-sectcreate,__TEXT,__info_plist,{plist}");
    }
}
