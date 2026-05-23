use std::path::Path;
use std::fs;
use std::env;
use sha2::{Sha256, Digest};

fn main() {
    // Embed and encrypt extra.exe (SIMPLY OUR PRIVATE MODULE)
    // We do not give source of our private modules but here is how they are implemented:
    // We simply do not give out source of private kernal functions because they are dangerous.
    // and will get patched very fast, its an cat and mouse game
    let out_dir = env::var("OUT_DIR").unwrap();
    let exe_path = Path::new("assets/extra.exe");
    
    if exe_path.exists() {
        println!("cargo:rerun-if-changed=assets/extra.exe");
        
        // Read the exe
        let exe_data = fs::read(exe_path).expect("Failed to read assets/extra.exe");
        println!("cargo:warning=Embedding extra.exe ({} bytes)", exe_data.len());
        
        // Encrypt with SHA256 key (same as decryption in main.rs)
        let key_source = b"hwidspoof.net_2025_encryption_key_v1.4.2";
        let mut hasher = Sha256::new();
        hasher.update(key_source);
        let key = hasher.finalize();
        
        let encrypted: Vec<u8> = exe_data
            .iter()
            .enumerate()
            .map(|(i, &byte)| byte ^ key[i % key.len()])
            .collect();
        
        // Write encrypted binary
        let dest_path = Path::new(&out_dir).join("embedded_exe.bin");
        fs::write(&dest_path, encrypted).expect("Failed to write embedded_exe.bin");
        
        println!("cargo:warning=✓ extra.exe encrypted and embedded");
    } else {
        println!("cargo:warning=⚠ assets/extra.exe not found - creating empty placeholder");
        
        // Create empty file so compilation doesn't fail
        let dest_path = Path::new(&out_dir).join("embedded_exe.bin");
        fs::write(&dest_path, []).expect("Failed to write empty embedded_exe.bin");
    }
    
    // Only run Windows resource compilation on Windows
    if cfg!(target_os = "windows") {
        let mut res = winres::WindowsResource::new();

        // Optional icon
        if Path::new("assets/icon.ico").exists() {
            res.set_icon("assets/icon.ico");
            println!("cargo:rerun-if-changed=assets/icon.ico");
        }

        // Standard metadata
        res.set_language(0x0409) // en-US
            .set("ProductName", "HWID Spoofer")
            .set("FileDescription", "Hardware ID Modification Tool")
            .set("LegalCopyright", "Copyright (c) 2025 hwidspoof.net")
            .set("CompanyName", "hwidspoof.net")
            .set("ProductVersion", "1.4.2.0")
            .set("FileVersion", "1.4.2.0");

        if Path::new("assets/manifest.xml").exists() {
            res.set_manifest_file("assets/manifest.xml");
            println!("cargo:rerun-if-changed=assets/manifest.xml");
        }

        if let Err(e) = res.compile() {
            eprintln!("Warning: failed to compile Windows resources: {}", e);
        }
    }
}