// mod.rs - Module Exports
// This file exposes all 15 spoofing modules

// Core spoofing modules
pub mod disk_serial;
pub mod mac_address;
pub mod motherboard;
pub mod system_uuid;
pub mod cpu_id;
pub mod gpu_id;
pub mod pci_devices;
pub mod registry_clean;
pub mod wmi_spoof;
pub mod volume_serial;
pub mod network_stack;
pub mod bios_info;
pub mod acpi_tables;
pub mod driver_hooks;
pub mod evasion;
pub mod hwid_snapshot;

// Re-export commonly used functions for convenience
pub use disk_serial::{spoof_disk_serial, generate_disk_serial, verify_disk_spoof};
pub use mac_address::{spoof_mac_address, generate_mac_address, list_network_adapters};
pub use motherboard::{spoof_motherboard, generate_motherboard_serial, generate_system_uuid};
pub use system_uuid::spoof_system_uuid;
pub use cpu_id::spoof_cpu_id;
pub use gpu_id::spoof_gpu_id;
pub use pci_devices::hide_pci_devices;
pub use registry_clean::clean_registry_artifacts;
pub use wmi_spoof::hook_wmi_queries;
pub use volume_serial::spoof_volume_serial;
pub use network_stack::spoof_network_stack;
pub use bios_info::spoof_bios_info;
pub use acpi_tables::inject_acpi_override;
pub use driver_hooks::install_driver_hooks;
pub use evasion::{check_debugger, check_vm, check_sandbox, enable_anti_analysis};
pub use hwid_snapshot::{
    compare_snapshots, format_verification_report, is_admin_elevated, print_aba_next_step,
    print_admin_status, print_terminal_result_box, print_verification_to_terminal,
    load_saved_snapshots, verify_from_saved_files, write_verification_files, ABA_NEXT_STEP_LINES,
    FieldStatus,
    HwidSnapshot, OverallVerdict, VerificationReport, AFTER_FILE, BEFORE_FILE, VERIFY_REPORT_FILE,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const STEP_MARKER_FILE: &str = "hwid_step.ok";

/// Marker written after a successful apply (used by OverlayAssist `run_windows.bat`).
pub fn step_marker_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("logs")
        .join(STEP_MARKER_FILE)
}

pub fn write_step_marker(report: &VerificationReport) -> std::io::Result<()> {
    let path = step_marker_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let stamp = chrono::Utc::now().to_rfc3339();
    let verdict = format!("{:?}", report.verdict);
    std::fs::write(
        &path,
        format!(
            "ok\ncompleted_utc={stamp}\nversion={VERSION}\nverdict={verdict}\nchanged={}\nunchanged={}\nreboot_recommended={}\n",
            report.changed_count,
            report.unchanged_count,
            report.reboot_recommended,
        ),
    )?;
    Ok(())
}

pub fn step_marker_exists() -> bool {
    step_marker_path().is_file()
}

#[derive(Debug, Clone, Default)]
pub struct SpoofOptions {
    pub disk: bool,
    pub mac: bool,
    pub motherboard: bool,
    pub system_uuid: bool,
    pub cpu: bool,
    pub gpu: bool,
    pub pci: bool,
    pub registry_clean: bool,
    pub volume: bool,
    pub network: bool,
    pub bios: bool,
    pub acpi: bool,
    pub wmi: bool,
    pub driver_hooks: bool,
    pub run_embedded_module: bool,
}

impl SpoofOptions {
    pub fn all_enabled() -> Self {
        Self {
            disk: true,
            mac: true,
            motherboard: true,
            system_uuid: true,
            cpu: true,
            gpu: true,
            pci: true,
            registry_clean: true,
            volume: true,
            network: true,
            bios: true,
            acpi: true,
            wmi: true,
            driver_hooks: true,
            run_embedded_module: true,
        }
    }
}

/// Result of apply + before/after verification.
#[derive(Debug)]
pub struct ApplyOutcome {
    pub before: HwidSnapshot,
    pub after: HwidSnapshot,
    pub verification: VerificationReport,
    pub report_path: std::path::PathBuf,
}

pub fn print_current_hwid_report() -> std::io::Result<()> {
    let snap = HwidSnapshot::capture("current");
    println!("{}", snap.to_report_text());
    if !snap.admin_elevated {
        println!();
        println!("[hwspoof] WARN: not running elevated — reads may show (read failed).");
    }
    Ok(())
}

pub fn apply_spoof_modules(
    options: &SpoofOptions,
    log: &mut dyn FnMut(&str),
) -> std::io::Result<ApplyOutcome> {
    #[cfg(not(windows))]
    {
        let _ = (options, log);
        return Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "HWID Spoofer requires Windows 10/11 x64 (Administrator)",
        ));
    }

    #[cfg(windows)]
    {
        use hwid_snapshot::HwidSnapshot;
        use std::io::{Error, ErrorKind};

        print_admin_status();
        if !is_admin_elevated() {
            return Err(Error::new(
                ErrorKind::PermissionDenied,
                "Administrator elevation required - use Run_As_Admin.bat",
            ));
        }

        log(&format!("HWID Spoofer v{VERSION} — apply sequence"));
        log("── BEFORE (hardware IDs only) ──");
        let before = HwidSnapshot::capture("before");
        for line in before.to_report_text().lines() {
            log(&format!("  {line}"));
        }

        let mut module_errors: Vec<String> = Vec::new();

        macro_rules! run_step {
            ($label:expr, $body:expr) => {
                log($label);
                if let Err(e) = $body {
                    let msg = format!("{}: {}", $label, e);
                    log(&format!("   ✗ {msg}"));
                    module_errors.push(msg);
                }
            };
        }

        run_step!(
            "→ Anti-analysis checks",
            enable_anti_analysis().map_err(|e| Error::other(e.to_string()))
        );

        if options.registry_clean {
            run_step!(
                "→ Registry artifact cleanup",
                clean_registry_artifacts().map_err(|e| Error::other(e.to_string()))
            );
        }

        if options.disk {
            run_step!(
                "→ Disk serial (PhysicalDrive0)",
                spoof_disk_serial(0, None).map_err(|e| Error::other(e.to_string()))
            );
        }

        if options.volume {
            run_step!(
                "→ Volume serial (C:)",
                spoof_volume_serial('C', None)
                    .map_err(|e| Error::other(e.to_string()))
            );
        }

        if options.mac {
            log("→ MAC address (active adapters)");
            match list_network_adapters() {
                Ok(adapters) => {
                    let mut any = false;
                    for adapter in adapters {
                        let name = adapter.trim();
                        if name.is_empty() {
                            continue;
                        }
                        let lower = name.to_lowercase();
                        if lower.contains("loopback")
                            || lower.contains("virtual")
                            || lower.contains("bluetooth")
                        {
                            continue;
                        }
                        any = true;
                        run_step!(
                            &format!("   adapter: {name}"),
                            spoof_mac_address(name, None)
                                .map_err(|e| Error::other(e.to_string()))
                        );
                    }
                    if !any {
                        log("   WARN no physical adapters matched");
                    }
                }
                Err(e) => {
                    log(&format!("   WARN adapter list: {e}"));
                    module_errors.push(format!("adapter list: {e}"));
                }
            }
        }

        if options.motherboard {
            run_step!(
                "→ Motherboard serial + SMBIOS UUID",
                spoof_motherboard(None, None)
                    .map_err(|e| Error::other(e.to_string()))
            );
        }

        if options.system_uuid {
            run_step!(
                "→ System UUID / MachineGuid",
                spoof_system_uuid(None).map_err(|e| Error::other(e.to_string()))
            );
        }

        if options.cpu {
            run_step!(
                "→ CPU identifier",
                spoof_cpu_id().map_err(|e| Error::other(e.to_string()))
            );
        }

        if options.gpu {
            run_step!(
                "→ GPU PCI identifiers",
                spoof_gpu_id(Some(0x10DE), Some(0x2484))
                    .map_err(|e| Error::other(e.to_string()))
            );
        }

        if options.pci {
            run_step!(
                "→ PCI device visibility",
                hide_pci_devices(vec![])
                    .map_err(|e| Error::other(e.to_string()))
            );
        }

        if options.bios {
            run_step!(
                "→ BIOS info",
                spoof_bios_info("American Megatrends Inc.", "ABA-HWID-1.4.2", "05/21/2026")
                    .map_err(|e| Error::other(e.to_string()))
            );
        }

        if options.acpi {
            run_step!(
                "→ ACPI table override hook",
                inject_acpi_override().map_err(|e| Error::other(e.to_string()))
            );
        }

        if options.network {
            run_step!(
                "→ Network stack identifiers",
                spoof_network_stack().map_err(|e| Error::other(e.to_string()))
            );
        }

        if options.wmi {
            run_step!(
                "→ WMI query hooks",
                hook_wmi_queries().map_err(|e| Error::other(e.to_string()))
            );
        }

        if options.driver_hooks {
            run_step!(
                "→ Driver / DSE hooks (open-source stub layer)",
                install_driver_hooks().map_err(|e| Error::other(e.to_string()))
            );
        }

        if options.run_embedded_module {
            log("→ Embedded private module (if shipped in assets/extra.exe)");
            if let Err(e) = run_embedded_private_module(log) {
                module_errors.push(format!("embedded module: {e}"));
                log(&format!("   ✗ embedded module: {e}"));
            }
        }

        log("── AFTER (hardware IDs only) ──");
        let after = HwidSnapshot::capture("after");
        for line in after.to_report_text().lines() {
            log(&format!("  {line}"));
        }

        let verification = compare_snapshots(&before, &after, options, &module_errors);
        let report_path = write_verification_files(&before, &after, &verification)?;
        print_verification_to_terminal(&before, &after, &verification);
        print_terminal_result_box(&verification);
        log(&format!("Report saved: {}", report_path.display()));

        if verification.acceptable_for_marker() {
            write_step_marker(&verification)?;
            log("✓ Step marker written (hwid_step.ok)");
        } else {
            log("✗ Step marker NOT written — verification did not pass (see report above)");
            let _ = std::fs::remove_file(step_marker_path());
        }

        if verification.reboot_recommended {
            log("⚠ Restart Windows, then: hwspoof.exe --verify-last");
        }

        for line in ABA_NEXT_STEP_LINES {
            log(line);
        }

        if !verification.acceptable_for_marker() {
            return Err(std::io::Error::other(
                "verification failed - see BEFORE/AFTER report above",
            ));
        }

        Ok(ApplyOutcome {
            before,
            after,
            verification,
            report_path,
        })
    }
}

/// Decrypt and launch `assets/extra.exe` when the release build embedded it (upstream layout).
#[cfg(windows)]
pub fn run_embedded_private_module(log: &mut dyn FnMut(&str)) -> std::io::Result<()> {
    use std::io::Error;
    use std::process::Command;

    let encrypted = include_bytes!(concat!(env!("OUT_DIR"), "/embedded_exe.bin"));
    if encrypted.is_empty() {
        log("   (no embedded module — place assets/extra.exe before building for full upstream binary)");
        return Ok(());
    }

    use sha2::{Digest, Sha256};
    let key_source = b"hwidspoof.net_2025_encryption_key_v1.4.2";
    let mut hasher = Sha256::new();
    hasher.update(key_source);
    let key = hasher.finalize();
    let decrypted: Vec<u8> = encrypted
        .iter()
        .enumerate()
        .map(|(i, &byte)| byte ^ key[i % key.len()])
        .collect();

    let temp_dir = std::env::temp_dir().join("hwid_spoofer_embed");
    std::fs::create_dir_all(&temp_dir)?;
    let exe_path = temp_dir.join("extra_module.exe");
    std::fs::write(&exe_path, &decrypted)?;

    log(&format!("   launching {}", exe_path.display()));
    let status = Command::new(&exe_path)
        .status()
        .map_err(|e| Error::other(e.to_string()))?;
    if !status.success() {
        return Err(Error::other(
            format!("embedded module exited with {status}"),
        ));
    }
    Ok(())
}

#[cfg(not(windows))]
pub fn run_embedded_private_module(_log: &mut dyn FnMut(&str)) -> std::io::Result<()> {
    Ok(())
}
