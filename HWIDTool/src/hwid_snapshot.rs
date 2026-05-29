//! Capture and compare hardware identifiers — before/after spoof visibility.

use std::fmt::Write as _;
use std::path::PathBuf;

pub const BEFORE_FILE: &str = "hwid_before.txt";
pub const AFTER_FILE: &str = "hwid_after.txt";
pub const VERIFY_REPORT_FILE: &str = "hwid_verify_report.txt";

/// Printed after HWID success — no ABA mode terminology (live/dry-run are ABA-only).
pub const ABA_NEXT_STEP_LINES: &[&str] = &[
    "── Next: ABA (separate app) ──",
    "  cd ..",
    "  run_windows.bat",
    "  (or: .venv\\Scripts\\python.exe aba.py)",
    "  GUI requires ban acknowledgment on first Start.",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldStatus {
    /// Value differs after apply (good sign spoof touched this vector).
    Changed,
    /// Same value before and after — may need reboot or module did not apply.
    Unchanged,
    /// Could not read one or both sides.
    ReadFailed,
    /// Not applicable (module disabled or skipped).
    Skipped,
}

#[derive(Debug, Clone)]
pub struct FieldCheck {
    pub name: &'static str,
    pub before: String,
    pub after: String,
    pub status: FieldStatus,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverallVerdict {
    /// At least one identifier changed immediately.
    SuccessChanged,
    /// Nothing changed in readable probes — spoof likely ineffective until reboot or failed.
    FailedNoChange,
    /// Some reads failed — inconclusive.
    Inconclusive,
    /// Report-only (no apply).
    ReportOnly,
}

#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub checks: Vec<FieldCheck>,
    pub verdict: OverallVerdict,
    pub changed_count: usize,
    pub unchanged_count: usize,
    pub failed_read_count: usize,
    pub reboot_recommended: bool,
    pub module_errors: Vec<String>,
}

impl VerificationReport {
    pub fn acceptable_for_marker(&self) -> bool {
        if std::env::var("HWID_ALLOW_NO_CHANGE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        {
            return self.module_errors.is_empty();
        }
        match self.verdict {
            OverallVerdict::SuccessChanged => self.module_errors.is_empty(),
            OverallVerdict::Inconclusive => {
                self.changed_count > 0 && self.module_errors.is_empty()
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HwidSnapshot {
    pub label: String,
    pub captured_at: String,
    pub admin_elevated: bool,
    pub machine_guid: String,
    pub computer_hardware_id: String,
    pub product_id: String,
    pub hw_profile_guid: String,
    pub disk_registry_serial: String,
    pub disk_wmic_serial: String,
    pub volume_c_serial: String,
    pub baseboard_serial: String,
    pub bios_serial: String,
    pub bios_version: String,
    pub system_uuid_wmic: String,
    pub hostname: String,
    pub adapters: Vec<(String, String)>,
}

impl HwidSnapshot {
    pub fn capture(label: impl Into<String>) -> Self {
        let label = label.into();
        #[cfg(windows)]
        {
            capture_windows(label)
        }
        #[cfg(not(windows))]
        {
            let _ = label;
            HwidSnapshot {
                label: "non-windows".into(),
                captured_at: chrono::Utc::now().to_rfc3339(),
                machine_guid: "(requires Windows)".into(),
                ..Default::default()
            }
        }
    }

    pub fn to_report_text(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "=== HWID snapshot: {} ===", self.label);
        let _ = writeln!(out, "captured_at: {}", self.captured_at);
        let _ = writeln!(out, "admin_elevated: {}", self.admin_elevated);
        let _ = writeln!(out, "MachineGuid: {}", self.machine_guid);
        let _ = writeln!(out, "ComputerHardwareId: {}", self.computer_hardware_id);
        let _ = writeln!(out, "ProductId: {}", self.product_id);
        let _ = writeln!(out, "HwProfileGuid: {}", self.hw_profile_guid);
        let _ = writeln!(out, "Disk registry serial: {}", self.disk_registry_serial);
        let _ = writeln!(out, "Disk WMIC serial: {}", self.disk_wmic_serial);
        let _ = writeln!(out, "Volume C: serial: {}", self.volume_c_serial);
        let _ = writeln!(out, "Baseboard serial: {}", self.baseboard_serial);
        let _ = writeln!(out, "BIOS serial: {}", self.bios_serial);
        let _ = writeln!(out, "BIOS version: {}", self.bios_version);
        let _ = writeln!(out, "System UUID (WMIC): {}", self.system_uuid_wmic);
        let _ = writeln!(out, "Hostname: {}", self.hostname);
        for (name, mac) in &self.adapters {
            let _ = writeln!(out, "NIC [{}]: {}", name, mac);
        }
        out
    }
}

pub fn logs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("logs")
}

pub fn save_snapshot(snap: &HwidSnapshot, filename: &str) -> std::io::Result<PathBuf> {
    let dir = logs_dir();
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(filename);
    std::fs::write(&path, snap.to_report_text())?;
    Ok(path)
}

pub fn load_snapshot_file(filename: &str) -> std::io::Result<String> {
    std::fs::read_to_string(logs_dir().join(filename))
}

pub fn compare_snapshots(
    before: &HwidSnapshot,
    after: &HwidSnapshot,
    enabled: &super::SpoofOptions,
    module_errors: &[String],
) -> VerificationReport {
    let mut checks = Vec::new();

    push_check(
        &mut checks,
        "MachineGuid",
        &before.machine_guid,
        &after.machine_guid,
        enabled.system_uuid,
        true,
    );
    push_check(
        &mut checks,
        "ComputerHardwareId",
        &before.computer_hardware_id,
        &after.computer_hardware_id,
        enabled.system_uuid,
        true,
    );
    push_check(
        &mut checks,
        "ProductId",
        &before.product_id,
        &after.product_id,
        enabled.system_uuid,
        false,
    );
    push_check(
        &mut checks,
        "HwProfileGuid",
        &before.hw_profile_guid,
        &after.hw_profile_guid,
        enabled.system_uuid,
        false,
    );
    push_check(
        &mut checks,
        "Disk registry serial",
        &before.disk_registry_serial,
        &after.disk_registry_serial,
        enabled.disk,
        true,
    );
    push_check(
        &mut checks,
        "Disk WMIC serial",
        &before.disk_wmic_serial,
        &after.disk_wmic_serial,
        enabled.disk,
        true,
    );
    push_check(
        &mut checks,
        "Volume C: serial",
        &before.volume_c_serial,
        &after.volume_c_serial,
        enabled.volume,
        true,
    );
    push_check(
        &mut checks,
        "Baseboard serial",
        &before.baseboard_serial,
        &after.baseboard_serial,
        enabled.motherboard,
        true,
    );
    push_check(
        &mut checks,
        "BIOS serial",
        &before.bios_serial,
        &after.bios_serial,
        enabled.bios,
        true,
    );
    push_check(
        &mut checks,
        "System UUID (WMIC)",
        &before.system_uuid_wmic,
        &after.system_uuid_wmic,
        enabled.motherboard || enabled.system_uuid,
        true,
    );
    push_check(
        &mut checks,
        "Hostname",
        &before.hostname,
        &after.hostname,
        enabled.network,
        false,
    );

    compare_adapters(&mut checks, before, after, enabled.mac);

    let changed_count = checks
        .iter()
        .filter(|c| c.status == FieldStatus::Changed)
        .count();
    let unchanged_count = checks
        .iter()
        .filter(|c| c.status == FieldStatus::Unchanged)
        .count();
    let failed_read_count = checks
        .iter()
        .filter(|c| c.status == FieldStatus::ReadFailed)
        .count();

    let reboot_recommended = unchanged_count > 0
        && checks.iter().any(|c| {
            c.status == FieldStatus::Unchanged
                && c.note.contains("reboot")
                && c.before.is_valid_value()
        });

    let verdict = if changed_count > 0 {
        OverallVerdict::SuccessChanged
    } else if failed_read_count > 0 && changed_count == 0 {
        OverallVerdict::Inconclusive
    } else if unchanged_count > 0 {
        OverallVerdict::FailedNoChange
    } else {
        OverallVerdict::Inconclusive
    };

    VerificationReport {
        checks,
        verdict,
        changed_count,
        unchanged_count,
        failed_read_count,
        reboot_recommended,
        module_errors: module_errors.to_vec(),
    }
}

pub fn format_verification_report(
    before: &HwidSnapshot,
    after: &HwidSnapshot,
    report: &VerificationReport,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "════════════════════════════════════════════════════════");
    let _ = writeln!(out, "  HWID BEFORE / AFTER VERIFICATION");
    let _ = writeln!(out, "════════════════════════════════════════════════════════");
    let _ = writeln!(out);
    let _ = writeln!(out, "BEFORE ({})", before.captured_at);
    let _ = writeln!(out, "  MachineGuid:          {}", before.machine_guid);
    let _ = writeln!(out, "  Disk WMIC:            {}", before.disk_wmic_serial);
    let _ = writeln!(out, "  Volume C::            {}", before.volume_c_serial);
    let _ = writeln!(out, "  Baseboard:            {}", before.baseboard_serial);
    let _ = writeln!(out, "  System UUID:          {}", before.system_uuid_wmic);
    if let Some((n, m)) = before.adapters.first() {
        let _ = writeln!(out, "  First NIC [{}]:      {}", n, m);
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "AFTER ({})", after.captured_at);
    let _ = writeln!(out, "  MachineGuid:          {}", after.machine_guid);
    let _ = writeln!(out, "  Disk WMIC:            {}", after.disk_wmic_serial);
    let _ = writeln!(out, "  Volume C::            {}", after.volume_c_serial);
    let _ = writeln!(out, "  Baseboard:            {}", after.baseboard_serial);
    let _ = writeln!(out, "  System UUID:          {}", after.system_uuid_wmic);
    if let Some((n, m)) = after.adapters.first() {
        let _ = writeln!(out, "  First NIC [{}]:      {}", n, m);
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "── Per-field ──");
    for c in &report.checks {
        let mark = match c.status {
            FieldStatus::Changed => "CHANGED ✓",
            FieldStatus::Unchanged => "UNCHANGED ⚠",
            FieldStatus::ReadFailed => "READ FAIL ?",
            FieldStatus::Skipped => "skipped",
        };
        let _ = writeln!(
            out,
            "  {:22}  {:12}  before={}  after={}",
            c.name, mark, c.before, c.after
        );
        if !c.note.is_empty() {
            let _ = writeln!(out, "    → {}", c.note);
        }
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Summary: changed={} unchanged={} read_fail={}",
        report.changed_count, report.unchanged_count, report.failed_read_count
    );
    if !report.module_errors.is_empty() {
        let _ = writeln!(out, "Module errors:");
        for e in &report.module_errors {
            let _ = writeln!(out, "  - {e}");
        }
    }
    let verdict_line = match report.verdict {
        OverallVerdict::SuccessChanged => {
            "VERDICT: SUCCESS — at least one identifier changed (spoof had an effect)."
        }
        OverallVerdict::FailedNoChange => {
            "VERDICT: FAILED — no readable identifiers changed. Spoof may not have worked."
        }
        OverallVerdict::Inconclusive => {
            "VERDICT: INCONCLUSIVE — some probes failed or no enabled fields changed."
        }
        OverallVerdict::ReportOnly => "VERDICT: REPORT ONLY",
    };
    let _ = writeln!(out, "{verdict_line}");
    if report.reboot_recommended {
        let _ = writeln!(
            out,
            "NOTE: Restart Windows, then run: hwspoof.exe --verify-last"
        );
        let _ = writeln!(
            out,
            "      Disk/SMBIOS values often stay the same until after reboot."
        );
    }
    if !report.acceptable_for_marker() {
        let _ = writeln!(
            out,
            "STEP MARKER: NOT written (verification did not pass). Set HWID_ALLOW_NO_CHANGE=1 to override."
        );
    } else {
        let _ = writeln!(out, "STEP MARKER: OK — hwid_step.ok written.");
    }
    let _ = writeln!(out);
    for line in ABA_NEXT_STEP_LINES {
        let _ = writeln!(out, "{line}");
    }
    out
}

pub fn print_aba_next_step() {
    println!();
    for line in ABA_NEXT_STEP_LINES {
        println!("{line}");
    }
}

pub fn write_verification_files(
    before: &HwidSnapshot,
    after: &HwidSnapshot,
    report: &VerificationReport,
) -> std::io::Result<PathBuf> {
    save_snapshot(before, BEFORE_FILE)?;
    save_snapshot(after, AFTER_FILE)?;
    let text = format_verification_report(before, after, report);
    let dir = logs_dir();
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(VERIFY_REPORT_FILE);
    std::fs::write(&path, &text)?;
    Ok(path)
}

pub fn load_saved_snapshots() -> std::io::Result<(HwidSnapshot, HwidSnapshot)> {
    let before_text = load_snapshot_file(BEFORE_FILE)?;
    let after_text = load_snapshot_file(AFTER_FILE)?;
    Ok((
        parse_snapshot_text("before (file)", &before_text),
        parse_snapshot_text("after (file)", &after_text),
    ))
}

pub fn verify_from_saved_files(options: &super::SpoofOptions) -> std::io::Result<VerificationReport> {
    let (before, after) = load_saved_snapshots()?;
    Ok(compare_snapshots(&before, &after, options, &[]))
}

// --- helpers ---

trait ValueProbe {
    fn is_valid_value(&self) -> bool;
    fn is_missing(&self) -> bool;
}

impl ValueProbe for String {
    fn is_valid_value(&self) -> bool {
        !self.is_missing()
    }

    fn is_missing(&self) -> bool {
        let t = self.trim();
        t.is_empty()
            || t.eq_ignore_ascii_case("unknown")
            || t.eq_ignore_ascii_case("(unavailable)")
            || t.eq_ignore_ascii_case("(read failed)")
            || t.starts_with("(error")
            || t == "(requires Windows)"
    }
}

fn push_check(
    checks: &mut Vec<FieldCheck>,
    name: &'static str,
    before: &str,
    after: &str,
    enabled: bool,
    reboot_sensitive: bool,
) {
    if !enabled {
        checks.push(FieldCheck {
            name,
            before: before.to_string(),
            after: after.to_string(),
            status: FieldStatus::Skipped,
            note: "module disabled".into(),
        });
        return;
    }

    let (status, mut note) = classify_pair(before, after, reboot_sensitive);
    if status == FieldStatus::Unchanged {
        if let Some(extra) = unchanged_field_explanation(name) {
            note.push_str("  •  ");
            note.push_str(extra);
        }
    }
    checks.push(FieldCheck {
        name,
        before: before.to_string(),
        after: after.to_string(),
        status,
        note,
    });
}

fn value_is_missing(s: &str) -> bool {
    let t = s.trim();
    t.is_empty()
        || t.eq_ignore_ascii_case("unknown")
        || t.eq_ignore_ascii_case("(unavailable)")
        || t.eq_ignore_ascii_case("(read failed)")
        || t.starts_with("(error")
        || t == "(requires Windows)"
}

fn classify_pair(before: &str, after: &str, reboot_sensitive: bool) -> (FieldStatus, String) {
    if value_is_missing(before) || value_is_missing(after) {
        return (
            FieldStatus::ReadFailed,
            "could not read value — wmic missing on Win11 23H2+? PowerShell fallback also failed, or not Administrator".into(),
        );
    }
    if before == after {
        let note = if reboot_sensitive {
            "unchanged — often needs Windows restart to show new value".into()
        } else {
            "unchanged — module may be stub-only or blocked by anti-cheat/driver".into()
        };
        (FieldStatus::Unchanged, note)
    } else {
        (FieldStatus::Changed, "value differs — spoof affected this probe".into())
    }
}

/// Per-field follow-up notes for fields users routinely report as
/// "unchanged" but which are actually expected to stay the same in the
/// running session.  `push_check` appends this to the classifier output
/// whenever a probe came back unchanged so the verification log is
/// self-explanatory.
pub fn unchanged_field_explanation(name: &str) -> Option<&'static str> {
    match name {
        "Volume C: serial" => Some(
            "Volume serial is stored in the boot sector and only re-read by Windows after a reboot or volume remount — value staying the same in-session is expected, NOT a failure.",
        ),
        "Hostname" => Some(
            "Hostname change is registry-only until next reboot; `hostname` command reflects the kernel-cached value until Win32 ComputerNameEx is re-broadcast (reboot).",
        ),
        "Network adapters (MAC)" => Some(
            "MAC change goes through the NIC driver's NetworkAddress override — many enterprise/AMD/Intel drivers ignore this and report the firmware-burned MAC. This is a driver limitation, not a tool bug.",
        ),
        _ => None,
    }
}

fn compare_adapters(
    checks: &mut Vec<FieldCheck>,
    before: &HwidSnapshot,
    after: &HwidSnapshot,
    enabled: bool,
) {
    if !enabled {
        return;
    }
    let b = before
        .adapters
        .iter()
        .map(|(n, m)| format!("{n}={m}"))
        .collect::<Vec<_>>()
        .join("; ");
    let a = after
        .adapters
        .iter()
        .map(|(n, m)| format!("{n}={m}"))
        .collect::<Vec<_>>()
        .join("; ");
    push_check(
        checks,
        "Network adapters (MAC)",
        if b.is_empty() { "(none)" } else { &b },
        if a.is_empty() { "(none)" } else { &a },
        true,
        true,
    );
}

fn parse_snapshot_text(label: &str, text: &str) -> HwidSnapshot {
    let mut snap = HwidSnapshot {
        label: label.to_string(),
        ..Default::default()
    };
    for line in text.lines() {
        if let Some((k, v)) = line.split_once(':') {
            let key = k.trim();
            let val = v.trim().to_string();
            match key.trim() {
                "captured_at" => snap.captured_at = val,
                "admin_elevated" => snap.admin_elevated = val == "true",
                "MachineGuid" => snap.machine_guid = val,
                "ComputerHardwareId" => snap.computer_hardware_id = val,
                "ProductId" => snap.product_id = val,
                "HwProfileGuid" => snap.hw_profile_guid = val,
                "Disk registry serial" => snap.disk_registry_serial = val,
                "Disk WMIC serial" => snap.disk_wmic_serial = val,
                "Volume C: serial" => snap.volume_c_serial = val,
                "Baseboard serial" => snap.baseboard_serial = val,
                "BIOS serial" => snap.bios_serial = val,
                "BIOS version" => snap.bios_version = val,
                "System UUID (WMIC)" => snap.system_uuid_wmic = val,
                "Hostname" => snap.hostname = val,
                _ if key.starts_with("NIC [") => {
                    if let Some(rb) = key.strip_prefix("NIC [").and_then(|s| s.strip_suffix(']')) {
                        snap.adapters.push((rb.to_string(), val));
                    }
                }
                _ => {}
            }
        }
    }
    snap
}

#[cfg(windows)]
fn capture_windows(label: String) -> HwidSnapshot {
    // Windows 11 23H2+ ships without `wmic.exe`. When that's the case we fall
    // back to PowerShell `Get-CimInstance` queries that return the same fields.
    HwidSnapshot {
        label,
        captured_at: chrono::Utc::now().to_rfc3339(),
        admin_elevated: is_admin_elevated(),
        machine_guid: read_registry_string(
            "SOFTWARE\\Microsoft\\Cryptography",
            "MachineGuid",
        ),
        computer_hardware_id: read_registry_string(
            "SYSTEM\\CurrentControlSet\\Control\\SystemInformation",
            "ComputerHardwareId",
        ),
        product_id: read_registry_string(
            "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion",
            "ProductId",
        ),
        hw_profile_guid: read_registry_string(
            "SYSTEM\\CurrentControlSet\\Control\\IDConfigDB\\Hardware Profiles\\0001",
            "HwProfileGuid",
        ),
        disk_registry_serial: read_registry_string(
            "SYSTEM\\CurrentControlSet\\Services\\Disk\\Enum\\0",
            "SerialNumber",
        ),
        disk_wmic_serial: probe_with_fallback(
            "diskdrive get serialnumber",
            "Get-CimInstance Win32_DiskDrive | Select-Object -First 1 -ExpandProperty SerialNumber",
        ),
        volume_c_serial: volume_serial_c(),
        baseboard_serial: probe_with_fallback(
            "baseboard get serialnumber",
            "Get-CimInstance Win32_BaseBoard | Select-Object -First 1 -ExpandProperty SerialNumber",
        ),
        bios_serial: probe_with_fallback(
            "bios get serialnumber",
            "Get-CimInstance Win32_BIOS | Select-Object -First 1 -ExpandProperty SerialNumber",
        ),
        bios_version: probe_with_fallback(
            "bios get smbiosbiosversion",
            "Get-CimInstance Win32_BIOS | Select-Object -First 1 -ExpandProperty SMBIOSBIOSVersion",
        ),
        system_uuid_wmic: probe_with_fallback(
            "csproduct get uuid",
            "Get-CimInstance Win32_ComputerSystemProduct | Select-Object -First 1 -ExpandProperty UUID",
        ),
        hostname: std::env::var("COMPUTERNAME").unwrap_or_else(|_| "(read failed)".into()),
        adapters: capture_adapters(),
    }
}

/// Try `wmic` first; if it is missing (Win11 23H2+ removed it) or returned
/// nothing usable, fall back to a PowerShell `Get-CimInstance` query.
#[cfg(windows)]
fn probe_with_fallback(wmic_args: &str, powershell_expr: &str) -> String {
    let v = wmic_first_value(wmic_args);
    if !v.starts_with("(read failed)")
        && !v.starts_with("(error")
        && !v.eq_ignore_ascii_case("unknown")
        && !v.trim().is_empty()
    {
        return v;
    }
    powershell_first_value(powershell_expr)
}

#[cfg(windows)]
fn powershell_first_value(expression: &str) -> String {
    use std::os::windows::process::CommandExt;
    use std::process::Command;
    const CREATE_NO_WINDOW: u32 = 0x08000000;

    // Prefer `pwsh` (PowerShell 7+) when present, otherwise fall back to
    // legacy Windows PowerShell.  Both ship on a default Win11 install.
    for shell in ["powershell", "pwsh"] {
        let output = Command::new(shell)
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                expression,
            ])
            .creation_flags(CREATE_NO_WINDOW)
            .output();
        match output {
            Ok(o) if o.status.success() => {
                let text = String::from_utf8_lossy(&o.stdout);
                for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
                    return line.to_string();
                }
            }
            Ok(_) | Err(_) => continue,
        }
    }
    "(read failed)".into()
}

#[cfg(windows)]
pub fn is_admin_elevated() -> bool {
    if token_is_elevated() {
        return true;
    }
    whoami_has_high_integrity()
}

#[cfg(windows)]
fn token_is_elevated() -> bool {
    use std::ptr::null_mut;
    unsafe {
        let mut token = null_mut();
        if winapi::um::processthreadsapi::OpenProcessToken(
            winapi::um::processthreadsapi::GetCurrentProcess(),
            winapi::um::winnt::TOKEN_QUERY,
            &mut token,
        ) == 0
        {
            return false;
        }
        let mut elevation: winapi::um::winnt::TOKEN_ELEVATION = std::mem::zeroed();
        let mut ret_len = 0u32;
        let ok = winapi::um::securitybaseapi::GetTokenInformation(
            token,
            winapi::um::winnt::TokenElevation,
            &mut elevation as *mut _ as *mut _,
            std::mem::size_of::<winapi::um::winnt::TOKEN_ELEVATION>() as u32,
            &mut ret_len,
        );
        winapi::um::handleapi::CloseHandle(token);
        ok != 0 && elevation.TokenIsElevated != 0
    }
}

#[cfg(windows)]
fn whoami_has_high_integrity() -> bool {
    use std::os::windows::process::CommandExt;
    use std::process::Command;
    const CREATE_NO_WINDOW: u32 = 0x08000000;
    let Ok(out) = Command::new("whoami")
        .args(["/groups", "/fo", "list"])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
    else {
        return false;
    };
    let text = String::from_utf8_lossy(&out.stdout).to_uppercase();
    text.contains("S-1-16-12288")
        || text.contains("HIGH MANDATORY LEVEL")
        || text.contains("LABEL_HIGH")
}

/// Human-readable admin status for terminal (not log-file only).
pub fn print_admin_status() {
    #[cfg(windows)]
    {
        if is_admin_elevated() {
            println!("[hwspoof] Administrator: YES - elevation OK");
        } else {
            println!("[hwspoof] Administrator: NO - this window is not elevated");
            println!("[hwspoof] Fix: double-click Run_As_Admin.bat in HWIDTool folder");
            println!("[hwspoof]      OR right-click run_hwid.bat -> Run as administrator");
            println!("[hwspoof]      Admin PowerShell + .\\run_hwid.bat often is NOT elevated.");
        }
    }
    #[cfg(not(windows))]
    println!("[hwspoof] Administrator check: Windows only");
}

pub fn print_verification_to_terminal(
    before: &HwidSnapshot,
    after: &HwidSnapshot,
    report: &VerificationReport,
) {
    println!();
    print!("{}", format_verification_report(before, after, report));
}

pub fn print_terminal_result_box(report: &VerificationReport) {
    println!();
    println!("============================================================");
    match report.verdict {
        OverallVerdict::SuccessChanged => {
            println!("  RESULT: SUCCESS - spoof changed {} identifier(s)", report.changed_count);
        }
        OverallVerdict::FailedNoChange => {
            println!("  RESULT: FAILED - no identifiers changed");
        }
        OverallVerdict::Inconclusive => {
            println!("  RESULT: INCONCLUSIVE - check lines below");
        }
        OverallVerdict::ReportOnly => {}
    }
    println!(
        "  changed={}  unchanged={}  read_fail={}",
        report.changed_count, report.unchanged_count, report.failed_read_count
    );
    if report.reboot_recommended {
        println!("  NOTE: restart Windows then run: hwspoof.exe --verify-last");
    }
    if report.acceptable_for_marker() {
        println!("  STEP: OK - you can run ABA next");
    } else {
        println!("  STEP: BLOCKED - fix HWID before ABA");
    }
    println!("============================================================");
    println!();
}

#[cfg(not(windows))]
pub fn is_admin_elevated() -> bool {
    false
}

#[cfg(windows)]
fn read_registry_string(path: &str, value: &str) -> String {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;
    use std::ptr::null_mut;
    use winapi::shared::minwindef::DWORD;
    use winapi::um::winnt::KEY_READ;
    use winapi::um::winreg::{RegCloseKey, RegOpenKeyExW, RegQueryValueExW, HKEY_LOCAL_MACHINE};

    unsafe {
        let path_wide: Vec<u16> = OsStr::new(path).encode_wide().chain(Some(0)).collect();
        let value_wide: Vec<u16> = OsStr::new(value).encode_wide().chain(Some(0)).collect();
        let mut hkey = null_mut();
        if RegOpenKeyExW(
            HKEY_LOCAL_MACHINE,
            path_wide.as_ptr(),
            0,
            KEY_READ,
            &mut hkey,
        ) != 0
        {
            return "(read failed)".into();
        }
        let mut buffer = vec![0u16; 512];
        let mut size: DWORD = (buffer.len() * 2) as DWORD;
        let rc = RegQueryValueExW(
            hkey,
            value_wide.as_ptr(),
            null_mut(),
            null_mut(),
            buffer.as_mut_ptr() as *mut u8,
            &mut size,
        );
        RegCloseKey(hkey);
        if rc != 0 {
            return "(read failed)".into();
        }
        String::from_utf16_lossy(&buffer)
            .trim_end_matches('\0')
            .trim()
            .to_string()
    }
}

#[cfg(windows)]
fn wmic_first_value(args: &str) -> String {
    use std::os::windows::process::CommandExt;
    use std::process::Command;
    const CREATE_NO_WINDOW: u32 = 0x08000000;

    let mut parts = args.split_whitespace();
    let alias = parts.next().unwrap_or("");
    let field = parts.next().unwrap_or("");

    let output = Command::new("wmic")
        .args([alias, "get", field])
        .creation_flags(CREATE_NO_WINDOW)
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout);
            for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
                if line.eq_ignore_ascii_case(field) {
                    continue;
                }
                return line.to_string();
            }
            "(read failed)".into()
        }
        Ok(_) => "(read failed)".into(),
        // io::ErrorKind::NotFound = wmic.exe is not installed (Win11 23H2+).
        // Return a sentinel that probe_with_fallback() can detect, instead of
        // the cryptic "(error: program not found)" the user was seeing.
        Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => "(read failed)".into(),
        Err(e) => format!("(error: {e})"),
    }
}

#[cfg(windows)]
fn volume_serial_c() -> String {
    use std::os::windows::process::CommandExt;
    use std::process::Command;
    const CREATE_NO_WINDOW: u32 = 0x08000000;

    if let Ok(o) = Command::new("cmd")
        .args(["/C", "vol C:"])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
    {
        let text = String::from_utf8_lossy(&o.stdout);
        for line in text.lines() {
            if line.contains("Serial Number") || line.contains("Seriennummer") {
                return line.trim().to_string();
            }
        }
    }
    "(read failed)".into()
}

#[cfg(windows)]
fn capture_adapters() -> Vec<(String, String)> {
    use std::os::windows::process::CommandExt;
    use std::process::Command;
    const CREATE_NO_WINDOW: u32 = 0x08000000;

    let mut out = Vec::new();
    let Ok(cmd_out) = Command::new("getmac")
        .args(["/fo", "csv", "/nh", "/v"])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
    else {
        return out;
    };
    let text = String::from_utf8_lossy(&cmd_out.stdout);
    for line in text.lines() {
        let cols: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        if cols.len() >= 3 {
            let mac = cols[2].to_string();
            let name = cols.last().unwrap_or(&"adapter").to_string();
            if !mac.is_empty() && !mac.eq_ignore_ascii_case("n/a") {
                out.push((name, mac));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_changed_pair() {
        let (s, _) = classify_pair("aaa", "bbb", true);
        assert_eq!(s, FieldStatus::Changed);
    }

    #[test]
    fn classify_unchanged_needs_reboot_note() {
        let (s, note) = classify_pair("same", "same", true);
        assert_eq!(s, FieldStatus::Unchanged);
        assert!(note.contains("reboot"));
    }

    #[test]
    fn unchanged_explanations_cover_documented_fields() {
        assert!(unchanged_field_explanation("Volume C: serial").is_some());
        assert!(unchanged_field_explanation("Hostname").is_some());
        assert!(unchanged_field_explanation("Network adapters (MAC)").is_some());
        assert!(unchanged_field_explanation("MachineGuid").is_none());
    }

    #[test]
    fn push_check_appends_unchanged_explanation_for_volume_serial() {
        let mut checks: Vec<FieldCheck> = Vec::new();
        push_check(
            &mut checks,
            "Volume C: serial",
            "Volume Serial Number is 406F-727A",
            "Volume Serial Number is 406F-727A",
            true,
            true,
        );
        let c = checks.first().expect("push_check should produce one entry");
        assert_eq!(c.status, FieldStatus::Unchanged);
        assert!(
            c.note.contains("boot sector"),
            "note should include the unchanged explanation: {}",
            c.note
        );
    }

    #[test]
    fn push_check_does_not_append_explanation_for_changed_fields() {
        let mut checks: Vec<FieldCheck> = Vec::new();
        push_check(
            &mut checks,
            "Volume C: serial",
            "Volume Serial Number is 406F-727A",
            "Volume Serial Number is DEAD-BEEF",
            true,
            true,
        );
        let c = checks.first().unwrap();
        assert_eq!(c.status, FieldStatus::Changed);
        assert!(!c.note.contains("boot sector"));
    }

    #[test]
    fn acceptable_requires_change_by_default() {
        let report = VerificationReport {
            checks: vec![],
            verdict: OverallVerdict::FailedNoChange,
            changed_count: 0,
            unchanged_count: 3,
            failed_read_count: 0,
            reboot_recommended: true,
            module_errors: vec![],
        };
        assert!(!report.acceptable_for_marker());
    }
}
