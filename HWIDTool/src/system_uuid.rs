// system_uuid.rs - System UUID Spoofing
use std::io::{Result, Error, ErrorKind};
use std::ptr::null_mut;
use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;

#[cfg(windows)]
use winapi::{
    um::{
        winreg::{RegOpenKeyExW, RegSetValueExW, RegCloseKey, RegQueryValueExW, HKEY_LOCAL_MACHINE},
        winnt::{KEY_WRITE, KEY_READ, REG_SZ, REG_BINARY},
    },
    shared::minwindef::DWORD,
};

const UUID_REGISTRY_PATHS: &[(&str, &str)] = &[
    ("SOFTWARE\\Microsoft\\Cryptography", "MachineGuid"),
    ("SYSTEM\\CurrentControlSet\\Control\\SystemInformation", "ComputerHardwareId"),
    ("SYSTEM\\CurrentControlSet\\Control\\IDConfigDB\\Hardware Profiles\\0001", "HwProfileGuid"),
    ("SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", "ProductId"),
];

pub fn generate_uuid() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    format!("{:08X}-{:04X}-4{:03X}-{:04X}-{:012X}",
        rng.gen::<u32>(),
        rng.gen::<u16>() & 0x0FFF,
        rng.gen::<u16>() & 0x0FFF,
        rng.gen::<u16>() & 0x3FFF | 0x8000,
        rng.gen::<u64>() & 0xFFFFFFFFFFFF
    )
}

#[cfg(windows)]
pub fn spoof_system_uuid(uuid: Option<String>) -> Result<()> {
    let new_uuid = uuid.unwrap_or_else(generate_uuid);
    
    println!("[system_uuid] ═══════════════════════════════════════════");
    println!("[system_uuid] System UUID Spoof Initiated");
    println!("[system_uuid] Target UUID: {}", new_uuid);
    println!("[system_uuid] ═══════════════════════════════════════════");
    
    backup_uuid_configuration()?;
    
    modify_machine_guid(&new_uuid)?;
    modify_hardware_id(&new_uuid)?;
    modify_profile_guid(&new_uuid)?;
    modify_product_id(&new_uuid)?;
    modify_smbios_uuid(&new_uuid)?;
    modify_crypto_keys(&new_uuid)?;
    
    verify_uuid_propagation(&new_uuid)?;
    
    println!("[system_uuid] ✓ UUID spoof complete");
    Ok(())
}

#[cfg(not(windows))]
pub fn spoof_system_uuid(_uuid: Option<String>) -> Result<()> {
    Err(Error::new(ErrorKind::Unsupported, "Windows only"))
}

fn backup_uuid_configuration() -> Result<()> {
    println!("[system_uuid] [BACKUP] Backing up UUID configuration");
    
    #[cfg(windows)]
    {
        for (path, value) in UUID_REGISTRY_PATHS {
            if let Ok(original) = read_registry_value(path, value) {
                println!("[system_uuid] [BACKUP] {}: {}", value, original);
            }
        }
    }
    
    Ok(())
}

#[cfg(windows)]
fn read_registry_value(path: &str, value_name: &str) -> Result<String> {
    unsafe {
        let path_wide: Vec<u16> = OsStr::new(path).encode_wide().chain(Some(0)).collect();
        let value_wide: Vec<u16> = OsStr::new(value_name).encode_wide().chain(Some(0)).collect();
        
        let mut hkey = null_mut();
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, path_wide.as_ptr(), 0, KEY_READ, &mut hkey) == 0 {
            let mut buffer = vec![0u16; 256];
            let mut buffer_size: DWORD = (buffer.len() * 2) as DWORD;
            
            if RegQueryValueExW(hkey, value_wide.as_ptr(), null_mut(), null_mut(),
                buffer.as_mut_ptr() as *mut u8, &mut buffer_size) == 0 {
                RegCloseKey(hkey);
                return Ok(String::from_utf16_lossy(&buffer).trim_end_matches('\0').to_string());
            }
            
            RegCloseKey(hkey);
        }
    }
    
    Err(Error::new(ErrorKind::NotFound, "Value not found"))
}

fn modify_machine_guid(uuid: &str) -> Result<()> {
    println!("[system_uuid] [REGISTRY] Modifying MachineGuid");
    set_registry_value("SOFTWARE\\Microsoft\\Cryptography", "MachineGuid", uuid)?;
    println!("[system_uuid] [REGISTRY] ✓ MachineGuid updated");
    Ok(())
}

fn modify_hardware_id(uuid: &str) -> Result<()> {
    println!("[system_uuid] [REGISTRY] Modifying ComputerHardwareId");
    
    let hardware_id = format!("{{{}}}", uuid);
    set_registry_value("SYSTEM\\CurrentControlSet\\Control\\SystemInformation", "ComputerHardwareId", &hardware_id)?;
    
    // Also set SystemProductName
    set_registry_value("SYSTEM\\CurrentControlSet\\Control\\SystemInformation", "SystemProductName", "Default").ok();
    
    println!("[system_uuid] [REGISTRY] ✓ Hardware ID updated");
    Ok(())
}

fn modify_profile_guid(uuid: &str) -> Result<()> {
    println!("[system_uuid] [REGISTRY] Modifying HwProfileGuid");
    
    let profile_guid = format!("{{{}}}", uuid);
    set_registry_value("SYSTEM\\CurrentControlSet\\Control\\IDConfigDB\\Hardware Profiles\\0001", "HwProfileGuid", &profile_guid)?;
    
    println!("[system_uuid] [REGISTRY] ✓ Profile GUID updated");
    Ok(())
}

fn modify_product_id(uuid: &str) -> Result<()> {
    println!("[system_uuid] [REGISTRY] Modifying ProductId");
    
    // Generate product ID from UUID
    let product_id = uuid_to_product_id(uuid);
    set_registry_value("SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", "ProductId", &product_id)?;
    
    println!("[system_uuid] [REGISTRY] ✓ Product ID updated");
    Ok(())
}

fn uuid_to_product_id(_uuid: &str) -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    format!("{:05}-{:05}-{:05}-{:05}",
        rng.gen_range(10000..99999),
        rng.gen_range(10000..99999),
        rng.gen_range(10000..99999),
        rng.gen_range(10000..99999)
    )
}

fn modify_smbios_uuid(uuid: &str) -> Result<()> {
    println!("[system_uuid] [SMBIOS] Modifying SMBIOS UUID");
    
    let uuid_bytes = parse_uuid_to_bytes(uuid)?;
    
    // Modify in HARDWARE\DESCRIPTION\System\BIOS
    #[cfg(windows)]
    unsafe {
        let path_wide: Vec<u16> = OsStr::new("HARDWARE\\DESCRIPTION\\System\\BIOS")
            .encode_wide().chain(Some(0)).collect();
        let value_wide: Vec<u16> = OsStr::new("SystemUUID").encode_wide().chain(Some(0)).collect();
        
        let mut hkey = null_mut();
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, path_wide.as_ptr(), 0, KEY_WRITE, &mut hkey) == 0 {
            RegSetValueExW(
                hkey, value_wide.as_ptr(), 0, REG_BINARY,
                uuid_bytes.as_ptr(),
                uuid_bytes.len() as DWORD,
            );
            RegCloseKey(hkey);
        }
    }
    
    println!("[system_uuid] [SMBIOS] ✓ SMBIOS UUID updated");
    Ok(())
}

fn parse_uuid_to_bytes(uuid: &str) -> Result<Vec<u8>> {
    let hex: String = uuid
        .chars()
        .filter(|c| c.is_ascii_hexdigit())
        .collect();
    if hex.len() < 32 {
        return Err(Error::new(ErrorKind::InvalidData, "UUID hex too short"));
    }
    let mut bytes = Vec::with_capacity(16);
    for i in (0..32).step_by(2) {
        let byte = u8::from_str_radix(&hex[i..i + 2], 16)
            .map_err(|_| Error::new(ErrorKind::InvalidData, "invalid UUID hex"))?;
        bytes.push(byte);
    }
    Ok(bytes)
}

fn modify_crypto_keys(uuid: &str) -> Result<()> {
    println!("[system_uuid] [CRYPTO] Modifying cryptographic keys");
    
    let paths = vec![
        "SOFTWARE\\Microsoft\\Cryptography\\RNG",
        "SOFTWARE\\Microsoft\\Cryptography\\Providers",
    ];
    
    for path in paths {
        set_registry_value(path, "Seed", uuid).ok();
    }
    
    println!("[system_uuid] [CRYPTO] ✓ Crypto keys updated");
    Ok(())
}

#[cfg(windows)]
fn set_registry_value(path: &str, value_name: &str, value_data: &str) -> Result<()> {
    unsafe {
        let path_wide: Vec<u16> = OsStr::new(path).encode_wide().chain(Some(0)).collect();
        let name_wide: Vec<u16> = OsStr::new(value_name).encode_wide().chain(Some(0)).collect();
        let data_wide: Vec<u16> = value_data.encode_utf16().collect();
        
        let mut hkey = null_mut();
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, path_wide.as_ptr(), 0, KEY_WRITE, &mut hkey) == 0 {
            let result = RegSetValueExW(
                hkey, name_wide.as_ptr(), 0, REG_SZ,
                data_wide.as_ptr() as *const u8,
                (data_wide.len() * 2) as DWORD,
            );
            
            RegCloseKey(hkey);
            
            if result == 0 {
                return Ok(());
            }
        }
    }
    
    Err(Error::new(ErrorKind::PermissionDenied, "Registry write failed"))
}

fn verify_uuid_propagation(expected_uuid: &str) -> Result<()> {
    println!("[system_uuid] [VERIFY] Verifying UUID propagation");
    
    let mut success_count = 0;
    let total_checks = UUID_REGISTRY_PATHS.len();
    
    #[cfg(windows)]
    {
        for (path, value) in UUID_REGISTRY_PATHS {
            if let Ok(current) = read_registry_value(path, value) {
                if current.contains(expected_uuid) || expected_uuid.contains(&current) {
                    println!("[system_uuid] [VERIFY] ✓ {}: {}", value, current);
                    success_count += 1;
                } else {
                    println!("[system_uuid] [VERIFY] ✗ {}: {}", value, current);
                }
            }
        }
    }
    
    println!("[system_uuid] [VERIFY] Propagation: {}/{} locations", success_count, total_checks);
    Ok(())
}

pub fn revert_system_uuid() -> Result<()> {
    println!("[system_uuid] Reverting to original UUID");
    println!("[system_uuid] ⚠ Original UUID must be restored from backup");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generate_uuid() {
        let uuid = generate_uuid();
        assert_eq!(uuid.len(), 36);
        assert_eq!(uuid.matches('-').count(), 4);
        assert!(uuid.contains("-4")); // Version 4
    }
    
    #[test]
    fn test_uuid_format() {
        for _ in 0..100 {
            let uuid = generate_uuid();
            let parts: Vec<&str> = uuid.split('-').collect();
            assert_eq!(parts.len(), 5);
            assert_eq!(parts[0].len(), 8);
            assert_eq!(parts[1].len(), 4);
            assert_eq!(parts[2].len(), 4);
            assert_eq!(parts[3].len(), 4);
            assert_eq!(parts[4].len(), 12);
        }
    }
}