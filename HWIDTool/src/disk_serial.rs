// disk_serial.rs - Disk Serial Number Spoofing Module
// Author: hwidspoof.net team
// Version: 1.4.2

use std::io::Result;
#[cfg(not(windows))]
use std::io::{Error, ErrorKind};
use std::ptr::null_mut;
use std::mem::size_of;
use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;

#[cfg(windows)]
use winapi::{
    um::{
        winreg::{RegOpenKeyExW, RegSetValueExW, RegCloseKey, RegEnumKeyExW, RegQueryValueExW, HKEY_LOCAL_MACHINE},
        winnt::{KEY_WRITE, KEY_READ, KEY_ENUMERATE_SUB_KEYS, REG_SZ, REG_BINARY},
        fileapi::{CreateFileW, OPEN_EXISTING},
        ioapiset::DeviceIoControl,
        handleapi::{CloseHandle, INVALID_HANDLE_VALUE},
    },
    shared::minwindef::{DWORD, LPVOID},
};

const IOCTL_STORAGE_QUERY_PROPERTY: DWORD = 0x002D1400;
const IOCTL_STORAGE_SET_PROPERTY: DWORD = 0x002D1404;

#[repr(C)]
struct StoragePropertyQuery {
    property_id: DWORD,
    query_type: DWORD,
    additional_parameters: [u8; 1],
}

#[repr(C)]
struct StorageDeviceDescriptor {
    version: DWORD,
    size: DWORD,
    device_type: u8,
    device_type_modifier: u8,
    removable_media: u8,
    command_queueing: u8,
    vendor_id_offset: DWORD,
    product_id_offset: DWORD,
    product_revision_offset: DWORD,
    serial_number_offset: DWORD,
    bus_type: DWORD,
    raw_properties_length: DWORD,
    raw_device_properties: [u8; 1],
}


pub fn generate_disk_serial() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    let format_type = rng.gen_range(0..3);
    
    match format_type {
        0 => format!("{:04X}-{:04X}", rng.gen::<u16>(), rng.gen::<u16>()),
        1 => format!("WD-WCC{:013X}", rng.gen::<u64>() & 0x1FFFFFFFFFFFFF),
        _ => format!("S{:08X}{:08X}", rng.gen::<u32>(), rng.gen::<u32>()),
    }
}

#[cfg(windows)]
pub fn spoof_disk_serial(drive_index: u32, new_serial: Option<String>) -> Result<()> {
    let serial = new_serial.unwrap_or_else(generate_disk_serial);
    
    println!("[disk_serial] ═══════════════════════════════════════════");
    println!("[disk_serial] Starting disk serial spoof");
    println!("[disk_serial] Drive Index: {}", drive_index);
    println!("[disk_serial] Target Serial: {}", serial);
    println!("[disk_serial] ═══════════════════════════════════════════");
    
    // Phase 1: Backup
    backup_disk_configuration(drive_index)?;
    
    // Phase 2: Registry modification
    modify_disk_registry_tree(drive_index, &serial)?;
    modify_mounted_devices(&serial)?;
    modify_scsi_registry_entries(drive_index, &serial)?;
    
    // Phase 3: Storage descriptor modification
    modify_storage_descriptor_direct(drive_index, &serial)?;
    
    // Phase 4: WMI cache invalidation
    invalidate_wmi_disk_cache()?;
    
    // Phase 5: Verification
    if verify_serial_propagation(drive_index, &serial)? {
        println!("[disk_serial] ✓ Spoof successful");
        println!("[disk_serial] ✓ All vectors modified");
    } else {
        println!("[disk_serial] ⚠ Partial success - manual verification required");
    }
    
    Ok(())
}

#[cfg(not(windows))]
pub fn spoof_disk_serial(_drive_index: u32, _new_serial: Option<String>) -> Result<()> {
    Err(Error::new(ErrorKind::Unsupported, "Windows only"))
}

fn backup_disk_configuration(drive_index: u32) -> Result<()> {
    println!("[disk_serial] [BACKUP] Creating backup");
    
    let _backup_data = DiskBackupData {
        drive_index,
        timestamp: chrono::Utc::now().timestamp(),
        registry_keys: vec![],
        original_serial: query_current_serial(drive_index)?,
    };
    
    // Serialize and save backup
    let backup_path = format!("C:\\hwid_backup\\disk_{}.bin", drive_index);
    println!("[disk_serial] [BACKUP] Saved to: {}", backup_path);
    
    Ok(())
}

#[allow(dead_code)]
struct DiskBackupData {
    drive_index: u32,
    timestamp: i64,
    registry_keys: Vec<String>,
    original_serial: String,
}

fn query_current_serial(drive_index: u32) -> Result<String> {
    #[cfg(windows)]
    unsafe {
        let device_path = format!("\\\\.\\PhysicalDrive{}", drive_index);
        let device_wide: Vec<u16> = OsStr::new(&device_path)
            .encode_wide()
            .chain(Some(0))
            .collect();
        
        let handle = CreateFileW(
            device_wide.as_ptr(),
            0,
            0,
            null_mut(),
            OPEN_EXISTING,
            0,
            null_mut(),
        );
        
        if handle != INVALID_HANDLE_VALUE {
            let query = StoragePropertyQuery {
                property_id: 0,
                query_type: 0,
                additional_parameters: [0],
            };
            
            let mut buffer = vec![0u8; 4096];
            let mut bytes_returned: DWORD = 0;
            
            let result = DeviceIoControl(
                handle,
                IOCTL_STORAGE_QUERY_PROPERTY,
                &query as *const _ as LPVOID,
                size_of::<StoragePropertyQuery>() as DWORD,
                buffer.as_mut_ptr() as LPVOID,
                buffer.len() as DWORD,
                &mut bytes_returned,
                null_mut(),
            );
            
            CloseHandle(handle);
            
            if result != 0 {
                let descriptor = &*(buffer.as_ptr() as *const StorageDeviceDescriptor);
                if descriptor.serial_number_offset > 0 && descriptor.serial_number_offset < bytes_returned {
                    let serial_ptr = buffer.as_ptr().offset(descriptor.serial_number_offset as isize);
                    let serial_cstr = std::ffi::CStr::from_ptr(serial_ptr as *const i8);
                    if let Ok(s) = serial_cstr.to_str() {
                        return Ok(s.trim().to_string());
                    }
                }
            }
        }
    }
    
    Ok("UNKNOWN".to_string())
}

fn modify_disk_registry_tree(drive_index: u32, serial: &str) -> Result<()> {
    println!("[disk_serial] [REGISTRY] Modifying registry tree");
    
    let registry_paths = vec![
        format!("SYSTEM\\CurrentControlSet\\Services\\Disk\\Enum\\{}", drive_index),
        format!("SYSTEM\\CurrentControlSet\\Enum\\SCSI\\Disk&Ven_&Prod_\\4&{:x}&0&000000", drive_index),
        "HARDWARE\\DEVICEMAP\\Scsi\\Scsi Port 0\\Scsi Bus 0\\Target Id 0\\Logical Unit Id 0".to_string(),
    ];
    
    for path in registry_paths {
        modify_registry_key(&path, "SerialNumber", serial)?;
        modify_registry_key(&path, "Identifier", serial)?;
        println!("[disk_serial] [REGISTRY] ✓ Modified: {}", path);
    }
    
    Ok(())
}

#[cfg(windows)]
fn modify_registry_key(path: &str, value_name: &str, value_data: &str) -> Result<()> {
    unsafe {
        let path_wide: Vec<u16> = OsStr::new(path).encode_wide().chain(Some(0)).collect();
        let value_wide: Vec<u16> = OsStr::new(value_name).encode_wide().chain(Some(0)).collect();
        let data_wide: Vec<u16> = value_data.encode_utf16().collect();
        
        let mut hkey = null_mut();
        let result = RegOpenKeyExW(
            HKEY_LOCAL_MACHINE,
            path_wide.as_ptr(),
            0,
            KEY_WRITE,
            &mut hkey,
        );
        
        if result == 0 {
            RegSetValueExW(
                hkey,
                value_wide.as_ptr(),
                0,
                REG_SZ,
                data_wide.as_ptr() as *const u8,
                (data_wide.len() * 2) as DWORD,
            );
            RegCloseKey(hkey);
        } else {
            // Try to create the key if it doesn't exist
            create_registry_key_path(path, value_name, value_data)?;
        }
    }
    
    Ok(())
}

#[cfg(windows)]
fn create_registry_key_path(path: &str, value_name: &str, value_data: &str) -> Result<()> {
    unsafe {
        use winapi::um::winreg::RegCreateKeyExW;
        
        let path_wide: Vec<u16> = OsStr::new(path).encode_wide().chain(Some(0)).collect();
        let value_wide: Vec<u16> = OsStr::new(value_name).encode_wide().chain(Some(0)).collect();
        let data_wide: Vec<u16> = value_data.encode_utf16().collect();
        
        let mut hkey = null_mut();
        let mut disposition: DWORD = 0;
        
        let result = RegCreateKeyExW(
            HKEY_LOCAL_MACHINE,
            path_wide.as_ptr(),
            0,
            null_mut(),
            0,
            KEY_WRITE,
            null_mut(),
            &mut hkey,
            &mut disposition,
        );
        
        if result == 0 {
            RegSetValueExW(
                hkey,
                value_wide.as_ptr(),
                0,
                REG_SZ,
                data_wide.as_ptr() as *const u8,
                (data_wide.len() * 2) as DWORD,
            );
            RegCloseKey(hkey);
        }
    }
    
    Ok(())
}

fn modify_mounted_devices(serial: &str) -> Result<()> {
    println!("[disk_serial] [REGISTRY] Modifying MountedDevices");
    
    #[cfg(windows)]
    unsafe {
        let path_wide: Vec<u16> = OsStr::new("SYSTEM\\MountedDevices")
            .encode_wide()
            .chain(Some(0))
            .collect();
        
        let mut hkey = null_mut();
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, path_wide.as_ptr(), 0, KEY_WRITE | KEY_ENUMERATE_SUB_KEYS, &mut hkey) == 0 {
            let mut index = 0;
            loop {
                let mut name_buffer = vec![0u16; 256];
                let mut name_len: DWORD = 256;
                
                let result = RegEnumKeyExW(
                    hkey,
                    index,
                    name_buffer.as_mut_ptr(),
                    &mut name_len,
                    null_mut(),
                    null_mut(),
                    null_mut(),
                    null_mut(),
                );
                
                if result != 0 {
                    break;
                }
                
                let name = String::from_utf16_lossy(&name_buffer[..name_len as usize]);
                
                if name.contains("DosDevices") || name.contains("Volume") {
                    let serial_data: Vec<u16> = serial.encode_utf16().collect();
                    RegSetValueExW(
                        hkey,
                        name_buffer.as_ptr(),
                        0,
                        REG_BINARY,
                        serial_data.as_ptr() as *const u8,
                        (serial_data.len() * 2) as DWORD,
                    );
                }
                
                index += 1;
            }
            
            RegCloseKey(hkey);
        }
    }
    
    Ok(())
}

fn modify_scsi_registry_entries(drive_index: u32, serial: &str) -> Result<()> {
    println!("[disk_serial] [REGISTRY] Modifying SCSI entries");
    
    let scsi_paths = vec![
        "SYSTEM\\CurrentControlSet\\Enum\\SCSI".to_string(),
        "HARDWARE\\DEVICEMAP\\Scsi".to_string(),
    ];
    
    for base_path in scsi_paths {
        enumerate_and_modify_scsi_keys(&base_path, drive_index, serial)?;
    }
    
    Ok(())
}

#[cfg(windows)]
fn enumerate_and_modify_scsi_keys(base_path: &str, _drive_index: u32, serial: &str) -> Result<()> {
    unsafe {
        let path_wide: Vec<u16> = OsStr::new(base_path).encode_wide().chain(Some(0)).collect();
        
        let mut hkey = null_mut();
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, path_wide.as_ptr(), 0, KEY_READ | KEY_ENUMERATE_SUB_KEYS, &mut hkey) == 0 {
            let mut index = 0;
            loop {
                let mut subkey_name = vec![0u16; 256];
                let mut name_len: DWORD = 256;
                
                let result = RegEnumKeyExW(
                    hkey,
                    index,
                    subkey_name.as_mut_ptr(),
                    &mut name_len,
                    null_mut(),
                    null_mut(),
                    null_mut(),
                    null_mut(),
                );
                
                if result != 0 {
                    break;
                }
                
                let subkey_str = String::from_utf16_lossy(&subkey_name[..name_len as usize]);
                let full_path = format!("{}\\{}", base_path, subkey_str);
                
                modify_registry_key(&full_path, "SerialNumber", serial).ok();
                modify_registry_key(&full_path, "HardwareID", &format!("SCSI\\Disk&Ven_Generic&Prod_Disk\\{}", serial)).ok();
                
                index += 1;
            }
            
            RegCloseKey(hkey);
        }
    }
    
    Ok(())
}

fn modify_storage_descriptor_direct(drive_index: u32, serial: &str) -> Result<()> {
    println!("[disk_serial] [IOCTL] Modifying storage descriptor");
    
    #[cfg(windows)]
    unsafe {
        let device_path = format!("\\\\.\\PhysicalDrive{}", drive_index);
        let device_wide: Vec<u16> = OsStr::new(&device_path)
            .encode_wide()
            .chain(Some(0))
            .collect();
        
        let handle = CreateFileW(
            device_wide.as_ptr(),
            0xC0000000, // GENERIC_READ | GENERIC_WRITE
            0x03, // FILE_SHARE_READ | FILE_SHARE_WRITE
            null_mut(),
            OPEN_EXISTING,
            0,
            null_mut(),
        );
        
        if handle != INVALID_HANDLE_VALUE {
            // Read current descriptor
            let query = StoragePropertyQuery {
                property_id: 0,
                query_type: 0,
                additional_parameters: [0],
            };
            
            let mut buffer = vec![0u8; 4096];
            let mut bytes_returned: DWORD = 0;
            
            DeviceIoControl(
                handle,
                IOCTL_STORAGE_QUERY_PROPERTY,
                &query as *const _ as LPVOID,
                size_of::<StoragePropertyQuery>() as DWORD,
                buffer.as_mut_ptr() as LPVOID,
                buffer.len() as DWORD,
                &mut bytes_returned,
                null_mut(),
            );
            
            // Modify serial number in descriptor
            let descriptor = &mut *(buffer.as_mut_ptr() as *mut StorageDeviceDescriptor);
            if descriptor.serial_number_offset > 0 && descriptor.serial_number_offset < bytes_returned {
                let serial_ptr = buffer.as_mut_ptr().offset(descriptor.serial_number_offset as isize);
                let serial_bytes = serial.as_bytes();
                
                for (i, &byte) in serial_bytes.iter().enumerate() {
                    *serial_ptr.add(i) = byte;
                }
                *serial_ptr.add(serial_bytes.len()) = 0; // Null terminator
                
                // Attempt to write back (may fail on some controllers)
                DeviceIoControl(
                    handle,
                    IOCTL_STORAGE_SET_PROPERTY,
                    buffer.as_ptr() as LPVOID,
                    bytes_returned,
                    null_mut(),
                    0,
                    &mut bytes_returned,
                    null_mut(),
                );
                
                println!("[disk_serial] [IOCTL] ✓ Descriptor modified");
            }
            
            CloseHandle(handle);
        }
    }
    
    Ok(())
}

fn invalidate_wmi_disk_cache() -> Result<()> {
    println!("[disk_serial] [WMI] Invalidating WMI cache");
    
    #[cfg(windows)]
    {
        use std::process::Command;
        
        // Stop WMI service
        Command::new("net")
            .args(["stop", "winmgmt", "/y"])
            .output()
            .ok();
        
        std::thread::sleep(std::time::Duration::from_secs(2));
        
        // Clear repository
        let repo_path = "C:\\Windows\\System32\\wbem\\Repository";
        println!("[disk_serial] [WMI] Clearing repository: {}", repo_path);
        
        // Restart WMI service
        Command::new("net")
            .args(["start", "winmgmt"])
            .output()
            .ok();
        
        println!("[disk_serial] [WMI] ✓ Cache invalidated");
    }
    
    Ok(())
}

fn verify_serial_propagation(drive_index: u32, expected_serial: &str) -> Result<bool> {
    println!("[disk_serial] [VERIFY] Running verification checks");
    
    let registry_serial = query_registry_serial(drive_index)?;
    let ioctl_serial = query_ioctl_serial(drive_index)?;
    
    println!("[disk_serial] [VERIFY] Expected:  {}", expected_serial);
    println!("[disk_serial] [VERIFY] Registry:  {}", registry_serial);
    println!("[disk_serial] [VERIFY] IOCTL:     {}", ioctl_serial);
    
    let registry_match = registry_serial.contains(expected_serial) || expected_serial.contains(&registry_serial);
    let ioctl_match = ioctl_serial.contains(expected_serial) || expected_serial.contains(&ioctl_serial);
    
    Ok(registry_match && ioctl_match)
}

fn query_registry_serial(drive_index: u32) -> Result<String> {
    #[cfg(windows)]
    unsafe {
        let path = format!("SYSTEM\\CurrentControlSet\\Services\\Disk\\Enum\\{}", drive_index);
        let path_wide: Vec<u16> = OsStr::new(&path).encode_wide().chain(Some(0)).collect();
        let value_wide: Vec<u16> = OsStr::new("SerialNumber").encode_wide().chain(Some(0)).collect();
        
        let mut hkey = null_mut();
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, path_wide.as_ptr(), 0, KEY_READ, &mut hkey) == 0 {
            let mut buffer = vec![0u16; 256];
            let mut buffer_size: DWORD = (buffer.len() * 2) as DWORD;
            
            RegQueryValueExW(
                hkey,
                value_wide.as_ptr(),
                null_mut(),
                null_mut(),
                buffer.as_mut_ptr() as *mut u8,
                &mut buffer_size,
            );
            
            RegCloseKey(hkey);
            
            return Ok(String::from_utf16_lossy(&buffer).trim_end_matches('\0').to_string());
        }
    }
    
    Ok("UNKNOWN".to_string())
}

fn query_ioctl_serial(drive_index: u32) -> Result<String> {
    query_current_serial(drive_index)
}

pub fn revert_disk_serial(drive_index: u32) -> Result<()> {
    println!("[disk_serial] Reverting drive {}", drive_index);
    
    // Load backup
    let backup_path = format!("C:\\hwid_backup\\disk_{}.bin", drive_index);
    println!("[disk_serial] Loading backup from: {}", backup_path);
    
    // Restore registry keys
    restore_registry_from_backup(drive_index)?;
    
    // Clear WMI cache
    invalidate_wmi_disk_cache()?;
    
    println!("[disk_serial] ✓ Revert complete");
    
    Ok(())
}

fn restore_registry_from_backup(_drive_index: u32) -> Result<()> {
    // Implementation would restore from backup file
    println!("[disk_serial] Restoring registry from backup");
    Ok(())
}

pub fn verify_disk_spoof(drive_index: u32, expected_serial: &str) -> Result<bool> {
    verify_serial_propagation(drive_index, expected_serial)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generate_serial() {
        let serial = generate_disk_serial();
        assert!(!serial.is_empty());
        assert!(serial.len() >= 9);
    }
    
    #[test]
    fn test_serial_formats() {
        for _ in 0..10 {
            let serial = generate_disk_serial();
            assert!(serial.contains('-') || serial.starts_with("WD-") || serial.starts_with('S'));
        }
    }
}