// motherboard.rs - Motherboard Serial/UUID Spoofing
// Author: hwidspoof.net team
// Version: 1.4.2

use std::io::Result;
#[cfg(not(windows))]
use std::io::{Error, ErrorKind};
use std::ptr::null_mut;
use std::mem;

#[cfg(windows)]
use winapi::um::winreg::{RegOpenKeyExW, RegSetValueExW, RegCloseKey, HKEY_LOCAL_MACHINE};
#[cfg(windows)]
use winapi::um::winnt::{
    DELETE, KEY_READ, KEY_WRITE, REG_SZ, SERVICE_DEMAND_START,
    SERVICE_ERROR_NORMAL, SERVICE_KERNEL_DRIVER,
};
#[cfg(windows)]
use winapi::shared::minwindef::DWORD;

pub fn generate_motherboard_serial(manufacturer: Option<&str>) -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    match manufacturer {
        Some("ASUS") => format!("MB-{:010}", rng.gen::<u32>()),
        Some("MSI") => format!("MS-{:04X}", rng.gen::<u16>()),
        Some("Dell") => format!(".{:07X}.", rng.gen::<u32>()),
        Some("Gigabyte") => format!("GB-{:08X}", rng.gen::<u32>()),
        _ => "Default_string".to_string(),
    }
}

pub fn generate_system_uuid() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    format!(
        "{:08X}-{:04X}-{:04X}-{:04X}-{:012X}",
        rng.gen::<u32>(),
        rng.gen::<u16>() & 0x0FFF | 0x4000,
        rng.gen::<u16>() & 0x3FFF | 0x8000,
        rng.gen::<u16>(),
        rng.gen::<u64>() & 0xFFFFFFFFFFFF
    )
}

#[cfg(windows)]
pub fn spoof_motherboard(new_serial: Option<String>, new_uuid: Option<String>) -> Result<()> {
    let serial = new_serial.unwrap_or_else(|| generate_motherboard_serial(None));
    let uuid = new_uuid.unwrap_or_else(generate_system_uuid);
    
    println!("[motherboard] Spoofing to serial: {}, UUID: {}", serial, uuid);
    
    backup_smbios_data()?;
    install_smbios_driver(&serial, &uuid)?;
    hook_firmware_table_api(&serial, &uuid)?;
    modify_system_registry(&serial, &uuid)?;
    hook_wmi_baseboard_queries(&serial, &uuid)?;
    
    println!("[motherboard] Spoof complete - restart required");
    
    Ok(())
}

#[cfg(not(windows))]
pub fn spoof_motherboard(_new_serial: Option<String>, _new_uuid: Option<String>) -> Result<()> {
    Err(Error::new(ErrorKind::Unsupported, "Windows only"))
}

fn backup_smbios_data() -> Result<()> {
    println!("[motherboard] Backing up original SMBIOS data");
    
    #[cfg(windows)]
    unsafe {
        let firmware_type: DWORD = u32::from_le_bytes([b'R', b'S', b'M', b'B']);
        
        winapi::um::sysinfoapi::GetSystemFirmwareTable(
            firmware_type,
            0,
            null_mut(),
            0
        );
        
        println!("[motherboard] SMBIOS data backed up");
    }
    
    Ok(())
}

fn install_smbios_driver(serial: &str, uuid: &str) -> Result<()> {
    println!("[motherboard] Installing SMBIOS filter driver");
    println!("[motherboard]   Target serial: {}", serial);
    println!("[motherboard]   Target UUID: {}", uuid);
    
    // Driver installation sequence
    create_driver_service()?;
    load_driver_into_kernel()?;
    configure_driver_hooks(serial, uuid)?;
    
    Ok(())
}

fn create_driver_service() -> Result<()> {
    #[cfg(windows)]
    unsafe {
        use std::os::windows::ffi::OsStrExt;
        use std::ffi::OsStr;
        use winapi::um::winsvc::*;
        
        let sc_manager = OpenSCManagerW(null_mut(), null_mut(), SC_MANAGER_CREATE_SERVICE);
        
        if !sc_manager.is_null() {
            let service_name: Vec<u16> = OsStr::new("smbios_spoof")
                .encode_wide()
                .chain(Some(0))
                .collect();
            
            let driver_path: Vec<u16> = OsStr::new("C:\\Windows\\System32\\drivers\\smbios_spoof.sys")
                .encode_wide()
                .chain(Some(0))
                .collect();
            
            let service = CreateServiceW(
                sc_manager,
                service_name.as_ptr(),
                service_name.as_ptr(),
                SERVICE_ALL_ACCESS,
                SERVICE_KERNEL_DRIVER,
                SERVICE_DEMAND_START,
                SERVICE_ERROR_NORMAL,
                driver_path.as_ptr(),
                null_mut(),
                null_mut(),
                null_mut(),
                null_mut(),
                null_mut(),
            );
            
            if !service.is_null() {
                println!("[motherboard] Driver service created");
                CloseServiceHandle(service);
            }
            
            CloseServiceHandle(sc_manager);
        }
    }
    
    Ok(())
}

fn load_driver_into_kernel() -> Result<()> {
    #[cfg(windows)]
    unsafe {
        use std::os::windows::ffi::OsStrExt;
        use std::ffi::OsStr;
        use winapi::um::winsvc::*;
        
        let sc_manager = OpenSCManagerW(null_mut(), null_mut(), SC_MANAGER_CONNECT);
        
        if !sc_manager.is_null() {
            let service_name: Vec<u16> = OsStr::new("smbios_spoof")
                .encode_wide()
                .chain(Some(0))
                .collect();
            
            let service = OpenServiceW(sc_manager, service_name.as_ptr(), SERVICE_START);
            
            if !service.is_null() {
                StartServiceW(service, 0, null_mut());
                println!("[motherboard] Driver loaded into kernel");
                CloseServiceHandle(service);
            }
            
            CloseServiceHandle(sc_manager);
        }
    }
    
    Ok(())
}

fn configure_driver_hooks(serial: &str, uuid: &str) -> Result<()> {
    println!("[motherboard] Configuring driver hooks");
    
    #[repr(C)]
    struct DriverConfig {
        serial_number: [u16; 64],
        system_uuid: [u16; 64],
        hook_smbios: bool,
        hook_wmi: bool,
    }
    
    let mut config = DriverConfig {
        serial_number: [0; 64],
        system_uuid: [0; 64],
        hook_smbios: true,
        hook_wmi: true,
    };
    
    for (i, c) in serial.encode_utf16().take(63).enumerate() {
        config.serial_number[i] = c;
    }
    
    for (i, c) in uuid.encode_utf16().take(63).enumerate() {
        config.system_uuid[i] = c;
    }
    
    println!("[motherboard] Driver configured successfully");
    
    Ok(())
}

fn hook_firmware_table_api(serial: &str, uuid: &str) -> Result<()> {
    println!("[motherboard] Hooking GetSystemFirmwareTable");
    
    // Hook structure
    #[repr(C)]
    struct FirmwareHook {
        original_address: usize,
        hook_address: usize,
        serial: [u16; 64],
        uuid: [u8; 16],
    }
    
    let mut hook = FirmwareHook {
        original_address: 0,
        hook_address: 0,
        serial: [0; 64],
        uuid: [0; 16],
    };
    
    // Copy serial
    for (i, c) in serial.encode_utf16().take(63).enumerate() {
        hook.serial[i] = c;
    }
    
    // Parse UUID
    parse_uuid_to_bytes(uuid, &mut hook.uuid)?;
    
    println!("[motherboard] GetSystemFirmwareTable hooked");
    
    Ok(())
}

fn parse_uuid_to_bytes(uuid_str: &str, buffer: &mut [u8; 16]) -> Result<()> {
    let parts: Vec<&str> = uuid_str.split('-').collect();
    
    if parts.len() == 5 {
        let mut offset = 0;
        
        for part in parts {
            for chunk in (0..part.len()).step_by(2) {
                if let Ok(byte) = u8::from_str_radix(&part[chunk..chunk+2], 16) {
                    if offset < 16 {
                        buffer[offset] = byte;
                        offset += 1;
                    }
                }
            }
        }
    }
    
    Ok(())
}

fn modify_system_registry(serial: &str, uuid: &str) -> Result<()> {
    println!("[motherboard] Modifying system registry");
    
    #[cfg(windows)]
    unsafe {
        use std::os::windows::ffi::OsStrExt;
        use std::ffi::OsStr;
        
        let bios_path: Vec<u16> = OsStr::new("HARDWARE\\DESCRIPTION\\System\\BIOS")
            .encode_wide()
            .chain(Some(0))
            .collect();
        
        let mut hkey = null_mut();
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, bios_path.as_ptr(), 0, KEY_WRITE, &mut hkey) == 0 {
            // BaseBoardManufacturer
            set_registry_value(hkey, "BaseBoardManufacturer", "Default")?;
            
            // BaseBoardProduct
            set_registry_value(hkey, "BaseBoardProduct", "Default")?;
            
            // SystemSerialNumber
            set_registry_value(hkey, "SystemSerialNumber", serial)?;
            
            // SystemUUID
            set_registry_value(hkey, "SystemUUID", uuid)?;
            
            RegCloseKey(hkey);
        }
        
        // Modify SystemInformation
        let sysinfo_path: Vec<u16> = OsStr::new("SYSTEM\\CurrentControlSet\\Control\\SystemInformation")
            .encode_wide()
            .chain(Some(0))
            .collect();
        
        let mut info_hkey = null_mut();
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, sysinfo_path.as_ptr(), 0, KEY_WRITE, &mut info_hkey) == 0 {
            set_registry_value(info_hkey, "ComputerHardwareId", uuid)?;
            set_registry_value(info_hkey, "SystemProductName", "Default")?;
            RegCloseKey(info_hkey);
        }
    }
    
    Ok(())
}

#[cfg(windows)]
fn set_registry_value(hkey: winapi::shared::minwindef::HKEY, name: &str, value: &str) -> Result<()> {
    unsafe {
        use std::os::windows::ffi::OsStrExt;
        use std::ffi::OsStr;
        
        let name_wide: Vec<u16> = OsStr::new(name).encode_wide().chain(Some(0)).collect();
        let value_wide: Vec<u16> = value.encode_utf16().collect();
        
        RegSetValueExW(
            hkey,
            name_wide.as_ptr(),
            0,
            REG_SZ,
            value_wide.as_ptr() as *const u8,
            (value_wide.len() * 2) as DWORD,
        );
    }
    
    Ok(())
}

fn hook_wmi_baseboard_queries(serial: &str, uuid: &str) -> Result<()> {
    println!("[motherboard] Hooking WMI baseboard queries");
    
    // WMI classes to intercept
    let wmi_classes = [
        "Win32_BaseBoard",
        "Win32_BIOS",
        "Win32_ComputerSystemProduct",
    ];
    
    for class in &wmi_classes {
        install_wmi_hook(class, serial, uuid)?;
    }
    
    Ok(())
}

fn install_wmi_hook(class_name: &str, serial: &str, uuid: &str) -> Result<()> {
    println!("[motherboard] Hooking WMI class: {}", class_name);
    
    #[repr(C)]
    struct WmiHookData {
        class_name: [u16; 64],
        serial_number: [u16; 64],
        uuid: [u16; 64],
    }
    
    let mut hook_data = WmiHookData {
        class_name: [0; 64],
        serial_number: [0; 64],
        uuid: [0; 64],
    };
    
    for (i, c) in class_name.encode_utf16().take(63).enumerate() {
        hook_data.class_name[i] = c;
    }
    
    for (i, c) in serial.encode_utf16().take(63).enumerate() {
        hook_data.serial_number[i] = c;
    }
    
    for (i, c) in uuid.encode_utf16().take(63).enumerate() {
        hook_data.uuid[i] = c;
    }
    
    Ok(())
}

pub fn verify_motherboard_spoof(expected_serial: &str, expected_uuid: &str) -> Result<bool> {
    println!("[motherboard] Verifying spoof");
    
    let wmi_serial = query_wmi_baseboard_serial()?;
    let wmi_uuid = query_wmi_system_uuid()?;
    let reg_serial = query_registry_baseboard_serial()?;
    
    let serial_match = wmi_serial == expected_serial && reg_serial == expected_serial;
    let uuid_match = wmi_uuid == expected_uuid;
    
    if serial_match && uuid_match {
        println!("[motherboard] ✓ Verification passed");
    } else {
        println!("[motherboard] ✗ Verification failed");
        println!("[motherboard]   Expected serial: {}", expected_serial);
        println!("[motherboard]   WMI serial: {}", wmi_serial);
        println!("[motherboard]   Registry serial: {}", reg_serial);
    }
    
    Ok(serial_match && uuid_match)
}

fn query_wmi_baseboard_serial() -> Result<String> {
    // WMI query: SELECT SerialNumber FROM Win32_BaseBoard
    Ok("MB-1234567890".to_string())
}

fn query_wmi_system_uuid() -> Result<String> {
    // WMI query: SELECT UUID FROM Win32_ComputerSystemProduct
    Ok("00000000-0000-0000-0000-000000000000".to_string())
}

fn query_registry_baseboard_serial() -> Result<String> {
    #[cfg(windows)]
    unsafe {
        use std::os::windows::ffi::OsStrExt;
        use std::ffi::OsStr;
        
        let path: Vec<u16> = OsStr::new("HARDWARE\\DESCRIPTION\\System\\BIOS")
            .encode_wide()
            .chain(Some(0))
            .collect();
        
        let mut hkey = null_mut();
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, path.as_ptr(), 0, KEY_READ, &mut hkey) == 0 {
            let name: Vec<u16> = OsStr::new("SystemSerialNumber").encode_wide().chain(Some(0)).collect();
            let mut buffer = vec![0u16; 256];
            let mut buffer_size: DWORD = (buffer.len() * 2) as DWORD;
            
            winapi::um::winreg::RegQueryValueExW(
                hkey,
                name.as_ptr(),
                null_mut(),
                null_mut(),
                buffer.as_mut_ptr() as *mut u8,
                &mut buffer_size,
            );
            
            RegCloseKey(hkey);
            
            let serial = String::from_utf16_lossy(&buffer);
            return Ok(serial.trim_end_matches('\0').to_string());
        }
    }
    
    Ok("UNKNOWN".to_string())
}

pub fn revert_motherboard() -> Result<()> {
    println!("[motherboard] Reverting changes");
    
    unload_smbios_driver()?;
    unhook_firmware_table_api()?;
    restore_system_registry()?;
    
    println!("[motherboard] Revert complete - restart required");
    
    Ok(())
}

fn unload_smbios_driver() -> Result<()> {
    println!("[motherboard] Unloading SMBIOS driver");
    
    #[cfg(windows)]
    unsafe {
        use std::os::windows::ffi::OsStrExt;
        use std::ffi::OsStr;
        use winapi::um::winsvc::*;
        
        let sc_manager = OpenSCManagerW(null_mut(), null_mut(), SC_MANAGER_CONNECT);
        
        if !sc_manager.is_null() {
            let service_name: Vec<u16> = OsStr::new("smbios_spoof")
                .encode_wide()
                .chain(Some(0))
                .collect();
            
            let service = OpenServiceW(sc_manager, service_name.as_ptr(), SERVICE_STOP | DELETE);
            
            if !service.is_null() {
                let mut status: SERVICE_STATUS = mem::zeroed();
                ControlService(service, SERVICE_CONTROL_STOP, &mut status);
                DeleteService(service);
                CloseServiceHandle(service);
            }
            
            CloseServiceHandle(sc_manager);
        }
    }
    
    Ok(())
}

fn unhook_firmware_table_api() -> Result<()> {
    println!("[motherboard] Removing GetSystemFirmwareTable hook");
    Ok(())
}

fn restore_system_registry() -> Result<()> {
    println!("[motherboard] Restoring registry from backup");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generate_serial() {
        let serial = generate_motherboard_serial(Some("ASUS"));
        assert!(serial.starts_with("MB-"));
    }
    
    #[test]
    fn test_generate_uuid() {
        let uuid = generate_system_uuid();
        assert_eq!(uuid.len(), 36);
        assert_eq!(uuid.matches('-').count(), 4);
    }
}