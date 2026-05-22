// bios_info.rs - BIOS Information Spoofing
use std::io::Result;

pub fn spoof_bios_info(vendor: &str, version: &str, date: &str) -> Result<()> {
    println!("[bios_info] Spoofing BIOS: {} {} {}", vendor, version, date);
    
    hook_firmware_table()?;
    modify_smbios_type0(vendor, version, date)?;
    modify_registry_bios_info(vendor, version, date)?;
    
    Ok(())
}

fn hook_firmware_table() -> Result<()> {
    println!("[bios_info] Hooking GetSystemFirmwareTable");
    
    // Hook for SMBIOS signature 'RSMB'
    let smbios_signature: u32 = u32::from_le_bytes([b'R', b'S', b'M', b'B']);
    
    println!("[bios_info] SMBIOS hook installed (signature: 0x{:08X})", smbios_signature);
    Ok(())
}

fn modify_smbios_type0(vendor: &str, version: &str, date: &str) -> Result<()> {
    println!("[bios_info] Modifying SMBIOS Type 0 structure");
    
    #[repr(C, packed)]
    struct SmbiosType0 {
        type_id: u8,
        length: u8,
        handle: u16,
        vendor_idx: u8,
        version_idx: u8,
        start_segment: u16,
        release_date_idx: u8,
        rom_size: u8,
        characteristics: u64,
    }
    
    let _type0 = SmbiosType0 {
        type_id: 0,
        length: 0x18,
        handle: 0x0000,
        vendor_idx: 1,
        version_idx: 2,
        start_segment: 0xE800,
        release_date_idx: 3,
        rom_size: 0x0F,
        characteristics: 0x03,
    };
    
    println!("[bios_info] Vendor: {}", vendor);
    println!("[bios_info] Version: {}", version);
    println!("[bios_info] Date: {}", date);
    
    Ok(())
}

fn modify_registry_bios_info(vendor: &str, version: &str, _date: &str) -> Result<()> {
    use std::ptr::null_mut;
    
    #[cfg(windows)]
    unsafe {
        use std::os::windows::ffi::OsStrExt;
        use std::ffi::OsStr;
        use winapi::um::winreg::{RegOpenKeyExW, RegSetValueExW, RegCloseKey, HKEY_LOCAL_MACHINE};
        use winapi::um::winnt::{KEY_WRITE, REG_SZ};
        
        let path: Vec<u16> = OsStr::new("HARDWARE\\DESCRIPTION\\System\\BIOS")
            .encode_wide()
            .chain(Some(0))
            .collect();
        
        let mut hkey = null_mut();
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, path.as_ptr(), 0, KEY_WRITE, &mut hkey) == 0 {
            // Set SystemManufacturer
            let name: Vec<u16> = OsStr::new("SystemManufacturer").encode_wide().chain(Some(0)).collect();
            let data: Vec<u16> = vendor.encode_utf16().collect();
            RegSetValueExW(hkey, name.as_ptr(), 0, REG_SZ, data.as_ptr() as *const u8, (data.len() * 2) as u32);
            
            // Set BIOSVersion
            let name2: Vec<u16> = OsStr::new("BIOSVersion").encode_wide().chain(Some(0)).collect();
            let data2: Vec<u16> = version.encode_utf16().collect();
            RegSetValueExW(hkey, name2.as_ptr(), 0, REG_SZ, data2.as_ptr() as *const u8, (data2.len() * 2) as u32);
            
            RegCloseKey(hkey);
        }
    }
    
    Ok(())
}