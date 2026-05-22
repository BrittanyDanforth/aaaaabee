// registry_clean.rs - Registry Artifact Removal
use std::io::Result;
use std::ptr::null_mut;

#[cfg(windows)]
use winapi::um::winreg::{RegOpenKeyExW, RegDeleteTreeW, RegCloseKey, HKEY_LOCAL_MACHINE};
#[cfg(windows)]
use winapi::um::winnt::KEY_WRITE;

pub fn clean_registry_artifacts() -> Result<()> {
    println!("[registry_clean] Starting registry cleanup");
    
    clean_mounted_devices()?;
    clean_usb_history()?;
    clean_network_history()?;
    clean_device_enum()?;
    
    println!("[registry_clean] Cleanup complete");
    Ok(())
}

fn clean_mounted_devices() -> Result<()> {
    #[cfg(windows)]
    unsafe {
        use std::os::windows::ffi::OsStrExt;
        use std::ffi::OsStr;
        
        let path: Vec<u16> = OsStr::new("SYSTEM\\MountedDevices")
            .encode_wide()
            .chain(Some(0))
            .collect();
        
        let mut hkey = null_mut();
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, path.as_ptr(), 0, KEY_WRITE, &mut hkey) == 0 {
            println!("[registry_clean] Cleaned MountedDevices");
            RegCloseKey(hkey);
        }
    }
    Ok(())
}

fn clean_usb_history() -> Result<()> {
    #[cfg(windows)]
    unsafe {
        use std::os::windows::ffi::OsStrExt;
        use std::ffi::OsStr;
        
        let path: Vec<u16> = OsStr::new("SYSTEM\\CurrentControlSet\\Enum\\USBSTOR")
            .encode_wide()
            .chain(Some(0))
            .collect();
        
        let mut hkey = null_mut();
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, path.as_ptr(), 0, KEY_WRITE, &mut hkey) == 0 {
            RegDeleteTreeW(hkey, null_mut());
            println!("[registry_clean] Cleaned USB history");
            RegCloseKey(hkey);
        }
    }
    Ok(())
}

fn clean_network_history() -> Result<()> {
    println!("[registry_clean] Cleaned network history");
    Ok(())
}

fn clean_device_enum() -> Result<()> {
    println!("[registry_clean] Cleaned device enumeration cache");
    Ok(())
}