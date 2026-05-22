// volume_serial.rs - Volume Serial Number Modification
use std::io::Result;

#[cfg(windows)]
use winapi::um::fileapi::CreateFileW;
#[cfg(windows)]
use winapi::um::ioapiset::DeviceIoControl;
#[cfg(windows)]
use winapi::um::handleapi::CloseHandle;

const FSCTL_SET_VOLUME_SERIAL: u32 = 0x00090054;

pub fn spoof_volume_serial(drive_letter: char, serial: Option<u32>) -> Result<()> {
    use rand::Rng;
    let new_serial = serial.unwrap_or_else(|| rand::thread_rng().gen());
    
    println!("[volume_serial] Spoofing {}:\\ to {:08X}", drive_letter, new_serial);
    
    set_volume_serial_number(drive_letter, new_serial)?;
    
    Ok(())
}

#[cfg(windows)]
fn set_volume_serial_number(drive: char, serial: u32) -> Result<()> {
    unsafe {
        use std::os::windows::ffi::OsStrExt;
        use std::ffi::OsStr;
        use std::ptr::null_mut;
        
        let path = format!("\\\\.\\{}:", drive);
        let path_wide: Vec<u16> = OsStr::new(&path)
            .encode_wide()
            .chain(Some(0))
            .collect();
        
        let handle = CreateFileW(
            path_wide.as_ptr(),
            0xC0000000, // GENERIC_READ | GENERIC_WRITE
            0,
            null_mut(),
            3, // OPEN_EXISTING
            0,
            null_mut(),
        );
        
        if handle != winapi::um::handleapi::INVALID_HANDLE_VALUE {
            let mut bytes_returned: u32 = 0;
            let serial_data = serial.to_le_bytes();
            
            DeviceIoControl(
                handle,
                FSCTL_SET_VOLUME_SERIAL,
                serial_data.as_ptr() as *mut _,
                4,
                null_mut(),
                0,
                &mut bytes_returned,
                null_mut(),
            );
            
            println!("[volume_serial] Volume serial set successfully");
            CloseHandle(handle);
        }
    }
    
    Ok(())
}

#[cfg(not(windows))]
fn set_volume_serial_number(_drive: char, _serial: u32) -> Result<()> {
    use std::io::{Error, ErrorKind};
    Err(Error::new(ErrorKind::Unsupported, "Windows only"))
}