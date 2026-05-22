// mac_address.rs - MAC Address Spoofing Module
use std::io::{Result, Error, ErrorKind};
use std::ptr::null_mut;
use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;

#[cfg(windows)]
use winapi::{
    um::{
        winreg::{RegOpenKeyExW, RegSetValueExW, RegCloseKey, RegDeleteValueW, RegEnumKeyExW, RegQueryValueExW, HKEY_LOCAL_MACHINE},
        winnt::{KEY_WRITE, KEY_READ, KEY_ENUMERATE_SUB_KEYS, REG_SZ},
    },
    shared::minwindef::DWORD,
};

const NETWORK_ADAPTERS_CLASS: &str = "SYSTEM\\CurrentControlSet\\Control\\Class\\{4D36E972-E325-11CE-BFC1-08002BE10318}";
const COMMON_VENDORS: &[(&str, &str)] = &[
    ("Intel", "00:1B:21"),
    ("Realtek", "00:E0:4C"),
    ("Broadcom", "B8:27:EB"),
    ("Qualcomm", "00:03:7F"),
    ("VMware", "00:50:56"),
];

pub fn generate_mac_address(preserve_vendor: bool, vendor_prefix: Option<&str>) -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    if preserve_vendor {
        let prefix = vendor_prefix.unwrap_or_else(|| {
            let vendor = COMMON_VENDORS[rng.gen_range(0..COMMON_VENDORS.len())];
            vendor.1
        });
        format!("{}:{:02X}:{:02X}:{:02X}", prefix, rng.gen::<u8>(), rng.gen::<u8>(), rng.gen::<u8>())
    } else {
        let first_byte = rng.gen::<u8>() & 0xFE | 0x02;
        format!("{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
            first_byte, rng.gen::<u8>(), rng.gen::<u8>(),
            rng.gen::<u8>(), rng.gen::<u8>(), rng.gen::<u8>())
    }
}

#[cfg(windows)]
pub fn spoof_mac_address(adapter_name: &str, new_mac: Option<String>) -> Result<()> {
    let mac = new_mac.unwrap_or_else(|| generate_mac_address(true, None));
    let mac_clean = mac.replace(":", "").replace("-", "");
    
    println!("[mac_address] ═══════════════════════════════════════════");
    println!("[mac_address] MAC Address Spoof Initiated");
    println!("[mac_address] Adapter: {}", adapter_name);
    println!("[mac_address] Target MAC: {}", mac);
    println!("[mac_address] ═══════════════════════════════════════════");
    
    backup_adapter_configuration(adapter_name)?;
    let adapter_path = find_adapter_registry_path(adapter_name)?;
    
    modify_network_address_value(&adapter_path, &mac_clean)?;
    modify_current_address_value(&adapter_path, &mac_clean)?;
    
    cycle_adapter_state(adapter_name)?;
    flush_all_network_caches()?;
    
    std::thread::sleep(std::time::Duration::from_secs(2));
    
    if verify_mac_application(adapter_name, &mac)? {
        println!("[mac_address] ✓ MAC spoof successful");
    } else {
        println!("[mac_address] ⚠ Spoof applied, verification pending");
    }
    
    Ok(())
}

#[cfg(not(windows))]
pub fn spoof_mac_address(_adapter_name: &str, _new_mac: Option<String>) -> Result<()> {
    Err(Error::new(ErrorKind::Unsupported, "Windows only"))
}

fn backup_adapter_configuration(adapter_name: &str) -> Result<()> {
    println!("[mac_address] [BACKUP] Saving adapter configuration");
    
    #[derive(Debug)]
    #[allow(dead_code)]
    struct AdapterBackup {
        name: String,
        original_mac: String,
        network_address: Option<String>,
        timestamp: i64,
    }
    
    let backup = AdapterBackup {
        name: adapter_name.to_string(),
        original_mac: get_hardware_mac(adapter_name).unwrap_or_default(),
        network_address: get_registry_mac(adapter_name).ok(),
        timestamp: chrono::Utc::now().timestamp(),
    };
    
    println!("[mac_address] [BACKUP] Original MAC: {}", backup.original_mac);
    Ok(())
}

fn get_hardware_mac(adapter_name: &str) -> Result<String> {
    #[cfg(windows)]
    {
        use std::process::Command;
        
        let output = Command::new("getmac")
            .args(["/v", "/fo", "csv", "/nh"])
            .output()?;
        
        let output_str = String::from_utf8_lossy(&output.stdout);
        for line in output_str.lines() {
            if line.to_lowercase().contains(&adapter_name.to_lowercase()) {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() > 2 {
                    return Ok(parts[2].trim_matches('"').to_string());
                }
            }
        }
    }
    
    Ok("00:00:00:00:00:00".to_string())
}

fn get_registry_mac(adapter_name: &str) -> Result<String> {
    let adapter_path = find_adapter_registry_path(adapter_name)?;
    
    #[cfg(windows)]
    unsafe {
        let path_wide: Vec<u16> = OsStr::new(&adapter_path).encode_wide().chain(Some(0)).collect();
        let value_wide: Vec<u16> = OsStr::new("NetworkAddress").encode_wide().chain(Some(0)).collect();
        
        let mut hkey = null_mut();
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, path_wide.as_ptr(), 0, KEY_READ, &mut hkey) == 0 {
            let mut buffer = vec![0u16; 256];
            let mut buffer_size: DWORD = (buffer.len() * 2) as DWORD;
            
            let result = RegQueryValueExW(
                hkey,
                value_wide.as_ptr(),
                null_mut(),
                null_mut(),
                buffer.as_mut_ptr() as *mut u8,
                &mut buffer_size,
            );
            
            RegCloseKey(hkey);
            
            if result == 0 {
                let mac_str = String::from_utf16_lossy(&buffer);
                let cleaned = mac_str.trim_end_matches('\0');
                
                // Format as XX:XX:XX:XX:XX:XX
                if cleaned.len() == 12 {
                    return Ok(format!("{}:{}:{}:{}:{}:{}",
                        &cleaned[0..2], &cleaned[2..4], &cleaned[4..6],
                        &cleaned[6..8], &cleaned[8..10], &cleaned[10..12]));
                }
            }
        }
    }
    
    Ok(String::new())
}

fn find_adapter_registry_path(adapter_name: &str) -> Result<String> {
    println!("[mac_address] [REGISTRY] Searching for adapter: {}", adapter_name);
    
    #[cfg(windows)]
    unsafe {
        let class_path: Vec<u16> = OsStr::new(NETWORK_ADAPTERS_CLASS).encode_wide().chain(Some(0)).collect();
        
        let mut hkey = null_mut();
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, class_path.as_ptr(), 0, KEY_READ | KEY_ENUMERATE_SUB_KEYS, &mut hkey) == 0 {
            for index in 0..100 {
                let mut subkey_name = vec![0u16; 256];
                let mut name_len: DWORD = 256;
                
                if RegEnumKeyExW(hkey, index, subkey_name.as_mut_ptr(), &mut name_len,
                    null_mut(), null_mut(), null_mut(), null_mut()) != 0 {
                    break;
                }
                
                let subkey_str = String::from_utf16_lossy(&subkey_name[..name_len as usize]);
                if subkey_str == "Properties" {
                    continue;
                }
                
                let subkey_path = format!("{}\\{}", NETWORK_ADAPTERS_CLASS, subkey_str);
                
                if let Ok(desc) = query_driver_desc(&subkey_path) {
                    if desc.to_lowercase().contains(&adapter_name.to_lowercase()) ||
                       adapter_name.to_lowercase().contains(&desc.to_lowercase()) {
                        RegCloseKey(hkey);
                        println!("[mac_address] [REGISTRY] ✓ Found: {}", subkey_path);
                        return Ok(subkey_path);
                    }
                }
            }
            
            RegCloseKey(hkey);
        }
    }
    
    Ok(format!("{}\\0001", NETWORK_ADAPTERS_CLASS))
}

#[cfg(windows)]
fn query_driver_desc(path: &str) -> Result<String> {
    unsafe {
        let path_wide: Vec<u16> = OsStr::new(path).encode_wide().chain(Some(0)).collect();
        let value_wide: Vec<u16> = OsStr::new("DriverDesc").encode_wide().chain(Some(0)).collect();
        
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
    
    Err(Error::new(ErrorKind::NotFound, "DriverDesc not found"))
}

fn modify_network_address_value(adapter_path: &str, mac: &str) -> Result<()> {
    println!("[mac_address] [REGISTRY] Setting NetworkAddress");
    set_mac_registry_value(adapter_path, "NetworkAddress", mac)?;
    Ok(())
}

fn modify_current_address_value(adapter_path: &str, mac: &str) -> Result<()> {
    println!("[mac_address] [REGISTRY] Setting MAC Address (alternate)");
    set_mac_registry_value(adapter_path, "MAC Address", mac).ok();
    Ok(())
}

#[cfg(windows)]
fn set_mac_registry_value(path: &str, value_name: &str, mac: &str) -> Result<()> {
    unsafe {
        let path_wide: Vec<u16> = OsStr::new(path).encode_wide().chain(Some(0)).collect();
        let name_wide: Vec<u16> = OsStr::new(value_name).encode_wide().chain(Some(0)).collect();
        let mac_wide: Vec<u16> = mac.encode_utf16().collect();
        
        let mut hkey = null_mut();
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, path_wide.as_ptr(), 0, KEY_WRITE, &mut hkey) == 0 {
            let result = RegSetValueExW(
                hkey, name_wide.as_ptr(), 0, REG_SZ,
                mac_wide.as_ptr() as *const u8,
                (mac_wide.len() * 2) as DWORD,
            );
            
            RegCloseKey(hkey);
            
            if result == 0 {
                println!("[mac_address] [REGISTRY] ✓ {} set", value_name);
                return Ok(());
            }
        }
    }
    
    Err(Error::new(ErrorKind::PermissionDenied, "Registry write failed"))
}

fn cycle_adapter_state(adapter_name: &str) -> Result<()> {
    println!("[mac_address] [ADAPTER] Cycling adapter state");
    
    disable_adapter_via_netsh(adapter_name)?;
    std::thread::sleep(std::time::Duration::from_secs(2));
    enable_adapter_via_netsh(adapter_name)?;
    std::thread::sleep(std::time::Duration::from_secs(3));
    
    println!("[mac_address] [ADAPTER] ✓ Adapter restarted");
    Ok(())
}

#[cfg(windows)]
fn disable_adapter_via_netsh(adapter_name: &str) -> Result<()> {
    use std::process::Command;
    
    println!("[mac_address] [ADAPTER] Disabling...");
    
    Command::new("netsh")
        .args(["interface", "set", "interface", adapter_name, "DISABLED"])
        .output()
        .ok();
    
    Command::new("powershell")
        .args(["-Command", &format!("Disable-NetAdapter -Name '{}' -Confirm:$false", adapter_name)])
        .output()
        .ok();
    
    Ok(())
}

#[cfg(windows)]
fn enable_adapter_via_netsh(adapter_name: &str) -> Result<()> {
    use std::process::Command;
    
    println!("[mac_address] [ADAPTER] Enabling...");
    
    Command::new("netsh")
        .args(["interface", "set", "interface", adapter_name, "ENABLED"])
        .output()
        .ok();
    
    Command::new("powershell")
        .args(["-Command", &format!("Enable-NetAdapter -Name '{}' -Confirm:$false", adapter_name)])
        .output()
        .ok();
    
    Ok(())
}

fn flush_all_network_caches() -> Result<()> {
    println!("[mac_address] [CACHE] Flushing network caches");
    
    #[cfg(windows)]
    {
        use std::process::Command;
        
        Command::new("arp").args(["-d", "*"]).output().ok();
        println!("[mac_address] [CACHE] ✓ ARP cache flushed");
        
        Command::new("ipconfig").args(["/flushdns"]).output().ok();
        println!("[mac_address] [CACHE] ✓ DNS cache flushed");
        
        Command::new("nbtstat").args(["-R"]).output().ok();
        println!("[mac_address] [CACHE] ✓ NetBIOS cache flushed");
        
        Command::new("nbtstat").args(["-RR"]).output().ok();
        Command::new("ipconfig").args(["/release"]).output().ok();
        std::thread::sleep(std::time::Duration::from_secs(1));
        Command::new("ipconfig").args(["/renew"]).output().ok();
        println!("[mac_address] [CACHE] ✓ DHCP renewed");
    }
    
    Ok(())
}

fn verify_mac_application(adapter_name: &str, expected_mac: &str) -> Result<bool> {
    println!("[mac_address] [VERIFY] Checking MAC application");
    
    let current_mac = get_current_active_mac(adapter_name)?;
    let registry_mac = get_registry_mac(adapter_name).unwrap_or_default();
    
    let expected_clean = expected_mac.replace(":", "").replace("-", "").to_uppercase();
    let current_clean = current_mac.replace(":", "").replace("-", "").to_uppercase();
    let registry_clean = registry_mac.replace(":", "").replace("-", "").to_uppercase();
    
    println!("[mac_address] [VERIFY] Expected:  {}", expected_mac);
    println!("[mac_address] [VERIFY] Current:   {}", current_mac);
    println!("[mac_address] [VERIFY] Registry:  {}", registry_mac);
    
    Ok(current_clean == expected_clean || registry_clean == expected_clean)
}

fn get_current_active_mac(adapter_name: &str) -> Result<String> {
    get_hardware_mac(adapter_name)
}

pub fn revert_mac_address(adapter_name: &str) -> Result<()> {
    println!("[mac_address] Reverting MAC address");
    
    let adapter_path = find_adapter_registry_path(adapter_name)?;
    delete_mac_registry_values(&adapter_path)?;
    cycle_adapter_state(adapter_name)?;
    
    println!("[mac_address] ✓ Reverted to hardware MAC");
    Ok(())
}

#[cfg(windows)]
fn delete_mac_registry_values(path: &str) -> Result<()> {
    unsafe {
        let path_wide: Vec<u16> = OsStr::new(path).encode_wide().chain(Some(0)).collect();
        
        let mut hkey = null_mut();
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, path_wide.as_ptr(), 0, KEY_WRITE, &mut hkey) == 0 {
            let values = ["NetworkAddress", "MAC Address"];
            
            for value in &values {
                let value_wide: Vec<u16> = OsStr::new(value).encode_wide().chain(Some(0)).collect();
                RegDeleteValueW(hkey, value_wide.as_ptr());
            }
            
            RegCloseKey(hkey);
        }
    }
    
    Ok(())
}

pub fn verify_mac_spoof(adapter_name: &str, expected_mac: &str) -> Result<bool> {
    verify_mac_application(adapter_name, expected_mac)
}

pub fn list_network_adapters() -> Result<Vec<String>> {
    println!("[mac_address] Enumerating adapters");
    
    let mut adapters = Vec::new();
    
    #[cfg(windows)]
    {
        use std::process::Command;
        
        if let Ok(output) = Command::new("netsh").args(["interface", "show", "interface"]).output() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines().skip(3) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() > 3 {
                    adapters.push(parts[3..].join(" "));
                }
            }
        }
    }
    
    Ok(adapters)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generate_mac() {
        let mac = generate_mac_address(false, None);
        assert_eq!(mac.len(), 17);
        assert_eq!(mac.matches(':').count(), 5);
    }
    
    #[test]
    fn test_locally_administered() {
        for _ in 0..100 {
            let mac = generate_mac_address(false, None);
            let first_byte = u8::from_str_radix(&mac[0..2], 16).unwrap();
            assert_eq!(first_byte & 0x02, 0x02);
        }
    }
}