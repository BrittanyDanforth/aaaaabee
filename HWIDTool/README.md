# HWID Tool - Hardware Identifier Management Utility

## Overview
A Rust-based hardware identifier management tool designed for privacy-conscious gamers and developers. This utility allows users to modify system identifiers while maintaining **0/70 detections on VirusTotal**, demonstrating clean code practices and legitimate system interaction methods. Created as a proof-of-concept for hardware privacy.

## Before / after verification (this repo)

Every apply writes:

- `logs/hwid_before.txt` — identifiers **before** spoof
- `logs/hwid_after.txt` — identifiers **after** spoof  
- `logs/hwid_verify_report.txt` — per-field **CHANGED ✓** / **UNCHANGED ⚠**

`logs/hwid_step.ok` is created only when at least one probe changes (see report **VERDICT**). After reboot, run `hwspoof.exe --verify-last`.

```bat
run_hwid.bat              REM apply + report (Administrator)
hwspoof.exe --report      REM current IDs only
hwspoof.exe --verify-last REM re-check after restart
```

## 🔍 Security Verification
**Status: 0/70 Detections (Clean)**
- **VirusTotal Report**: [View Analysis](https://www.virustotal.com/gui/file/c1e16a1d5ffaef5835e343862520062a118ebf815fe37ab6beac92390881b9bb/detection)
- **SHA-256**: `c1e16a1d5ffaef5835e343862520062a118ebf815fe37ab6beac92390881b9bb`
- **File Size**: 18.821 MB (x64 only)
- **Detection Rate**: Exceptionally low due to proper Rust coding and compilation standards, our private modules are very sophisticated.

## 🚀 Capabilities

### Hardware Identifier Management
- **System ID Adjustment**: Modifies hardware identifiers at multiple system layers
- **Network Identifier Randomization**: Updates MAC addresses across all adapters
- **Storage Serial Management**: Handles GPT/MBR disk identifiers with driver-level access
- **Component ID Configuration**: Manages GPU, CPU, and motherboard identifiers
- **Registry Management**: System-level registry modifications for ID persistence

### Compatibility Testing
Tested in various environments including:
- **EasyAntiCheat** environments (Fortnite, Apex Legends, Rust)
- **Vanguard** systems (will likely require additional configuration)
- **BattlEye** protected titles (Rainbow Six Siege, Escape from Tarkov, PUBG)
- **Ricochet** implementations (Call of Duty series)
- **FACEIT** and other competitive platforms (variable success rates)

### Technical Architecture
- **Language**: Rust with direct system bindings
- **Distribution**: Statically linked, zero runtime dependencies
- **Operation**: Kernel-level access, WMI integration, registry management
- **Persistence Model**: Single execution pattern, no resident processes

## 🛠️ Technical Implementation

### System Integration Layers
1. **WMI Interface**: Communicates with Win32_DiskDrive, Win32_BaseBoard, Win32_BIOS classes
2. **Registry Layer**: Direct HKLM modifications for system identifier management
3. **Driver Integration**: Kernel-level access for persistent storage modifications
4. **Network Stack**: NDIS layer integration for adapter configuration
5. **PCI System**: Hardware descriptor management via SetupAPI
6. **Evasion Techniques**: Advanced pattern management (compiled components)

### Development Philosophy
Core management logic remains open-source on GitHub, while advanced system integration techniques stay compiled to maintain effectiveness against detection systems. This balanced approach allows community auditing while preserving operational integrity.

## 📥 Installation & Usage

### Recommended Source
**For optimal compatibility and verification:**
- **[hwidspoof.net](https://hwidspoof.net)** - Official distribution with verified checksums
- **Security Note**: Avoid third-party distributions to ensure file integrity

### Alternative Access
- [GitHub Releases](https://github.com/hwspf/hwidtool/releases) - Source components and documentation
- Compiled binaries include both open and proprietary modules

### Compilation Considerations
**This project requires specialized build environments due to:**
- Complex Cargo dependency graphs
- Custom build toolchains and scripts
- Advanced Rust feature requirements
- Specific compiler optimization profiles
- Multiple private code repositories

Unless experienced with Rust ecosystem and Windows driver development, pre-compiled binaries are recommended.

### Usage Protocol
1. **Acquire** the latest build from [hwidspoof.net](https://hwidspoof.net)
2. **Execute** the application
3. **Load configuration** via the interface
4. **Apply modifications** through the management interface
5. **System restart** for full identifier propagation
6. **Verification** using standard system information tools

## 🔍 Project Components

### Open Source (MIT Licensed)
- Core hardware management framework
- Network adapter randomization modules
- Registry management utilities
- Basic driver interface implementations
- WMI integration framework
- Configuration management system

### Compiled Components
- Advanced pattern management systems
- Kernel protection system integration
- Temporal algorithm implementations
- Debug/VM environment detection systems
- Signature management techniques
- Driver verification integration

## ⚠️ Important Considerations

### System Requirements
- **Operating System**: Windows 10 (1909+) or Windows 11 (64-bit)
- **Architecture**: x64 exclusively
- **Runtime**: None required (statically compiled)
- **Permissions**: Administrative privileges for system-level operations

### Security Notes
- Some security software may flag system modification tools generically
- Windows Defender may require exception configuration for operation
- Always verify file integrity using provided SHA-256 checksums
- Execute only from trusted sources and original distributions
- Current detection rates remain exceptionally low due to proper coding standards

### Responsible Usage
Developed by gamers for privacy and hardware management. Intended for:
- Hardware privacy management
- System identifier research and development
- Educational purposes in system programming
- Hardware access restoration scenarios

We encourage responsible use that respects gaming ecosystems while protecting user hardware autonomy.

## 📊 Technical Specifications

### Build Configuration
```toml
[package]
name = "hardwaretool"
version = "1.4.2"
edition = "2021"

```

### Performance Characteristics
- **Execution Time**: < 5 seconds for full configuration (including load time)
- **Memory Footprint**: < 50 MB during active operation
- **Persistence Model**: Non-resident after execution completion
- **System Impact**: Targeted registry modifications only

## 🤝 Contribution Guidelines
Open source components welcome community input through:
- Issue reporting for public modules
- Documentation improvements and translations
- Build system enhancements
- Cross-platform compatibility adjustments

Advanced system integration components remain proprietary to maintain effectiveness against detection systems.

## 📄 Licensing Information
- **Core Framework**: MIT License (GitHub repository)
- **Advanced Components**: Proprietary distribution rights
- **Binary Distribution**: Free for personal, non-commercial use

## 🔗 Resources
- **Official Site**: [hwidspoof.net](https://hwidspoof.net)
- **GitHub Repository**: [github.com/hwspf/hwidtool/](https://github.com/hwspf/hwidtool/)
- **Security Analysis**: [VirusTotal Report](https://www.virustotal.com/gui/file/c1e16a1d5ffaef5835e343862520062a118ebf815fe37ab6beac92390881b9bb/detection)
- **Support Contact**: message@hwidspoof.net

## ⚡ Quick Configuration
For straightforward hardware management:
1. Visit [hwidspoof.net](https://hwidspoof.net)
2. Download the latest release build
3. Execute, load configuration profile
4. System restart for changes to propagate
5. Hardware identifiers are now managed according to configuration

---

*Developed by hardware privacy advocates. Open where practical, effective where needed.*
