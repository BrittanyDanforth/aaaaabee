//! HWID Spoofer — GUI + CLI entry (upstream hwidtool layout).

use std::sync::{Arc, Mutex};

use iced::{Application, Command, Element, Settings};

use hwid_spoof::{
    apply_spoof_modules, is_admin_elevated, load_saved_snapshots, print_aba_next_step,
    print_admin_status, print_current_hwid_report, print_terminal_result_box,
    print_verification_to_terminal, step_marker_exists, verify_from_saved_files, SpoofOptions,
    HwidSnapshot, VERSION,
};

fn main() -> iced::Result {
    let args: Vec<String> = std::env::args().collect();
    let headless = args.iter().any(|a| a == "--apply-all" || a == "--no-gui");
    let verify_marker = args.iter().any(|a| a == "--verify-marker");
    let report_only = args.iter().any(|a| a == "--report");
    let verify_last = args.iter().any(|a| a == "--verify-last");
    let check_admin = args.iter().any(|a| a == "--check-admin");

    if check_admin {
        return run_check_admin();
    }

    if report_only {
        return run_report();
    }

    if verify_last {
        return run_verify_last();
    }

    if verify_marker {
        if step_marker_exists() {
            println!("[hwspoof] Step marker present - ABA may proceed.");
            std::process::exit(0);
        }
        println!("[hwspoof] Step marker missing - run HWID apply first.");
        std::process::exit(1);
    }

    if headless {
        return run_cli_apply();
    }

    App::run(Settings {
        window: iced::window::Settings {
            size: iced::Size::new(920.0, 720.0),
            ..Default::default()
        },
        ..Settings::default()
    })
}

fn run_check_admin() -> iced::Result {
    print_admin_status();
    if is_admin_elevated() {
        std::process::exit(0);
    }
    std::process::exit(1);
}

fn run_report() -> iced::Result {
    print_admin_status();
    if let Err(e) = print_current_hwid_report() {
        eprintln!("[hwspoof] report failed: {e}");
        std::process::exit(1);
    }
    std::process::exit(0);
}

fn run_verify_last() -> iced::Result {
    print_admin_status();
    let options = SpoofOptions::all_enabled();
    match load_saved_snapshots() {
        Ok((before, after)) => match verify_from_saved_files(&options) {
            Ok(report) => {
                println!("[hwspoof] Re-verification from saved BEFORE/AFTER:");
                print_verification_to_terminal(&before, &after, &report);
                print_terminal_result_box(&report);
                if report.acceptable_for_marker() {
                    print_aba_next_step();
                    std::process::exit(0);
                }
                std::process::exit(2);
            }
            Err(e) => {
                eprintln!("[hwspoof] verify failed: {e}");
                std::process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("[hwspoof] No saved BEFORE/AFTER - run --apply-all first. ({e})");
            std::process::exit(1);
        }
    }
}

fn run_cli_apply() -> iced::Result {
    match apply_spoof_modules(&SpoofOptions::all_enabled(), &mut |msg| println!("{msg}")) {
        Ok(_outcome) => {
            print_aba_next_step();
            std::process::exit(0);
        }
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("verification failed") || msg.contains("elevation") {
                if msg.contains("verification failed") {
                    let options = SpoofOptions::all_enabled();
                    if let Ok((b, a)) = load_saved_snapshots() {
                        if let Ok(r) = verify_from_saved_files(&options) {
                            print_verification_to_terminal(&b, &a, &r);
                            print_terminal_result_box(&r);
                        }
                    }
                }
                if msg.contains("elevation") {
                    eprintln!("[hwspoof] {msg}");
                    std::process::exit(1);
                }
                std::process::exit(2);
            }
            eprintln!("[hwspoof] FAILED: {msg}");
            std::process::exit(1);
        }
    }
}

#[derive(Debug, Clone)]
struct App {
    options: SpoofOptions,
    log: Arc<Mutex<Vec<String>>>,
    status: String,
    busy: bool,
}

impl App {
    fn push_log(&self, line: impl Into<String>) {
        if let Ok(mut g) = self.log.lock() {
            g.push(line.into());
        }
    }
}

#[derive(Debug, Clone)]
enum Message {
    ToggleDisk(bool),
    ToggleMac(bool),
    ToggleBoard(bool),
    ToggleUuid(bool),
    ToggleCpu(bool),
    ToggleGpu(bool),
    TogglePci(bool),
    ToggleReg(bool),
    ToggleVol(bool),
    ToggleNet(bool),
    ToggleBios(bool),
    ToggleAcpi(bool),
    ToggleWmi(bool),
    ToggleDriver(bool),
    ToggleEmbedded(bool),
    EnableAll,
    ReportCurrent,
    VerifyLast,
    Apply,
    ReportDone(Result<String, String>),
    ApplyFinished(Result<(), String>),
}

impl Application for App {
    type Executor = iced::executor::Default;
    type Message = Message;
    type Theme = iced::Theme;
    type Flags = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        let mut log_lines = vec![
            "HWID Spoofer - use Run_As_Admin.bat if Admin PowerShell fails.".to_string(),
            "Full BEFORE/AFTER prints in this log on Apply.".to_string(),
        ];
        if !is_admin_elevated() {
            log_lines.push("WARN: Not elevated.".to_string());
        }
        let app = Self {
            options: SpoofOptions::all_enabled(),
            log: Arc::new(Mutex::new(log_lines)),
            status: "Ready".to_string(),
            busy: false,
        };
        (app, Command::none())
    }

    fn title(&self) -> String {
        format!("HWID Spoofer v{VERSION}")
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::ToggleDisk(v) => self.options.disk = v,
            Message::ToggleMac(v) => self.options.mac = v,
            Message::ToggleBoard(v) => self.options.motherboard = v,
            Message::ToggleUuid(v) => self.options.system_uuid = v,
            Message::ToggleCpu(v) => self.options.cpu = v,
            Message::ToggleGpu(v) => self.options.gpu = v,
            Message::TogglePci(v) => self.options.pci = v,
            Message::ToggleReg(v) => self.options.registry_clean = v,
            Message::ToggleVol(v) => self.options.volume = v,
            Message::ToggleNet(v) => self.options.network = v,
            Message::ToggleBios(v) => self.options.bios = v,
            Message::ToggleAcpi(v) => self.options.acpi = v,
            Message::ToggleWmi(v) => self.options.wmi = v,
            Message::ToggleDriver(v) => self.options.driver_hooks = v,
            Message::ToggleEmbedded(v) => self.options.run_embedded_module = v,
            Message::EnableAll => self.options = SpoofOptions::all_enabled(),
            Message::ReportCurrent => {
                self.status = "Capturing…".to_string();
                return Command::perform(
                    async {
                        std::thread::spawn(|| -> Result<String, String> {
                            let snap = HwidSnapshot::capture("current");
                            let mut text = snap.to_report_text();
                            if !snap.admin_elevated {
                                text.push_str(
                                    "\n[hwspoof] WARN: not elevated — reads may show (read failed).",
                                );
                            }
                            Ok(text)
                        })
                        .join()
                        .unwrap_or_else(|_| Err("report thread panicked".to_string()))
                    },
                    Message::ReportDone,
                );
            }
            Message::VerifyLast => {
                self.status = "Verifying…".to_string();
                let options = self.options.clone();
                return Command::perform(
                    async move {
                        std::thread::spawn(move || -> Result<String, String> {
                            let (before, after) =
                                load_saved_snapshots().map_err(|e| e.to_string())?;
                            let report =
                                verify_from_saved_files(&options).map_err(|e| e.to_string())?;
                            Ok(format!(
                                "{}{}",
                                hwid_spoof::format_verification_report(&before, &after, &report),
                                format_terminal_box_text(&report)
                            ))
                        })
                        .join()
                        .unwrap_or_else(|_| Err("verify thread panicked".to_string()))
                    },
                    Message::ReportDone,
                );
            }
            Message::ReportDone(result) => match result {
                Ok(text) => {
                    self.status = "Done".to_string();
                    for line in text.lines() {
                        self.push_log(line);
                    }
                }
                Err(e) => {
                    self.status = format!("Failed: {e}");
                    self.push_log(format!("✗ {e}"));
                }
            },
            Message::Apply => {
                if self.busy {
                    return Command::none();
                }
                self.busy = true;
                self.status = "Applying…".to_string();
                let options = self.options.clone();
                let log_sink = Arc::clone(&self.log);
                return Command::perform(
                    async move {
                        std::thread::spawn(move || {
                            let mut log = |msg: &str| {
                                if let Ok(mut g) = log_sink.lock() {
                                    g.push(msg.to_string());
                                }
                            };
                            apply_spoof_modules(&options, &mut log)
                                .map(|_| ())
                                .map_err(|e| e.to_string())
                        })
                        .join()
                        .unwrap_or_else(|_| Err("apply thread panicked".to_string()))
                    },
                    Message::ApplyFinished,
                );
            }
            Message::ApplyFinished(result) => {
                self.busy = false;
                match result {
                    Ok(()) => self.status = "Done - see log above".to_string(),
                    Err(e) => {
                        self.status = format!("Failed: {e}");
                        self.push_log(format!("✗ {e}"));
                    }
                }
            }
        }
        Command::none()
    }

    fn view(&self) -> Element<'_, Message> {
        use iced::widget::{button, checkbox, column, container, row, scrollable, text, Space};

        let header = column![
            text(format!("HWID Spoofer v{VERSION}")).size(26),
            text("Use Run_As_Admin.bat if elevation fails from PowerShell").size(14),
            text(format!("Status: {}", self.status)).size(14),
        ]
        .spacing(6);

        let toggles = column![
            text("Modules").size(18),
            checkbox("Disk serial", self.options.disk).on_toggle(Message::ToggleDisk),
            checkbox("Volume serial (C:)", self.options.volume).on_toggle(Message::ToggleVol),
            checkbox("MAC addresses", self.options.mac).on_toggle(Message::ToggleMac),
            checkbox("Motherboard / SMBIOS", self.options.motherboard)
                .on_toggle(Message::ToggleBoard),
            checkbox("System UUID", self.options.system_uuid).on_toggle(Message::ToggleUuid),
            checkbox("CPU ID", self.options.cpu).on_toggle(Message::ToggleCpu),
            checkbox("GPU ID", self.options.gpu).on_toggle(Message::ToggleGpu),
            checkbox("PCI hide", self.options.pci).on_toggle(Message::TogglePci),
            checkbox("Registry cleanup", self.options.registry_clean)
                .on_toggle(Message::ToggleReg),
            checkbox("Network stack", self.options.network).on_toggle(Message::ToggleNet),
            checkbox("BIOS info", self.options.bios).on_toggle(Message::ToggleBios),
            checkbox("ACPI override", self.options.acpi).on_toggle(Message::ToggleAcpi),
            checkbox("WMI hooks", self.options.wmi).on_toggle(Message::ToggleWmi),
            checkbox("Driver hooks (stub)", self.options.driver_hooks)
                .on_toggle(Message::ToggleDriver),
            checkbox("Embedded private module", self.options.run_embedded_module)
                .on_toggle(Message::ToggleEmbedded),
        ]
        .spacing(4);

        let log_text = self
            .log
            .lock()
            .map(|g| g.join("\n"))
            .unwrap_or_default();

        let controls = row![
            button("Report current IDs").on_press(Message::ReportCurrent),
            button("Verify last run").on_press(Message::VerifyLast),
            button("Enable all").on_press(Message::EnableAll),
            button("Apply selected").on_press_maybe((!self.busy).then_some(Message::Apply)),
        ]
        .spacing(8);

        let layout = column![
            header,
            Space::with_height(8),
            row![
                container(toggles.width(iced::Length::FillPortion(1))).padding(10),
                container(
                    column![
                        text("Log").size(18),
                        scrollable(text(log_text).size(13)).height(iced::Length::Fill),
                    ]
                    .spacing(8),
                )
                .width(iced::Length::FillPortion(2))
                .padding(10),
            ]
            .spacing(12)
            .height(iced::Length::Fill),
            controls,
        ]
        .spacing(10)
        .padding(16);

        container(layout).into()
    }
}

fn format_terminal_box_text(report: &hwid_spoof::VerificationReport) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(s);
    let _ = writeln!(s, "============================================================");
    let _ = match report.verdict {
        hwid_spoof::OverallVerdict::SuccessChanged => {
            writeln!(s, "  RESULT: SUCCESS - changed {}", report.changed_count)
        }
        hwid_spoof::OverallVerdict::FailedNoChange => writeln!(s, "  RESULT: FAILED"),
        hwid_spoof::OverallVerdict::Inconclusive => writeln!(s, "  RESULT: INCONCLUSIVE"),
        hwid_spoof::OverallVerdict::ReportOnly => writeln!(s, "  RESULT: REPORT"),
    };
    let _ = writeln!(s, "============================================================");
    s
}
