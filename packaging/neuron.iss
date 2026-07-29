; NEURON — Windows installer (Inno Setup 6).
; Build:  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\neuron.iss
; Produces dist\installer\NEURON-Setup-<ver>.exe
;
; Per-user install (no admin / no UAC): program -> %LOCALAPPDATA%\Programs\NEURON,
; writable state (config, log, model slice) -> %LOCALAPPDATA%\NEURON (set in the app code).
; Installs the tray app, Start-menu + optional desktop shortcut, optional auto-start,
; and on uninstall deregisters the node + deletes its slice/config.

#define AppName "NEURON"
#define AppVersion "0.15.0"
#define AppExe "neuron-agent.exe"

[Setup]
AppId={{A7E3C9F1-2B4D-4E6A-8C1F-9D0E5B7A3C21}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=NEURON Labs
AppSupportURL=https://github.com/raman011sharma-code/neuron-network
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist\installer
OutputBaseFilename=NEURON-Setup-{#AppVersion}
SetupIconFile=neuron.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
InfoBeforeFile=DISCLOSURE.txt
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "startup"; Description: "Start NEURON automatically when I sign in"; GroupDescription: "Startup:"; Flags: unchecked

[Files]
; the whole PyInstaller onedir output (neuron-agent.exe + _internal\)
Source: "..\dist\neuron-agent\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"; IconFilename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; IconFilename: "{app}\{#AppExe}"; Tasks: desktopicon
Name: "{userstartup}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: startup

[Run]
Filename: "{app}\{#AppExe}"; Description: "Start {#AppName} now"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; deregister the node and delete its slice/config before the files are removed
Filename: "{app}\{#AppExe}"; Parameters: "--deregister"; Flags: runhidden; RunOnceId: "DeregisterNode"
