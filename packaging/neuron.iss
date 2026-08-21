; NEURON — Windows installer (Inno Setup 6).
; Build:  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\neuron.iss
; Produces dist\installer\NEURON-Setup-<ver>.exe
;
; Per-user install (no admin / no UAC): program -> %LOCALAPPDATA%\Programs\NEURON,
; writable state (config, log, model slice) -> %LOCALAPPDATA%\NEURON (set in the app code).
; Installs the tray app, Start-menu + optional desktop shortcut, optional auto-start,
; and on uninstall deregisters the node + deletes its slice/config.

#define AppName "NEURON"
#define AppVersion "0.20.16"
#define AppExe "neuron-agent.exe"

[Setup]
AppId={{A7E3C9F1-2B4D-4E6A-8C1F-9D0E5B7A3C21}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=NEURON Labs
AppSupportURL=https://github.com/neuron-network-ai/neuron
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
; STOP THE RUNNING APP BEFORE OVERWRITING IT ([P24], [P46]).
; Installing over a live agent is how a partial copy happens: the running exe holds files open,
; those specific files are skipped, and the install completes "successfully" with a directory
; that is half one build and half another. On 2026-08-17 that left an index.html asking for a
; bundle that was never copied — a blank page, with every other signal green.
; Restart Manager asks the app to close rather than killing it, so a node mid-request finishes
; rather than dropping the chain it is serving.
CloseApplications=yes
CloseApplicationsFilter=*.exe,*.dll,*.pyd
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"
; Checked by DEFAULT. A node that does not come back after a reboot leaves the network on its
; first power cycle and never returns, and the earn-rate the installer promised quietly becomes
; zero ([P21]). A stranger will not go hunting for this checkbox, so the safe default is on;
; unticking it is still one click for anyone who wants to start it by hand.
Name: "startup"; Description: "Start NEURON automatically when I sign in"; GroupDescription: "Startup:"

[InstallDelete]
; CLEAR THE CONTENT-HASHED BUNDLES BEFORE COPYING ([P46]).
; Vite names every bundle with a content hash, so index.html references a DIFFERENT filename on
; each build, and `ignoreversion` overwrites same-named files while NEVER pruning ones that
; vanished. Left alone, an upgrade merges two builds: last build's index-OLD.js sits beside this
; build's index.html, which asks for index-NEW.js. Both files look fine; the page renders
; nothing.
; Deleting the directory first makes the copy a REPLACE rather than a merge, which is the only
; state in which the reference and the file are guaranteed to agree. Scoped to `assets` alone —
; the rest of {app} is version-stable filenames that ignoreversion handles correctly, and a
; broader delete would throw away files this installer does not put back.
Type: filesandordirs; Name: "{app}\_internal\ui\static\app\assets"

[Files]
; the whole PyInstaller onedir output (neuron-agent.exe + _internal\)
Source: "..\dist\neuron-agent\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
; Licence obligations. NEURON's own terms, plus the notices for every bundled third-party
; component — MIT, BSD and Apache-2.0 all require their text to travel with a redistribution,
; and the installer IS a redistribution. Both are installed next to the exe and removed with it.
Source: "..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion
Source: "..\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"; IconFilename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; IconFilename: "{app}\{#AppExe}"; Tasks: desktopicon
Name: "{userstartup}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: startup

[Run]
Filename: "{app}\{#AppExe}"; Description: "Start {#AppName} now"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; deregister the node and delete its slice/config before the files are removed
Filename: "{app}\{#AppExe}"; Parameters: "--deregister"; Flags: runhidden; RunOnceId: "DeregisterNode"
