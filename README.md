# DiRT Showdown Music Manager (Windows)

A Windows-only tool for replacing DiRT Showdown WIM music streams
with custom audio while preserving original loudness characteristics
and byte-accurate file sizing.

Author: David Holt  
License: MIT  
Contact: creativesamurai1982@gmail.com  

---

## Overview

This tool allows you to:

• Replace DiRT Showdown WIM music streams  
• Preserve original loudness (LUFS / True Peak / LRA)  
• Inject audio with exact byte-size matching  
• Automatically create backups before modification  
• Use musicPlayer.xml for intelligent stream filtering  
• Generate reproducible injection plans  
• Create distributable ZIP music packs  

This tool does NOT modify WIP streams. Only WIM streams are injected.

---

## Windows Requirements

• Windows 10 or Windows 11  
• Python 3.10 or newer  
• No additional external installs required (FFmpeg + vgmstream included)

---

## How To Run

1. Open Command Prompt inside the project folder.
2. Run:

   python app\showdown_music_manager_windows.py

Alternatively, create a simple batch file:

   run_windows.bat

Containing:

   @echo off
   python app\showdown_music_manager_windows.py
   pause

---

## What The Tool Does

• Scans DIC files
• Detects WIM streams
• Matches original loudness using ffmpeg loudnorm
• Converts custom audio to 48kHz stereo mulaw
• Pads or trims to match original byte size
• Injects safely after backup
• Provides progress bars for long operations
• Cleans up temporary WAV files automatically

---

## Safety

Before any injection:

• Original WIM files are backed up into:
  backups/

You can restore all backups at any time using the built-in restore option.

---

## Important Notes

• This tool is Windows-only.
• Do not use with modified DIC files unless you understand the structure.
• Always keep backups of your game install.

---

## Legal Notice

This project is not affiliated with Codemasters or Electronic Arts.

DiRT Showdown is a trademark of its respective owners.

This tool modifies game files at your own risk.
