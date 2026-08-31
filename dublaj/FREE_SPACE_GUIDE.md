# How to Free Up Space on Windows

You need **40-50 GB free** to run Dublaj. Here's how to free up space:

## Quick Wins (Safe & Easy)

### 1. Empty Recycle Bin
- Right-click Recycle Bin on desktop → Empty Recycle Bin

### 2. Disk Cleanup
```
1. Press Win + R
2. Type: cleanmgr
3. Select C: drive
4. Check all boxes (especially "Previous Windows installations" if available)
5. Click OK
```
**Can free: 5-20 GB**

### 3. Clear Temp Files
```powershell
# Run as Administrator
Remove-Item -Path "$env:TEMP\*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "C:\Windows\Temp\*" -Recurse -Force -ErrorAction SilentlyContinue
```
**Can free: 2-10 GB**

### 4. Uninstall Unused Programs
```
Settings → Apps → Apps & features → Sort by Size
```
Remove:
- Old games
- Unused software
- Trial programs

**Can free: 5-50 GB**

### 5. Clear Browser Cache
- **Chrome**: Settings → Privacy → Clear browsing data → All time
- **Firefox**: Settings → Privacy → Clear Data
- **Edge**: Settings → Privacy → Clear browsing data

**Can free: 1-5 GB**

### 6. Move Large Files
Move to external drive or cloud:
- Videos in `Documents`, `Downloads`, `Desktop`
- Large game installations
- Old project files

**Can free: 10-100 GB**

### 7. Windows Storage Sense
```
Settings → System → Storage → Configure Storage Sense
```
Enable automatic cleanup

### 8. Check Large Folders
```powershell
Get-ChildItem C:\ -Recurse -Directory | 
  Where-Object {$_.GetFiles().Count -gt 0} | 
  Select-Object FullName, @{Name="SizeGB";Expression={
    [math]::Round((Get-ChildItem $_.FullName -Recurse -File | 
    Measure-Object -Property Length -Sum).Sum / 1GB, 2)
  }} | 
  Sort-Object SizeGB -Descending | 
  Select-Object -First 20
```

Common space hogs:
- `C:\Users\{YourName}\AppData\Local` (app caches)
- `C:\Program Files` (old software)
- `C:\Windows\WinSxS` (Windows components - USE CAUTION)

---

## After Freeing Space

Check available space:
```powershell
Get-PSDrive C | Select-Object @{Name="FreeGB";Expression={[math]::Round($_.Free/1GB,2)}}
```

You need **at least 40 GB free** to proceed with Docker installation.

---

## Alternative: Use External Drive

If you have an external drive with space, you can move Docker's storage location:

1. Install Docker Desktop
2. Settings → Resources → Advanced
3. Change "Disk image location" to external drive
4. Click "Apply & Restart"

**Note**: External drive must be formatted as NTFS and stay connected while Docker runs.
