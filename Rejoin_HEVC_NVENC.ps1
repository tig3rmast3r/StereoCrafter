# --- CONFIG ---
$DirSbs  = "G:\work\sbs"
$Pattern = "*_sbs.mp4"
$Out     = "G:\work\final_sbs_1080_hevc_nvenc.mp4"
$FFMPEG = "path\to\ffmpeg.exe"
$Preset = "p7"
$CQ     = 16

$files = Get-ChildItem -LiteralPath $DirSbs -File -Filter $Pattern | Sort-Object Name
if (-not $files) { throw "Nessun file trovato in $DirSbs con pattern $Pattern" }

# concat list su STDIN (con newline finale)
$list = ($files | ForEach-Object {
  $p = $_.FullName -replace "'", "\'"
  "file '$p'"
}) -join "`n"
$list += "`n"

$vf = "pad=iw:1080:0:(1080-ih)/2:black"
# remove vf is it's already 1080p 

$ffArgs = @(
  "-hide_banner","-y",
  "-f","concat","-safe","0",
  "-protocol_whitelist","file,pipe,crypto,data",
  "-i","pipe:0",
  "-vf",$vf,
  "-c:v","hevc_nvenc",
  "-preset",$Preset,
  "-rc","vbr","-cq","$CQ","-b:v","0",
  "-pix_fmt","yuv420p",
  "-an",
  "-multipass","fullres","-spatial_aq","1","-temporal_aq","1","-aq-strength","12","-rc-lookahead","32","-bf","3",
  $Out
)

$list | & $FFMPEG @ffArgs
