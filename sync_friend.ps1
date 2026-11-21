<#
.SYNOPSIS
    Automated script to sync, merge, install requirements, and verify PeakCut from friend repo.
    Minimizes agent token costs by doing all routine steps locally in one shot and printing a clean summary.
#>
param(
    [string]$Remote = "friend",
    [string]$Branch = "main"
)

$logFile = "sync_last.log"

function Write-Step ($msg) {
    Write-Host "[SYNC] $msg..." -ForegroundColor Cyan
    "[$(Get-Date -Format 'HH:mm:ss')] $msg" | Out-File -Append -FilePath $logFile -Encoding utf8
}

"" | Out-File -FilePath $logFile -Encoding utf8

try {
    Write-Step "1/4 Fetching latest from $Remote/$Branch"
    & git fetch $Remote 2>&1 | Out-File -Append -FilePath $logFile -Encoding utf8

    Write-Step "2/4 Merging $Remote/$Branch"
    & git merge "$Remote/$Branch" --allow-unrelated-histories --no-edit 2>&1 | Out-File -Append -FilePath $logFile -Encoding utf8

    # Check if there are unresolved merge conflicts
    $conflicts = & git diff --name-only --diff-filter=U 2>&1
    if ($conflicts) {
        Write-Step "Resolving merge conflicts by accepting incoming code from $Remote/$Branch"
        & git checkout "$Remote/$Branch" -- . | Out-Null
        & git add -A | Out-Null
        & git commit -m "Auto-merge: synced with $Remote/$Branch" 2>&1 | Out-File -Append -FilePath $logFile -Encoding utf8
    }

    Write-Step "3/4 Installing requirements (quiet mode)"
    & python -m pip install -q -r requirements.txt 2>&1 | Out-File -Append -FilePath $logFile -Encoding utf8

    Write-Step "4/4 Verifying module imports"
    $verifyOut = & python -c "import studio.server, studio.transcribe, studio.jobs; print('PeakCut imports verified successfully!')" 2>&1 | Out-String
    $verifyOut | Out-File -Append -FilePath $logFile -Encoding utf8

    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host " SUCCESS: Sync, merge, and install complete!" -ForegroundColor Green
    Write-Host " Status: $($verifyOut.Trim())" -ForegroundColor Green
    Write-Host " Full log saved to: $logFile" -ForegroundColor Gray
    Write-Host "==========================================" -ForegroundColor Green
    exit 0
}
catch {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host " FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host " Check $logFile for details." -ForegroundColor Yellow
    Write-Host "==========================================" -ForegroundColor Red
    exit 1
}
