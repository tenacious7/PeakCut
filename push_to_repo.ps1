<#
.SYNOPSIS
    Commits modified/untracked files feature by feature or file by file with clear messages,
    then pushes all commits to your remote repository (origin/main).
    Minimizes agent token cost.
#>
param(
    [string]$Remote = "origin",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Continue"

function Write-Step ($msg) {
    Write-Host "[PUSH] $msg" -ForegroundColor Cyan
}

# 1. Check for uncommitted changes
$statusItems = git status --porcelain
if ($statusItems) {
    Write-Step "Found uncommitted changes. Staging and committing feature-by-feature..."
    
    # Meaningful feature categorization
    $categoryRules = @{
        "studio/transcribe" = "feat(transcribe): speech recognition and ASR updates"
        "studio/server"     = "feat(server): backend server API and routing updates"
        "studio/jobs"       = "feat(jobs): worker queue and job pipeline enhancements"
        "studio/render"     = "feat(render): video rendering, layout, and caption processing"
        "studio/media"      = "feat(media): video download, audio extraction, and probing"
        "studio/qc"         = "feat(qc): quality check and video metrics updates"
        "studio/ai"         = "feat(ai): AI prompt and provider integration updates"
        "web/"              = "feat(ui): frontend interface and styling updates"
        "tests/"            = "test: update unit and integration test suite"
        "requirements.txt"  = "build: update project dependencies"
        ".gitignore"        = "chore: update git ignore rules"
    }

    # Group changed files
    $lines = $statusItems -split "`r?`n" | Where-Object { $_.Trim() -ne "" }
    $handledFiles = @()

    foreach ($pattern in $categoryRules.Keys) {
        $matched = @()
        foreach ($line in $lines) {
            $filePath = $line.Substring(3).Trim()
            if ($filePath -like "*$pattern*" -and $handledFiles -notcontains $filePath) {
                $matched += $filePath
            }
        }
        if ($matched.Count -gt 0) {
            $msg = $categoryRules[$pattern]
            Write-Step "Committing feature: $msg"
            foreach ($f in $matched) {
                git add $f | Out-Null
                $handledFiles += $f
            }
            git commit -m "$msg" | Out-Null
        }
    }

    # Commit any remaining files individually
    $remaining = git status --porcelain
    if ($remaining) {
        $remLines = $remaining -split "`r?`n" | Where-Object { $_.Trim() -ne "" }
        foreach ($rLine in $remLines) {
            $file = $rLine.Substring(3).Trim()
            if ($file) {
                Write-Step "Committing individual change: $file"
                git add $file | Out-Null
                $cleanName = [System.IO.Path]::GetFileName($file)
                git commit -m "feat/chore: update $cleanName" | Out-Null
            }
        }
    }
} else {
    Write-Step "Working tree clean, no uncommitted local changes."
}

# 2. Check how many commits are pending push to Remote/Branch
$unpushed = git log "$Remote/$Branch..HEAD" --oneline 2>&1
if ($LASTEXITCODE -eq 0 -and $unpushed) {
    $commitCount = ($unpushed -split "`r?`n" | Where-Object { $_.Trim() -ne "" }).Count
    Write-Host "[PUSH] Found $commitCount commit(s) ready to push to $Remote/$Branch." -ForegroundColor Yellow
}

# 3. Push to remote
Write-Step "Pushing to $Remote $Branch..."
$pushResult = git push $Remote $Branch 2>&1 | Out-String

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host " SUCCESS: All commits pushed to $Remote/$Branch!" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    exit 0
} else {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host " PUSH FAILED: $pushResult" -ForegroundColor Red
    Write-Host "==========================================" -ForegroundColor Red
    exit 1
}
