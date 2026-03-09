param(
    [ValidateSet("setup", "demo", "train", "score", "update", "api", "dashboard", "app", "test-api")]
    [string]$Step = "demo",
    [string]$Python = "python",
    [int]$Port = 8000,
    [switch]$InstallDeps
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Run-Process {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host ">> $Executable $($Arguments -join ' ')" -ForegroundColor Cyan
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao executar: $Executable $($Arguments -join ' ')"
    }
}

function Setup-Deps {
    Run-Process -Executable $Python -Arguments @("-m", "pip", "install", "-U", "pip", "setuptools", "wheel")
    Run-Process -Executable $Python -Arguments @("-m", "pip", "install", "-r", "requirements.txt")
}

function Generate-Sample {
    Run-Process -Executable $Python -Arguments @("-m", "src.data.generate_sample")
}

function Train-Model {
    Run-Process -Executable $Python -Arguments @("src/main.py", "train", "--train-path", "data/raw/churn_train.csv")
}

function Score-Model {
    Run-Process -Executable $Python -Arguments @("src/main.py", "score", "--input-path", "data/raw/churn_score.csv")
}

function Update-Model {
    Run-Process -Executable $Python -Arguments @("src/main.py", "update", "--labeled-path", "data/raw/churn_feedback.csv")
}

function Run-Api {
    $env:PORT = "$Port"
    Run-Process -Executable $Python -Arguments @("src/api/app.py")
}

function Run-Dashboard {
    $streamlitDir = Join-Path $env:USERPROFILE ".streamlit"
    if (-not (Test-Path $streamlitDir)) {
        New-Item -Path $streamlitDir -ItemType Directory | Out-Null
    }

    $credentialsContent = @"
[general]
email = ""
"@
    $credentialsPath = Join-Path $streamlitDir "credentials.toml"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($credentialsPath, $credentialsContent, $utf8NoBom)

    $configContent = @"
[browser]
gatherUsageStats = false
"@
    $configPath = Join-Path $streamlitDir "config.toml"
    [System.IO.File]::WriteAllText($configPath, $configContent, $utf8NoBom)

    $env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
    Run-Process -Executable $Python -Arguments @(
        "-m", "streamlit", "run", "src/dashboard/app.py",
        "--server.port", "$Port",
        "--browser.gatherUsageStats", "false"
    )
}

function Test-Api {
    $healthUrl = "http://127.0.0.1:$Port/health"
    $predictUrl = "http://127.0.0.1:$Port/predict"
    $payload = @{
        customers = @(
            @{
                customer_id = 1
                tenure_months = 12
                recency_days = 45
                purchase_frequency_90d = 3
                avg_ticket = 120.5
                support_tickets_90d = 2
                payment_delay_days = 5
                failed_payments_90d = 1
                login_days_30d = 8
                engagement_30d = 0.27
                usage_ratio = 0.34
                nps_score = 5
                satisfaction_score = 3
                plan_value = 99.9
                plan_type = "padrao"
                contract_type = "mensal"
                payment_method = "cartao"
                region = "sudeste"
            }
        )
    } | ConvertTo-Json -Depth 6

    Write-Host ""
    Write-Host ">> GET $healthUrl" -ForegroundColor Cyan
    $health = Invoke-RestMethod -Uri $healthUrl -Method Get
    Write-Host ($health | ConvertTo-Json -Compress)

    Write-Host ""
    Write-Host ">> POST $predictUrl" -ForegroundColor Cyan
    $predict = Invoke-RestMethod -Uri $predictUrl -Method Post -ContentType "application/json" -Body $payload
    Write-Host ($predict | ConvertTo-Json -Compress)
}

Write-Host "Step: $Step | Python: $Python | Port: $Port" -ForegroundColor Yellow

if ($InstallDeps -or $Step -eq "setup") {
    Setup-Deps
}

switch ($Step) {
    "setup" {
        Write-Host "`nDependências instaladas." -ForegroundColor Green
    }
    "demo" {
        Generate-Sample
        Train-Model
        Score-Model
        Update-Model
        Write-Host "`nDemo concluída." -ForegroundColor Green
        Write-Host "Saídas principais:" -ForegroundColor Green
        Write-Host "- models/churn_model.joblib"
        Write-Host "- reports/model_metrics.json"
        Write-Host "- data/processed/churn_scores.csv"
    }
    "train" {
        Train-Model
    }
    "score" {
        Score-Model
    }
    "update" {
        Update-Model
    }
    "api" {
        Run-Api
    }
    "dashboard" {
        Run-Dashboard
    }
    "app" {
        Run-Dashboard
    }
    "test-api" {
        Test-Api
    }
}
