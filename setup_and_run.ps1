Write-Host "🌀 Step 1: Creating virtual environment..."
python -m venv venv
if (!$?) { Write-Error "❌ Failed to create virtual environment"; exit }

Write-Host "🚀 Step 2: Activating virtual environment..."
$env:VENV_PATH = "$PSScriptRoot\venv\Scripts\Activate.ps1"
if (Test-Path $env:VENV_PATH) {
    & $env:VENV_PATH
} else {
    Write-Error "❌ Activation script not found"; exit
}

Write-Host "📦 Step 3: Installing dependencies..."
pip install -r requirements.txt
if (!$?) { Write-Error "❌ Failed to install requirements"; exit }

Write-Host "▶️ Step 4: Running main.py..."
python main.py
if (!$?) { Write-Error "❌ Failed to execute main.py"; exit }

Write-Host "🌐 Step 5: Launching Streamlit dashboard..."
streamlit run deduplicator.py
