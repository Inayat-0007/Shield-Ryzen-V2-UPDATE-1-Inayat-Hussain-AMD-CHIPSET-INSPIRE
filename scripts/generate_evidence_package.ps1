# Shield-Ryzen V2 - Evidence Generation Script (Windows)
# Task 13.4

Write-Host "Generating Comprehensive Evidence Package..." -ForegroundColor Cyan

$ROOT = $PSScriptRoot\..
$OUT_DIR = "$ROOT\evidence_package"

if (Test-Path $OUT_DIR) { Remove-Item $OUT_DIR -Recurse -Force }
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

# 1. Run Validation
Write-Host "Running Unit Tests..."
try {
    pytest $ROOT\tests -v --tb=short > "$OUT_DIR\test_results.txt" 
} catch {
    Write-Host "Tests failed or pytest missing" -ForegroundColor Red
}

# 2. Run Benchmarks (if present)
# Assumes benchmarks already run or run them now?
# Script says "Generates evidence package containing... benchmarks"
# We copy existing ones.
if (Test-Path "$ROOT\benchmarks") {
    Copy-Item "$ROOT\benchmarks" "$OUT_DIR\benchmarks" -Recurse
}

# 3. Documentation
Copy-Item "$ROOT\README.md" "$OUT_DIR\"
Copy-Item "$ROOT\MODEL_CARD.md" "$OUT_DIR\" -ErrorAction SilentlyContinue
Copy-Item "$ROOT\docs" "$OUT_DIR\docs" -Recurse

# 4. Logs (Audit Trail)
if (Test-Path "$ROOT\logs\shield_audit.jsonl") {
    Copy-Item "$ROOT\logs\shield_audit.jsonl" "$OUT_DIR\audit_trail_sample.jsonl"
}

# 5. Network Proof (Simulated or Real)
if (Test-Path "$ROOT\network_during_inference.txt") {
    Copy-Item "$ROOT\network_during_inference.txt" "$OUT_DIR\"
} else {
    # Generate network proof now?
    # netstat -an | findstr "ESTABLISHED" > ...
    netstat -an > "$OUT_DIR\network_snapshot.txt"
}

# 6. Compliance
if (Test-Path "$ROOT\docs\COMPLIANCE.md") {
    Copy-Item "$ROOT\docs\COMPLIANCE.md" "$OUT_DIR\"
}

# Zip it
$ZIP_FILE = "$ROOT\ShieldRyzenV2_Evidence_$(Get-Date -Format 'yyyyMMdd').zip"
Compress-Archive -Path "$OUT_DIR\*" -DestinationPath $ZIP_FILE -Force

Write-Host "EVIDENCE PACKAGE CREATED: $ZIP_FILE" -ForegroundColor Green
