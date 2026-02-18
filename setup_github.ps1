
$ErrorActionPreference = "SilentlyContinue"
Write-Host "Cleaning up artifacts..."
Remove-Item step*.py
Remove-Item run_output*
Remove-Item test_model_init.py
Remove-Item final_audit.py
Remove-Item *.zip
Remove-Item *.rar
Remove-Item step4_class_test.py

$ErrorActionPreference = "Continue"
Write-Host "Initializing Git..."
git init

# Configure Identity (Local scope to be safe)
git config user.name "Inayat Hussain"
git config user.email "inayat@shield.dev"

# Reset Remote
Write-Host "Setting Remote..."
git remote remove origin 2> $null
git remote add origin https://github.com/Inayat-0007/Shield-Ryzen-V2-UPDATE-1-Inayat-Hussain-AMD-CHIPSET-INSPIRE.git

# Branch
git branch -M main

# Add & Commit
Write-Host "Staging Files..."
git add .
Write-Host "Committing..."
git commit -m "SHIELD-RYZEN V2: The Diamond Tier Protocol 💎 | Authored by Inayat Hussain"

# Push
Write-Host "Pushing to GitHub..."
git push -u origin main

Write-Host "DONE."
