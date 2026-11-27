# CPU smoke test of every training script (1-2 steps each). Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File training\smoke_test.ps1
$ErrorActionPreference = "Stop"
$py = if (Test-Path .\.venv\Scripts\python.exe) { ".\.venv\Scripts\python.exe" } else { "python" }
$steps = @(
  @("-m", "training.build_datasets", "--smoke"),
  @("-m", "training.verify_datasets", "--dir", "training/runs/smoke_datasets"),
  @("-m", "training.train_ppo", "--smoke", "--run-name", "smoke"),
  @("-m", "training.tune_ppo", "--smoke", "--run-name", "smoke"),
  @("-m", "training.evaluate_ppo", "--smoke", "--model", "training/runs/ppo_smoke/smoke/ppo_policy.npz"),
  @("-m", "training.train_sft", "--smoke", "--run-name", "smoke"),
  @("-m", "training.tune_sft", "--smoke", "--run-name", "smoke"),
  @("-m", "training.train_grpo", "--smoke", "--run-name", "smoke"),
  @("-m", "training.evaluate_llm", "--smoke", "--run-name", "smoke"),
  @("-m", "training.fe_benchmark", "--smoke", "--run-name", "smoke")
)
foreach ($s in $steps) {
  Write-Host ">>> python $($s -join ' ')"
  & $py @s
  if ($LASTEXITCODE -ne 0) { throw "failed: $($s -join ' ')" }
}
Write-Host "all training smoke tests passed"
