param(
  [string]$LlamaRoot = $(if ($env:ANIMUS_LLAMA_ROOT) { $env:ANIMUS_LLAMA_ROOT } else { Join-Path $env:USERPROFILE 'local-llm' }),
  [string]$ModelPath = '',
  [int]$Port = 8081,
  [int]$ContextSize = 8192,
  [int]$GpuLayers = 99
)

$resolvedRoot = (Resolve-Path -LiteralPath $LlamaRoot).Path
$serverPath = Join-Path $resolvedRoot 'bin\llama-server.exe'
if (-not $ModelPath) {
  $ModelPath = Join-Path $resolvedRoot 'models\Qwen3-4B-Q4_K_M.gguf'
}
$resolvedModel = (Resolve-Path -LiteralPath $ModelPath).Path

if (-not (Test-Path -LiteralPath $serverPath -PathType Leaf)) {
  throw "llama-server.exe was not found at $serverPath"
}
if (-not (Test-Path -LiteralPath $resolvedModel -PathType Leaf)) {
  throw "The Animus model was not found at $resolvedModel"
}

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
  throw "Port $Port already has a listening process. Choose another port."
}

Write-Host "Starting the Animus local actor model on 127.0.0.1:$Port"
Write-Host "This foreground process stops when this terminal closes."

& $serverPath `
  -m $resolvedModel `
  --ctx-size $ContextSize `
  -np 1 `
  --flash-attn on `
  --cache-type-k q8_0 `
  --cache-type-v q8_0 `
  -ngl $GpuLayers `
  --jinja `
  --host 127.0.0.1 `
  --cors-origins localhost `
  --port $Port
