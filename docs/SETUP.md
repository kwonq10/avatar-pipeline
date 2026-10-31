# セットアップ（別の PC で再構築する手順）

このリポジトリには、重み・仮想環境・公式リポジトリ・入力素材は含まれていません。以下の手順で再構築します。2026-10-31 時点の元の PC（`D:\Avatar-Pipeline`）での構成をもとにしています。手順の一部（特に新しい PC での一括の再現）は**まだ実行して確かめていません**。

## 0. 前提
- Windows 11・NVIDIA GPU（RTX 3060 12GB で確認）・ドライバ（元の PC は 621.01、CUDA 12.8 の wheel が動くもの）
- ffmpeg・ffprobe が PATH にあること（元の PC は ffmpeg 8.1 full_build・gyan.dev）
- git、[uv](https://docs.astral.sh/uv/)（元の PC は `tools\uv.exe` 0.12.9。リポジトリには含めない）
- ディスク：重み 約 6.9GB ＋ 仮想環境・キャッシュで 15GB 以上の空き
- パスは `D:\Avatar-Pipeline` を前提にしたコードがある（`gui\job_manager.py`・`flashhead\tools\fh_worker.py`・`gui\start_gui*.cmd` など）。別の場所に置く場合は、それらのパスの見直しが必要（未対応）

## 1. リポジトリを取得
```
git clone <このリポジトリの URL> D:\Avatar-Pipeline
cd /d D:\Avatar-Pipeline
```

## 2. uv の設定（キャッシュと Python を D: に置く）
`env.sh`（bash 用）と同じ設定です。PowerShell なら：
```
$env:UV_CACHE_DIR='D:\Avatar-Pipeline\cache\uv'; $env:UV_PYTHON_INSTALL_DIR='D:\Avatar-Pipeline\python'
$env:UV_PYTHON_PREFERENCE='only-managed'; $env:HF_HOME='D:\Avatar-Pipeline\cache\hf'; $env:TORCH_HOME='D:\Avatar-Pipeline\cache\torch'; $env:PYTHONUTF8='1'
uv python install 3.10
```

## 3. GUI 用の venv（torch なし）
```
uv venv --python 3.10 gui\.venv
uv pip install --python gui\.venv\Scripts\python.exe --no-deps -r gui\requirements.lock.txt
uv pip check --python gui\.venv\Scripts\python.exe
```
- 元の PC：Python 3.10.21・56 個・gradio 5.50.0（PROJECT_STATUS 31節）

## 4. FlashHead の公式リポジトリとパッチ
```
git clone https://github.com/Soul-AILab/SoulX-FlashHead.git flashhead\SoulX-FlashHead
git -C flashhead\SoulX-FlashHead checkout 9bc03de06bb0de82cd6bc477804512ae06144bf2
cd flashhead\SoulX-FlashHead
git apply --ignore-whitespace ..\..\patches\flashhead\fh_01-03_windows_lowmem.patch
git apply --check -R --ignore-whitespace ..\..\patches\flashhead\fh_01-03_windows_lowmem.patch   ← 成功すれば適用済み
cd ..\..
```
- FH-01：xfuser の import を保護／FH-02：torch.compile をオフ／FH-03：重みを1テンソルずつ GPU に読み込む（`flash_head/utils/lowmem_load.py`・`FLASHHEAD_LOWMEM_LOAD=1`）。詳細は CODEX_HANDOFF.md 5章

## 5. FlashHead 用の venv（torch 2.7.1+cu128）
取得元を分けて2段階で、`--no-deps` で入れます（元の PC と同じ方法・PROJECT_STATUS 27節）。
```
uv venv --python 3.10 flashhead\.venv
uv pip install --python flashhead\.venv\Scripts\python.exe --no-deps --index-url https://download.pytorch.org/whl/cu128 -r patches\flashhead\lock_step1_pytorch.txt
uv pip install --python flashhead\.venv\Scripts\python.exe --no-deps --index-url https://pypi.org/simple -r patches\flashhead\lock_step2_pypi.txt
uv pip check --python flashhead\.venv\Scripts\python.exe
flashhead\.venv\Scripts\python.exe flashhead\env_check.py
```
- 元の PC：83 個（`patches/flashhead/installed_freeze.txt` と一致）・torch 2.7.1+cu128・torchvision 0.22.1+cu128・diffusers 0.36.0・transformers 4.57.3・numpy 1.26.4・opencv 4.11.0.86・mediapipe 0.10.9・librosa 0.11.0

## 6. モデルの重み（Pro と wav2vec2 だけ）
取得先と配置：
| Hugging Face | 配置先 | 必要なファイル |
|---|---|---|
| `Soul-AILab/SoulX-FlashHead-1_3B` | `flashhead\models\SoulX-FlashHead-1_3B\` | `Model_Pro/`（config.json・diffusion_pytorch_model.safetensors 約 6.0GB）、`VAE_Wan/Wan2.1_VAE.pth`（約 0.5GB）、`config.json`、`model_index.json`、`README.md` |
| `facebook/wav2vec2-base-960h` | `flashhead\models\wav2vec2-base-960h\` | `model.safetensors`（約 0.38GB）、config・preprocessor・feature_extractor・tokenizer・special_tokens_map・vocab の json、README.md |
- **取得しないもの**：`Model_Lite`・`VAE_LTX`（LTX-Video 0.9 の VAE。研究目的のみの条件で商用不可）・wav2vec2 の `pytorch_model.bin`・`tf_model.h5`（重複）
- 例（公式 README の方法。必要なファイルだけに絞ること）：
  ```
  huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --include "Model_Pro/*" "VAE_Wan/*" "config.json" "model_index.json" "README.md" --local-dir flashhead\models\SoulX-FlashHead-1_3B
  huggingface-cli download facebook/wav2vec2-base-960h --include "*.json" "model.safetensors" "README.md" --local-dir flashhead\models\wav2vec2-base-960h
  ```
  （`huggingface-cli` は FlashHead 用 venv の huggingface_hub に含まれる。このコマンドの組み合わせは元の PC では実行していない）
- **SHA256 の照合**（14 件すべて一致すること）：`licenses\flashhead_weights_sha256.txt`。GUI の開始前の検査（`job_manager.weights_ok()`）でも照合する

## 7. 動作確認（GPU を使わない）
```
cd /d D:\Avatar-Pipeline\gui
set PYTHONUTF8=1
ffmpeg -v error -y -f lavfi -i "sine=frequency=330:duration=8" -ar 16000 -ac 1 tmp\ui_test\dummy_tone_8s.wav
.venv\Scripts\python.exe -X utf8 tools\stream_dryrun_test.py D:\Avatar-Pipeline\gui\tmp\ui_test\dummy_tone_8s.wav
.venv\Scripts\python.exe -X utf8 -c "import job_manager as jm; p=jm.gpu_precheck(); [print(ok, n, v) for n,ok,v in p['items']]"
```
（`tmp\ui_test\` が無ければ先に作る）。そのほかのテストは [DEVELOPMENT.md](DEVELOPMENT.md)

## 8. GUI の起動
- `gui\start_gui.cmd`（通常）／`gui\start_gui_gpu.cmd`（GPU 生成モード）→ http://127.0.0.1:7861
- ポートが使われていると「前の GUI が動いています」と表示して終了する
