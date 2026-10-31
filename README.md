# Avatar-Pipeline

任意のキャラクター画像 1枚と、外部で作った音声（作成元は問わない・API キー不要）から、**音声と同じ長さの「しゃべる」動画**（512×512・25fps・H.264＋AAC）を作る、Windows ローカルのツールです。モデルは **SoulX-FlashHead Pro**（Apache-2.0）。GUI は Gradio 5.50。

- 対象 PC：Windows 11・NVIDIA RTX 3060 12GB・RAM 16GB クラス（ほかの構成は未確認）
- GPU 生成の実績：T1・T1b（4秒）・T1c・GUI-G1（日本語 8.333秒）・**T2（日本語 60秒・1,500 フレーム）**。検証済みの長さは **1,500 フレーム（60.000秒）**。120秒以上は未検証
- 品質：発音単位のリップシンク、長時間の表情・頭の動きの自然さ、別の入力での品質は**未評価**

> **このリポジトリにはモデルの重み・生成動画・仮想環境・入力素材は含まれていません。** 別の PC で使うときは [docs/SETUP.md](docs/SETUP.md) の手順で再構築します。

## 最初に読むもの
| 文書 | 内容 |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | 環境の再構築（Python・依存関係・公式リポジトリ・Windows 用パッチ・モデルの取得と配置） |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | GPU 生成の安全基準・テスト・開発記録の更新ルール・ブランチ運用（Claude Code / Codex 共通） |
| [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md) | 第三者のコード・モデルと利用条件、このリポジトリに含めないもの |
| [CODEX_HANDOFF.md](CODEX_HANDOFF.md) | 全体の引き継ぎ資料（仕組み・結果・不具合・次の作業） |
| [RESTART_PROMPT.md](RESTART_PROMPT.md) | 再開用の要約 |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | 作業記録（節ごとに追記。最新の節から読む） |

## フォルダ構成（リポジトリに含まれるもの）
```
gui/                      GUI（app.py・job_manager.py・static/app.css・起動用 .cmd・依存関係のロック・tools/）
flashhead/tools/          ワーカー fh_worker.py・逐次書き出し frame_sink.py・長さの判定 fh_length.py・fh_limits.json・解析ツール
flashhead/env_check.py    FlashHead 用 venv の確認
tools/run_stage.py        監視つきの実行（メモリ・VRAM・GPU 温度・停止）
patches/flashhead/        Windows 用パッチ（FH-01〜03）と依存関係のロック
licenses/                 重みの SHA256・モデルのライセンス台帳
materials/                素材台帳
tests_cpu/                GPU を使わないテスト
docs/                     セットアップ・開発ルール・第三者の利用条件
```
ローカルだけにあるもの（Git 管理外）：`flashhead/SoulX-FlashHead/`（公式リポジトリ）・`flashhead/models/`（重み）・`flashhead/.venv/`・`gui/.venv/`・`python/`・`cache/`・`outputs/`・`inputs/`・`gui/jobs/`・`gui/tmp/`・`secrets/`・以前の実験（`liveportrait/`・`musetalk/`・`motion-ltx/`・`tts/`・`runs/`）・`*.bak_*`

## 使い方（概要）
- 通常モード：`gui\start_gui.cmd` → http://127.0.0.1:7861（接続テストのみ。GPU 生成は無効）
- GPU 生成モード：`gui\start_gui_gpu.cmd` → 画像と音声を選ぶ →「動画を生成」→ 開始前の8項目の確認画面 →「この内容で1回だけ生成する」
- 検証済み（1,500 フレーム）を超える音声は、「未検証の長さで実行する」の確認が必要

## 重要な注意
- GPU 生成は、毎回ユーザーの明示的な承認のもとで1回ずつ行う。安全基準は [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) を参照（変更禁止）
- 機密情報（API キーなど）をコミットしない。`secrets/` は Git 管理外
- ライセンス：このリポジトリの独自コードのライセンスは未設定（非公開リポジトリ）。第三者のものは [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md)
