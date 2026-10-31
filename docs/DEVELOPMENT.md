# 開発ルール（Claude Code / Codex 共通）

## 1. 作業を始める前に
1. `git pull` で最新にする（作業ブランチで）
2. `RESTART_PROMPT.md` → `PROJECT_STATUS.md` の最新の節 → 必要なら `CODEX_HANDOFF.md` を読む
3. 実行中のジョブが無いか（`gui\jobs\.gpu.lock`）、ユーザーの GUI（7861）が動いているかを読み取りだけで確かめる

## 2. 変更禁止・承認が必要なこと
- **GPU 生成（実推論）は、毎回ユーザーの明示的な承認が必要。** 1回ずつ・自動の再実行なし
- 安全基準（下の 3）の緩和、`AVATAR_ALLOW_GPU` の恒久的な設定、検証済みの長さ（`flashhead/tools/fh_limits.json`）の実機テストなしの変更
- モデル・重み・Python 環境（両方の venv）の追加や更新
- ユーザーの GUI（7861）の停止・再起動、ユーザーのブラウザのタブの操作、ユーザーのアプリの終了
- ほかの AI 環境（ComfyUI・LTX-Video・MuseTalk・EchoMimic・LivePortrait など）の変更
- 生成動画・モデル・ログ・ジョブの記録の削除
- 機密情報（API キー・トークン・パスワード）の読み書き・転記・コミット。`secrets/` は読まない
- Gemini など外部 API の呼び出し

## 3. GPU 生成の安全基準（コード：`gui/job_manager.py`・`tools/run_stage.py`・`flashhead/tools/fh_worker.py`）
| 段階 | 基準 |
|---|---|
| 開始前（`gpu_precheck`、permit の発行時と開始時の2回） | 実行中のジョブなし・システムのコミット残量 19GiB 以上・空きRAM 3GiB 以上・GPU 温度を読める・80℃ 未満・空き VRAM 8,192MiB 以上・FH-01〜03 適用済み・重みの SHA256 一致 |
| ワーカーの中 | 読み込み前：空きRAM 3GiB・コミット残量 12GiB／読み込み後：空きRAM 3GiB・コミット残量 8GiB・空き VRAM 2GiB |
| 実行中（run_stage が強制終了） | ジョブのコミット 12GiB 超・空きRAM 2GiB 未満・コミット残量 5GiB 未満・VRAM 11,800MiB 超・**GPU 83℃ 以上が 10秒続く**・温度を 15秒読めない・時間切れ（300＋チャンク数×16.5×1.5 秒）・停止ボタン |
| 許可 | 1段目：GUI の1回限りの permit（GPU 生成モード）または1プロセスだけの `AVATAR_ALLOW_GPU=1`／2段目：ワーカーへの `--allow-gpu`／3段目：検証済み（1,500 フレーム）を超えると `allow_unverified_length=True`（ユーザーの確認） |
- T2（60秒）の最高温度は 82.09℃。停止の基準まで約 0.9℃。長い生成の前に冷却を確認する（基準は変えない）

## 4. テスト（GPU を使わない）
```
cd /d D:\Avatar-Pipeline\gui
set PYTHONUTF8=1
.venv\Scripts\python.exe -X utf8 tools\stream_dryrun_test.py <WAV>          :: 接続テスト（推論なし・逐次書き出し・結合・7項目の検査）
.venv\Scripts\python.exe -X utf8 ..\tests_cpu\test_gpu_permit.py            :: GPU 許可と開始前の検査（リハーサル）
.venv\Scripts\python.exe -X utf8 ..\tests_cpu\test_length_t2.py             :: 長さの判定（job_manager とワーカーの一致・1,500 フレームの境界）
```
- 詳しくは `tests_cpu/README.md`。`tests_cpu/run_t1c.py` は **GPU を使う**参考用で、承認なしに実行しない
- GUI の確認は、別のポート（`set AVATAR_GUI_PORT=7862`）で起動し、専用のブラウザのタブで行う。`app.py --gpu-rehearsal` は GPU 生成モードの画面を推論なしで試せる
- 変更の前後で `python -c "import ast; ast.parse(open('<file>',encoding='utf-8').read())"` などで構文を確かめる

## 5. 開発記録の更新ルール
- 作業ごとに `PROJECT_STATUS.md` の末尾へ**新しい節**を追記する（既存の節は書き換えない。誤りは新しい節で訂正する）
  - 見出し：`## <番号>. <日付 時刻>：<内容>【実測】/【推定】`。時刻は実際の時計で確かめて書く
  - 実測と推定を分ける。承認の範囲・実行したこと・しなかったこと・結果・残りを書く
- 状態が変わったら `RESTART_PROMPT.md` の冒頭と該当行を更新する。大きな変化は `CODEX_HANDOFF.md` の「最新の状態」にも追記する
- 変更前のファイルはローカルに `*.bak_<日付>_<理由>` で残してよい（Git 管理外）。Git では差分で管理する
- 検証済みの長さを変えたときは、`fh_limits.json` の履歴に実測値と未確認の事項を書く

## 6. ブランチ運用と統合
- `main`：動作を確認したものだけ。直接 push しない
- 作業ブランチ：`claude/<内容>`（Claude Code）・`codex/<内容>`（Codex）・`<名前>/<内容>`（人）
- 手順
  1. `git switch main && git pull` → `git switch -c codex/<内容>`
  2. 変更 → 上のテスト → `PROJECT_STATUS.md` に節を追記
  3. `git add <ファイルを指定>`（`git add .` は allowlist の .gitignore で守られているが、追加するファイルは `git status` で確かめる）→ `git commit`
  4. `git push -u origin <ブランチ>` → GitHub で Pull Request を作り、ユーザーが確認して `main` へ統合
- 同時に2つの AI が同じファイルを変えない。`PROJECT_STATUS.md` は追記だけなので衝突しても、両方の節を残して番号を振り直す
- コミットの前の確認：機密情報が無いか（`git diff --cached` を見る）・大きなファイルが無いか・テストが通るか
- 新しい種類のファイルを共有するときは `.gitignore` に `!` の行を足す（重み・動画・音声・画像・仮想環境・secrets は足さない）
