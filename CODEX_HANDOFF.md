# CODEX_HANDOFF — Avatar-Pipeline（SoulX-FlashHead Pro・キャラクター音声同期動画）

作成：2026-10-31 15:20（Claude Code から OpenAI Codex への引き継ぎ資料。15:40 の時点では Claude Code で継続中。Claude Code の利用制限のあとに使う）
根拠：`PROJECT_STATUS.md`（特に 34〜55節）・`RESTART_PROMPT.md`・実際のコード（gui\app.py・gui\job_manager.py・gui\static\app.css・flashhead\tools\fh_worker.py・fh_length.py・fh_limits.json・tools\run_stage.py）と、この時点のプロセスの状態。
表記：**【済】**＝実測で確認した完了事項、**【未】**＝未完了、**【推定】**＝確認していない見込み。

> 最初に読む順番：この文書 → `RESTART_PROMPT.md` → `PROJECT_STATUS.md` の 55節から 45節へさかのぼる（さらに詳しい経緯は 34〜44節）

---

## 最新の状態（2026-10-31 15:56 以降の追記。下の各章より優先）
- **T2（日本語 60秒・1,500 フレーム・54 チャンク）は GPU 生成1回で成功済み**（PROJECT_STATUS 61節）：`outputs\t2\20261031-154143_fh_gpu.mp4`・全体 876.2秒・VRAM 最大 8,490MiB（54 チャンクで一定）・ジョブのコミット 10.29GiB・空きRAM 最小 8.41GiB・コミット残量 最小 11.42GiB・**GPU 最高 82.09℃**・安全停止なし・7項目の検査すべて合格
  - 実行は、ブラウザの拡張機能が使えなかったため、GUI と同じバックエンドの経路（issue_gpu_permit → start_job(permit)）で行った（迂回なし）
- **検証済みの長さは 1,500 フレーム（60.000秒）に更新済み**（62節・ユーザー承認）。`fh_limits.json` の gpu_verified_max_frames=1500・gpu_verified_max_seconds=60.0。1,501 フレーム以上は未検証の確認が必要（CPU テスト `tests_cpu\test_length_t2.py` で確認）
- **未確認**：発音単位のリップシンク精度・60秒間の表情と頭の動きの自然さ・別の画像や音声での品質・**120秒以上（T3・T4）の連続生成**
- **温度の余裕が小さい**：T2 は約 25 チャンク目から 80〜82℃ で横ばい。停止の基準（83℃・10秒）まで約 0.9℃。T3 の前に冷却（室温・ケースの風通し・ファン）を確認する。基準は変えない
- ミュートボタンの件は解決済み（7861 の古い GUI を停止し、ユーザーが起動し直して表示を確認。57・58節）。13章-1 は過去の記録
- 2026-10-31 の時点では Claude Code で開発を継続中。この文書は Claude Code の利用制限のあとに使う
- **GitHub の準備**（64節）：`README.md`・`docs/SETUP.md`・`docs/DEVELOPMENT.md`（ブランチ運用・記録のルール）・`docs/THIRD_PARTY.md` を作成。`.gitignore` は allowlist 方式（共有は 49 ファイル）。`git init` 済み・未 commit・未 push。push 後は GitHub のリポジトリ（非公開 avatar-pipeline）から作業する。0章-6・17章の「git の管理下ではない」は、それ以前の記述

## 0. いちばん大事な注意（作業を始める前に）
1. **GPU 生成（実推論）は、毎回ユーザーの明示的な承認が必要。** 承認は1回ごとで、使い切ったら終わり。これまでの承認はすべて使い切っている
2. **ユーザーが使っている GUI（127.0.0.1:7861）を、許可なく止めたり再起動したりしない。** テストは別のポート（`AVATAR_GUI_PORT=7862` など）と、専用のブラウザのタブで行う
3. **ユーザーのブラウザのタブで、動画の状態（`video.muted` など）を変えない**（53節の事故。Gradio 5.50 のプレーヤーでは画面から戻せない）
4. ユーザーのアプリ（Chrome・ChatGPT・Discord など）を無断で終了しない。メモリの余裕が足りなければ、ユーザーに頼む
5. 機密情報：`secrets\` は読まない・書き写さない（`.gitignore` 済み）。**`secrets\README.txt` に API キーのような文字列が平文で書かれている**ことを 2026-10-31 に確認した（中身はこの文書に書いていない）。ユーザーに、無効化・削除を検討するよう伝えてある。Gemini API は使わない
6. このフォルダは **git の管理下ではない**（`.git` が無い）。変更前のファイルは `*.bak_<日付>_<理由>` として同じ場所に残す運用（17章）

---

## 1. プロジェクトの目的と完成イメージ
- 任意のキャラクター画像 1枚と、外部で作った音声（作成元は問わない・API キー不要・Gemini は使わない）から、**音声と同じ長さの「しゃべる」動画**を作るローカルツール
  - 出力：512×512・25fps・H.264＋AAC の MP4（頭部のアップ）
  - モデル：**SoulX-FlashHead Pro**（Apache-2.0。Lite は VAE が LTX-Video 0.9 の RAIL-M で商用不可のため使わない）
  - 用途：SNS 向けの商用利用を想定。商用可の構成だけを使う（素材台帳 `materials\MATERIALS_REGISTER.json`、モデル台帳 `licenses\`）
- 当面の目標：**1〜3分の動画を1本ずつ**作る（結合は別の編集ソフト）
- **ユーザーが Claude Code / Codex を使わず、GUI だけで生成できること**（GUI-G1 で達成【済】）
- `inputs\avatar_bob.png` は比較用のサンプル。**キャラクターとして固定しない**
- PC：Windows 11・RTX 3060 12GB（ドライバ 621.01）・RAM 16GB クラス（コミットの上限 約 31.9GiB）

## 2. 現在の実装状況
| 項目 | 状態 |
|---|---|
| FlashHead Pro の Windows 専用環境・修正 FH-01〜03・重みの SHA256 | 【済】 |
| 逐次書き出し（全フレームを RAM にためない・目標フレーム数で打ち切り） | 【済】`flashhead\tools\frame_sink.py` |
| 実 GPU 生成（T1・T1b・T1c・GUI-G1） | 【済】すべて成功（11章） |
| GUI（Gradio 5.50・1画面・虹色の枠・進捗表示） | 【済】42節 |
| GUI の「動画を生成」（1回限りの許可＋開始前の検査＋確認画面） | 【済】48節・GUI-G1 で実際に成功（52節） |
| 長さの判定の統一（WAV のサンプル数・フレーム数で判定） | 【済】51節 |
| 完成動画のミュート・解除ボタン | コードは【済】（54・55節。テスト用の GUI で表示・動作を確認）。**ユーザーの 7861 では【未】表示されない**（13章-1） |
| リップシンクの品質評価（ユーザーが再生して判断） | 【未】 |
| 60秒（T2）の GPU 生成 | 【済】61節・検証済み 1,500 フレーム（62節） |
| 120秒（T3）・180秒（T4）の GPU 生成 | 【未】（18章） |

## 3. フォルダ構成（主なもの）
```
D:\Avatar-Pipeline\
├─ CODEX_HANDOFF.md        この文書
├─ PROJECT_STATUS.md       作業記録（1〜55節。追記していく）
├─ RESTART_PROMPT.md       再開用の要約
├─ env.sh                  bash 用の環境変数（uv・キャッシュのパス。機密なし）
├─ gui\                    GUI（Gradio 5.50・GUI 専用 venv）
│  ├─ app.py               画面・イベント・ミュートのスクリプト・起動時のポート確認
│  ├─ job_manager.py       ジョブの作成・GPU 許可（permit）・開始前の検査・長さの判定・履歴
│  ├─ static\app.css       デザイン（白・虹色の枠・進捗バー・ボタン）
│  ├─ start_gui.cmd        通常モードで起動（GPU 生成は無効）
│  ├─ start_gui_gpu.cmd    GPU 生成モードで起動（app.py --gpu-enabled）
│  ├─ tools\stream_dryrun_test.py  接続テスト（推論なし）をコマンドで1本
│  ├─ tools\dummy_job.py   GUI だけの確認用
│  ├─ jobs\<job_id>\       ジョブごとの記録（job.json・status.json・result.json・settings.json・job_memory.csv・worker_mem.csv・job_stdout.txt・inputs\）
│  ├─ jobs\.gpu.lock       実行中のジョブ（同時に1つ）
│  ├─ jobs\.weights_sha256_ok.json  重みの SHA256 の合格の記録
│  └─ tmp\                 Gradio の一時ファイル・長さの判定用の WAV（length_check\）・テスト素材（ui_test\）
├─ flashhead\
│  ├─ .venv\               FlashHead 専用 venv（Python 3.10.21・83個）
│  ├─ SoulX-FlashHead\     公式リポジトリ @9bc03de ＋ FH-01〜03
│  ├─ models\              SoulX-FlashHead-1_3B（Pro・VAE_Wan）・wav2vec2-base-960h
│  └─ tools\
│     ├─ fh_worker.py      ワーカー（入力の検査 → 計画 → 生成 or ダミー → 逐次書き出し → 音声と結合 → 7項目の検査）
│     ├─ frame_sink.py     ffmpeg の stdin へ逐次書き出し
│     ├─ fh_length.py      長さの測り方と「検証済みか」の判定（GUI・job_manager・ワーカー共通）
│     ├─ fh_limits.json    検証済みの長さ・実測値・時間切れの係数
│     ├─ image_check.py    画像の事前チェック（MediaPipe・CPU）
│     ├─ verify_frames.py・test_frame_sink.py・load_test.py
│     └─ lipsync_chart.py・audio_tail_adjust.py・a1_context_check.py（CPU の解析）
├─ tools\run_stage.py      監視つきの実行（ジョブのコミット上限・RAM・コミット残量・VRAM・温度・停止ファイル・時間切れ）
├─ tests_cpu\              CPU テストの写し（README.md 参照）
├─ patches\flashhead\      fh_01-03_windows_lowmem.patch・依存関係のロック
├─ licenses\               flashhead_weights_sha256.txt など
├─ inputs\                 avatar_bob.png・T1c の日本語 MP3
├─ outputs\                t1・t1b・t1c・gui_g1・stream_test・connection_test・gui_gpu_rehearsal・ui_screenshots_*
├─ secrets\                読まない（0章-5）
└─ liveportrait\ musetalk\ motion-ltx\ tts\ runs\  以前の実験（現在の方式では使わない。変更しない）
```
- `python\`（uv 管理の CPython 3.10）・`cache\`（uv・HF・torch）は環境の一部。消さない

## 4. Python 環境・パッケージ・モデル（2026-10-31 に実測）
| 環境 | 内容 |
|---|---|
| GUI 専用 venv `gui\.venv` | Python 3.10.21・56個・**gradio 5.50.0**。ロックは `gui\requirements.lock.txt`。torch は入っていない（GUI はモデルを読み込まない） |
| FlashHead 専用 venv `flashhead\.venv` | Python 3.10.21・83個・**torch 2.7.1+cu128**・torchvision 0.22.1+cu128・diffusers 0.36.0・transformers 4.57.3・numpy 1.26.4・opencv-python 4.11.0.86・mediapipe 0.10.9・librosa 0.11.0。解決の記録は `patches\flashhead\resolved_win_pro_np1.txt` など |
| ffmpeg | 8.1 full_build（gyan.dev）。PATH 上 |
| モデル | `flashhead\models\SoulX-FlashHead-1_3B\Model_Pro`・`VAE_Wan\Wan2.1_VAE.pth`・`wav2vec2-base-960h`。SHA256 は `licenses\flashhead_weights_sha256.txt`（14件。2026-10-31 に一致を確認） |
- 環境・パッケージの追加・更新は禁止（16章）

## 5. Windows 専用の修正（FH-01〜03）
- パッチ：`patches\flashhead\fh_01-03_windows_lowmem.patch`（SoulX-FlashHead @9bc03de に適用済み）
  - FH-01：`flash_head\src\modules\flash_head_model.py` の xfuser の import を try で保護（1GPU・Windows では使わない）
  - FH-02：`flash_head\src\pipeline\flash_head_pipeline.py` の torch.compile をオフ（Windows に Triton が無い。速度だけが下がる）
  - FH-03：重みを1テンソルずつ GPU に読み込む（`flash_head\utils\lowmem_load.py`、環境変数 `FLASHHEAD_LOWMEM_LOAD=1`。job_manager が設定）
- 適用の確認：`cd D:\Avatar-Pipeline\flashhead\SoulX-FlashHead` → `git apply --check -R --ignore-whitespace ..\..\patches\flashhead\fh_01-03_windows_lowmem.patch`（成功＝適用済み）
- attention は SDPA（公式は FlashAttention があれば使う）。数値はわずかに違いうるが、時間の対応には影響しない（46節）

## 6. GPU 生成の実装方法
1. `job_manager.start_job("fh_gpu", payload, permit=...)` がジョブのフォルダを作り、画像と音声をコピーする。音声は ffmpeg で PCM 16bit の WAV にする（サンプルレート・ch はそのまま）。画像チェックをして、job.json を書く
2. `tools\run_stage.py` の監視の下で、`flashhead\.venv` の python が `fh_worker.py --job-dir <dir> --allow-gpu` を実行する（Windows の Job Object でプロセスツリーごと管理）
3. ワーカー：入力の検査（長さは fh_length）→ 計画 → 開始の検査（空きRAM 3GiB・コミット残量 12GiB）→ モデルの読み込み → 読み込み後の検査（空きRAM 3GiB・コミット残量 8GiB・空き VRAM 2GiB）→ 公式の **stream 方式**で 1チャンク＝33 フレーム（つなぎ 5 ＋ 新しい 28）を生成 → `frame_sink` で ffmpeg に逐次書き出し（目標フレーム数 round(秒×25) で打ち切り）→ 元の WAV と結合（映像はコピー・音声は AAC 192k）→ 7項目の検査（フレーム数・512・25fps・h264・aac・音声と映像の差が1フレーム未満・音声の欠けなし）→ 合格したときだけ `.incomplete.mp4` を `.mp4` にする
4. 出力：`<output_dir>\<job_id>.mp4`（上書きしない）。記録：`gui\jobs\<job_id>\`
- 1チャンク（28 フレーム＝1.12秒）は約 15〜17秒（約 1.7〜1.8FPS）。起動（import＋読み込み）は 35〜105秒（OS のキャッシュ次第）
- seed が同じなら、同じ入力から**まったく同じ MP4** になる（T1c と GUI-G1 の SHA256 が一致・52節）

## 7. GUI の起動方法
- 通常モード：`D:\Avatar-Pipeline\gui\start_gui.cmd` をダブルクリック → ブラウザで http://127.0.0.1:7861
- GPU 生成モード：`D:\Avatar-Pipeline\gui\start_gui_gpu.cmd`（中身は `app.py --gpu-enabled`）
- ポートは環境変数 `AVATAR_GUI_PORT`（既定 7861）。127.0.0.1 のみ・share なし
- **起動時に、ポートがすでに使われていれば「前の GUI が動いています」と表示して終了する（exit 3）。** .cmd はエラーのとき `pause` してメッセージを残す（55節）
- **ユーザーが使う GUI は、ユーザーが .cmd で起動する。** AI が裏で起動するとウィンドウが無く、ユーザーが閉じられない（55節の原因）
- 起動時の DeprecationWarning（css_paths・js・theme を Blocks に渡す書き方は Gradio 6 で launch へ移る）は 5.50 では無害

## 8. 通常モードと GPU 生成モードの違い
| | 通常モード | GPU 生成モード（`--gpu-enabled`） | リハーサル（`--gpu-rehearsal`・テスト用） |
|---|---|---|---|
| バッジ | GPU 生成：無効（通常モード） | GPU 生成モード（このウィンドウの間だけ・1回ごとに確認） | GPU 生成：リハーサル（GPU 未使用） |
| 「動画を生成」 | 無効（無理に押してもバックエンドが拒否） | 使える | 使える（許可と検査は本番どおり・ワーカーは dry_run） |
| 「未検証の長さで実行する」 | 押せない | 検証済みを超える音声のときだけ押せる | 同左 |
| 接続テスト（推論なし） | 使える | 使える | 使える |
| ミュートボタン | あり | あり | あり |

## 9. GPU 生成を許可する仕組み（3段階＋開始前の検査）
- 1段目（どちらか）
  - (a) そのプロセスだけの環境変数 `AVATAR_ALLOW_GPU=1`（コマンドラインからの1回の実行用。**恒久的に設定しない**）
  - (b) GUI の **1回限りの許可（permit）**：GPU 生成モードの GUI で「動画を生成」→ バックエンド `jm.issue_gpu_permit()` が入力と開始前の条件を検査 → 合格したらメモリの中だけに乱数の ID を発行 → 確認画面の「この内容で1回だけ生成する」→ `start_job(permit=...)`
    - permit は画像・音声の SHA256・長さ・seed・顔の切り出し・出力先・未検証の確認に結び付く。期限は 180秒で、**1回使うと（不合格でも）消える**。確認後に入力が変われば拒否する
    - `prepare_gpu`・`confirm_gpu`・`cancel_gpu` は `api_name=False`（API から呼べない）
    - **最後の確認ボタンはユーザー本人が押す。AI が自動でクリックしない**
- 2段目：ワーカーへの `--allow-gpu`（ワーカーも確認する。リハーサルでは付けない）
- 3段目：検証済みの長さを超える音声は `allow_unverified_length=True` が必要（GUI では「未検証の長さで実行する」にチェックして発行した permit に限る）。ワーカーも確認する
- **開始前の検査（`jm.gpu_precheck()`）は、permit の発行時と開始時の2回、バックエンドが強制する**（環境変数の経路にも適用）

## 10. 安全基準（変更禁止）
| 段階 | 基準 |
|---|---|
| 開始前（gpu_precheck） | 実行中のジョブなし・**システムのコミット残量 19GiB 以上**・空きRAM 3GiB 以上・GPU 温度を読める・80℃ 未満・空き VRAM 8,192MiB 以上・FH-01〜03 適用済み・重みの SHA256 一致 |
| ワーカーの中 | 読み込み前：空きRAM 3GiB・コミット残量 12GiB／読み込み後：空きRAM 3GiB・コミット残量 8GiB・空き VRAM 2GiB |
| 実行中（run_stage が強制終了） | **ジョブのコミット 12GiB 超**・空きRAM 2GiB 未満・システムのコミット残量 5GiB 未満・**VRAM 11,800MiB 超**・**GPU 83℃ 以上が 10秒続く**・**温度を 15秒以上読めない**・時間切れ（長さから計算）・停止ボタン（STOP ファイル） |
| 運用 | 同時に1ジョブ（`.gpu.lock`）・**自動の再実行なし**・毎回ユーザーの承認 |
- 実際の値の目安：この PC のコミット残量は、常駐アプリがあると **18.4〜19.4GiB** で、基準ぎりぎり（GUI 自体が約 0.8GiB 使う）。T1c・GUI-G1 の実行中のコミット残量の最小は 9GiB 前後

## 11. 検証結果（実 GPU 生成）
| テスト | 入力 | 結果 | 全体 | VRAM 最大 | ジョブのコミット最大 | 最高温度 | 出力 |
|---|---|---|---|---|---|---|---|
| T1（10-30・37節） | 英語 4秒・avatar_bob・face crop | 成功・100 フレーム | 173秒 | 8,624MiB | 10.46GiB | （記録なし） | `outputs\t1\20261030-221116_fh_gpu.mp4` |
| T1b（10-30・39節） | 同上・逐次書き出し | 成功・100 フレーム | 120秒 | 8,486MiB | 10.26GiB | 70.3℃ | `outputs\t1b\20261030-223842_fh_gpu.mp4` |
| T1c（10-31・45節） | **日本語 8.333061秒（208 フレーム）**・seed 42・face crop | 成功・208 フレーム・差 0.013秒 | 179秒 | 8,666MiB | 10.28GiB | 76.5℃ | `outputs\t1c\20261031-134405_fh_gpu.mp4` |
| GUI-G1（10-31・52節） | T1c と同じ。**GUI から、ユーザーが最終ボタンを押して実行** | 成功・208 フレーム・差 0.013秒 | 173秒 | 8,436MiB | 10.29GiB | 76.1℃ | `outputs\gui_g1\20261031-143923_fh_gpu.mp4`（**T1c と SHA256 が同じ**） |
| T2（10-31・61節） | **日本語 60.000秒（1,500 フレーム・54 チャンク）**・avatar_bob・seed 42・face crop。GUI と同じ permit の経路 | 成功・1,500 フレーム・差 0.000秒 | 876秒 | 8,490MiB | 10.29GiB | **82.09℃** | `outputs\t2\20261031-154143_fh_gpu.mp4` |
- チャンクごとの VRAM・ワーカーのメモリは一定で、増え続けない。温度はチャンクごとに少しずつ上がる（71→76℃／8チャンク）。T2 は約 25 チャンク目から 80〜82℃ で横ばい（83℃ の停止まで約 0.9℃）
- 失敗の記録：GUI-G1 の1回目（50節・ジョブ 20261031-142654_fh_gpu）は、長さの判定の食い違いでワーカーが 0.5秒で拒否（GPU は使っていない）→ 51節で修正済み
- `fh_limits.json`：`gpu_verified_max_frames` = **1500**・`gpu_verified_max_seconds` = **60.0**（T2・62節で更新。以前は 208・8.333061＝T1c）。履歴に未確認の事項（リップシンク・表情と頭の動き・別の入力・120秒以上）と温度の注意を記録

## 12. リップシンクの品質評価と改善候補
- **ユーザーによる再生での評価は【未】**（T1c と GUI-G1 は同じ映像）。数値や静止画だけで「正確」と判断しないこと
- 調査結果（46・47節。CPU のみ）
  - 全体の時間のずれを生む不具合は**無い**：44.1k→16k のずれ 0ms・wav2vec2 の特徴量とフレームの差は 1フレーム未満・口と音の相互相関の最大はずらし 0 フレーム
  - 観察：0.82〜1.72秒の短い無音で口が半開き（息や余韻〈-50〜-60dB〉に反応・時間の細かさの限界と推定）、発話の始まりで約 80ms 前に口が動く（モデルの ±2 フレームの参照どおり）
  - 資料：`outputs\t1c\analysis\`（lipsync_chart.png など）・`outputs\t1c\audio_comparison\`
- 改善候補（**ユーザーの判断で当面保留**）
  - B1：音声の余韻を軽く弱める（保護つきで最大 -12dB）。調整済みの WAV は `outputs\t1c\audio_comparison\adjusted_light_tail_-12dB_44k.wav`。効果は小さい見込み。標準の処理には採用しない
  - A1：stream 方式に先読みを足す。チャンクの終わりで特徴量が劣化することは確認したが、観察した問題との一致は一部だけ → 優先度は下げた
  - 顔の切り出しの比率（face_ratio）・参照画像・seed・話す速さ（効果は小〜中の見込み）
  - モデル自体の限界：wav2vec2 は英語で学習・時間の細かさ（1 潜在＝160ms）・4ステップへの蒸留

## 13. 現在の不具合と未完了の作業（2026-10-31 15:20 時点）
1. **【未解決（ユーザーの画面）】ミュートボタンがユーザーの GUI（7861）に表示されない**
   - 原因は特定済み（55節）：7861 で動いているのは、**Claude Code が 14:37 に裏で起動した古い GUI**（プロセス：venv の python **PID 19832** → 本体 **PID 24008**、どちらも `app.py --gpu-enabled`、2026-10-31 14:37 起動）。ミュートのコード（app.py の更新は 15:04）より前のコード。ウィンドウが無いのでユーザーが閉じられず、起動し直しても 7861 が使用中で新しい GUI は起動できなかった
   - コードは【済】：テスト用の GUI（7862 通常・7863 GPU 生成モードの画面）で、ボタンが動画の直下に表示され、実際のクリックでミュート・解除でき、`video.muted` と表示が一致することを確認した（写真 `outputs\ui_screenshots_20261031_mute\03〜05`）
   - **この文書を書いた時点でも、7861 は PID 24008 のまま（古いコード）**。ユーザーの操作（古い GUI を止めて `start_gui_gpu.cmd` で起動し直す）が終わったかは未確認
   - **追記 2026-10-31 15:40：ユーザーの承認で PID 24008 を停止し、7861 は解放された（PROJECT_STATUS 57節）。** 残りは、ユーザーが `start_gui_gpu.cmd` で起動し直して、ボタンが画面に出るかを確認すること（結果は PROJECT_STATUS の新しい節を見る）
   - Codex がすること：(1) 7861 のプロセスの開始時刻と `gui\app.py` の更新時刻を比べる（下のコマンド）。(2) まだ古ければ、ユーザーに止め方を伝える。**止めるのはユーザーの許可を得てから**。(3) 新しい GUI になったら、ユーザーの画面にボタンが出るかをユーザーに確認してもらう。**同じ修正をコードに繰り返さないこと**（ボタンのコードは動く）
   - 確認のコマンド（PowerShell）：
     ```
     $p=(Get-NetTCPConnection -LocalPort 7861 -State Listen).OwningProcess | Select-Object -Unique
     Get-CimInstance Win32_Process -Filter "ProcessId=$p" | Select-Object ProcessId, CreationDate, CommandLine
     (Get-Item D:\Avatar-Pipeline\gui\app.py).LastWriteTime
     ```
2. 【未】リップシンクの品質評価（ユーザーが再生して判断）
3. 【未】T2（60秒）以降（18章）
4. 小さな点
   - リハーサルの状況の文が、ワーカーの dry_run の文（「接続テスト…」）のまま（種類の欄には「GPU 生成のリハーサル」と出る）
   - 確認画面を開いている間は、その分だけ処理状況の欄が下にずれる（1920×1080 で一時的にスクロールが必要）
   - 1366×768 では、MP4 の保存先・音声の情報が少しはみ出す
   - コミット残量が基準ぎりぎり（10章）
5. 古い記録の注意：`tts\gemini_tts.py`・`tools\run_jp_test.sh` は MuseTalk 時代のもので、今の方式では使わない（Gemini は使わない）

## 14. 次に Codex がすること（順番）
1. 13章-1 の状態を確かめる（読み取りだけ）→ 必要ならユーザーに古い GUI の止め方を伝える → 新しい GUI でボタンが見えるかを、ユーザーに確認してもらう
2. ユーザーに T1c（＝GUI-G1）の動画を再生して評価してもらい、結果を PROJECT_STATUS.md に新しい節として記録する
3. 評価しだいで、改善候補（12章）を1つずつ、同じ入力・seed で比べる（GPU は毎回承認）
4. T2（60秒）の準備：60秒の日本語の会話音声と画像をユーザーに用意してもらう → 承認 → GUI から実行（ユーザーが押す）
5. 小さな点（13章-4）は、ユーザーが望めば

## 15. テスト方法と実行コマンド（GPU を使わないもの）
```
cd /d D:\Avatar-Pipeline\gui
set PYTHONUTF8=1
:: 接続テスト（推論なし。逐次書き出し・結合・7項目の検査・通し番号の検査）
.venv\Scripts\python.exe -X utf8 tools\stream_dryrun_test.py D:\Avatar-Pipeline\gui\tmp\ui_test\dummy_tone_8s.wav
:: 開始前の検査の今の値（読み取りだけ）
.venv\Scripts\python.exe -X utf8 -c "import job_manager as jm; p=jm.gpu_precheck(); [print(ok, n, v) for n,ok,v in p['items']]"
:: GPU 許可・長さの判定のテスト（tests_cpu\README.md）
.venv\Scripts\python.exe -X utf8 ..\tests_cpu\test_gpu_permit.py
.venv\Scripts\python.exe -X utf8 ..\tests_cpu\test_length_agree.py
:: テスト用の GUI（ユーザーの 7861 と別のポート）
set AVATAR_GUI_PORT=7862
.venv\Scripts\python.exe app.py              ← 通常モード
.venv\Scripts\python.exe app.py --gpu-rehearsal   ← GPU 生成モードの画面（推論なし）
```
- dry run 用の音声が無ければ：`ffmpeg -v error -y -f lavfi -i "sine=frequency=330:duration=8" -ar 16000 -ac 1 gui\tmp\ui_test\dummy_tone_8s.wav`
- 重みの SHA256：`jm.weights_ok()`（合格の記録を使う。ファイルが変わると全体を計算し直す、約 16秒）
- フレームの検査：`flashhead\.venv\Scripts\python.exe flashhead\tools\verify_frames.py <mp4> <目標フレーム数>`（接続テストの MP4 用）
- 口の開きの資料（CPU）：`flashhead\.venv\Scripts\python.exe flashhead\tools\lipsync_chart.py <mp4> <出力フォルダ> "開始-終了:ラベル" ...`
- ブラウザのテストは**専用のタブ**で。音声つきの再生は、スクリプトからのクリックではブラウザに止められることがある（実際のマウス操作では問題ない）

## 16. 変更禁止事項
- GPU 推論・動画生成を承認なしに行う／自動で再実行する
- モデル・重み・Python 環境・パッケージの追加や更新（両方の venv）
- 安全基準（10章）の緩和。`AVATAR_ALLOW_GPU` を恒久的に設定する
- 検証済みの長さ（`fh_limits.json`）を、実機テストの合格なしに変える。値は丸めずに実測のまま・フレーム数でも記録する（51節の教訓）
- 全フレームを RAM にためる方式に戻す（frame_sink を使う）
- 既存の ComfyUI・LTX-Video・MuseTalk・EchoMimic・LivePortrait など、ほかの AI 環境を変更する
- 生成動画・モデル・ログ・ジョブの記録を削除する
- ユーザーの GUI（7861）を許可なく止める・再起動する。ユーザーのアプリを終了する。ffmpeg.exe などを名前で強制終了する（止めるのは自分が起動したプロセスだけ）
- GUI の大幅なデザイン変更（白・虹色の枠・1画面を保つ）
- Gemini API の呼び出し。`secrets\` の読み書き・書き写し

## 17. バックアップ・復旧
- git は無い。変更前のファイルは同じ場所に `*.bak_<日付>_<理由>` で残っている。主なもの（2026-10-31）：
  - `gui\app.py.bak_20261031_pre_ui` / `_pre_gpubtn` / `_pre_lenfix` / `_pre_mute` / `_pre_portcheck`
  - `gui\job_manager.py.bak_20261031_pre_gpubtn` / `_pre_lenfix`
  - `gui\static\app.css.bak_20261031_pre_ui` / `_pre_gpubtn` / `_pre_mute`
  - `gui\start_gui.cmd.bak_20261031_pre_gpubtn` / `_pre_portcheck`、`gui\start_gui_gpu.cmd.bak_20261031_pre_portcheck`
  - `flashhead\tools\fh_worker.py.bak_20261031_pre_lenfix`、`fh_limits.json.bak_20261031_pre_t1c` / `_pre_lenfix`
  - `tools\run_stage.py.bak_20261031_pre_ui`（flush を足す前）
  - `RESTART_PROMPT.md.bak_*`、`PROJECT_STATUS.md.bak_20261031_pre_ui`
  - 10-30 以前：`*.bak_20261030_pre_stream`・`*.bak_20261030_pre_temp`
- 戻すとき：対象のファイルを別名で残してから、`.bak` を元の名前にコピーする。GUI は起動し直す（動いている GUI は古いコードのまま）
- FlashHead 本体：`cd flashhead\SoulX-FlashHead` → `git apply -R ..\..\patches\flashhead\fh_01-03_windows_lowmem.patch` で FH-01〜03 を外せる（通常は不要）
- 失敗したジョブ：`gui\jobs\<id>\result.json` の `incomplete_files` と `.incomplete.mp4` を見る（完成品と区別される）。ロックが残ったら、`jm.current_job()` が PID の生存を見て自動で外す
- 止め方：GUI の停止ボタン（STOP ファイル → run_stage がプロセスツリーごと止める）

## 18. 60秒・120秒・180秒の検証計画（すべて【未】・それぞれ承認が必要）
計算式（`job_manager.plan_for`・`fh_limits.json`）：チャンク数＝ceil(秒×16000 / 17920)、見込み＝105 ＋ チャンク数×16.5 秒、時間切れ＝300 ＋ チャンク数×16.5×1.5 秒。下の値は**この式からの推定**
| テスト | 長さ | フレーム | チャンク | 見込み【推定】 | 時間切れ |
|---|---|---|---|---|---|
| T2 | 60秒 | 1,500 | 54 | 約 17分（996秒）→ **実測 876秒で成功（61節）** | 1,636秒 |
| T3 | 120秒 | 3,000 | 108 | 約 31分（1,887秒） | 2,973秒 |
| T4 | 180秒 | 4,500 | 161 | 約 46分（2,762秒） | 4,284秒 |
- 順番：T2 は合格済み（61節）。次は T3（120秒）、T3 が合格してから T4。1回ずつ、自動の再実行なし。**T3 の前に冷却の確認**（T2 の最高 82.09℃）
- 前提：ユーザーが評価した品質（12章）・その長さの日本語の会話音声・ユーザーが指定する画像・開始前の検査に合格（コミット残量 19GiB 以上）
- 実行：GUI（GPU 生成モード）で「未検証の長さで実行する」にチェック → 「動画を生成」→ 確認画面 →「この内容で1回だけ生成する」（ユーザーが押す）
- 見ること：`worker_mem.csv` で VRAM・ワーカーのメモリ・コミット残量が**チャンクごとに増え続けないか**、GPU 温度の上がり方（8チャンクで 71→76℃。長時間で 83℃ に届かないか）、音声の末尾が欠けないか、チャンクの境目の動き、顔の一貫性、GUI でのプレビュー
- 合格したときだけ `fh_limits.json` の `gpu_verified_max_frames`（と秒）を、**実測のフレーム数・WAV の秒数のまま**更新し、履歴に追記する

---
最終の確認（この文書の作成時）：GPU 推論・生成なし。コード・環境・安全設定の変更なし（`tests_cpu\` にテストの写しを追加しただけ）。ユーザーの GUI（7861・PID 24008）は止めていない。
