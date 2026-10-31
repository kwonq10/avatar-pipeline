# 再開用の引き継ぎ（SoulX-FlashHead Pro・キャラクター音声同期動画）

> 2026-10-31：GitHub（非公開 avatar-pipeline）の準備をローカルで実施（64節）。`.gitignore` は allowlist 方式・`git init` 済み・まだ commit・push していない（ユーザーの確認待ち）
> 2026-10-31 15:40：**週の利用制限までは Claude Code で続ける**（ユーザー）。`CODEX_HANDOFF.md` は、制限のあとに Codex へ引き継ぐときの資料（PROJECT_STATUS 56・57節）。
> 7861 の古い GUI は停止済み・ミュートボタンはユーザーが確認済み（57・58節）。
> **2026-10-31 15:56 T2（日本語 60秒・1,500 フレーム）成功**（61節）：outputs\t2\20261031-154143_fh_gpu.mp4・876秒・VRAM 8,490MiB・ジョブのコミット 10.29GiB・最高 82.09℃（83℃ まで約 1℃）。GPU の承認は使い切った。**検証済みの長さは 1,500 フレーム（60.000秒）に更新済み**（62節・ユーザー承認）。未確認：発音単位のリップシンク・60秒間の表情と頭の動きの自然さ・別の画像や音声での品質・120秒以上。T3・T4 は冷却の確認と別の承認が必要（T2 の最高 82.09℃・停止の基準 83℃ まで約 0.9℃）

最終更新：2026-10-31 15:10（GUI-G1 成功〈52節〉・ミュート切り替えボタン〈54・55節〉。7861 で古い GUI が動いたままだったのが表示されない原因 → 起動時にポートの使用中を検知するよう修正。GPU の承認は使い切った）
プロジェクト：`D:\Avatar-Pipeline`（Windows 11・RTX 3060 12GB・RAM 16GB クラス）

## 最初にすること
1. `PROJECT_STATUS.md` の **51節**（長さの判定の統一）・**48〜50節**（GUI の生成ボタン・GUI-G1）と **45〜47節**（T1c・リップシンクの調査）を読む。全体の要約は 41節
2. 実行中のジョブが無いことを確かめる
   - `gui\jobs\.gpu.lock` が無いこと
   - Avatar-Pipeline 関連の Python・ffmpeg のプロセスが無いこと
3. **ユーザーの承認なしに GPU 生成をしない**（これまでの承認はすべて使い切った）

## 何を作っているか
- 任意のキャラクター画像 1枚と、外部で作った音声（作成元は問わない・API キーは不要・Gemini は使わない）から、音声と同じ長さの「しゃべる」動画を作る
  - 512×512・25fps・H.264＋AAC の MP4
  - モデルは SoulX-FlashHead Pro（Apache-2.0。Lite は VAE のライセンスで商用不可なので使わない）
- 当面の目標：1〜3分の動画を1本ずつ作る（結合は別の編集ソフトで行う）。**ユーザーが Claude Code を使わず、GUI だけで生成できること**
- `inputs\avatar_bob.png` は比較用のサンプル。**キャラクターとして固定しない**

## 今の状態（要点）
- 実 GPU 生成の成功：T1・T1b（英語 4秒）、**T1c（日本語 8.333秒・45節）**
  - T1c：全体 179秒・VRAM 8,666MiB・ジョブのコミット 10.28GiB・最高 76.5℃
- 検証済みの長さ（`fh_limits.json`）：**1,500 フレーム（60.000秒・T2）**（以前は 208 フレーム＝T1c）。判定はフレーム数で、GUI・job_manager・ワーカーの全部が `flashhead\tools\fh_length.py` を使う。長さはいつも変換後の WAV のサンプル数で測る（リップシンクの品質評価は未完了）
- リップシンク：全体の時間のずれを生む不具合は無い（46節）。改善候補 B1（音声の余韻の軽い調整・47節）と A1（先読み）は**当面保留**。元の T1c 音声を標準の入力とする
- GUI（1画面・虹色の枠）：**「動画を生成」を実装済み**（48節）
  - 通常の起動 `gui\start_gui.cmd`：GPU 生成は無効（接続テストだけ）
  - GPU 生成モード `gui\start_gui_gpu.cmd`（`app.py --gpu-enabled`。環境変数は使わない）：「動画を生成」→ バックエンドが開始前の条件を検査 → 確認パネル →「この内容で1回だけ生成する」で、そのジョブ1回だけの許可（permit）を使って開始する
  - リハーサル `app.py --gpu-rehearsal`：同じ許可と検査を通すが、推論はしない（GUI のテスト用）
- 完成動画のプレビューの下に「🔊 音声オン／🔇 ミュート中」のボタン（54節）。Gradio 5.50 のプレーヤーにはミュートのボタンが無いため、完成動画の video 要素の muted を実際に切り替える（入力音声の試聴には触れない）
- 開始前の検査（バックエンドが強制。GUI でもコマンドラインでも）：実行中のジョブなし・コミット残量 19GiB 以上・空きRAM 3GiB 以上・GPU 温度を読める・80℃ 未満・空き VRAM 8,192MiB 以上・FH-01〜03・重みの SHA256
- 実行中の監視（変更なし）：ジョブのコミット 12GiB・空きRAM 2GiB・コミット残量 5GiB・VRAM 11,800MiB・83℃ が 10秒・温度を 15秒読めない → 停止。同時に1ジョブ・自動の再実行なし

## 次の作業
### 1. GUI からの実 GPU 生成 → 2026-10-31 GUI-G1 で成功（52節）。以下はその手順（次に生成するときも毎回承認が必要）
- 入力の候補：元の T1c 音声（`inputs\落ち着いた女性-2026-10-31-13-36-…mp3`、8.333秒＝検証済み）と、ユーザーが指定する画像
- 手順
  1. 常駐アプリ（Chrome・ChatGPT・Discord など）を閉じて、コミット残量を 19GiB 以上にする（直近は 18.5〜19.3GiB で、ぎりぎり）
  2. `gui\start_gui_gpu.cmd` で起動して、ブラウザで http://127.0.0.1:7861 を開く
  3. 画像と音声を選ぶ →「動画を生成」→ 確認パネルの8項目がすべて ✅ であることを見る →「この内容で1回だけ生成する」
  4. 生成中の8項目（進捗・フレーム・経過・残り・温度・VRAM・空きRAM・コミット残量）と、停止ボタンを確かめる
  5. 完成後：プレビュー・保存先・`gui\jobs\<id>\`（result.json・worker_mem.csv・job_result.json）
- 検証済みの長さを超える音声のときは、「未検証の長さで実行する」の確認が必要（確認した permit のジョブに限り allow_unverified_length=True）。検証済みの長さは自動では更新しない
### 2. ユーザーによる T1c の再生評価（リップシンク・表情・頭の動き・人物の一貫性）
### 3. T2（60秒）は別の承認。B1・A1 は保留中

## GUI の起動について（55節）
- ユーザーが使う GUI は、ユーザーが `gui\start_gui.cmd` か `gui\start_gui_gpu.cmd` で起動する（コンソールのウィンドウを閉じれば止まる）。Claude Code が裏で起動した GUI はウィンドウが無く、ユーザーが閉じられない
- 7861 番が使用中なら、新しい GUI は「前の GUI が動いています」と表示して終了する（exit 3）。表示が古いときは、前の GUI を止めてから起動し直す
- テストは別のポート（`AVATAR_GUI_PORT=7862` など）と、専用のタブで行う

## 守ること
- GPU 生成・モデルのダウンロード・パッケージの更新・Windows の設定の変更は、ユーザーの承認なしに行わない
- `AVATAR_ALLOW_GPU` を恒久的に設定しない。GUI の GPU 生成モードは起動引数だけで有効にする
- 既存の ComfyUI・LTX-Video・MuseTalk・EchoMimic などは変更しない
- 全フレームを RAM にためる方式に戻さない（必ず frame_sink.py を使う）
- 生成動画・モデル・ログは削除しない
- 他の作業を止めうる操作（ffmpeg.exe を名前で強制終了するなど）はしない。止めるのは自分が起動したプロセスだけ
- 実測と推定を分けて記録する。結果は `PROJECT_STATUS.md` に新しい節として追記する

## 主要なファイル
| 用途 | 場所 |
|---|---|
| 状況の記録 | `PROJECT_STATUS.md`（34〜64節）、引き継ぎ資料 `CODEX_HANDOFF.md`、`README.md`・`docs/`（GitHub 用。64節） |
| GUI | `gui\app.py`、`gui\static\app.css`、`gui\job_manager.py`（permit・開始前の検査）、`gui\start_gui.cmd`、`gui\start_gui_gpu.cmd` |
| ワーカー（推論・逐次書き出し・結合・検査） | `flashhead\tools\fh_worker.py` |
| ffmpeg への逐次書き出し | `flashhead\tools\frame_sink.py` |
| 長さ・実測値の設定 | `flashhead\tools\fh_limits.json`、判定 `flashhead\tools\fh_length.py` |
| 画像のチェック | `flashhead\tools\image_check.py` |
| 解析（CPU） | `flashhead\tools\lipsync_chart.py`、`audio_tail_adjust.py`、`a1_context_check.py` |
| 接続テストの実行 | `gui\tools\stream_dryrun_test.py` |
| 監視（メモリ・VRAM・温度・停止） | `tools\run_stage.py` |
| Windows 対応のパッチ | `patches\flashhead\fh_01-03_windows_lowmem.patch` |
| 重みの SHA256 | `licenses\flashhead_weights_sha256.txt`（合格の記録 `gui\jobs\.weights_sha256_ok.json`） |
| 実 GPU の出力 | `outputs\t1\`、`outputs\t1b\`、`outputs\t1c\20261031-134405_fh_gpu.mp4` |
| 画面の写真 | `outputs\ui_screenshots_20261031\`、`outputs\ui_screenshots_20261031_gpubtn\` |
| ジョブの記録 | `gui\jobs\<job_id>\` |
