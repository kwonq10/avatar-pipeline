# CPU テスト（GPU を使わない）

Claude Code が 2026-10-31 に使ったテスト用のスクリプトの写しです（元は Claude の一時フォルダ）。どれも GUI の Python で、`gui` フォルダから実行します。

```
cd /d D:\Avatar-Pipeline\gui
set PYTHONUTF8=1
.venv\Scripts\python.exe -X utf8 ..\tests_cpu\<スクリプト>
```

| スクリプト | 内容 | GPU |
|---|---|---|
| `test_gpu_permit.py` | GUI の1回限りの許可（permit）と、開始前の検査の拒否 18 件・成功・停止。リハーサル（ワーカーは dry_run）。条件の不合格は、このプロセスの中だけ sys_mem・gpu_status を差し替えて作る | 使わない |
| `test_length_agree.py` | 長さの判定が job_manager とワーカーで一致するか。ワーカーは mode=gpu・`--allow-gpu` なしで直接起動し、torch を読み込む前に止まる | 使わない |
| `test_length_t2.py` | 検証済み 1,500 フレーム（T2）での長さの判定：job_manager とワーカーの一致（60.000・60.019・60.04・65・120秒）と、1,500 を超えると確認が必要なこと。テスト用の音声は `lentest_t2\` に自動で作る | 使わない |
| `gui_rehearsal_harness.py` | テスト専用：リハーサルの GUI を 7862 に起動する（このプロセスの中だけ、コミット残量に +3GiB） | 使わない |
| `run_t1c.py` | **参考だけ**：T1c を Claude Code から実行したときのスクリプト。`AVATAR_ALLOW_GPU=1` で **GPU を使う**。ユーザーの承認なしに実行しないこと | **使う** |

`test_length_agree.py` は、スクリプトと同じフォルダの `lentest\` にテスト用の音声が必要です（先に作る）：

```
mkdir lentest
ffmpeg -v error -y -f lavfi -i "sine=frequency=300:duration=8.339" -ar 44100 -ac 1 -c:a pcm_s16le lentest\tone_8.339.wav
ffmpeg -v error -y -f lavfi -i "sine=frequency=300:duration=8.36"  -ar 44100 -ac 1 -c:a pcm_s16le lentest\tone_8.36.wav
ffmpeg -v error -y -f lavfi -i "sine=frequency=300:duration=9"     -ar 44100 -ac 1 -c:a pcm_s16le lentest\tone_9.wav
```

`test_gpu_permit.py` と `test_length_agree.py` は、`gui\tmp\ui_test\dummy_tone_12s.wav` も使います（無ければ `ffmpeg -f lavfi -i "sine=frequency=440:duration=12" -ar 16000 -ac 1` で作る）。

## T2のCPU頭部解析・参考安定化ツール

以下はGUI・GPU生成エンジンから独立したCPUツールです。モデルの読み込み、
ダウンロード、外部API、GPU推論は行いません。上の古いGUI用テストとは別に、
**依存関係が既にそろっているFlashHead用Python環境**を使ってください。
新しいvenvやパッケージのインストールは不要です。

必要な既存ライブラリ：NumPy、OpenCV、MediaPipe（solutions.face_mesh対応）、
Matplotlib、Pillow。動画・音声の検証にはPATH上のFFmpegとFFprobeも使います。
Python 3.10の既存環境で検証しています。GPUを使う`run_t1c.py`などを一括実行せず、
以下の対象スクリプトだけを実行してください。

| ファイル | 目的 | 素材・出力 |
|---|---|---|
| `flashhead/tools/t2_motion_analysis.py` | 頭部位置・推定角度・EAR/MAR・画像変化・チャンク境界を計測 | T2動画が必要。新規ディレクトリへCSV・JSON・画像を保存 |
| `flashhead/tools/cpu_stabilization_trial.py` | 位置・画面内傾きの弱／中補正を比較する**参考試作** | T2動画と対応する解析が必要。実行するとCPUで比較動画を新規作成 |
| `tests_cpu/test_t2_motion_analysis.py` | 既知3軸姿勢・空CSV・出力保護を検証。オプションで既存T2資料も検証 | 引数なしは素材不要。素材依存時の結果は別の新規ディレクトリ |
| `tests_cpu/test_cpu_stabilization_trial.py` | 静止・緩やかな動き・剛体変換・位置／角度上限を検証 | 引数なしは素材不要。オプションは**既存**比較動画を復号して検証するだけで、動画を生成しない |

**CPU安定化は正式不採用です。** 頭の移動は部分的に減っても背景揺れ・端の人工補完・
ぼけが発生します。GUI・標準生成には組み込まず、推奨フィルターとして扱いません。
顔の局所変形や時間補間は行わず、2D補正で頭の3D回転を直すこともできません。

### 前提と入出力保護

- T2専用：25fps、1500フレーム、28フレーム単位の新規書き出し。
  フレーム番号は0始まり、f308=12.32秒です。PnP角度は未校正カメラと一般顔モデルによる推定値です。
- 安定化試作は512×512・44100Hz音声を前提とし、12秒・45秒・49秒付近の小区間だけを比較します。
- 解析CSVのハッシュが動画と一致することを検査します。素材・モデル・解析結果はGitHubに含めません。
- 出力先は**存在しない新しいディレクトリ**にしてください。既存出力先は拒否します。
  入力の解析ディレクトリ／比較動画ディレクトリの内部にも出力できません。
- CSVは既存ファイルを上書きしません。候補が0件でも列名付きの空CSVを保存します。
  列名なしの空データは明確なValueErrorで拒否し、ファイルを作りません。
- 元の解析資料をテスト出力先に指定しないでください。素材依存テストは入力のSHA256を前後で照合します。

### 素材不要の数値テスト（動画生成なし）

PowerShellで、新しいworktreeのルートから実行します。変数の値は自分の既存環境を指定してください。

```powershell
$cpuPython = '既存FlashHead環境のpython.exeの絶対パス'
& $cpuPython -B -X utf8 tests_cpu/test_t2_motion_analysis.py
& $cpuPython -B -X utf8 tests_cpu/test_cpu_stabilization_trial.py
```

引数なしでは素材依存検証を実行せず、その旨を表示します。数値テストは一時ディレクトリを使います。
モジュールをimportするだけではテストや動画生成を実行しません。

### ローカル素材が必要な検証（動画生成なし）

```powershell
$video = '元のT2動画のパス'
$analysis = '既存motion_analysisディレクトリのパス'
$mouthCsv = '既存mouth_open_per_frame.csvのパス'
$comparisons = '既存stabilization_testディレクトリのパス'
& $cpuPython -B -X utf8 tests_cpu/test_t2_motion_analysis.py --video $video --analysis $analysis --mouth-csv $mouthCsv --out outputs/cpu_checks/motion_run_01
& $cpuPython -B -X utf8 tests_cpu/test_cpu_stabilization_trial.py --comparisons $comparisons --out outputs/cpu_checks/clips_run_01
```

最初の検証は全1500行・53境界・既存MARの整合性と38枚の独立顔検出を確認します。
次の検証は既存12本の枚数・fps・音声長・開始時刻・復号・A/B/C音声同一性を確認します。
素材がないPCでは、これらは未実施として扱い、ダウンロードや生成で補いません。
再実行時は`run_02`など別の新規出力先にしてください。

### 解析ツールの実行例

```powershell
& $cpuPython -B -X utf8 flashhead/tools/t2_motion_analysis.py $video outputs/cpu_checks/motion_analysis_run_01
```

参考安定化ツールの構文は次のとおりです。**これは比較動画を作るコマンドなので、
動画生成を禁止した作業や通常テストでは実行しないでください。**

```powershell
& $cpuPython -B -X utf8 flashhead/tools/cpu_stabilization_trial.py $video $analysis outputs/cpu_checks/stabilization_trial_run_01 --weak 0.35 --medium 0.65
```

第66～69節の開発記録・引き継ぎ更新は別の文書PRで扱います。
未完了の`t2_lipsync_review.py`は今回の共有対象に含みません。
