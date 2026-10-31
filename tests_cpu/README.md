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
