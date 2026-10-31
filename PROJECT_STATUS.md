# Avatar-Pipeline 作業状況

最終更新：2026-10-30 12:10（Claude Code）

> **2026-10-30 12:10 時点の最新状態（この節が最優先。下の「現在の状態」は 11:20 時点）**
> - 方針変更：**音声は Google Gemini TTS（gemini-3.8-flash-tts）。AivisSpeech は使わない（未導入）**
> - Gemini TTS：モジュール `tts/gemini_tts.py` 完成（独立 venv `tts/.venv`、google-genai 2.33.0）。**APIキー未設定のため音声は未生成**（費用 0 円）
> - MuseTalk 1.5：独立環境 `musetalk/.venv` を構築し、商用可の重み3種だけを取得。**S3FD・DWPose・face-parse を MediaPipe に置き換えた**（`musetalk/avatar_ext/`、元ファイルは変更なし）。マスクの確認は成功
> - **短い推論は、モデル読み込み中に空きRAMが 1.47GiB まで下がり、安全装置が強制終了した**（自動再実行はしていない）。原因と対策は「7. MuseTalk」の節を参照
> - 3〜5秒の日本語アバター動画は**まだ完成していない**

目的：1枚の人物画像から、まばたき・頭の動き・表情・日本語AI音声に合った口の動きを持つ、SNS（Shorts / TikTok / Reels / X）向けの縦型アバター動画を作る。**商用利用（商品紹介・PR・収益化）できる構成だけを本番に採用する。**

記録の区別：**【実測】**＝このPCで測った値／**【公式】**＝公式の記載／**【推定】**＝見込み

---

## 現在の状態（要約）

| 段階 | 状態 |
|---|---|
| STEP 1 ライセンス確認 | 完了。台帳 `licenses/MODEL_LICENSES.json` |
| LivePortrait（商用可の構成） | **導入・内部テスト成功**（InsightFace を MediaPipe に置き換え） |
| MuseTalk 1.5 | **未導入で停止中**。標準構成に商用利用できない／権利未確認のモデルが含まれる（下記） |
| AivisSpeech | 未導入（導入順序により MuseTalk の後） |
| SNS 書き出し | `tools/compose_sns.py` 完成・内部テスト成功 |
| 権利の記録の仕組み | 台帳2つ＋書き出し時の manifest＋公開モードの自動拒否。動作確認済み |

**停止理由：** MuseTalk 1.5 の推論に必要な顔パーツ分割モデル（face-parse BiSeNet）が CelebAMask-HQ で学習されており、そのデータセット規約は「派生データの商用利用禁止」。前処理の顔検出 S3FD は重みにライセンス表記がない。指示「商用利用できる構成が確定できない場合は停止して報告」に従い、MuseTalk は導入していない。

---

## 保護している既存環境（変更なし）
- `D:\EchoMimicV3-Flash-Test`（完成動画・ログ含む）、既存 ComfyUI（8188）、GPUドライバー、ページファイル
- EchoMimic からは `tools\uv.exe` をコピーしただけ（読み取りのみ）

## フォルダ構成
```
D:\Avatar-Pipeline\
  env.sh                 共通の環境変数（uv・HF・torch のキャッシュをすべてこの中に置く）
  tools\uv.exe           uv 0.12.9
  tools\run_stage.py     工程を安全に実行する仕組み（下記）
  tools\compose_sns.py   SNS 用 1080x1920 書き出し＋manifest＋公開前の権利チェック
  python\                uv 管理の Python 3.10.21（PATH・レジストリに登録しない）
  liveportrait\          KlingAIResearch/LivePortrait @9b294b3 ＋ .venv（独立環境）
  patches\               lp_01_mediapipe_face_detector.patch、依存関係の固定ファイル
  licenses\              MODEL_LICENSES.json、lp_weights_sha256.txt
  materials\             MATERIALS_REGISTER.json（素材台帳）
  assets\fonts\          NotoSansJP-Bold.ttf ＋ OFL.txt
  runs\<日時>_<名前>\    各実行の出力・メモリCSV・結果JSON
  logs\                  自己テストのログ
```

---

## 1. LivePortrait（商用可の構成）

### 構成【実測】
- コード：KlingAIResearch/LivePortrait（旧 KwaiVGI）commit `9b294b3d`（2026-06-02）、MIT
- Python 3.10.21、**torch 2.3.0+cu118**、torchvision 0.18.0+cu118、onnxruntime-gpu 1.18.0（CUDA が使える）、numpy 1.26.4、opencv 4.10.0.84、transformers 4.38.0、mediapipe 0.10.14
  - 依存関係は `patches/lp_resolved.txt` で固定（uv pip sync）
  - mediapipe は `--no-deps` で入れた。そのまま入れると opencv-contrib 4.11 と jax が入り、cv2 が二重になるため。absl-py・attrs・flatbuffers・protobuf 4.25.9 は別に固定した
- 重み：HF `KlingTeam/LivePortrait` の `liveportrait/` 以下の6ファイルだけ（約0.6GB）。**InsightFace・動物用・X-Pose は取得していない**
- 顔検出：**MediaPipe Face Landmarker**（`pretrained_weights/mediapipe/face_landmarker.task`、Apache-2.0）

### 置き換えの方法（patch 01）
- `src/utils/face_analysis_mediapipe.py` を追加。`src/utils/cropper.py` と `src/config/crop_config.py` を修正した
- 処理の流れ：MediaPipe の 478点 → 5点（両目の虹彩中心・鼻先・口角）→ LivePortrait 同梱の `landmark.onnx`（MIT）で 203点 → 以降は公式と同じ処理（crop.py は 203点を正式に扱える）
- `FaceAnalysisDIY` の import 自体を外した。起動ログに「Face detector: MediaPipe Face Landmarker …, InsightFace is not used.」と出ることを確認した
- 参考にしたもの：ComfyUI-LivePortraitKJ（MIT）の MediaPipe 版 cropper。ComfyUI は使っていない
- 公式は 478点のうち先頭101点を「101点形式」とみなす（点の並びが違う）。そのため KJ 方式（478点をそのまま渡す）ではなく、5点を経由する方式にした

### 内部テスト【実測】 `runs/20261030-110600_lp_internal_s9_d0/`
- 入力：公式サンプル s9.jpg（720×1280）＋ d0.mp4（512×512、25fps、78フレーム、3.12秒）。**内部確認専用**
- 結果：成功（exit 0）。出力は `out/s9--d0.mp4`（H.264、720×1280、25fps、78フレーム）
- 時間：全体 31.1秒（起動・モデル読み込み込み）。うち動きの抽出 2秒、生成 12秒（78フレーム）
- メモリ：

| 項目 | 値 |
|---|---|
| 空きRAM | 6.79 → 最小 5.17GiB |
| ジョブのコミット最大 | 4.55GiB |
| システムのコミット最大 | 16.67GiB |
| VRAM | 開始時 約1.1GB → 最大 2,734MiB（LivePortrait 分は約1.6GB） |

- 見た目（4コマを目視）：顔の切り出し・頭の向き・目・表情は見本動画に追従した。**d0 は笑って歯を見せるので、その口の動きもそのまま写る**（MuseTalk との干渉に注意）
- MediaPipe の顔検出（画像幅に対する顔の幅）：s9 28%・s0 15%・s6 22%・EchoMimic 静止画 30%。すべて検出できた

### 本番での設定（予定）
- `animation_region=all`（第一候補）、`flag_normalize_lip=True`（初期設定で有効）、`flag_stitching=True`、`driving_option=expression-friendly`
- 見本動画：**口を閉じたまま**まばたき・小さなうなずき・首をかしげる・軽い微笑みを入れる。最初の1秒は正面・無表情で静止。25fpsにそろえる
- 公式サンプルの見本動画は、すべて口を開けて話している（d0・d11・d12・d20 を目視で確認）。**本番用には自分で撮る必要がある**
- 出力の fps は見本動画の fps と同じになる（コードで確認）。25fps の見本動画なら MuseTalk の推奨と合う

---

## 2. MuseTalk 1.5（未導入・停止中）

### 推論に必要なものとライセンス
| 部品 | 判定 | 根拠 |
|---|---|---|
| コード | 商用可 | MIT |
| musetalkV15 UNet | 商用可 | MIT。README「any purpose, even commercially」 |
| sd-vae-ft-mse | 商用可 | MIT |
| whisper-tiny | 商用可 | Apache-2.0 |
| DWPose dw-ll_ucoco_384 | 注意付きで可 | 作者が Apache-2.0 を付けている |
| **face-parse 79999_iter.pth** | **不可** | CelebAMask-HQ 学習。規約は「派生データを商用に利用しない」 |
| **S3FD** | **権利未確認** | 重みにも出典の repo にもライセンスがない |
| resnet18（ImageNet） | 権利未確認 | face-parse の初期値。face-parse ごと外せば不要 |
| mmcv / mmdet / mmpose | 商用可 | Apache-2.0 |

### 商用可にする案（ユーザー判断待ち。未実施）
- **案1：MuseTalk の前処理と合成を MediaPipe に置き換える（コード修正）**
  - S3FD＋DWPose → MediaPipe Face Landmarker の点から顔の範囲（bbox）を作る
  - face-parse → MediaPipe の点（あご・口のまわりの輪郭）から下半分の顔のマスクを作り、境目をぼかす
  - 置き換え後に残るのは UNet・VAE・whisper（すべて商用可）だけになる
  - 口元のなじみの品質は検証が必要
- **案2：別のリップシンク方式を調べる。** ただし、多くの方式が InsightFace などを使っているため、同じ確認が必要

---

## 3. AivisSpeech（未導入）
- エンジンは LGPL-3.0。生成音声は商用利用可。初回起動時に BERT（CC BY-SA 4.0）と既定のモデルを自動でダウンロードする
- 音声モデルは **ACML 1.0 または CC0 だけ**を使う。ACML-NC は不可
- **以前の標準モデル Anneli は、声優の声の無断学習が問題になり2025年9月に削除された。** ライセンスの表記だけでなく、学習音声の提供者と許諾がモデルのページに明記されているかも確認する
- ACML 1.0 で禁止されている使い方：なりすまし、公式と誤認させること、誤情報、誇大・虚偽の広告、政治・宗教の宣伝など。**商品紹介で誇大な表現をすると ACML 違反になる**点に注意

---

## 4. 工程の安全な実行（`tools/run_stage.py`）
EchoMimic では、ComfyUI の /interrupt（協調的な中断）が Sampler の実行中に効かなかった。そこで「止めてと頼む」のではなく、**OS の仕組みで止める**方式にした。
1. 子プロセスを **Windows Job Object** に入れ、ジョブ全体のコミット上限（JobMemoryLimit）を掛ける。上限を超える確保は OS が失敗させる
2. 0.5秒ごとに空きRAM・システムのコミット・ジョブのコミット・VRAM（nvidia-smi で1秒ごと）を CSV に記録する。しきい値・時間切れで **TerminateJobObject を使い、プロセスツリーごと即時終了**する
3. 監視側が落ちても KILL_ON_JOB_CLOSE で子プロセスも終了する。工程ごとにプロセスが終わるので、メモリも毎回解放される

自己テスト【実測】 `logs/selftest_run_stage2/`
- 上限1GiBで3GiBを確保させた → 子プロセスで MemoryError（OS が拒否）
- 時間切れ5秒 → 5.1秒で終了。**孫プロセスも含めて消えた**ことを PID で確認

既定値：空きRAM 2.0GiB 未満で終了／ジョブのコミット 9GiB／VRAM 11,264MiB／30分

---

## 5. SNS 用の書き出し（`tools/compose_sns.py`）
- 出力：1080×1920、H.264 High、yuv420p、bt709、CRF18、AAC 192kbps 48kHz ステレオ、faststart。fps は入力のまま（25fps）
- 9:16 の入力は全面に拡大（lanczos）。それ以外は幅1080で上寄せに置き、背景は元の動画をぼかしたもの（または単色）
- 字幕：SRT → ASS（Noto Sans JP Bold 66px、白文字・黒縁）。**安全域**（上250・下480・左90・右150px）の内側に置く
- `--guides` で安全域を赤く表示できる（確認用。公開モードでは使えない）
- 内部テスト【実測】：9:16 の入力・正方形の入力の両方で、日本語字幕が安全域の内側に表示された（目視）
- **公開モード（`--mode public`）**：2つの台帳を照合し、次のどれかに当てはまると書き出しを拒否する（exit 3）
  - status が commercial_ok／ok_with_note 以外のモデル
  - internal_only の素材
  - SHA256 が台帳と違う素材
  - 許諾の証拠・肖像の同意・AI加工の可否が欠けている素材
  - 公式サンプルで公開モードを試し、拒否されることを確認した
- 書き出すたびに `<出力名>.manifest.json` を作る。使ったモデル・素材・SHA256・ffmpeg コマンド・各SNSでのAI表示の手順を記録する

## 6. SNS の AI 表示ルール（2026-10 時点の二次情報。公開時に各社のヘルプで再確認すること）
- **YouTube**：写実的な合成・改変コンテンツは、アップロード時に「改変または合成されたコンテンツ」で開示する。2026-05 から Shorts では動画の上にラベルが重なって表示される。未開示でも自動でラベルが付くことがある
- **TikTok**：写実的な AI 人物・シーンには「AI生成コンテンツ」ラベルが必要。C2PA の情報から自動で付くこともある。2026-07 から広告では必須
- **Instagram/Facebook**：写実的な動画・音声には「AI情報」ラベルが必要。AI 人物を前面に出すアカウントは「AI-generated profile」の表示が求められ、付けないと表示が制限される
- **X**：2026年から「Made with AI」の表示がある（任意）。合成・操作されたメディアのポリシーがある
- **日本の法律**：商品紹介・PR で対価を受けた場合は、景品表示法のステマ規制により「PR／広告」の表示が必要。各SNSの有料パートナーシップ機能も使う
- 運用：manifest の `ai_disclosure` を投稿チェックリストとして使う

---

## 残課題
1. **MuseTalk の商用可の構成を決める**（案1・案2のどちらにするか、ユーザー判断）
2. 口を閉じた見本動画を撮影する（自分で）。素材台帳に登録する
3. 公開用の人物画像を決め、許諾の証拠を用意する（自分の写真／許可を得た人物／商用可のAI生成画像）
4. AivisSpeech の導入と音声モデルの選定（ACML／CC0、提供者の許諾が明記されているもの）
5. アップスケーラーは未導入。Real-ESRGAN などを使う場合は、重みと学習データのライセンスを同じ基準で確認する。GFPGAN などの顔補正は facexlib の検出器（RetinaFace）の権利も確認が必要
6. 15秒・30秒へ伸ばしたときのメモリ（LivePortrait はフレームをメモリ上に置く。30秒＝750フレームで増える）【推定】
7. 公式サンプルの見本動画で作った `assets/examples/driving/d0.pkl` が repo 内に生成された（公式の挙動。内部用）

---

# 2026-10-30 午後：Gemini TTS への切り替えと MuseTalk の MediaPipe 対応

## 6. Google Gemini TTS（音声）
### 公式仕様（【公式】2026-10-30 確認：ai.google.dev の speech-generation / pricing / terms）
- モデル：**gemini-3.8-flash-tts（GA）**、gemini-3.8-flash-lite-tts（GA）。以前のもの（3.1 flash preview / 2.5 pro preview）もある
- 日本語：対応（入力の言語を自動で判別。Flash は130言語以上）
- 声：30種（Zephyr, Puck, Charon, Kore, Fenrir, Leda, Orus, Aoede, Callirrhoe, Autonoe, Enceladus, Iapetus, Umbriel, Algieba, Despina, Erinome, Algenib, Rasalgethi, Laomedeia, Achernar, Alnilam, Schedar, Gacrux, Pulcherrima, Achird, Zubenelgenubi, Vindemiatrix, Sadachbia, Sadaltager, Sulafat）
- 話し方の調整：`speech_metadata.style` に自然文で書く（例「落ち着いた、親しみやすい話し方」「ゆっくり」）。文の途中に `<short pause>` `<sigh>` などのタグも入れられる。**速さを数値で指定するパラメータはない**
- 出力：WAV（24kHz・モノラル・16bit）。16k / 8k も選べる
- API：Interactions API（`client.interactions.create`）、google-genai 2.25.0 以上が必要
- 料金：**無料枠は0円**。ただし入力・出力が Google の改善に使われ、人がレビューすることもある。有料枠は入力 $0.50 と音声出力 $9.00（100万トークンあたり、2026-12-31 まで。2027-01-01 から倍額）。音声は1秒＝25トークン
  - 見込み：4秒の音声1本 ≒ **$0.0012**、5種類の声で比べても ≒ **$0.006**（有料枠の場合）
- 規約：生成物の所有権を Google は主張しない（同じ・似た出力が他の人にも生成され得る）。18歳以上。EEA・英国・スイス向けのサービスに使う場合は有料枠のみ
- 禁止使用ポリシー：人間だけが作ったと偽ること、実在人物へのなりすまし、健康・金融などで誤解を招く主張を禁止 → **SNSでは AI 音声・AI アバターであることを表示する**

### 作ったもの
- `tts/gemini_tts.py`：複数の声を1回で作って比べられる（`--voices Kore,Aoede,...` または `all`）。`--style` で話し方を指定、`--dry-run` で料金の見込みだけを表示する。WAV を保存し、長さ・サンプルレート・チャンネル数を検査して `tts_manifest.json` に記録する
  - `store=False`（サーバーに会話を保存しない）。1つ失敗したら次を試さずに止まる
- APIキーは環境変数 `GEMINI_API_KEY`、または `secrets/gemini.env` から読む。ソース・ログ・manifest には書かない。例外メッセージにキーが含まれていても `***` に置き換える
- `secrets/` は本人と SYSTEM だけが読めるように権限を設定し、`.gitignore` 済み（`secrets/` `*.env` `.env*` `runs/` `logs/` `.venv/` も除外）
- 動作確認【実測】：`--dry-run` で見込みが表示された。キーが無いときは API を呼ばずに終了した（exit 1）

### APIキーの設定（ユーザーが行う）
1. https://aistudio.google.com/apikey で「Create API key」を押す
2. 次のどちらかに設定する
   - 推奨：PowerShell で `[Environment]::SetEnvironmentVariable('GEMINI_API_KEY','<キー>','User')` を実行し、Claude Code を起動し直す
   - または：`D:\Avatar-Pipeline\secrets\gemini.env` に `GEMINI_API_KEY=<キー>` の1行を書く（起動し直す必要はない）
3. キーをチャットに貼らない
- 有料枠（請求先アカウントを紐付けたキー）の場合、上の見込み額がかかる

### MuseTalk に渡す音声
- MuseTalk は内部で librosa を使い 16kHz に変換して読む（`audio_processor.py` で確認）。**Gemini の 24kHz WAV はそのまま渡せる**（変換は不要）

## 7. MuseTalk 1.5（MediaPipe 対応）
### 環境【実測】
- コード：TMElyralab/MuseTalk @`0a89dec4`（2025-09-26）。`musetalk/.venv`：Python 3.10.21、**torch 2.3.0+cu118**（公式は 2.0.1）、diffusers 0.30.2、transformers 4.39.2、librosa 0.11.0、mediapipe 0.10.14（`--no-deps`）、opencv 4.9.0.80。版は `patches/mt_resolved.txt` で固定
- **mmcv / mmpose / mmdet / tensorflow / gradio は入れていない**（MediaPipe で代替したため不要）
- 重み（`musetalk/models/`、SHA256 は `licenses/mt_weights_sha256.txt`）：musetalkV15（unet.pth 3.17GiB + musetalk.json）、sd-vae（safetensors）、whisper-tiny（safetensors）、mediapipe/face_landmarker.task（LivePortrait 側からコピー）
- **S3FD・DWPose・face-parse・resnet18・SyncNet は取得していない**
- LivePortrait 環境は変更していない（出力を読むだけ）

### 置き換え（patch 02、新規ファイルだけ：`musetalk/avatar_ext/`、控えは `patches/mt_02_mediapipe/`）
| 公式 | 置き換え |
|---|---|
| S3FD（顔検出）＋ DWPose の顔68点 → 顔の範囲 | MediaPipe 478点から同じ規則で作る：x はあごの線（目の高さから下の輪郭）の左右端、y は「鼻筋の点29 −（あご − 点29）」からあごまで。点29 ≒ 168→4 の 2/3 の位置。v1.5 と同じく下に extra_margin=10px |
| face-parse 'jaw' モード（肌・唇・口、鼻を除く） | MediaPipe の顔の輪郭 − 鼻（凸包）− 唇 を「肌」とし、公式と同じ形態処理（下向きのコーン状の膨張・頬の削り・頬幅90）を掛け、唇を足す。上半分を消し、ガウスぼかし（公式 get_image と同じ） |

- `guard.py`：S3FD・preprocessing（読み込んだ瞬間に DWPose と S3FD を初期化する）・dwpose・mm 系・face_parsing・insightface を **import した時点で例外**にする。動作確認済み（ブロックされることと、必要なモジュールは読めること）
- オフラインで実行（`HF_HUB_OFFLINE=1`）。権利未確認の重みを自動でダウンロードしない
- 出力の長さは**音声の長さ**で決める。人物動画が音声より長ければ先頭から使い、往復再生はしない。短いときだけ公式と同じ往復再生にする。音声は切らずに AAC で結合する
- 生成したフレームはそのまま ffmpeg に流し込む（全フレームをメモリにためない）

### 内部テスト `runs/20261030-1200_integration_internal/`（公式サンプル・内部確認専用）
1. LivePortrait【実測】：s9.jpg ＋ d11 の先頭5秒（25fps、125フレーム）を使った
   - 1回目は rich の 🚀 を cp932 で書けずに失敗した（メモリとは無関係）。`run_stage.py` で子プロセスを UTF-8 にする修正をしてから実行し、成功した
   - 所要 34.9秒、ジョブのコミット最大 4.93GiB、空きRAM 5.24 → 最小 3.12GiB、VRAM 最大 2,763MiB
2. MuseTalk マスクだけの確認【実測】：4.1秒、125フレームすべてで顔を検出（失敗0）。`musetalk/mask_check.png` を目で確認した
   - 顔の範囲は公式と同じ構図（目が上半分、口〜あごが下半分）。マスクは鼻から下（口・頬・あご）で、目・鼻は元のまま。VRAM の増加はなし
3. **MuseTalk の短い推論（英語のテスト音声4秒）：安全装置で強制終了**【実測】
   - 開始時の空きRAMは 5.32GiB（Chrome・ChatGPT・Discord などが常駐）
   - モデル読み込みの 9秒目にコミットが +4GB 増えた → 空きRAMが下がり続け、17.4秒で **1.47GiB < 1.5** になり TerminateJobObject で終了した。ジョブのコミット最大 8.30GiB、VRAM 3,983MiB
   - **自動の再実行はしていない**
   - 原因（ログの推移からの推定）：Windows では `torch.load(mmap=True)` のマッピングがコピーオンライトで、コミットに計上される。GPU に送る途中で fp32 の 3.4GB がそのまま RAM に載り、torch と CUDA の分も重なった
   - 対策案（未実施・承認待ち）：
     - (a) unet.pth を1回だけ **fp16 の safetensors（約1.7GB）に変換**し、`safetensors` で **GPU に直接**読み込む → RAM の消費は数百MBになる見込み【推定】。変換作業そのものも、run_stage の監視の下で1回だけ行う
     - (b) 生成の前に常駐アプリ（Chrome・ChatGPT・Discord）を閉じ、空きRAMを 7GiB 以上にする

## 8. 次の操作
1. **Gemini の APIキーを設定**する（上の手順。有料枠かどうかも教えてほしい）
2. MuseTalk の読み込み方法を (a) に変える許可 → 変換 → 英語のテスト音声で短い推論 → 成功したら Gemini の日本語音声で統合テスト
3. 統合テストが成功したら compose_sns.py で 1080×1920 に書き出す（公式サンプルを使うので内部確認専用）

---

# 2026-10-30 13:00〜：UNet の fp16 化と省メモリ読み込み（対策 (a) 承認済み）

## 9. fp16 化の方法（`musetalk/avatar_ext/convert_unet_fp16.py`）
- `torch.load` を使わない。PyTorch の zip 形式の中の `data.pkl` を**許可したクラスだけを通す Unpickler**で読み、各テンソルの位置・形・dtype だけを取り出す（コードは実行しない）
- safetensors のヘッダー（各テンソルの位置）は形から事前に計算できる。そこで、**ヘッダーを先に書き、テンソルを1個ずつ zip から読んで fp16 にして追記する**。CPU 上にあるのは常に1テンソル分だけで、fp32 のモデル全体は作らない
- 元の `models/musetalkV15/unet.pth` は読むだけ。SHA256 `7ebf6c98…` は変換の前後で同じ
- 出力：`models/musetalkV15_fp16/unet_fp16.safetensors`（別のファイル。既にあれば上書きしない）

| | 元 | 変換後 |
|---|---|---|
| 形式 | PyTorch zip（pickle） | safetensors |
| dtype | float32 ×686 | float16 ×686 |
| 容量 | 3,400,074,924 B（3.17GiB） | 1,699,979,144 B（1.58GiB） |
| SHA256 | 7ebf6c98… | 3cf54e55… |

- 変換【実測】`runs/20261030-1300_fp16/`：29.4秒。**ジョブのコミット最大 0.97GiB**（上限 2GiB で実行）、空きRAM 5.21 → 最小 4.75GiB
- 検証【実測】（`verify_fp16.py`、上限 2GiB で全テンソルを1個ずつ比較）
  - キーは 686/686 で一致、不足・余分・形の違いは0、すべて F16、Inf/NaN は0
  - 元との最大の差は 4.88e-4（fp16 の丸め）
  - 0 に潰れた値は 1,599 個（全 849,947,844 パラメータの 0.00019%、|x|<6e-8 の極小値）
  - 1回目の検証は safetensors の numpy 読み込みがファイル全体を確保し、2GiB の上限で止まった（変換物の問題ではない）。読み方をヘッダー直読みに変えて成功した
- **fp16 による精度低下の可能性**：重みの相対誤差は最大約 0.05%。公式の推論も `--use_float16` で同じ fp16 化をしているので、同等と考えられる【推定】。ただし、口元の画質を fp32 と比べたことはない（未確認）

## 10. GPU への読み込み（`musetalk/avatar_ext/unet_loader.py`）
1. `accelerate.init_empty_weights` で中身の無い（meta）モデルを作る → CPU に fp32 3.4GB を作らない
2. safetensors のヘッダーを直接読み、テンソルを1個ずつ `bytearray` に読む → `torch.frombuffer` → すぐ `.to(cuda)`。safetensors ライブラリの CPU 経路と、ファイル全体の mmap は使わない
3. `load_state_dict(strict=True, assign=True)` で GPU 上のテンソルをそのままパラメータにする（追加のコピーなし）
4. meta のまま残ったパラメータ・バッファが無いこと、デバイスと dtype を検査する
- `infer_mp.py` は fp16 のとき（既定）この方式を使う。fp32 は旧方式のまま（RAM を大きく使うので使わない）

## 11. 読み込みテスト（推論なし）【実測】 `runs/20261030-1300_fp16/03_load_test_*`
| 項目 | 前回（fp32 + torch.load mmap） | 今回（fp16 safetensors を1テンソルずつ GPU へ） |
|---|---|---|
| 実行前の空きRAM | 5.32GiB | 5.40GiB |
| 読み込み中の最小空きRAM | **1.47GiB（17.4秒で強制終了）** | **4.53GiB**（低下は 0.87GiB） |
| ジョブのコミット最大 | 8.30GiB（途中で終了） | **3.91GiB**（大半は CUDA と torch の DLL 分【推定】） |
| システムのコミット最大 | 21.92GiB | 17.71GiB |
| VRAM 最大（nvidia-smi） | 3,983MiB（途中で終了） | 3,331MiB（完了） |
| 読み込み時間 | — | UNet 5.99秒、VAE・whisper・PE 2.27秒、CUDA の初期化を含め全体 13.2秒（ジョブ 14.2秒） |
| GPU への配置 | — | UNet 849,947,844 パラメータすべて cuda:0 / float16。VAE・whisper も cuda:0 / float16。torch の確保 1.84GiB |

- **判定：改善した**（実測）。空きRAMの低下は約1/4以下になり、上限 6GiB のジョブで成功した

## 12. Gemini TTS
- **2026-10-30 13:10 の確認で APIキーが未設定**（環境変数のプロセス・ユーザー両方とも無し。`secrets/gemini.env` も無し）。キーの値は表示していない
- そのため **API は呼んでいない。費用は0**。Kore の音声は未生成

## 13. MuseTalk の短い推論・SNS 用動画
- **未実施**。指示「最終的な完成判定は Gemini TTS の音声で行う」に従い、キーが設定されるまで止めている
- 準備したもの：`tools/run_jp_test.sh`
  - 流れ：Kore で1本 → 音声の長さとデコードを検査（1〜5秒の範囲外なら中止）→ MuseTalk（fp16・batch 4、run_stage で空きRAM 1.5GiB・コミット 8GiB の上限）→ compose_sns で 1080×1920（internal）
  - 失敗したら止まり、自動で再実行しない
- 人物の映像は `runs/20261030-1200_integration_internal/lp_out/s9--driving_d11_5s.mp4`（5秒・125フレーム、読むだけ）
- 素材台帳に `lp_sample_d11`・`musetalk_sample_eng` を internal_only として追加した

## 残っている問題（13:10 時点）
1. Gemini の APIキーの設定（ユーザーが行う）→ その後 `bash /d/Avatar-Pipeline/tools/run_jp_test.sh`
2. 口の動きの品質（日本語との同期・口元と頬の境目・ちらつき）は推論していないので未確認
3. 常駐アプリによって、開始時の空きRAMは約5.2〜5.4GiB。今回の方式なら足りる見込み【推定】

---

# 2026-10-30 12:15〜：方針変更 → LTX-Video（Image-to-Video）で全身の自然な動きを作る

背景：LivePortrait の出力は顔が引き伸ばされて見え、期待した自然さではなかった（ユーザーの評価）。最重要目標は、**頭・首・肩・姿勢・髪が人間らしく動く実写風の女性アバター**。LivePortrait・MuseTalk・EchoMimic・ComfyUI は変更していない。

## 14. LTX-Video の調査（【公式】Lightricks/LTX-Video README・configs・HF、2026-10-30 確認）
- 2B 系：0.9.6 dev／**0.9.6 distilled**（2025-04-15）、**0.9.8 distilled 2B**（2025-07-16、fp8 版あり）。13B 系（0.9.7・0.9.8）は 12GB では厳しいため対象外
- 0.9.6 distilled の公式設定（`configs/ltxv-2b-0.9.6-distilled.yaml`）
  - 8ステップ、guidance 1（CFG なし）、STG 0、bf16
  - サンプラーはチェックポイントの設定に従う、stochastic_sampling 有効
  - decode_timestep 0.05、decode_noise_scale 0.025
- 0.9.8 2B distilled との違い
  - 2段階のマルチスケール生成（2/3 の解像度で生成 → 空間アップスケーラーで拡大して仕上げる）。タイムステップは固定
  - 最大60秒の長尺、IC-LoRA（ポーズ・深度）に対応、プロンプトの理解が向上
  - 必要なもの：本体 5.91GB と空間アップスケーラー 0.47GB
- 解像度は32の倍数、フレーム数は 8n+1。推奨の上限は 720×1280・257フレーム未満。公式の既定値は 30fps・1216×704
- テキストエンコーダー：公式設定は **PixArt-alpha/PixArt-XL-2-1024-MS の T5-XXL**（fp32、2ファイル計 14.45GB）
- プロンプト強化：プロンプトが120語未満だと、Florence-2 と Llama-3.2-3B が自動で有効になる → **使わない**（取得していない）
- Windows：公式は Linux・CUDA 12.2 で検証している。今回は torch 2.3.0+cu118 で import と CUDA を確認した
- **公式の読み込み方法では RAM 16GB で動かない**（コードで確認）
  - transformer を CPU の fp32（約8GB）、VAE を CPU、T5 を CPU の fp32（17.7GB）で読んでから GPU に移す作りになっている

## 15. 商用利用の条件（LTXV Open Weights License 0.X、`licenses/LTX-Video-Open-Weights-License-0.X.txt`）
- **年間売上1,000万ドル未満なら無償で商用利用できる**。以上の事業者は有料の商用ライセンスが必要
- **添付A(e)：生成物を、機械生成だと明示せずに公開してはならない** → SNS では AI 生成の表示が必須（compose_sns の manifest に記載済み）
- (g) 本人の同意なくなりすまし（ディープフェイク）をしてはならない。(c) 害を与える目的の偽情報を作ってはならない、ほか
- 第5条：出力について権利を主張しない。第6条：最新版を使うよう努める（旧版の利用は自己責任）→ 0.9.6 より 0.9.8 が推奨される立場
- T5（PixArt 配布）：元は Google の Apache-2.0。配布リポジトリは openrail++ の表記で、利用制限に従う

## 16. 独立環境 `D:\Avatar-Pipeline\motion-ltx`【実測】
- `LTX-Video/`：Lightricks/LTX-Video @`4b2d053`（2026-01-05）。`--no-deps -e` で導入した
- `.venv`：Python 3.10.21、torch 2.3.0+cu118、diffusers 0.32.2、transformers 4.51.3、accelerate 1.6.0。版は `requirements.lock.txt` で固定
- モデル（`motion-ltx/models/`、SHA256 は `licenses/ltx_weights_sha256.txt`）

| ファイル | 容量 | 中身 |
|---|---|---|
| `ltxv/ltxv-2b-0.9.6-distilled-04-25.safetensors` | 6.34GB（5.91GiB） | transformer 3.58GiB ＋ VAE 2.32GiB、すべて bf16 |
| `t5/text_encoder/` 2ファイル | 14.45GB（13.5GiB） | fp32 |
| `t5/tokenizer/` | 小 | — |

- キャッシュはすべて D: に置いている（uv・HF・torch）。C: には保存していない
- 自作の読み込み部品（`motion-ltx/tools/`）
  - `gpu_loader.py`：safe_open の代わりに、1テンソルずつ GPU に送る。transformer と VAE は同じファイルに入っているので、接頭辞でキーを分ける
  - `encode_prompt.py`（工程1）：T5 を1テンソルずつ bf16 で GPU に置き、公式の `LTXVideoPipeline.encode_prompt` で埋め込みを作って保存し、終了する
  - `generate_i2v.py`（工程2）：公式の transformer・VAE・scheduler・pipeline・prepare_conditioning を使う
    - transformer は meta で作って assign、VAE は GPU の bf16 で直接作る。T5 は読まない
    - 既定値：384×640・49フレーム・24fps（約2.04秒）・seed 42・image_cond_noise_scale 0.15
    - `--load-only` で読み込みだけの確認ができる。構文チェック済み、未実行

## 17. 工程1（プロンプトの埋め込み）の結果：**異常終了（自動の再実行はしていない）**【実測】`runs/20261030-1240_ltx_motion/`
- 開始時：空きRAM 5.65GiB、VRAM 1,190MiB
- T5 を GPU に読み込んでいる途中（約50秒、VRAM 5,284MiB の時点）で、run_stage のジョブのコミット上限 **6GiB** に当たり、`MemoryError` になった
- **わかったこと：Windows（WDDM）では、GPU に確保したメモリもプロセスのコミットに計上される**
  - ジョブのコミットが VRAM の使用量とほぼ同じ割合で増えた（0.9 → 5.4GiB）
  - 一方、空きRAMはほとんど減らなかった（最小 4.78GiB）。物理RAMの不足ではない
- つまり、上限の設定が小さすぎた。T5（bf16 で約8.9GiB）を GPU に置くには、ジョブのコミットが約10.5〜11GiB 必要になる見込み【推定】
- システム全体：コミット上限 31.87GiB、使用中 14.30GiB、残り **17.58GiB**（終了後）
- 終了後：残ったプロセスはなし。VRAM は 1,192MiB に戻った

## 18. 入力画像
- 「ショートボブの女性アバター画像」は、D:\Avatar-Pipeline・ダウンロード・デスクトップ・ピクチャ・D: 全体の名前検索では見つからなかった
- 権利：ユーザーの申告では「AI 生成・商用 OK」。素材台帳に `avatar_bob_v1`（パスは未指定、internal_only）として登録した
  - 公開用にするには、生成サービス名・規約の URL・実在の人物に似せていないことの記入が必要

## 次の操作（ユーザーの承認・情報待ち）
1. 入力画像のフルパスを教えてもらう
2. 工程1をジョブのコミット上限 **12GiB**（VRAM 上限 11,800MiB、空きRAM 1.5GiB 未満で停止）で1回だけ再実行する承認
3. 工程2：まず `--load-only` で読み込みだけを確認（コミット上限 11GiB）→ 問題がなければ 384×640・49フレームで1本生成

## 19. 2026-10-30 13:30〜：ユーザー画像で再開（`runs/20261030-1330_ltx_bob/`）
### 入力画像【実測】
- `inputs/avatar_bob.png`：PNG・941×1672・RGB、開けることを確認。SHA256 `7df6b8b5…be3e0`
- 生成元：OpenAI ChatGPT（架空の成人女性、ユーザー申告）。素材台帳の `avatar_bob_v1` に、生成元・規約（OpenAI Terms of Use、生成物は利用者に帰属・商用可＝ユーザー申告、公開前に再確認）・SHA256 を記録した。**internal_only のまま**
- LTX 用のコピー：幅384に Lanczos 縮小（384×682）→ 下端42pxを切って **384×640**。`runs/.../inputs/avatar_bob_384x640.png`（SHA256 `a265da9a…5a28`）。元画像は変更していない

### run_stage.py の追加
- `--min-commit-headroom-gib`：システムのコミット残量（上限−使用中）がこれを下回ったら、プロセスツリーごと強制終了する。最小値を記録する

### プロンプト（v2：肯定的な表現だけ。`motion-ltx/prompts/motion_v2_positive.txt`）
- 前回の「No rubber-like … no stretching …」は、guidance 1（CFG なし・ネガティブプロンプトなし）の distilled モデルでは否定として効かない。かえって「stretching」などの語が描かれるおそれがあるので削った【推定】
- 文面：A photorealistic close-up video of a young Japanese woman with a short brown bob haircut and soft bangs, sitting in a warmly lit living room and looking into the camera with a gentle smile. She blinks slowly and naturally, then gently tilts her head to one side, and her neck and shoulders follow the movement in a relaxed, subtle way. Her upper body shifts slightly as she breathes, and her short hair sways softly with each small head movement. Her face keeps the same proportions and features throughout. The camera is static at eye level. Soft natural window light and a warm lamp in the background, shallow depth of field, realistic skin texture, smooth and lifelike motion.

### 工程1：T5 の埋め込み → **成功**【実測】
- 開始前：空きRAM 5.80GiB、コミット 13.99/31.87GiB（残量 17.88）、VRAM 1,179MiB
- 上限：ジョブのコミット 12GiB、空きRAM 2GiB、コミット残量 5GiB、VRAM 11,800MiB
- 結果：exit 0、**80.1秒**（T5 の GPU への読み込み 約68秒）
  - 空きRAM 最小 **4.04GiB**、ジョブのコミット最大 **10.44GiB**、システムのコミット最大 24.78GiB、コミット残量 最小 **7.10GiB**、VRAM 最大 **10,627MiB**（torch の確保 8.94GiB）
- T5：4,762,310,656 パラメータすべて cuda:0 / bf16。`encoder.embed_tokens.weight` は `shared.weight` と共有している（tie_weights）ので、ファイルに無くても問題ない
- 出力：`prompt_embeds_v2.pt`（1×256×4096 bf16、使ったトークン 155、Inf/NaN なし）
- 終了後：プロセスは残っていない。空きRAM 5.54GiB、VRAM 1,257MiB、コミット残量 17.46GiB → **解放を確認した**

### 工程2：transformer と VAE の読み込み → **異常終了（自動の再実行はしていない）**【実測】
- transformer：12.73秒、1,923,385,472 パラメータ、cuda:0 / bf16、3.58GiB
- VAE：9.03秒、1,246,913,522 パラメータ、cuda:0 / bf16。合計の確保 5.93GiB、VRAM 7,757MiB
- **その直後、公式の `RectifiedFlowScheduler.from_pretrained` で OS error 1455（ページング ファイルが小さすぎる）が出て終了した**
  - 原因：公式コードは**設定（メタデータ）を読むだけのために、全テンソル 5.9GB を CPU に読み込む**（rf.py 264〜274行）。これがジョブのコミット上限 12GiB に当たり、OS が確保を拒否した（コミット最大 13.65GiB を記録）。物理RAMの空きは最小 4.76GiB で、システムのコミット残量も 10.04GiB あった
  - **安全装置（ジョブの上限）が、CPU への 5.9GB の無駄な読み込みを止めた形になった**
- 修正（実施済み）：`generate_i2v.py` で、`rf.safe_open` をテンソルを1つも見せない代わりのもの（メタデータだけ）に差し替えた。自分のメタデータ読み込みもヘッダーの直読みに変えた
  - 単体の確認（`--scheduler-only`、上限 2GiB）：7.6秒、ジョブのコミット 0.96GiB、`RectifiedFlowScheduler`（LinearQuadratic、allowed_inference_steps 8 個）の作成に成功
- **生成（工程5）は実施していない。** 指示「モデルの読み込みが正常に完了した場合のみ生成へ進む」「自動の再実行は禁止」に従い、ここで停止している

### 見込み（修正後の工程2）【推定】
- transformer ＋ VAE で VRAM 約 7.8GB（実測）。生成時の中間データは 384×640×49 なので小さい（数百MB〜1GB 程度）
- ジョブのコミット：約 8〜10GiB の見込みで、上限 12GiB の範囲内

### 次の操作（承認待ち）
- 修正版で、工程2の読み込みだけをもう一度行う → 正常で余裕があれば、そのまま1本生成する（384×640・49フレーム・24fps・8ステップ・guidance 1・seed 42）
- 埋め込み `prompt_embeds_v2.pt` は再利用できる（T5 の再実行は不要）

## 20. 2026-10-30 14:00〜：読み込みの再実行と生成（承認済み・1回だけ）→ **生成の直前で異常終了（メモリとは無関係）**
### 開始前の確認【実測】
- 画像：SHA256 は元・コピーとも台帳と一致
- scheduler の修正が反映され、単体の確認（exit 0）の記録がある
- `prompt_embeds_v2.pt`：保存されているプロンプト文と `prompts/motion_v2_positive.txt` が**完全に一致**（1×256×4096 bf16、155 トークン、Inf/NaN なし）→ T5 の再実行は不要
- T5 のプロセスは残っていない。空きRAM 5.31GiB、コミット 14.91/31.87GiB（残量 16.97GiB）、VRAM 1,365MiB
- 追加：`generate_i2v.py` に、読み込み後の安全確認を入れた（空きRAM 3GiB 以上、コミット残量 8GiB 以上、空き VRAM 2GiB 以上のときだけ生成する）

### 結果（`runs/20261030-1330_ltx_bob/03_generate_*`）【実測】
- **モデルの読み込み：成功**
  - transformer 12.16秒、VAE 8.81秒、どちらも cuda:0 / bf16。確保 5.93GiB（読み込み中のピーク 8.28GiB）
- **scheduler の修正後の動作：正常**（OS error 1455 は再発しなかった。全テンソルを CPU に読まなくなった）
- 安全確認：通過（空きRAM 4.80GiB、コミット残量 9.60GiB、空き VRAM 4.68GiB）
- **生成：開始直後の encode_prompt で `AttributeError: 'NoneType' object has no attribute 'strip'` が出て終了した**（exit 1、全体 28.0秒）
- 全体の実測：空きRAM 最小 **4.49GiB**、ジョブのコミット最大 **9.74GiB**、システムのコミット最大 23.41GiB、コミット残量 最小 **8.47GiB**、VRAM 最大 **10,130MiB**
- 動画は出力されていない。終了後に残ったプロセスはなく、VRAM は 1,259MiB に戻った
- **自動の再実行はしていない**

### 原因（コードで確認）
- 公式の `LTXVideoPipeline.__call__` は、`encode_prompt(prompt, True, ...)` と **CFG=True を固定で渡して**呼ぶ（pipeline_ltx_video.py 1029行付近）
- そのため、負のプロンプトの埋め込みを渡していないと、T5 が無い状態で負のプロンプト（None）の文字列処理をしようとして落ちる
- 一方、guidance 1・STG 0 では、各ステップで使われるのは `prompt_embeds_batch[slice(1,2)]`＝**正のプロンプトだけ**（1125行付近）。負の埋め込みはモデルに入らない
- 公式も、負の埋め込みが無いときはゼロで埋める（1049〜1057行）

### 修正（実施済み・未実行）
- `generate_i2v.py` で、`negative_prompt_embeds` と `negative_prompt_attention_mask` に**ゼロ**を渡すようにした → guidance 1 では公式と同じ結果になり、T5 の再実行も不要
- 単体の確認：埋め込みだけを渡して `encode_prompt` を CPU で呼び、通ることを確認した（モデルは読み込んでいない）

### 次の候補
1. 修正版で、読み込みと生成をもう1回だけ行う（承認が必要）。設定は同じ（384×640・49フレーム・24fps・8ステップ・guidance 1・seed 42、プロンプト v2）
2. 生成に成功したら：MP4 を検査し（49フレーム・約2.04秒）、並べた確認画像を作る。動きの自然さは目視での確認待ちとする

## 21. 2026-10-30 14:40〜：修正版で生成 → **成功（1本）**【実測】`runs/20261030-1330_ltx_bob/`
### 修正内容の確認（単体テスト、GPU 上、モデルは読み込まない）
- `generate_i2v.py` の 116〜117 行に、負の埋め込みとマスクへゼロを渡す処理が入っていることを確認した
- 形・dtype・device：正の埋め込み・負の埋め込みとも (1,256,4096) bf16 cuda:0、マスクはどちらも (1,256) int64 cuda:0
- 公式の `encode_prompt(CFG=True)` が通った。guidance 1・STG 0 で使われるスロット `batch[slice(1,2)]` が正の埋め込みと一致し、負の埋め込みはすべて 0 であることを確認した
- `prompt_embeds_v2.pt` は変更していない（再利用。SHA256 は 54741fb8… で始まる）

### 実行（`04_generate_fixed_*`、1回だけ）
- 開始前：空きRAM 4.70GiB、コミット 14.72/31.87GiB（残量 17.16GiB）、VRAM 1,255MiB。残っていたプロセスはなし
- 上限：ジョブのコミット 12GiB、空きRAM 2GiB、コミット残量 5GiB、VRAM 11,800MiB
- 読み込み：transformer **12.52秒**、VAE **8.97秒**（どちらも cuda:0 / bf16、確保 5.93GiB）
- 生成前の安全確認：通過（空きRAM 4.73GiB、コミット残量 9.79GiB、空き VRAM 4.68GiB）
- **生成：5.65秒**（8ステップ＋VAE デコード）。保存までの合計 34.19秒（ジョブ全体 35.1秒）
- 実測の最大・最小：空きRAM 最小 **4.13GiB**、ジョブのコミット最大 **9.86GiB**、システムのコミット最大 24.22GiB、コミット残量 最小 **7.66GiB**、VRAM 最大 **9,697MiB**（torch の確保のピーク 8.28GiB）
- エラー：なし（CUDA OOM・OS error 1455 も出なかった）

### 出力と検証
- **`D:\Avatar-Pipeline\runs\20261030-1330_ltx_bob\ltx_i2v_bob_384x640_49f_seed42.mp4`**（1,019,276 B、SHA256 a2e86cc5…d349）
- ffprobe：H.264 High、yuv420p、**384×640、24fps、49フレーム、2.0417秒**、映像のみ（音声トラックなし）
- 全フレームの読み取り：`-count_frames` で 49 を確認。`ffmpeg -f null` でデコードエラー 0
- 確認用の画像（フレーム 0・8・16・24・32・40・48）
  - `check_frames_0-8-16-24-32-40-48.png`：全体
  - `check_head_shoulder_crop.png`：上の 400px（頭〜肩）を拡大
  - `check_input_vs_last.png`：入力画像と最後のフレームの比較
- 機械的な指標：フレーム0との平均の差は、8 → 0.9、16 → 4.6、24 → 12.6、32 → 23.3、40 → 23.8、48 → 20.4。隣り合うフレームの差は平均 1.71、最大 7.07（フレーム26）
  - 前半（0〜16）は動きが小さく、24〜32 で大きく動いている

### 静止画で見えたこと（参考。動きの自然さはユーザーが動画を見て判断する＝**目視確認待ち**）
- 頭は、最初は左に傾き、その後まっすぐ〜反対側へ動いている。肩・上半身・髪も一緒に位置が変わっている
- 最後のフレームでは、人物が少し大きく写り、背景（額縁・ランプ・棚）の位置も変わっている → **カメラが少しズームまたは移動したように見える**（プロンプトでは static camera と指定）【静止画からの観察】
- 最後のフレームでは、目の形・笑顔の大きさ・顔の輪郭が入力画像と少し違って見える → **顔立ちが途中で変わっている可能性がある**【静止画からの観察・要目視】
- 隣り合うフレームの差の最大がフレーム26にある → この辺りで動きが速くなっている可能性がある【要目視】

### 残っている問題・次の改善候補
1. ユーザーの目視評価（頭・首・肩・髪の自然さ、顔の一貫性、カメラの動き）
2. カメラが動く場合：プロンプトの先頭近くで「locked-off tripod shot / the camera does not move」などとカメラ固定を強める。または image_cond_noise_scale（今は 0.15）を下げる
3. 顔立ちが変わる場合：動きの量を小さくする表現にする（「very subtle」「slight」）。0.9.8 2B distilled（2段階生成）を試す
4. 解像度を上げる（例：480×832、704×1216）。1080×1920 の SNS 用には、後で拡大する必要がある
5. 公開用：素材台帳の avatar_bob_v1 を公開用に切り替えるには、OpenAI の規約と、実在の人物に似ていないことの確認が必要。LTXV のライセンスにより AI 生成の表示が必須

## 22. 2026-10-30 15:10〜：LTX-Video **2B 0.9.8 distilled** との比較（1本だけ）【実測】`runs/20261030-1330_ltx_bob/`
### 公式仕様と互換性（HF のヘッダーと公式コードで確認）
- 0.9.8 2B distilled（2025-07-16）は、2B 系列の更新版。LTX 全体の最新世代（13B など）とは別のもの
- チェックポイントの構造は 0.9.6 と完全に同じ
  - キー 944/944 で、形の違いは0、すべて BF16
  - VAE と scheduler の設定は同じ。allowed_inference_steps も同じ
  - transformer の設定で違うのは `causal_temporal_positioning`（0.9.6 は True、0.9.8 は記載なし＝既定の False）だけ
- 推論の方法は大きく違う：**公式設定はマルチスケール**（`ltxv-2b-0.9.8-distilled.yaml`）
  - 1パス目：0.667 倍に縮小して、7ステップ（固定のタイムステップ）
  - 空間アップスケーラー（LatentUpsampler、fp32、1.26億パラメータ）で潜在を2倍に拡大し、AdaIN で色合いをそろえる
  - 2パス目：3ステップ
  - 最後に指定の解像度へ bilinear で縮小する
  - guidance 1・STG 0・stochastic_sampling は false（0.9.6 は true）
- 384×640 の場合：**1パス目 224×416 → 2パス目 448×832 → 384×640 に縮小**
- 必要なもの：`ltxv-2b-0.9.8-distilled.safetensors`（6.34GB）と `ltxv-spatial-upscaler-0.9.8.safetensors`（0.51GB）だけを取得。SHA256 は `licenses/ltx_weights_sha256.txt`
- T5（同じ PixArt T5-XXL）と `prompt_embeds_v2.pt` はそのまま再利用した
- 商用利用：LTXV Open Weights License（本体は 0.X、アップスケーラーのファイルに埋め込まれたものは 2025-05-05 版）。年間売上1,000万ドル未満なら無償で商用可。AI 生成の明示が必須など、条件は 0.9.6 と同じ
- 省メモリの読み込み：そのまま使えた（transformer は meta から assign、VAE は GPU の bf16 で直接作る、scheduler はメタデータだけ）
- コードの変更は小さい：`tools/generate_i2v_098.py` を新しく作った（0.9.6 用の `generate_i2v.py` は変更していない）
  - 公式のアップスケーラーは fp32 の重みに bf16 の潜在をそのまま入れるので、型が合わない（公式コードに型の変換がない）。そこで、拡大する部分だけ潜在を fp32 にして通し、元の型に戻す（アップスケーラーの精度は公式どおり fp32）

### 実行
- メモリの確保：ユーザーが Chrome・Discord・Gemini を終了した（ChatGPT Classic は 6 プロセスが残っていた）
- 開始前：空きRAM 8.44GiB、コミット 10.83/31.87GiB（**残量 21.05GiB**、目標の 19GiB 以上を満たした）、VRAM 931MiB
- 安全基準は変えていない：ジョブの上限 12GiB・空きRAM 2GiB・コミット残量 5GiB・VRAM 11,800MiB、読み込み後の確認（コミット残量 8GiB 以上）
- 読み込み：transformer 10.10秒、VAE 6.44秒、アップスケーラー 1.24秒（確保 6.42GiB）
  - 読み込み後の確認：空きRAM 7.94GiB、コミット残量 **13.08GiB**、空き VRAM 4.19GiB → 通過
- **生成：6.13秒**（2パス＋拡大＋デコード）。保存までの合計 30.21秒（ジョブ全体 31.1秒）
- 最大・最小：空きRAM 最小 **7.53GiB**、ジョブのコミット最大 **11.43GiB**（上限 12GiB に近い）、システムのコミット最大 22.05GiB、コミット残量 最小 **9.82GiB**、VRAM 最大 **10,958MiB**（torch の確保のピーク 8.39GiB）
- エラー：なし

### 出力と検証
- **`D:\Avatar-Pipeline\runs\20261030-1330_ltx_bob\ltx098_i2v_bob_384x640_49f_seed42.mp4`**（SHA256 b6dc2b3d…e9ba）
  - H.264、384×640、24fps、49フレーム、2.0417秒。全フレームを読めた（デコードエラー 0）
- 0.9.6 の動画は変更していない（SHA256 a2e86cc5… で一致）
- 比較用のファイル
  - `compare_096_left_098_right.mp4`（768×640、49フレーム、2.04秒、左が 0.9.6・右が 0.9.8）
  - `compare_096_top_098_bottom_frames.png`（同じ時刻のフレーム 0・8…48、上が 0.9.6・下が 0.9.8）
  - `compare_096_top_098_bottom_head.png`（頭〜肩を拡大、フレーム 0・12・24・36・48）

### 品質の比較（数値と静止画からの観察。自然さの最終判断は**目視確認待ち**）
| 指標 | 0.9.6 | 0.9.8 |
|---|---|---|
| フレーム0との平均の差（8/16/24/32/40/48） | 0.9 / 4.6 / 12.6 / 23.3 / 23.8 / 20.4 | **25.2** / 29.7 / 32.8 / 33.5 / 33.5 / 32.2 |
| 隣り合うフレームの差（平均・最大） | 1.71・7.07（フレーム26） | 4.29・**12.02（フレーム3）** |
| 画面左端の帯（背景）の変化（フレーム48） | 10.6 | **30.0** |

- **0.9.8 は、最初の数フレーム（最大の変化はフレーム3）で絵が大きく変わり、顔立ちが入力画像と別人に近い顔になっている**（丸顔・まっすぐな前髪・顔が正面の中央に来る）。その後の頭・肩の動きは小さい【静止画からの観察】
- 0.9.8 は背景の帯の変化も大きく、最初にカメラの位置または構図が変わっている可能性がある
- 0.9.6 は前半は入力に近く、後半（24〜32）で頭と肩が動き、顔立ちが少しずつ変わる
- 両方とも目視確認待ち：まばたき・髪の自然さ・首と肩の連動・ちらつき
- 0.9.8 が悪く見える原因の候補【推定・未検証】
  1. 1パス目が 224×416 と低い解像度で、顔が小さく、入力画像の顔の情報が保たれにくい
  2. `causal_temporal_positioning` の違い
  3. 公式の i2v 設定（conditioning の強さ・ノイズ量）が、この入力・この解像度に合っていない

### 3〜5秒に伸ばす場合のメモリの見込み【推定】
- 潜在のフレーム数：49f → 7、73f（約3秒）→ 10、97f（約4秒）→ 13、121f（約5秒）→ 16（2.3倍）
- 0.9.6（384×640）：生成時の確保は今回 8.28GiB（大半は読み込み時の一時分）。121f では transformer のトークンと VAE デコードが増えて、**VRAM 9.5〜11GB、ジョブのコミット 11〜12.5GiB** の見込み → 上限 12GiB を超えるおそれがある。VAE のタイル分割デコード、または上限の再検討が必要
- 0.9.8（2パス目 448×832）：49f で既に VRAM 10.96GB・ジョブのコミット 11.43GiB → **3〜5秒は 12GB の GPU では難しい見込み**。2パス目の解像度を下げる工夫が必要
- どちらの場合も、開始前にコミット残量が 19GiB 以上あることが前提（常駐アプリを終了する）

### 次の判断（ユーザーの目視後）
- 0.9.6 の方が期待に近い場合：0.9.6 のまま、カメラ固定・動きを小さくする表現を加えたプロンプトで改善し、その後に延長を検討する
- 0.9.8 を使い続ける場合：入力解像度を上げて1パス目の顔を大きくする（例：512×896 → 1パス目 320×576）。ただしメモリが足りない可能性が高い

## 23. 2026-10-30 15:40〜：0.9.6 を採用し、プロンプト v3（カメラ固定・小さな動き）で1本生成【実測】
- 決定（ユーザー）：今後は **LTX-Video 2B 0.9.6 Distilled を採用する**。0.9.8 は追加で生成しない（環境・モデル・動画はそのまま残す）
- 変えたのはプロンプトだけ（384×640・49フレーム・24fps・8ステップ・guidance 1・seed 42・音声なし）
- プロンプト v3：`motion-ltx/prompts/motion_v3_fixedcam.txt`（112語。ユーザー指定の文面そのまま）
- 既存のファイルは変更していない：0.9.6 の v2 動画（SHA256 a2e86cc5…）、0.9.8 の動画（b6dc2b3d…）、`prompt_embeds_v2.pt`（54741fb8…）、モデル3ファイル

### 工程1：T5 で v3 の埋め込みを作成（`06_encode_prompt_v3_*`）
- 開始前：空きRAM 8.44GiB、コミット残量 21.08GiB、VRAM 933MiB
- 成功：74.6秒（T5 の GPU への読み込み 約67秒）
  - 空きRAM 最小 6.95GiB、ジョブのコミット最大 10.44GiB、コミット残量 最小 10.77GiB、VRAM 最大 10,243MiB
- 出力：`runs/20261030-1330_ltx_bob/prompt_embeds_v3.pt`（1×256×4096 bf16、153 トークン、Inf/NaN なし）
- 照合：保存されているプロンプト文と v3 のファイルが一致し、v2 とは文も埋め込みも違う → **v2 を誤って使っていない**
- 終了後：プロセスは残っていない。空きRAM 8.46GiB、コミット残量 21.12GiB、VRAM 925MiB → 解放を確認した

### 工程2：生成（`07_generate_096_v3_*`、別プロセスで1回だけ）
- 読み込み：transformer 12.79秒、VAE 9.12秒。読み込み後の確認は通過（空きRAM 7.94GiB、コミット残量 13.46GiB、空き VRAM 4.68GiB）
- **生成 4.99秒**、保存までの合計 33.81秒（ジョブ全体 34.6秒）
- 空きRAM 最小 7.55GiB、ジョブのコミット最大 9.87GiB、コミット残量 最小 11.53GiB、VRAM 最大 9,375MiB。エラーなし
- 出力：**`D:\Avatar-Pipeline\runs\20261030-1330_ltx_bob\ltx096_v3fixedcam_i2v_bob_384x640_49f_seed42.mp4`**（SHA256 5edf539b…de55）
  - H.264、384×640、24fps、49フレーム、2.0417秒。全フレームを読めた（デコードエラー 0）
- 比較用のファイル
  - `compare_096v2_left_096v3_right.mp4`（768×640、49フレーム、左が v2・右が v3）
  - `compare_input_096v2_096v3_frames.png`（1段目が入力画像、2段目が v2、3段目が v3。フレーム 0・8…48）
  - `compare_096v2_top_096v3_bottom_head.png`（頭〜肩を拡大、フレーム 0・12…48）

### 比較（数値と静止画からの観察。自然さの最終判断はユーザーの目視待ち）
| 指標 | v2（最初の動画） | v3（今回） |
|---|---|---|
| フレーム0との平均の差（8/16/24/32/40/48） | 0.9 / 4.6 / 12.6 / 23.3 / 23.8 / 20.4 | 10.5 / 13.1 / 17.8 / 18.4 / 18.8 / 18.8 |
| 隣り合うフレームの差（平均・最大） | 1.71・7.07（フレーム26） | **1.57・4.75**（フレーム20） |
| 画面左端の帯（背景）の変化（0→48） | 10.6 | 9.6 |
| 顔付近の入力画像との差（フレーム 0/24/48） | 6.9 / 25.9 / 37.2 | 7.2 / **40.0 / 39.5** |

- v3 は、急な変化（隣り合うフレームの差の最大）が小さく、動きはなめらかな傾向。背景の帯の変化もわずかに小さい【数値】
- v3 では、目を閉じている・伏せているフレームがある（フレーム 16 前後と 40 前後）→ まばたきが出ている可能性がある。プロンプトでは1回と指定したが、2回に見える【静止画からの観察】
- v3 は後半、前髪がまっすぐ下りて分け目が中央に寄り、顔がやや丸く見える → **顔立ちと髪型が入力画像から離れている可能性がある**。顔付近の差の数値も v2 より大きい（ただし、頭の動きによる位置のずれも含む指標）【要目視】
- カメラの固定：両方とも背景の大きなズームは静止画では見られない。v2 で疑われた後半の拡大は、v3 では小さい【要目視】
- 首と肩の連動・髪の自然さ・ちらつき：**目視確認待ち**

### 残っている問題
- 顔立ちの一貫性（特に前髪と分け目が中央に寄る変化）が、プロンプトでは抑えきれていない可能性がある
- 次の候補（未実施）
  - 動きの量を下げる（seed を変えて比べる）
  - image_cond_noise_scale を 0.15 から下げる
  - 最後のフレームにも入力画像を conditioning として置く（始点と終点の両方を固定する）
  - 解像度を上げて顔の情報を増やす（VRAM の見込みの確認が必要）

## 24. 2026-10-30 16:10〜：音声の方針（Gemini TTS 第1候補・Fish Audio 第2候補）と SoulX-FlashHead Lite の調査（**調査のみ。実行・ダウンロード・生成はしていない**）
### 既存の gemini_tts.py【実測】
- `tts/.venv`：google-genai 2.33.0。既定のモデルは `gemini-3.8-flash-tts`。`store=False`
- APIキーは環境変数 `GEMINI_API_KEY` か `secrets/gemini.env` から読む。キーをソース・ログに書かず、例外メッセージの中のキーも伏せる。`.gitignore` で `secrets/` `*.env` `.env*` を除外している
- `--dry-run` は正常に動いた（API は呼んでいない）
- **キーは現在も未設定**（プロセス・ユーザーの環境変数とも無し、`secrets/gemini.env` も無し）→ 構成は動く状態だが、実際の生成は未確認

### Gemini TTS の無料枠【公式、2026-10-30】
- `gemini-3.8-flash-tts`・`gemini-3.8-flash-lite-tts`：**無料枠で入力・出力とも無料**
  - 無料枠では、送った文と出力が Google の改善に使われる（人がレビューすることもある）
- 無料枠の回数制限（RPM・RPD）はドキュメントに数値が無く、AI Studio のレート制限画面で確認する
- **課金を防ぐ方法**：請求先アカウントを紐付けていないプロジェクトの APIキーを使う（紐付けると Tier 1＝有料になる）
- 規約：Google は生成物の所有権を主張しない。人が作ったと偽る・なりすましは禁止。EEA・英国・スイス向けは有料枠のみ
- 言語：入力の言語を自動で判別する（日本語・英語とも対応）。同じ声で両方の言語を話せる
- 女性の声（二次情報 geminitts.org など。公式の性別の表記は未確認）：Achernar（Soft）、Aoede（Breezy）、Autonoe（Bright）、Callirrhoe（Easy-going）、Despina（Smooth）、Erinome（Clear）、Gacrux（Mature）、Kore（Firm）、Laomedeia（Upbeat）、Leda（Youthful）、Pulcherrima（Forward）、Sulafat（Warm）、Vindemiatrix（Gentle）、Zephyr（Bright）

### Fish Audio【公式の利用規約 2024-08-18 版＋料金ページの情報（二次情報）】
- 規約：「無料ユーザーは個人の非商用利用に限る」「有料サービスの利用者だけが商用利用できる」
- API：従量課金（S2 Pro は100万文字あたり $15 と報告されている）。無料での API 利用は確認できなかった
- 生成物の所有権は利用者に与えられない書き方（『Services の権利は与えない』）。AI の開示は推奨（義務ではない）
- 「Drama 3 は年商50万ドル未満なら商用無料」という二次情報があるが、**公式では未確認**
- **判定：費用0円の条件では、商用 SNS 用の音声には使えない**（使わない）

### SoulX-FlashHead（Soul-AILab）【公式 GitHub・HF】
- 1.3B の音声駆動トーキングヘッド。**Lite**（RTX4090 で 96FPS）と Pro（4090 で 10.8FPS）がある
- 出力は 512×512・25fps。リアルタイムのストリーミングにも対応
- ライセンス
  - コード：Apache-2.0
  - DiT の重み：Apache-2.0
  - **Lite が使う VAE_LTX（LTX-Video 0.9 の VAE）は Open-RAIL-M**（利用制限付きで商用可）
  - Pro が使う VAE_Wan：Apache-2.0
  - wav2vec2-base-960h：Apache-2.0
  - HF の README に記載：「Lite は Open-RAIL-M の条件に従う必要がある」
- 重み：Model_Lite 5.69GB（fp32 と推定）、VAE_LTX 1.56GB、wav2vec2 0.35GB（合計 約7.6GB）
- 環境：Python 3.10・**torch 2.7.1＋CUDA 12.8**・diffusers 0.34 以上・transformers 4.57.3・xformers 0.0.31・**xfuser**（モジュールの先頭で import している）・decord・mediapipe 0.10.9（顔の切り出し）など
  - flash-attn は任意（無ければ PyTorch の SDPA を使う）
  - **Windows の動作は公式に書かれていない**（Linux と conda が前提）。xfuser と decord の Windows での動作は未確認
- 音声の入力：`librosa.load(audio_path, sr=16000, mono=True)` → **WAV なら何Hzでも内部で 16kHz モノラルに変換される**。fps は 25（infer_params）
  - 生成した動画には ffmpeg で元の音声を結合する（`-shortest`）
- RTX 3060 12GB・RAM 16GB：1.3B なので VRAM は足りる見込み【推定】
  - ただし公式の読み込みが CPU に fp32 で載せる作りなら、LTX と同じく1テンソルずつ GPU に送るローダーが必要になる可能性がある（未確認）

### 音声の受け渡し（提案。未実装）
- 原本：Gemini の WAV（24kHz・モノラル・16bit）を `runs/<日時>/01_audio_<声>.wav` に保存し、manifest に記録する（既存の gemini_tts.py そのまま）
- 共通の前処理（新規ツールの案）：無音のつめ物（先頭・末尾 0.2秒）・ラウドネスをそろえる・16kHz モノラル版も作る
- FlashHead Lite：`--audio_path` に原本か 16kHz 版を渡す（どちらでも内部で 16kHz に変換される）
- MuseTalk 1.5（MediaPipe 版）：同じ WAV を `--audio` に渡す（librosa で 16kHz に変換、25fps で一致する）
- 最終の SNS 書き出し（compose_sns.py）では、原本の 24kHz の WAV を AAC 48kHz にする

## 25. 2026-10-30 16:40〜：SoulX-FlashHead の実行環境とライセンスの調査（**調査のみ。ダウンロード・pip・生成・設定変更はしていない**）
- 読んだもの：公式リポジトリ @`9bc03de0`（2026-05-28）のコード（scratchpad に浅い clone、重みなし）、HF の README とファイル一覧、PyPI・PyTorch のパッケージ一覧（JSON の参照のみ）
- WSL：Ubuntu を一度だけ起動して読み取りの確認をし、すぐ終了した（元の Stopped 状態に戻した）。設定は変えていない

### 1. Windows で直接実行：**条件付きで可能**
| 部品 | Windows | 対応 |
|---|---|---|
| torch 2.7.1+cu128 | ○ win_amd64 の cp310 版がある | ドライバー 621.01 は CUDA 12.8 に対応 |
| xformers 0.0.31 | ○ win_amd64 版がある | — |
| mediapipe 0.10.9（顔の切り出し） | ○ cp310 版がある | — |
| decord | ○ 版がある | **コードでは使っていない**（requirements にあるだけ） |
| flash-attn 2.8 | × Windows 版なし | **任意**（無ければ PyTorch の SDPA を使う。flash_head_model.py で確認） |
| sageattention | × | 任意 |
| **Triton（torch.compile）** | 公式は × ／ 有志の triton-windows あり | **`COMPILE_MODEL`・`COMPILE_VAE` を False にすれば不要**（flash_head_pipeline.py 18〜22行。速度だけが下がる） |
| **nvidia-nccl-cu12==2.27.3** | × Linux のみ | 複数 GPU 用。**requirements から外す**（1 GPU では使わない） |
| **xfuser**（≥0.4.3、純 Python） | △ インストールはできるはず。import できるかは未確認 | flash_head_model.py が**先頭で import する**。1 GPU では使わない（use_usp=False）→ import 失敗の場合は try/except で代わりのものを入れる小さな修正 |
| torch.distributed | ○（gloo のみ） | 1 GPU では init しない（usp_device.py。world_size=1 なら "cuda" を返すだけ） |
| gradio / flask | ○ | 推論スクリプトには不要 |

### 2. WSL2 で実行：**条件付き**
- GPU：WSL2 の中から RTX 3060（ドライバー 621.01）が見える。libcuda も 4 件ある【実測】
- OS：Ubuntu 26.04、システムの Python は 3.14 → 3.10 の独立環境（uv 管理）が必要。既存の Ubuntu の Python には触れずに作れる
- **メモリ：WSL の上限は 7GiB**（.wslconfig が無いので既定の 50%）【実測】→ fp32 の重み 5.6GB を CPU に読む公式の方法では、ほぼ上限になる。上限を上げるには .wslconfig の変更が必要（今回は禁止）
- **ディスク：WSL の仮想ディスクは C:**（`%LOCALAPPDATA%\wsl\{…}\ext4.vhdx`、31.8GB。C: の空きは 65.9GB）→ そのまま作ると C: が増える。D: に置くには /mnt/d（遅い）か、D: に別のディストリビューションを作る（`wsl --import`。設定の追加にあたる）必要がある
- Windows との競合：WSL の VM は Windows の RAM から最大約 8GB を取るので、16GB の PC では両方の空きが減る
- 利点：Linux 用の公式手順・Triton・flash-attn がそのまま使える

### 3. 推奨する実行方法
- **Windows で直接、`D:\Avatar-Pipeline\flashhead\` に独立した uv の venv を作る**（LTX と同じ方式。C: も WSL のメモリ上限も使わない）
- 必要なコード変更は小さい（下記）。**モデルは Lite ではなく Pro を使う**（ライセンスの理由。下記 7）

### 4. 必要なディスク容量【推定】
- Pro を使う場合：Model_Pro 5.62GB ＋ VAE_Wan 0.47GB ＋ wav2vec2（safetensors のみ）0.35GB ≒ **6.5GB**
- venv（torch 2.7.1+cu128 ほか）約 7〜9GB、キャッシュ 約 3〜5GB → **合計 約 17〜20GB**（D: の空きは約 290GB）
- 参考：Lite を加えると ＋7.3GB（Model_Lite 5.69 ＋ VAE_LTX 1.56）

### 5. RAM と VRAM で予想される問題【コードを読んでの推定】
- 本体：`WanModelAudioProject.from_pretrained(model_dir)`（diffusers の ModelMixin）→ fp32 の safetensors（5.6GB）を **CPU に読み込んでから** `.to(cuda, bf16)` にする
  - Windows ではコミットに約 5.6GB 以上が計上される → **LTX と同じく1テンソルずつ GPU に送る方法に変えるべき**（使える：`tools/gpu_loader.py` の LazyGpuSafeOpen、または meta で作ってから assign）
- VAE_Wan（0.47GB）・wav2vec2（0.35GB）は小さい
- VRAM：1.3B を bf16 にすると約 2.6GB ＋ VAE ＋ wav2vec2 ＋ 中間データ（512×512、33フレームずつ）→ **12GB で足りる見込み**
- 長い動画：生成したフレームを `generated_list` に CPU のテンソルとしてためる
  - 10秒（250フレーム）×512×512×3×4B（float）≒ 0.8GB → 5〜10秒なら問題ない。もっと長い場合は逐次書き出しに変える
- 速度：Pro は RTX4090 で 10.8FPS → RTX 3060 では 2〜4FPS 程度【推定】。10秒（250フレーム）で 1〜2分程度（リアルタイムは不要なので問題ない）

### 6. 必要なコード変更（最小限・案）
1. `requirements`：nvidia-nccl-cu12・gradio・flask・decord を外す
2. `flash_head_pipeline.py`：`COMPILE_MODEL = COMPILE_VAE = False`（Triton を使わない）
3. `flash_head_model.py`：`xfuser` の import を try/except にする（1 GPU では使わない）
4. 本体の読み込みを1テンソルずつ GPU へ送る方式に変える（`torch_dtype=bf16` を指定し、CPU に fp32 の全体を作らない）
5. 監視：既存の `tools/run_stage.py` をそのまま使う（ジョブの上限・空きRAM・コミット残量・VRAM）

### 7. 商用利用の条件（原文で確認）
| 部品 | ライセンス | 商用 SNS |
|---|---|---|
| FlashHead のコード | Apache-2.0（GitHub の LICENSE） | ○ |
| FlashHead の DiT の重み（Lite・Pro） | Apache-2.0（HF の README「Component Licenses」） | ○（学習データ VividHead は CC-BY-4.0 で「研究・非商用が目的」と書かれているが、重みには作者が Apache-2.0 を付けている → 注意付きで可） |
| VAE_Wan（Wan2.1 VAE） | Apache-2.0 | ○ |
| wav2vec2-base-960h | Apache-2.0 | ○ |
| mediapipe（顔の切り出し） | Apache-2.0 | ○ |
| **VAE_LTX（LTX-Video 0.9 の VAE、Lite だけが使う）** | HF の README は「Open-RAIL-M」と書いている。しかし **Lightricks の原文（`ltx-video-2b-v0.9.license.txt`、RAIL-M、2024-11-22）は、許諾の範囲を「Permitted Purpose＝学術・研究目的のみ、商業化を除く」に限っている**（第2・3条、添付A 第1項） | **×（Lite は商用 SNS に使わない）** |
- Pro の構成：README にも「Pro だけ（VAE_Wan）なら全体が Apache-2.0」と書かれている
- 表示義務：Apache-2.0 は再配布するときに LICENSE と NOTICE を付けることだけ（成果物の動画に表示する義務はない）
  - ただし、SNS 側の AI 表示ルールと、Gemini の「人が作ったと偽らない」は別に守る
- 禁止用途：Apache には利用制限がない。Gemini の禁止使用ポリシー（なりすまし・誤解を招く主張）と各 SNS の規約に従う

### 8. 画像1枚と音声から作れる動き（公式の記述と設定からの判断）
- 画像1枚と音声から動画を作れる（`--cond_image`、`--audio_path`）
- 口の動き：wav2vec2（英語の LibriSpeech で学習）の音響特徴で口を動かす。**日本語は公式の例が無い**（例は四川語の podcast）→ 日本語の口の形の精度は未確認
- 出力：**512×512、頭を中心に切り出したもの**（`--use_face_crop` を使うと顔の約2倍の範囲。データセットも「頭の領域」）→ 肩は端が少し入る程度で、**上半身の動きは主な生成の対象ではない**
- まばたき・表情・頭の動き：公式ページは「高品質な portrait video」とだけ書き、個別の動きは明記していない（サンプル動画の目視が必要）
- 長い動画：33フレームずつ作り、前の部分の 2 潜在フレームをつなぎにする。「60秒でも誤差が積み重ならない」と書いている（Oracle-guided bidirectional distillation）
- SNS の 9:16 にするには、512×512 の頭部の映像を compose_sns で配置する必要がある（LTX のような上半身の構図は作れない）

### 9. 最終判断（提案）
- **導入を進める価値はある。ただし「Pro・Windows で直接」に限る**
  - Lite は LTX 0.9 の VAE が研究目的に限られるので、商用 SNS には使えない
  - Pro は全体が Apache-2.0 で、リアルタイムは不要なので速度の差も問題にならない
- 制約：出力は頭部中心の 512×512 で、肩・上半身の動きは限られる → **体の動きは LTX-Video 0.9.6、話す口と頭は FlashHead Pro** という分担、または FlashHead の結果を 9:16 に配置する構成を検討する
- 次の段階（承認が必要）：①Windows の venv を作り依存関係を解決するだけの試行（インストールはしない）→ ②導入 → ③Pro と wav2vec2 だけを取得 → ④読み込みテスト → ⑤5秒の生成（音声は Gemini、キーが必要）

## 26. 2026-10-30 17:10〜：FlashHead **Pro** を Windows で直接動かすための互換性確認（承認範囲：依存関係の解決だけ。インストール・ダウンロード・生成なし）
- 方針（ユーザーの決定）：**Lite は使わず Pro を候補にする。Windows 11 で直接動かすことを第一方針とする**
- 確認に使ったもの：公式コード @`9bc03de0`（scratchpad の clone）、HF の `Model_Pro/config.json`（353B のみ）、`uv pip compile`（Windows・Python 3.10 向けに依存関係を解決するだけ）
- SDPA の確認は既存の motion-ltx の torch で行った（変更していない）

### 確認結果
1. **torch 2.7.1+cu128・torchvision 0.22.1+cu128**：PyTorch の索引に win_amd64・cp310 版がある。依存関係の解決でも選ばれた。ドライバー 621.01 は CUDA 12.8 に対応
2. **xformers 0.0.31**：torch==2.7.1 に固定されていて整合する。ただし **Pro では不要**（`import xformers` があるのは Lite 用の `flash_head/ltx_video/models/transformers/attention.py` の try だけ）→ 入れない
3. **FlashAttention なし**：`flash_head/src/modules/flash_head_model.py` の `flash_attention()`（36〜66行）は FA3・FA2・Sage が import できないと、**最後の else（60〜65行）で `F.scaled_dot_product_attention` を使う**
   - 呼び出し元は `SelfAttention.forward`（174〜201行、`use_usp=False` のときは 194行で flash_attention を呼ぶ）と `CrossAttention.forward`（222〜237行、231・235行）
   - 以上から、FA なしで動く経路があることをコードで確認した
4. **SDPA の実装（RTX 3060、sm_86）**：既存の Windows 版 torch 2.3 で実測した
   - `FLASH_ATTENTION` はカーネルが無い（Windows 版には入っていない）
   - **`EFFICIENT_ATTENTION`（メモリ効率型）は動いた**（`MATH` も OK）
   - Pro の head_dim は 128（dim 1536 / 12 heads）で、メモリ効率型が対応する範囲 → 修正は不要（SDPA が自動で選ぶ）
   - 2.7.1 で FLASH が使えるかは導入後に確認する
5. **xfuser の回避**：`flash_head_model.py` の **11〜16行で無条件に import している**。実際の呼び出しは次のとおり
   - `use_usp = dist.is_initialized()`（170・258・393行）が False のとき（1 GPU で `init_process_group` をしない＝`usp_device.get_device()` は world_size=1 なら "cuda" を返すだけ）は、180〜192行・273行・473〜480行（xFuser・yunchang・get_sp_group）を**通らない**
   - `usp_device.py` の 15行・`flash_head_pipeline.py` の 118行の xfuser の import は条件付き（関数の中・use_usp のときだけ）
   - → **11〜16行を try/except にするだけで回避できる**
6. **Triton と torch.compile**：`flash_head_pipeline.py` の 19行 `COMPILE_MODEL = True`、20行 `COMPILE_VAE = True` → 130〜137行で `torch.compile`。**False にすれば Triton は不要**（遅くなるだけ）。コードの他の場所に torch.compile や triton は無い
7. **Windows で動かない依存関係**：解決した 83 個の中に nccl・triton・flash-attn・xfuser・xformers・decord は**含まれていない**（0 件）
   - その他の Linux 依存：`generate_video.py` 101行の `ffmpeg` は PATH 上の Windows 版で代わりになる
   - パスを `"/"` で分けている箇所（`flash_head_pipeline.py` 38・40行）は辞書のキーに使うだけなので害はない
   - `inference.py` 10行は設定ファイルを相対パスで開く → **作業フォルダはリポジトリの直下にする**
8. **Pro は Wan VAE だけ**：`flash_head_pipeline.py` 82行 `use_ltx = model_type == "lite"`
   - lite 以外は 95〜103行で `VAE_Wan/Wan2.1_VAE.pth` と `WanVAE` を使う
   - LtxVAE の import は lite の分岐の中（88行）だけ → **Pro では LTX 0.9 の VAE を読み込まない**（ダウンロードも不要）
9. **VRAM（推定）**
   - 重み：DiT 1.3B を bf16 にして約 2.8GB、wav2vec2（fp32）0.36GB、Wan VAE（bf16）約 0.25GB
   - 中間データ：33フレーム→潜在 9フレーム×32×32＝約 9,216 トークン、30層・dim 1536・FFN 8960 → 1回の forward で 1GB 未満（メモリ効率型の注意を使うので N² の行列は作らない）
   - VAE デコード（33×512×512）で数GB
   - → **合計 5〜8GB 程度で 12GB に収まる見込み**
10. **読み込み時の RAM とコミットのリスク**
    - **本体**：`flash_head_pipeline.py` の 111行 `WanModelAudioProject.from_pretrained(model_dir)` → diffusers が **fp32 の 5.62GB を CPU に読み込み**、113行で `.to(cuda, bf16)` にする
      - 一時的に CPU の fp32 と GPU の bf16 が同時にある → ジョブのコミットが 8〜10GiB 以上になるおそれがある＝**OS error 1455 の危険がいちばん高い箇所**
      - → **111〜113行を、meta で作り1テンソルずつ GPU（bf16）へ送って assign する方式に変える**（`motion-ltx/tools/gpu_loader.py` の LazyGpuSafeOpen を流用する）
    - **Wan VAE**：`wan/modules/vae.py` の `_video_vae()`（923行〜）は meta で作り（946行）、950行で `torch.load(map_location=device)` → assign。0.47GB と小さいのでそのままでよい
    - **wav2vec2**（139行）：0.36GB、そのままでよい
    - Windows では GPU の確保もコミットに計上される → 開始前の**コミット残量は 19GiB 以上**を前提にする（LTX と同じ運用）

### 依存関係の解決（`patches/flashhead/`）
- 案A（`requirements_win_pro.in` → `resolved_win_pro.txt`）：解決できた。ただし numpy 2.2.6、opencv 5.0.0.93 と opencv-contrib-python 5.0.0.93 が同時に入る → **mediapipe 0.10.9（numpy 1.x 向けのビルド）が実行時に失敗するおそれがある**
- **案B（採用）**（`requirements_win_pro_np1.in` → `resolved_win_pro_np1.txt`、83 個）：numpy 1.26.4、opencv-python と opencv-contrib-python の 4.11.0.86（同じ版にそろえる）、**diffusers 0.36.0（モデルを保存した版）**、mediapipe 0.10.9、protobuf 3.20.3、transformers 4.57.3、accelerate 1.15.0、torch 2.7.1+cu128、torchvision 0.22.1+cu128
- 公式の requirements.txt との違い：opencv は「4.12 以上」→ 4.11 に下げた。mediapipe は `--use_face_crop` を使うときだけ必要
- 備考：最初は numpy<2 を足した組み合わせが opencv 4.12 と衝突して解決できず、numpy を外して案A、opencv を 4.11 にして案B とした

### 次に構築するときの手順（案。**未実行**）
```bash
source /d/Avatar-Pipeline/env.sh
cd /d/Avatar-Pipeline
git clone https://github.com/Soul-AILab/SoulX-FlashHead.git flashhead/SoulX-FlashHead
git -C flashhead/SoulX-FlashHead checkout 9bc03de06bb0de82cd6bc477804512ae06144bf2
$UV venv --python 3.10 flashhead/.venv
$UV pip sync --python flashhead/.venv/Scripts/python.exe patches/flashhead/resolved_win_pro_np1.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
# 重み（Pro と必要なものだけ）：Model_Pro/*（約5.62GB）、VAE_Wan/Wan2.1_VAE.pth（0.47GB）、config.json、model_index.json、
#   facebook/wav2vec2-base-960h の config・preprocessor・tokenizer 系と model.safetensors（0.35GB）。Lite・VAE_LTX・tf/bin の重複は取得しない
```
- 変更するファイル（patch として `patches/flashhead/` に保存する。元の clone は git で差分を管理する）
  1. `flash_head/src/modules/flash_head_model.py` 11〜16行：xfuser の import を try/except にする（失敗したら None に。use_usp=False では呼ばれない）
  2. `flash_head/src/pipeline/flash_head_pipeline.py` 19・20行：`COMPILE_MODEL = False`、`COMPILE_VAE = False`
  3. 同じファイルの 111〜113行：本体を meta で作り、1テンソルずつ GPU の bf16 へ送って assign し、meta が残っていないことを確認する（ローダーの部品は LTX のものを流用）
  4. 新しいスクリプト：`run_stage.py` と組み合わせる読み込みテスト（`--load-only`）と、読み込み後の安全確認（空きRAM 3GiB 以上・コミット残量 8GiB 以上・空き VRAM 2GiB 以上）付きの生成スクリプト
- 実行の段階（各段階で承認をもらう。すべて run_stage で RAM・VRAM・システムのコミット残量を監視し、ジョブの上限 12GiB・空きRAM 2GiB・コミット残量 5GiB で停止、自動の再実行はしない）
  1. 独立環境の構築 → 2. 重みの取得（約6.5GB）→ 3. 読み込みテスト → 4. Gemini TTS で日本語の音声を作る（無料枠・請求先なしのキーが必要。**キーは未設定**）→ 5. 約3秒の会話動画（`--model_type pro --audio_encode_mode stream`）→ 6. リップシンク・表情・頭と首の評価

## 27. 2026-10-30 17:40〜：FlashHead Pro の独立環境を Windows で直接構築し、基本の動作を確認（承認済みの STEP 1。**重みのダウンロード・本体の読み込み・生成・ソースの修正はしていない**）【実測】
### 事前の確認
- C: の空き 65.71GB、D: の空き 291GB、uv 0.12.9、Python は uv 管理の 3.10.21（D:\Avatar-Pipeline\python）
- RTX 3060（ドライバー 621.01、VRAM 使用 1,059MiB、50℃）。`D:\Avatar-Pipeline\flashhead` は存在しなかった（新規）
- キャッシュ：env.sh で uv・HF・torch はすでに D:。システムの TEMP は C: → **このセッションだけ** `TMP`・`TEMP` を `D:\Avatar-Pipeline\cache\tmp` にした。ユーザー・システムの環境変数は変えていない。`UV_LINK_MODE=copy` も使った

### ソース
- `git ls-remote` で、`9bc03de06bb0de82cd6bc477804512ae06144bf2` が公式の main の HEAD であることを確認した
- `D:\Avatar-Pipeline\flashhead\SoulX-FlashHead` に clone し、その commit に checkout した（未修正、13MB、重みのファイルは含まない。LFS は取得しない）

### インストール（専用の venv だけ）
- `D:\Avatar-Pipeline\flashhead\.venv`（Python 3.10.21）
- **取得元を分けて2段階で入れた**ので、`--index-strategy unsafe-best-match` は使っていない
  1. `patches/flashhead/lock_step1_pytorch.txt`（torch 2.7.1+cu128・torchvision 0.22.1+cu128）を `--index-url https://download.pytorch.org/whl/cu128` から `--no-deps` で入れた
  2. `patches/flashhead/lock_step2_pypi.txt`（残りの 81 個）を `--index-url https://pypi.org/simple` から `--no-deps` で入れた
- 結果：**成功・83 個**。`uv pip list` を `patches/flashhead/installed_freeze.txt` に保存し、ロックファイル（`resolved_win_pro_np1.txt`）と**完全に一致**した
  - `uv pip check`：All installed packages are compatible
  - nccl・triton・flash-attn・xfuser・xformers・decord は 0 件
- 版：torch 2.7.1+cu128（CUDA 12.8、cuDNN 9.7.1）、torchvision 0.22.1+cu128、numpy 1.26.4、opencv-python と opencv-contrib-python 4.11.0.86、diffusers 0.36.0、transformers 4.57.3、accelerate 1.15.0、mediapipe 0.10.9、librosa 0.11.0、protobuf 3.20.3

### 基本の動作確認（`flashhead/env_check.py`、専用環境の中で、重みは使わない）
- Python 3.10.21 が起動した。torch、torchvision、diffusers、transformers、accelerate、safetensors、numpy、cv2、mediapipe、librosa、soundfile、pyloudnorm、imageio、imageio_ffmpeg、skimage、scipy、einops、easydict、ftfy、loguru、yaml、PIL、tokenizers、huggingface_hub を **すべて import できた**
- CUDA が使える。**RTX 3060（compute capability 8.6）を認識した**。torchvision の roi_align が CUDA で動いた
- mediapipe 0.10.9：`mp.solutions.face_detection` があり、真っ黒な画像で FaceDetection を実行できた（numpy 1.26 で問題なし）
- **SDPA（bf16、1×12×9216×128＝Pro の 33 フレーム分に当たる大きさ）**

| 実装 | 結果 |
|---|---|
| FLASH_ATTENTION | NG：No available kernel（Windows 版には 2.7.1 でも入っていない） |
| EFFICIENT_ATTENTION | **OK（ピーク 0.05GiB）** |
| CUDNN_ATTENTION | OK（ピーク 0.08GiB） |
| MATH | OK だが**ピーク 8.82GiB**（N² の行列を作る） |
| 既定（自動で選ぶ） | **OK（ピーク 0.09GiB）** → 効率型を自動で選んでいる |

  - FlashHead の fallback（`F.scaled_dot_product_attention` をそのまま呼ぶ）は自動選択になるので問題ない。**MATH に固定する設定は絶対に使わない**
- FFmpeg：PATH 上の Gyan の 8.1 full build（libx264・aac あり）と、imageio-ffmpeg に付いている v7.1（venv 内）の両方を使える

### ディスクと C: への影響
- `flashhead\.venv` 6.27GB、`flashhead\SoulX-FlashHead` 0.01GB、uv キャッシュ（D:）は今回 ＋5.98GB（合計 13.48GB）、D: の空き 277.8GB
- **C: の空き：65.71GB → 65.71GB（−0.002GB。誤差の範囲）**
  - 既存の `%LOCALAPPDATA%\uv\cache`（1.72GB、最終更新 10/29）と `pip\cache`（最終更新 10/28）には、今回書き込んでいない
- 後で消せるもの：D: の uv キャッシュ（13.48GB。LivePortrait・MuseTalk・LTX・FlashHead の分を含む）

### 変更・作成したファイル
- 新規作成
  - `D:\Avatar-Pipeline\flashhead\`（`.venv`、`SoulX-FlashHead\`、`env_check.py`）
  - `D:\Avatar-Pipeline\cache\tmp\`（一時フォルダ、`c_free_before.txt`）
  - `patches/flashhead/lock_step1_pytorch.txt`・`lock_step2_pypi.txt`・`installed_freeze.txt`
- 既存の環境（LTX-Video・MuseTalk・LivePortrait・EchoMimic・ComfyUI）、グローバルの Python と設定、Windows の設定は変更していない

### 残っている問題（次の段階）
1. FlashHead のソースは未修正なので、`flash_head_model.py` を import すると xfuser が無いために失敗するはず → 次の段階で 26節の修正 1〜3（xfuser の import の保護、compile を False に、1テンソルずつ GPU へ読み込む方式に）を入れる
2. 重み（Model_Pro・VAE_Wan・wav2vec2、約 6.5GB）は未取得 → 次の段階でユーザーの承認をもらってから
3. Gemini の APIキーが未設定（日本語音声の作成に必要）

## 28. 2026-10-30 18:10〜：ComfyUI で動かせるかの調査（**調査のみ。ダウンロード・インストール・ComfyUI の更新・生成はしていない**）
- 最終目標：無料・ローカルで、**8分以上の解説動画**を、自然に話す女性で作る
- 調べたもの：3つのリポジトリのコードを scratchpad に浅く clone（重みなし）、既存の ComfyUI の venv（`pip` の一覧を読むだけ）

### 既存の ComfyUI（`D:\D：ComfyUI\ComfyUI\ComfyUI`、Desktop 版の `.venv`）【読み取りのみ】
- Python **3.13.12**、torch **2.12.1+cu130**、numpy **2.5.2**、transformers **5.16.1**、av 18.1。diffusers と mediapipe は入っていない
- custom_nodes：ComfyUI-GGUF、ComfyUI_EchoMimic-main
- FlashHead の専用環境（Python 3.10・torch 2.7.1+cu128・numpy 1.26・transformers 4.57.3・diffusers 0.36・mediapipe 0.10.9）とは**主な版がすべて違う**
  - → **専用環境はそのまま流用できない**
  - ComfyUI に FlashHead のノードを入れると、transformers 5 系との互換性（Wav2Vec2 の継承クラス）、mediapipe 0.10.9 に Python 3.13 版が無い、numpy 2.5 といった衝突が起きる見込み

### 1. HM-RunningHub/ComfyUI_RH_FlashHead（@111bf7d、2026-02-26、Apache-2.0）
- 公式の flash_head を同梱して少し変えたもの。`COMPILE_MODEL/VAE=False`、xfuser の import は try で保護済み
- `requirements.txt` は版を固定していない（optimum-quanto・diffusers・transformers・mediapipe など）
- **Pro に 24GB 以上を推奨しているが、それを下げる工夫はコードに無い**
  - quanto は import しているだけで使っていない
  - 本体は公式と同じく `from_pretrained` で CPU に fp32 を読み込み、GPU に bf16 で置く
  - Pro と Lite の DiT はどちらも 1.3B（ファイルはほぼ同じ 5.6GB）なので、差が出るのは VAE だけ（Pro は Wan VAE）
  - 24GB の根拠は書かれていない。**この PC の実測（27節）では、注意の計算は効率型 SDPA で 0.09GiB と小さい**ので、12GB で動かすのに必要なのは次の4点だと考えられる【推定】
    1. 1テンソルずつ GPU へ送る読み込み（CPU の 5.6GB と GPU の同時保持を避ける）
    2. chunk ごとの `torch.cuda.empty_cache()`
    3. 必要なら Wan VAE のタイル分割デコード
    4. フレームをためない（下記）
- 長い音声：8秒の deque と 28フレームずつの chunk でストリーミングする。**末尾の 1 chunk に満たない部分は捨てる**（公式の generate_video.py は無音でつめ物をして残す）
- **全フレームを `generated_list` に CPU のテンソル（float）としてため、最後にまとめて書き出す**
  - 8分（25fps × 480秒＝12,000フレーム）× 512×512×3 × 4B ≒ **37.7GB** → RAM 16GB では**確実に足りない**
- 途中で失敗したときの再開：無い（最初からやり直し）
- 既定の model_type は lite → **Lite（LTX 0.9 の VAE＝研究目的のみ）は商用には使えない**ので、pro を選ぶ必要がある

### 2. UCreateAI/ComfyUI_iAskAsk_SoulX_FlashHead（@555f447、2026-06-30）
- **LICENSE ファイルが無い**（著作権者が許可していないので、改変・再配布・商用での利用の可否が不明）→ **商用の制作には使わない**
- 構造は RunningHub 版とほぼ同じ（同じ sampler の作り：deque・chunk・generated_list にためる・末尾を捨てる・再開なし）
- **`COMPILE_MODEL/VAE=True`** → Windows では Triton が無いと止まる
- `flash_head.*` を絶対 import している → RunningHub 版と一緒に入れるとモジュール名が衝突するおそれがある
- 「BF16/FP16 autocast で省メモリ」と書いているが、読み込みは公式と同じ

### 3. smthemex/ComfyUI_EchoMimic（@3a36b00、2026-02-28、Apache-2.0、Issue 106 件）
- EchoMimic v1・v2・v3、v3 flash に対応。v3 flash は「12GB で 768×768・8ステップ、block_offload を有効にする」とある
- モデルのライセンス：EchoMimicV3（Apache-2.0、生成物の権利は主張しない、違法・有害・誤情報などは禁止）、Wan2.1-Fun-1.3B-InP（Apache-2.0）、chinese-wav2vec2（MIT）、umt5（Apache-2.0）
  - v3（flash ではない方）は deepface の retinaface.h5 を使う（コードは MIT。重みは WIDER FACE で学習されていて、重み自体の条件は未確認）
- 長い動画：`length` の上限は 5,000 フレーム（25fps で 200秒）、partial_video_length（33〜193）で分割する
  - 出力は ComfyUI の IMAGE（全フレームを1つのテンソル）→ **8分は上限を超え、RAM にも載らない**
- **この PC での実績**：同じ系統の EchoMimicV3-Flash（D:\EchoMimicV3-Flash-Test）では、3秒の生成で**空きRAMが最小 0.28GiB まで下がった**（SETUP_LOG 第13工程）

### 共通の問題（公式の generate_video.py も同じ）
- 公式も `generated_list.append(video.cpu())` で**全フレームをためてから書き出す**
  - 8分の制作には、どの方式でも「**chunk ごとにすぐディスクへ書き出す**（ffmpeg への流し込み、または区切りごとの mp4）」改造が必要

### 方式の比較（8分以上・RTX 3060・RAM 16GB）
| 観点 | A：専用環境で FlashHead Pro を実行 | B：ComfyUI の FlashHead ノード | C：専用環境で実行し、ComfyUI から別プロセスとして呼ぶ |
|---|---|---|---|
| 品質 | 公式と同じ（Pro・4ステップ） | 同じモデルなら同じ。ただし末尾の音声を捨てる | A と同じ |
| 安定性 | ◎ 依存関係を固定済み（83個、照合済み）。既存の環境に影響しない | ✕ ComfyUI（Py3.13・torch2.12・transformers5）と衝突する。ComfyUI の更新で壊れる | ○ 実行部分は A と同じ。ComfyUI 側は呼び出すだけ |
| メモリ | ◎ 1テンソルずつ読み込む・逐次書き出し・run_stage の監視を自由に入れられる | ✕ ComfyUI 本体とモデルが同じプロセスに乗る。IMAGE や generated_list で全フレームをためる | ○ 別プロセスなので ComfyUI のメモリと分けられる。ただし ComfyUI を起動している分（数GB）は RAM を圧迫する |
| 長尺（8分以上） | ◎ 逐次書き出し・区切りごとのファイル・再開を作り込める | ✕ 現状は不可（全フレームをためる・再開なし・EchoMimic は 200秒が上限） | ○ A と同じだが、長時間の実行が ComfyUI のキューや時間切れに縛られやすい |
| 保守性 | ◎ 公式 commit の固定＋小さな patch 3〜5 個。ログ・記録が今の仕組みと同じ | ✕ 第三者のノード（iAskAsk はライセンスなし）と ComfyUI の更新を追う必要がある | △ A に加えて、ComfyUI 用の小さな呼び出しノードを保守する必要がある |
| Gemini の WAV との接続 | ◎ WAV のパスを直接渡す（内部で 16kHz に変換） | ○ LoadAudio → AUDIO → 一時 WAV | ○ パスを渡す |

### 推奨：**A（専用環境）を本命にし、必要になったら C を足す**。B は使わない
- 8分以上の制作に必須の「逐次書き出し・再開・メモリの監視」は、A でなければ確実に作れない
- 既存の ComfyUI（Py3.13・torch2.12）を壊す心配もない
- C の利点（GUI で操作できる）は、A が安定してから「ComfyUI のノードから A の CLI を subprocess で呼ぶだけ」の薄い連携で後から足せる
- A に加える改造（案。26節の3点に加える）
  1. **chunk ごとに ffmpeg の stdin へ書き出す**（または N 秒ごとの区切りファイル）。RAM にためない
  2. **区切りごとの再開**：音声を 30〜60秒単位に分け、各区切りの最後の motion frames（つなぎの潜在）と音声の位置を状態ファイルに保存する → 失敗した区切りからやり直せるようにする
  3. 末尾の音声は無音でつめ物をして残す（公式の generate_video.py の方式）
  4. 区切りごとに `empty_cache` をし、run_stage で監視する（ジョブの上限・空きRAM・コミット残量・VRAM）
- 見込み時間【推定】：3060 で 2〜4FPS → 8分（12,000フレーム）で **50〜100分**（再開できる前提）

## 29. 2026-10-30 18:40〜：方式A（FlashHead Pro を Windows の専用環境で）を正式な方針にし、STEP 2 を実施【実測】
- ユーザーの決定：**方式A を正式な方針にする**。B（ComfyUI のノード）は使わない。C（ComfyUI から別プロセスで呼ぶ）は将来の拡張候補として保留
- 今回の範囲：Windows 対応の修正・必要な重みの取得・SHA256 の記録・構文と import の確認・読み込みテストの準備（**モデルの読み込み・推論・生成はしていない**）

### 1. Windows 対応の修正（`flashhead/SoulX-FlashHead` @9bc03de。差分は `patches/flashhead/fh_01-03_windows_lowmem.patch`、104行）
| ID | ファイル | 内容 |
|---|---|---|
| FH-01 | `flash_head/src/modules/flash_head_model.py`（もとの 11〜16行） | xfuser の import を try/except にする。無い場合は 4 つの名前を None にする（1 GPU では use_usp=False なので呼ばれない） |
| FH-02 | `flash_head/src/pipeline/flash_head_pipeline.py`（19・20行） | `COMPILE_MODEL = False`、`COMPILE_VAE = False`（Triton を使わない） |
| FH-03 | 同じファイル（もとの 111〜113行）＋新規 `flash_head/utils/lowmem_load.py` | 環境変数 `FLASHHEAD_LOWMEM_LOAD=1`（既定）のとき、`accelerate.init_empty_weights` で作り、safetensors のヘッダーを直接読んで1テンソルずつ `param_dtype`（bf16）で GPU へ送り、`load_state_dict(strict=True, assign=True)` で読み込む。meta が残っていないか検査する。`0` にすると公式の from_pretrained に戻る |
- `self.freqs`（RoPE）は parameter ではない普通のテンソルなので、`init_empty_weights`（パラメータだけを meta にする）を使った。`torch.device("meta")` は使わない（freqs まで meta になってしまうため）

### 2. 取得した重み（`D:\Avatar-Pipeline\flashhead\models\`。SHA256 は `licenses/flashhead_weights_sha256.txt`）
| ファイル | 容量 | SHA256（HF の LFS の値と一致） |
|---|---|---|
| `SoulX-FlashHead-1_3B/Model_Pro/diffusion_pytorch_model.safetensors` | 6,030,864,656 B | e47e61b9…af250 ✓ |
| `SoulX-FlashHead-1_3B/Model_Pro/config.json` | 353 B | 94b480d3… |
| `SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth` | 507,609,880 B | 38071ab5…fd981 ✓ |
| `SoulX-FlashHead-1_3B/{config.json, model_index.json, README.md}` | 小 | 記録済み |
| `wav2vec2-base-960h/model.safetensors` | 377,607,901 B | 8aa76ab2…d74a ✓ |
| `wav2vec2-base-960h/{config, feature_extractor_config, preprocessor_config, tokenizer_config, special_tokens_map, vocab}.json, README.md` | 小 | 記録済み |
- 合計 約 6.92GB。**Model_Lite・VAE_LTX（研究目的のみ）・wav2vec2 の pytorch_model.bin と tf_model.h5（重複）は取得していない**
- Pro の重みの中身：fp32 で 843 テンソル、15.08 億パラメータ、最大のテンソルは `audio_proj.proj1_vf.weight`（151MB）。bf16 にすると GPU 上で約 **2.81GiB**
- ディスク：D: の空き 277.8GB → 272GB、C: の空きは 66〜67GB で変化なし

### 3. 構文と import の確認（専用 venv、作業フォルダはリポジトリの直下）
- `compileall`：flash_head の 34 ファイルと generate_video.py、すべて OK
- import：`flash_head.src.modules.flash_head_model`・`...pipeline.flash_head_pipeline`・`flash_head.inference`・`flash_head.wan.modules`・`flash_head.audio_analysis.wav2vec2`・`flash_head.utils.facecrop`・`flash_head.utils.utils`・`flash_head.utils.lowmem_load` を **すべて import できた**
  - xFuser は None（stub）。FA2・FA3・Sage は無い → `F.scaled_dot_product_attention` の fallback を使う
  - COMPILE_MODEL と COMPILE_VAE は False
- **重みを読まない構造の確認**：`init_empty_weights` で `WanModelAudioProject.from_config` → state_dict 843 キーが safetensors のヘッダーと**完全に一致**（不足 0・余分 0・形の違い 0）
  - パラメータはすべて meta、freqs は CPU の complex128 (1024,64) で通常どおり作られた

### 4. 次の読み込みテストの準備（**未実行**）
- `flashhead/tools/load_test.py`：`flash_head.inference.get_pipeline(world_size=1, model_type="pro")` で読み込み、本体・Wan VAE・wav2vec2 のデバイスと dtype、VRAM を記録して終了する（推論はしない）
- 実行方法（承認後）：`run_stage.py` を通す（ジョブの上限 12GiB・空きRAM 2GiB・コミット残量 5GiB・VRAM 11,800MiB・900秒）。作業フォルダは `D:\Avatar-Pipeline\flashhead\SoulX-FlashHead`
- 開始前の条件：コミット残量 19GiB 以上（常駐アプリを確認する）
- 見込み【推定】：VRAM は本体 2.81GiB ＋ Wan VAE（bf16）約 0.25GiB ＋ wav2vec2（fp32）0.35GiB ＋ CUDA の分。CPU 側は最大 151MB のテンソル1個分ずつ
- 残る確認：WanVAE の `torch.load(map_location=device)`（torch 2.7 の weights_only=True の既定）で `.pth` を読めるか
- 補足：WanVAE には `use_tiling`（タイル分割デコード）の引数がある → 長尺・高解像度のときのメモリ対策の候補

## 30. 8分以上の解説動画に向けた実装ロードマップ（設計方針のみ。**長尺対応のコードはまだ実装しない**）
### 用語を分ける
- **生成チャンク**（モデル内部の処理単位）：FlashHead は 33 フレームずつ作り、2 潜在フレーム（=5フレーム）のつなぎを次へ引き継ぐ（Pro は 33→28→28…）
  - 音声は 8 秒の deque でストリーミングして埋め込む。チャンクは編集とは無関係で、途中で切ってはいけない連続の単位
- **セグメント**（再開の単位）：例えば 30〜60 秒。セグメントの終わりに「つなぎの潜在・音声の位置・乱数の状態」を保存し、失敗したらそこから再開する
- **シーン**（編集の単位）：原稿の段落・話題ごと。シーンごとに Gemini で音声を作り、シーンの間に意図した間（無音・カット）を入れる。字幕・画角の切り替えもシーン単位

### 段階
1. **R1 短い動画を確実に動かす**（今の段階）：読み込みテスト → Gemini（無料枠・キーが必要）の日本語 3 秒 → Pro で 1 本 → 品質の評価（口の同期・表情・頭と首）
2. **R2 逐次書き出し**：チャンクごとのフレームを GPU から uint8 にして、すぐ ffmpeg の stdin（rawvideo → H.264）へ流す。`generated_list` は使わない（8分なら約 38GB になるため）。RAM の使用量はチャンク数に関係なく一定にする
3. **R3 フレームと時間の正確な管理**：25fps 固定、`フレーム数 = round(音声のサンプル数 / 16000 × 25)`
   - 末尾は無音でつめ物をして最後のチャンクまで作る（公式 generate_video.py の方式）。書き出しの後、映像を音声の長さで切りそろえる（`-shortest` に頼らない）
   - 映像と音声の長さの差は ±1 フレーム以内であることを検査する
4. **R4 セッション管理と再開**：`runs/<session>/` に manifest（原稿・音声・設定・seed・モデルの SHA256）と `progress.json`（終わったセグメント・チャンクの番号・状態ファイル）を置く
   - セグメントごとの mp4 と状態（`torch.save`：つなぎの潜在・audio deque・RNG）を書く。失敗したら最後に完了したセグメントの状態から再開する（自動の再実行はせず、ユーザーの指示で再開する）
5. **R5 時間的な状態の引き継ぎ**：pipeline の内部状態（つなぎのフレームの潜在・条件画像の埋め込み）を、セグメントをまたいで同じ値で渡す
   - 再開後に色合いや位置が飛ばないか、セグメントの境目の差を数値で検査する
6. **R6 シーンをまたいだ一貫性**：すべてのシーンで同じ条件画像・seed・color_correction を使う。シーンの始まりは条件画像から始める（つなぎを持ち越すかは、シーンの間をカットにするか連続にするかで選ぶ）
   - 顔と髪型の一貫性は、シーンの最初と最後のフレームを条件画像と比べて記録する
7. **R7 結合と書き出し**：シーンの mp4 を ffmpeg の concat で結合する（再エンコードは最後の1回だけ）
   - 16:9（1920×1080：512×512 の顔を配置し、背景やスライドと合成）と 9:16（1080×1920：compose_sns.py）を同じ素材から書き出す。字幕は SRT から焼き込む
8. **R8 8分の通し試験**：まず 30秒 → 2分 → 8分の順に、RAM・VRAM・コミットが一定であること、所要時間（見込みは 3060 で 50〜100分）、再開の動作を確認する
- 全段階で run_stage の監視（ジョブの上限・空きRAM・コミット残量・VRAM）を使う。異常時は停止し、自動では再実行しない

## 31. 2026-10-30 19:10〜：STEP 3（読み込みテストを1回）→ **開始条件を満たさなかったので、モデルを読み込まずに停止**【実測】
### 実行前の確認
| 項目 | 結果 | 判定 |
|---|---|---|
| システムのコミット残量 | **17.28GiB**（コミット 14.60/31.87GiB） | **✕（条件は 19GiB 以上）** |
| 空きRAM | 6.57GiB | — |
| VRAM 使用量 | 1,281MiB / 12,288MiB（使用率 14%） | ○ |
| 常駐アプリ | Chrome 12 プロセス・Gemini 9 プロセス・ChatGPT Classic 6 プロセスが起動中（Discord は 0） | 終了すれば約 2〜3GiB 空く見込み（前回の実績：終了後の残量 21.05GiB） |
| 残っている Avatar-Pipeline の python | なし | ○ |
| 重みの SHA256 | `sha256sum -c licenses/flashhead_weights_sha256.txt` で **14 ファイルすべて OK** | ○ |
| FH-01〜03 | HEAD は 9bc03de。今の `git diff` が `patches/flashhead/fh_01-03_windows_lowmem.patch` と**完全に一致** | ○ |
| 省メモリの読み込み | `FLASHHEAD_LOWMEM_LOAD` の既定値は "1"（flash_head_pipeline.py 114行） | ○ |
| run_stage.py | 準備済み（ジョブの上限 12GiB・空きRAM 2GiB・コミット残量 5GiB・VRAM 11,800MiB） | ○ |
- 指示「開始条件を満たさない場合は、モデルを読み込まず停止」に従い、**load_test.py は実行していない**（読み込みは0回）

### load_test.py の更新（実行はしていない。構文チェックのみ）
- 公式のコードは変えずに、`load_model_lowmem`（Pro 本体）・`WanVAE.__init__`・`Wav2Vec2Model.from_pretrained` を包んで、**モデルごとの読み込み時間と VRAM の増加**を記録するようにした
- 読み込みの後に、生成を始める条件（空きRAM 3GiB 以上・コミット残量 8GiB 以上・空き VRAM 2GiB 以上）を**判定だけ**して記録する（生成はしない）

### 公式 GUI の確認（読むだけ。gradio は入れていない・起動していない）
- `requirements.txt` は `gradio==5.50.0` を指定している → 専用 venv（83個）には入っていない。導入すると依存関係が増える（fastapi など）ので、別のロックファイルを作って解決できるか確かめる必要がある
- `gradio_app.py`（427行）
  - **Multi-GPU** の経路は `torchrun --nproc_per_node` で `generate_video.py` を実行する → nccl が無い Windows では使えない
  - Single GPU の経路は `get_pipeline` を呼び、`generated_list.append(video.cpu())`（237・275行）で**全フレームをためる**（長尺では RAM が足りない）
  - 既定の model_type は **"pro"**（327行）
  - `app.launch()`（引数なし＝127.0.0.1:7860、share なし）
- `gradio_app_streaming.py`（338行）
  - 推論を別スレッドで実行し、3 chunk ごと（`CHUNKS_PER_SEGMENT=3`）にプレビュー用の区切りの mp4 を `gradio_results/stream_preview` に書く
  - しかし最終の動画用に `accumulated.append`（225行）で**全フレームもためる** → 長尺では RAM が足りない
  - 既定の model_type は **"lite"**（310行）→ **商用では pro を選ぶ必要がある**（Lite の VAE は研究目的のみ）
- Windows 修正（FH-01〜03）との関係
  - どちらの GUI も `flash_head.inference.get_pipeline` を通るので、FH-01〜03（xfuser の保護・compile をオフ・1テンソルずつの読み込み）は**そのまま効く**
  - GUI の中で別の torch.compile や xfuser は使っていない
- GUI を使う場合の懸念と対策（案）
  1. gradio 5.50 の追加で依存関係が変わる → 案B のロックに gradio を足した別のロックを作って確認する
  2. 全フレームをためる → ロードマップの R2（ffmpeg への逐次書き出し）を GUI 側にも入れる（streaming 版の区切りの書き出しを流用し、`accumulated` を無くす）
  3. 既定の Lite → 選択肢を pro だけにする
  4. Multi-GPU の UI → 隠す
  5. GUI のプロセスはモデルを常駐させるので、run_stage の外で動く → GUI のプロセス自体を run_stage（Job Object とメモリ監視）の下で起動する
  6. 外部への公開（share）は使わない（127.0.0.1 のみ）

### 次の操作（ユーザー）
- Chrome・Gemini（・ChatGPT Classic）を保存してから終了する → コミット残量が 19GiB 以上であることを再確認する → load_test.py を1回だけ実行する（今回の承認範囲の残り）

## 32. 2026-10-30 19:30〜：STEP 3 の読み込みテスト → **成功**【実測】`runs/20261030-1930_fh_load_test/`
### 事前の確認（もう一度実施）
- ユーザーが不要なアプリを終了した（Chrome・Discord・Gemini は 0。ChatGPT Classic は 6 プロセスが残っていた）
- 空きRAM 9.12GiB、コミット 11.11/31.87GiB（**残量 20.77GiB ≥ 19GiB**）、VRAM 1,020MiB、残っている python なし
- 重みの SHA256：14 ファイルすべて OK。FH-01〜03：保存した patch と一致。省メモリの読み込みが既定で有効。load_test.py の構文 OK

### 実行
- **1回目の起動は、私のコマンドの書き方の誤りで load_test.py が始まらなかった**
  - bash で `..\tools\load_test.py` の `\` が消え、`..toolsload_test.py` を開こうとして、Python が 0.5秒で終了した
  - モデルは読み込んでいない。ジョブのコミット 0.01GiB
  - ログは `fh_load_test_try0_path_error_*` として残した
- 承認された読み込みテストは1度も実行されていなかったので、開始条件を確かめ直してから（空きRAM 9.11GiB・残量 20.73GiB・VRAM 1,013MiB）、**絶対パスで1回だけ実行した**
- 結果：**exit 0、ジョブ全体 22.4秒**

### モデルごとの結果
| モデル | 結果 | 時間 | 置き場所・dtype | GPU メモリ |
|---|---|---|---|---|
| SoulX-FlashHead Pro（FH-03 の1テンソルずつ読み込む方式） | 成功 | **5.37秒** | cuda:0 / **bfloat16**、15.08億パラメータ | **+2.81GiB** |
| Wan VAE（`torch.load(map_location=cuda)`、torch 2.7 の weights_only の既定） | 成功（**weights_only のエラーなし**） | 0.61秒 | cuda:0 / bfloat16、1.27億パラメータ | +0.25GiB |
| wav2vec2-base-960h | 成功 | 0.08秒（from_pretrained の部分） | cuda:0 / float32、9,437万パラメータ | 約 0.35GiB（CPU で作ったあと `.to(device)` で GPU に送るので、包んだ区間では 0 と出た。全体の確保から逆算した値） |
- pipeline 全体：6.69秒、`use_ltx=False`、VAE は WanVAE（**Pro は LTX の VAE を使っていない**）
- torch の確保：3.41GiB（ピークも 3.41GiB）
- 起動から CUDA の準備までが 5.6秒、import が約 9秒

### 実測の最大・最小（run_stage、0.5秒ごと）
- 空きRAM 最小 **7.78GiB**（開始時 9.10GiB）
- ジョブのコミット最大 **6.41GiB**（上限 12GiB）
- システムのコミット最大 17.16GiB、コミット残量 最小 **14.71GiB**
- VRAM 最大 **4,859MiB**（nvidia-smi。デスクトップの約 1.0GB を含む）
- エラー：なし（OOM・OS error 1455・weights_only・xfuser・Triton の問題はいずれも出なかった）
- 参考：LTX の T5 のとき（ジョブのコミット 10.44GiB）と比べて、FH-03 のおかげで CPU 側に fp32 の 6GB を作らず、ジョブのコミットは 6.41GiB で済んだ

### 生成を始める条件の判定（全部読み込んだ後。**判定だけで、生成はしていない**）
- 空きRAM 7.83GiB（≥3）、コミット残量 14.71GiB（≥8）、空き VRAM 7.33GiB（≥2）→ **満たした**

### 終了後の解放
- python プロセスは残っていない。空きRAM 9.07GiB、コミット 11.17GiB（残量 20.70GiB）、VRAM 1,014MiB → **CPU・GPU とも開始前の水準に戻った**

### 3〜5秒の生成へ進めるか
- **メモリの面では進められる状態**：読み込み後に VRAM 7.3GiB・コミット残量 14.7GiB の余裕がある。推論時の追加分（33フレーム・9,216 トークン・効率型 SDPA・Wan VAE のデコード）は数GB と見込む【推定】
- 残る前提：①Gemini の APIキー（請求先なしのプロジェクト。未設定）、または承認を得た別の音声。②推論中の VRAM・RAM の実測（Wan VAE のデコードのピーク）

## 33. 2026-10-30 19:50〜：GUI 専用環境と基本画面（承認範囲：環境の構築・基本画面・ダミーでの動作確認。**実際の生成と Gemini の呼び出しはしていない**）【実測】
- 注：ユーザーの指示文は「STEP 4：音声… APIキーは既存のsec」で途切れていた。STEP 4 は「キーは既存の secrets\gemini.env を使う（値は表示しない）」と解釈した。それ以降の指示は未受信

### GUI 専用環境 `D:\Avatar-Pipeline\gui\.venv`
- Python 3.10.21、**gradio 5.50.0**（公式と同じ版）。`gui/requirements.in` → `gui/requirements.lock.txt`（56個、torch・nvidia は 0）を PyPI から `--no-deps` で入れた。`uv pip check` OK
- **FlashHead の venv（83個）は変更なし**（`uv pip list` が `installed_freeze.txt` と一致）
- 容量：venv 360MB。C: の空きは 66.28GB → 66.28GB（変化なし）

### 構成（GUI と生成処理を別プロセスにする）
- `gui/app.py`（Gradio の画面）：**torch もモデルも読み込まない**（GUI のプロセスの WS は 131MB、GPU のプロセス一覧に出ない）
  - `127.0.0.1:7861` だけで待ち受け、share なし
  - アップロードの一時置き場は `GRADIO_TEMP_DIR=gui/tmp`（D:）
- `gui/job_manager.py`：1ジョブ＝`gui/jobs/<日時>_<種類>/`
  - `job.json`（入力）、`status.json`（ワーカーが書く進捗）、run_stage の記録（`job_result.json`・`job_memory.csv`・`job_stdout.txt`）、`STOP`（停止の要求）
  - **GPU 処理は同時に1つ**：`gui/jobs/.gpu.lock` に PID を書く。PID が終了していればロックを自動で解除する（Windows API の `GetExitCodeProcess` で確かめる）
  - ワーカーは **run_stage.py の監視の下**（ジョブの上限 12GiB・空きRAM 2GiB・コミット残量 5GiB・VRAM 11,800MiB・2時間）で起動する
- `tools/run_stage.py` に `--stop-file` を追加した（互換性あり。ファイルが現れたら TerminateJobObject でプロセスツリーごと止め、`aborted: ユーザーの停止` と記録する）。変更前の控えは `cache/tmp/run_stage.py.bak_before_stopfile`
- `gui/tools/dummy_job.py`：ダミーのワーカー（GPU・モデル・ネットワークを使わない）。本番と同じ約束で status.json に `{state, progress, message, chunk, chunks, output}` を書き、ffmpeg のテストパターンで 3秒の動画を作る
- 将来の本番のワーカー（未実装・ボタンは無効）：`flashhead/.venv` の Python で、`flash_head.inference` を使うワーカーを run_stage の下で起動する
  - **モデルは生成を実行するときだけ読み込み、ジョブが終わればプロセスごと解放する**（GUI に常駐させない）
  - 長尺はロードマップ R2〜R4（逐次書き出し・区切りごとの再開）をこのワーカーに実装する
- 起動：`gui\start_gui.cmd`（または `gui\.venv\Scripts\python.exe gui\app.py`）

### 画面（`http://127.0.0.1:7861`）
| # | 機能 | 状態 |
|---|---|---|
| 1 | 人物画像の選択とプレビュー（大きさを表示） | 使える |
| 2 | 日本語・英語の台本の入力（文字数・語数と読み上げ時間の目安） | 使える |
| 3 | Gemini TTS の女性の声（14種）と話し方の入力、APIキーの有無の表示（値は表示しない） | 選択は使える。**「音声を作成」ボタンは無効**（未承認・未実装と明記） |
| 4 | WAV の読み込みと再生（長さ・Hz・ch を表示） | 使える |
| 5 | 動画を生成（FlashHead Pro） | **ボタンは無効**（未承認と明記） |
| 5' | ダミーで動作確認（GPU を使わない） | 使える（確認用） |
| 6 | 状況と進捗（1秒ごとに更新、run_stage の実測値も表示） | 使える |
| 7 | 停止ボタン | 実行中だけ押せる |
| 8 | 完成動画のプレビューと MP4 の保存先 | 使える（ダミーで確認） |
| 9 | 生成履歴（ジョブ・状態・進捗・秒・VRAM・空きRAM・出力） | 使える |

### ダミーでの動作確認【実測】
- job_manager の単体テスト
  - 最後まで実行：exit 0・約 9秒・512×512・25fps・3秒の MP4
  - 二重起動：**拒否された**
  - 途中で停止：**STOP から 0.5秒で停止**、ロックも解除された
  - torch は import されていない
  - 1回目は `tasklist` の出力（cp932）を UTF-8 で読もうとして PID の確認が失敗した → Windows API に変えて解決
- 実際に GUI を起動して gradio_client（API）で確認した
  - HTTP 200。待ち受けは **127.0.0.1:7861 だけ**
  - 画像の情報（941×1672）、WAV の情報（ダミー 2.00秒・24000Hz・1ch）、台本の目安（6秒）
  - ダミーのジョブ：開始 → running → done（空きRAM最小 8.83GiB・ジョブのコミット 0.17GiB・VRAM 1,006MiB・8.7秒）
  - 実行中の二重起動は拒否された。停止 → stopped（ユーザーの停止、3.6秒）
  - 画面の設定（/config）で、「音声を作成」「動画を生成」は `interactive=False` であることを確かめた
- 一時ファイル
  - サーバーのアップロードと動画のキャッシュは `gui/tmp`（D:）に入った
  - 確認中に私のテストで C: の `%LOCALAPPDATA%\Temp\gradio` に作ったもの（最初の起動のときの空フォルダ、テスト用クライアントがダウンロードしたダミー MP4 340KB）は、中身を確かめてから削除した
  - `.gitignore` に `gui/tmp/`・`gui/jobs/`・`gui/*.log` を追加した
- 確認後は GUI を止めた。残っているプロセスはなし、VRAM 999MiB

### 次の段階（承認が必要）
1. 本番のワーカー（FlashHead Pro・短い動画・逐次書き出し）を作り、「動画を生成」を有効にする
2. Gemini TTS の呼び出し（キーが必要。無料枠のみ）を「音声を作成」に接続する
3. 長尺対応（R2〜R8）

## 34. 2026-10-30 20:20〜：GUI と FlashHead Pro のワーカーを接続（承認範囲：接続の実装とダミーでの確認。**推論・生成・Gemini の呼び出しはしていない**）【実測】
### 構成
```
GUI（gui\.venv、torch なし）── job_manager.start_job("fh_dryrun" | "fh_gpu")
   └─ run_stage.py（GUI の Python、ジョブの上限 12GiB・空きRAM 2GiB・コミット残量 5GiB・VRAM 11,800MiB・2時間・--stop-file）
        └─ flashhead\.venv\Scripts\python.exe flashhead\tools\fh_worker.py --job-dir gui\jobs\<job_id> [--allow-gpu]
```
- 環境は分けたまま：GUI に PyTorch は入っていない。ワーカーは FlashHead 専用の venv で動く（83個は変更なし）
- **ジョブごとにフォルダを分ける**：`gui\jobs\<日時>_<種類>\`
  - `inputs\image.*`・`inputs\audio.wav`（選んだファイルのコピー）
  - `job.json`（入力・出力フォルダ・seed・face crop）
  - `settings.json`（生成設定：Pro・commit・FH-01〜03・省メモリの読み込み・重みの SHA256・推論パラメータ・入力の SHA256・計画）
  - `status.json`（進捗）、`result.json`（実行結果）
  - run_stage の記録（`job_result.json`・`job_memory.csv`・`job_stdout.txt`）、`run_stage_console.txt`
  - 出力の MP4 は指定した出力フォルダに `<job_id>.mp4`
- ワーカーに API キーは渡さない（環境変数から `GEMINI_API_KEY` を除く）
- **Pro だけ**：GUI のモデル欄は「Pro（固定）」で編集不可。job_manager は `model_type="pro"` に固定。ワーカーは `get_pipeline(model_type="pro")` だけを呼ぶ
- **GPU での生成（fh_gpu）は二重に止めてある**
  1. GUI の「動画を生成」ボタンは無効
  2. job_manager は環境変数 `AVATAR_ALLOW_GPU=1` が無いと拒否する
  3. ワーカーは `--allow-gpu` が無いと拒否する

### fh_worker.py（`flashhead\tools\fh_worker.py`）
- 入力の検査：画像を開けるか、WAV（PCM）を開けるか、長さは 0.5〜20秒か（今は短い動画だけ）
- 計画（公式の stream モードと同じ切り方）
  - 1チャンク＝28フレーム（33−つなぎ5）＝1.12秒＝16kHz で 17,920 サンプル
  - チャンク数＝音声の長さを 1.12秒単位で切り上げた数（末尾は無音でつめ物）
  - 最終フレーム数＝round(音声の秒数 × 25)
- **長さの合わせ方**：生成した映像（チャンク数 × 28 フレーム）を、`-frames:v 最終フレーム数` で切り、元の WAV を `-t 元の秒数` で AAC 192k にして結合する（H.264・CRF16・faststart）
  - ffprobe で映像と音声の長さの差が 1 フレーム（0.04秒）以内であることを検査し、result.json に記録する
- `mode=dry_run`：**torch もモデルも import しない**。チャンクごとの進捗を見立て、テストパターンの映像（生成フレーム数ぶん）→ 同じ切り方と結合をする
- `mode=gpu`（**未承認・未実行**）
  - 開始前の確認：空きRAM 3GiB 以上・コミット残量 12GiB 以上
  - → FH-03 の省メモリ読み込みで Pro・Wan VAE・wav2vec2 を読み込む
  - → 読み込み後の確認：空きRAM 3GiB 以上・コミット残量 8GiB 以上・空き VRAM 2GiB 以上
  - → 公式の stream ループ（deque 8秒・`get_audio_embedding`・`run_pipeline`・`[motion:]`）
  - → フレームを uint8 にして（20秒まで）メモリに置き、書き出す → 結合 → `empty_cache`。プロセスが終わると GPU も解放される

### GUI（gui\app.py）の変更
- 「5. 生成の設定」を追加：モデル（Pro 固定・編集不可）、出力フォルダ（既定 `D:\Avatar-Pipeline\outputs`）、seed、顔の切り出し（use_face_crop）
- ボタン：**「接続テスト（FlashHead ワーカー・推論なし）」**（旧ダミーのボタンを置き換えた）。「動画を生成」と「音声を作成」は無効のまま
- 状況の表示：計画（チャンク数・生成フレーム数→切った後のフレーム数）、結果（映像のフレーム数と秒数・音声の秒数・差・1フレーム以内か）、run_stage の実測値
- プレビュー：Gradio の `allowed_paths` を「ジョブのフォルダ」と「既定の出力フォルダ」に限った。それ以外の出力フォルダでは、保存先だけを表示する（エラーにしない）

### ダミーでの確認【実測】
- 入力（ダミー）：テストパターンの画像 768×1024（`gui\tmp\dummy_inputs\dummy_person.png`）、220Hz のテスト音 4.3秒・24kHz・モノラル・16bit（Gemini の出力と同じ形式）
- job_manager から直接
  - fh_gpu は**拒否**、二重起動も**拒否**
  - fh_dryrun：exit 0・4.1秒
    - 計画はチャンク 4・生成 112 → **108 フレーム（4.32秒）に切った**。音声 4.30秒、**差 0.02秒（1フレーム以内）**、512×512
    - ジョブのコミット最大 0.18GiB・VRAM 1,026MiB（torch を読み込んでいない）
  - 途中で停止：STOP から **0.3秒で停止**、MP4 は作られない
- GUI を起動して gradio_client（API）で確認
  - 開始 → 進捗（計画を表示）→ 完了（結果の行を表示）→ プレビュー（`gui\tmp` 経由で配信）→ MP4 の保存先 → 履歴 → 停止（stopped・ユーザーの停止）
  - 画面の seed=7・face crop=True が settings.json に入った
  - ボタンの状態：「音声を作成」「動画を生成」は `interactive=False`。モデル欄は編集不可。Lite を選ぶ要素は無い
- **1回目は失敗**：出力フォルダ（`D:\Avatar-Pipeline\outputs\...`）が Gradio の許可パス外で、プレビューのときに `InvalidPathError` → 許可パスに既定の出力フォルダを加え、それ以外の場所はプレビューしないようにして解決した
- 後片付け：GUI を止めた。残っているプロセスなし、VRAM 1,019MiB、C: の gradio 一時フォルダなし。FlashHead の venv（83個）と FH-01〜03 は変更なし
- 出力：`D:\Avatar-Pipeline\outputs\connection_test\` に接続テストの MP4（テストパターン＋テスト音。人物の動画ではない）

### 音声ファイルの必要な形式（今の実装）
- **WAV（PCM。16bit または 32bit float）**。拡張子は .wav
- サンプルレートは任意（内部で librosa により 16kHz に変換）、モノラル推奨（ステレオは内部でモノラルに混ぜる）
- 長さは **0.5〜20秒**（今の段階の制限。長尺は R2〜R4 で拡張する）
- Gemini TTS の出力（24kHz・モノラル・16bit の WAV）はそのまま使える
- 先頭と末尾の無音は長さに含まれる（映像も同じ長さになる）

### 次の実機テストの手順（承認が必要）
1. 日本語の音声 3〜5秒を用意する（Gemini TTS：請求先なしのプロジェクトのキーを `secrets\gemini.env` に設定し、無料枠であることをユーザーが確認する → Kore で1本。または、ユーザーが用意した WAV）
2. 常駐アプリを終了し、コミット残量が 19GiB 以上であることを確認する
3. GUI の「動画を生成」を有効にする（`AVATAR_ALLOW_GPU=1` で GUI を起動する方式。承認後に実装）
4. 画像 `inputs\avatar_bob.png`・音声・出力フォルダを指定して1回だけ実行する（run_stage の監視の下。異常なら停止し、自動では再実行しない）
5. 確認する
   - MP4（512×512・25fps・フレーム数＝round(秒×25)・音声との差は1フレーム以内）
   - RAM・VRAM・コミットの最大最小、所要時間
   - リップシンク・表情・頭と首の動き（目視）

## 35. 2026-10-30 20:40〜：GUI を「任意のキャラクター × 外部の音声」の汎用ツールに変更（ダミーでの確認まで。**GPU での生成はしていない**）【実測】
### 仕様の変更（ユーザーの指示）
- 女性に限らない汎用ツールにする：任意のキャラクター画像を、外部で作った音声に合わせて FlashHead Pro でしゃべらせる
- キャラクター画像は**必須で、初期状態は未選択**。`inputs\avatar_bob.png` は動作確認用のサンプルとしてだけ残す（固定の入力にはしない）
- 音声の作成元（Gemini TTS・Fish Audio・録音など）に依存しない。**API キーがなくても使える**

### 変更したファイル
- **新規** `flashhead\tools\image_check.py`（FlashHead 専用 venv、CPU の MediaPipe だけ。torch・モデルは使わない）
  - 公式の入力の作り方を同じ手順で再現する
    - 中央切り：`resize_and_centercrop`
    - 顔の切り出し：`process_image`（MediaPipe・信頼度 0・**最初に見つかった顔 boxes[0]**・幅×2.0・上 55% / 下 45%）
  - 判定（ok / warn / ng）と理由（日本語）を出す
    - ng：開けない／短い辺が 256px 未満／**信頼度 0.5 以上の顔が見つからない**
    - warn：短い辺が 512px 未満／顔が複数／**中央切りで顔が外に出る・頭の上（髪）が切れる**（上の余白が顔の高さの 35% 未満）・顔が小さい／**顔の切り出しで範囲が画像の端で削られ、正方形にならず歪む**
    - どちらの切り方を勧めるか（`recommend_use_face_crop`）も出す
  - **モデルに入る 512×512 のプレビュー**を、中央切り・顔切りの 2 枚出力する
  - 結果には「顔検出は実写の人物向け。**アニメ・イラスト・動物は動作未確認**」と必ず書く
- `gui\job_manager.py`
  - `check_image()`（FlashHead の venv で image_check.py を別プロセスで実行。結果は `gui\tmp\image_check\<sha先頭>\`）
  - `audio_info()`（ffprobe。WAV 以外も読める）
  - ジョブの開始時：画像が必須・**ng なら拒否**（ジョブのフォルダは残さない）
    - 音声は WAV・MP3・FLAC・M4A・AAC・OGG・Opus に対応し、元のファイルを `inputs\audio_original.*` に残す。ffmpeg で `inputs\audio.wav`（PCM 16bit、Hz と ch はそのまま）に変換し、ワーカーに渡す
  - `job.json` に `character`（名前は任意・元のファイル名・SHA256・画像チェックの結果）と `audio_source`（元の名前・SHA256・形式）を記録する
  - 履歴に「キャラクター」の列を足した
- `flashhead\tools\fh_worker.py`：settings.json に character と audio_source を記録する。画像チェックが ng なら停止する。表記を「キャラクター画像」に変えた
- `gui\app.py`（作り直し）
  - タイトル「キャラクター音声同期動画（FlashHead Pro）」
  - **1. キャラクター画像（必須）**＋キャラクター名（任意）＋チェック結果＋「モデルに入る画像」2 枚
    - 顔の切り出しを勧められたら、チェックボックスを自動でオンにする
  - **2. 音声ファイル（必須）**：試聴、ffprobe の情報、目標のフレーム数、作成元は問わない・API キーは不要と表示
  - 3. 生成の設定（Pro 固定・出力フォルダ・seed・顔の切り出し）
  - 4. 生成（「動画を生成」は無効のまま／接続テスト（推論なし）／停止）
  - 5. 状況（キャラクター・監視・計画・結果）
  - 6. 完成動画（再生・ダウンロード）
  - 7. 履歴
  - **削除したもの**：Gemini の女性の声の選択・「音声を作成」ボタン・台本の入力（音声は外部で作る方針のため。将来、音声を作る補助機能を別のタブとして足す場合は、作成元に依存しない形で設計する）

### ダミーでの確認【実測】
- image_check.py
  - サンプル（avatar_bob.png・941×1672）：顔は 1 つ → **warn**（中央切りで頭の上が切れる）→ 顔の切り出しを推奨。2 枚のプレビューで、頭全体が入ることを目で確かめた
  - テストパターン（顔なし）→ **ng**
  - 150×200 の小さい画像 → **ng**（小さすぎる・顔なし）
- job_manager
  - **拒否**：顔のない画像（理由を表示）、画像なし（必須）、対応外の音声（.txt）。拒否したとき、ジョブのフォルダは残らない
  - MP3（44.1kHz・ステレオ・3.7秒）＋サンプル画像＋名前「サンプル（動作確認用）」→ done
    - **映像 92 フレーム／3.68秒・音声 3.668秒・差 0.012秒（1フレーム以内）**、512×512
    - inputs：`character.png`・`audio_original.mp3`・`audio.wav`・`image_check\`
    - settings.json に character（名前・元のファイル名・チェック warn・推奨）と audio_source（mp3・44100Hz・2ch・3.7秒・SHA256）が入った
  - 途中で停止：進捗 26% で停止を要求 → **0.2秒で stopped**、MP4 は作られず、ロックも解除された
    - 1回目の停止テストは待ち時間（4秒）の間にジョブが終わり、停止を確かめられなかった → 待ち時間を短くして確かめ直した
- GUI を起動して API で確認
  - 画像を選ぶとチェック結果（⚠️・理由・注意書き）とプレビュー 2 枚が出て、face crop が自動でオンになる
  - 顔のない画像では ⛔ と理由が出る
  - MP3 の情報と目標 92 フレームが表示された
  - 開始 → done → プレビュー → 保存先 → 履歴にキャラクター名
  - 顔のない画像で開始しても、ジョブは作られない
  - 画面の設定：「女性」の表記なし、Gemini の声の選択なし、Lite を選ぶ要素なし。「動画を生成」は `interactive=False`
- 後片付け：GUI を停止した。プロセスなし、VRAM 1,017MiB、C: の gradio 一時フォルダなし
  - FlashHead の venv（83個）と FH-01〜03 は変更なし。GUI の venv は 56個のまま（ライブラリの追加なし）

### 今後の設計（実装はしていない）
- **キャラクターの登録**：`characters\<キャラクターID>\`
  - `character.json`：名前・作成日・元の画像の SHA256・出典と権利（素材台帳の ID と結びつける）・画像チェックの結果・推奨の use_face_crop・seed・color_correction
  - `image.png`：登録した画像のコピー。変更しない
  - `preview_*.png`
  - GUI では「登録済みのキャラクターから選ぶ／新しく登録する」を選べるようにする。ジョブには `character_id` と SHA256 を記録し、後からの差し替えも検出できるようにする
- **長尺での一貫性**
  - 同じキャラクターのシーンでは、同じ登録画像・同じ seed・同じ crop の設定・同じ color_correction に固定する
  - シーンの最初と最後のフレームを登録画像と比べ（顔の領域の差・目視）、記録する
- **シーンごとに別のキャラクター**
  - 台本のシーン（R6 の単位）ごとに `character_id` を持てる形にする
  - 生成は「シーン × キャラクター」の単位でジョブにし、結合（R7）のときに並べる
- **音声**：作成元に依存しない（ファイルを読み込むだけ）。音声を作る補助（Gemini など）を足す場合も、出力の WAV をこの入力に渡すだけの別機能にする

## 36. 2026-10-30 21:00〜：要件の修正（音声と同じ長さの動画を1本ずつ作る。当面の目標は 1〜3分）→ 調査と設計（**コードの変更・GPU での推論はしていない**）
### 要件（ユーザー）
- 外部で作った音声を読み込み、**音声と同じ長さ**のキャラクター動画を1本作る。結合は別の編集ソフトで行う
- 当面は **1〜3分**を目標にする。ただし固定の上限にはしない（実測で安定を確認したら、さらに伸ばす）
- 不要になったもの：8分以上の一括生成・台本のシーン分割・複数シーンの自動結合・長尺の自動編集
- Gemini は必須にしない。キャラクターと音声の作成元は固定しない

### 1. 今の最大の長さを決めている箇所（コードで確認）
| 箇所 | 内容 | 種類 |
|---|---|---|
| `flashhead\tools\fh_worker.py` 25行 `MAX_SECONDS = 20.0`、83行の判定 | 20秒を超えると停止する | **私が入れた暫定の制限**（全フレームをメモリに置くため） |
| 同じファイル 169行 `frames.append(v.to(torch.uint8).cpu())` | 全フレームを RAM にためる（uint8 で 1フレーム 786KB → 3分＝4,500 フレームで約 3.5GB。書き出しまで一時ファイルも含めて重なる） | 長尺を止めている本当の理由 |
| `gui\app.py` 54行・121行 | 「20秒まで」と表示する | 暫定の表示 |
| `gui\job_manager.py` の `LIMITS` | run_stage の時間切れ 7,200秒（固定） | 長さに応じて決めるべき |
- **モデル側の制約ではない**
  - FlashHead の stream モードは、33フレーム（つなぎ5＋新しい28）ずつ生成し、音声は**最新 8秒の固定窓（deque）**で埋め込む → 1チャンクの計算量と VRAM は長さによらず一定
  - RoPE の位置もチャンクの中だけ（33フレーム＝潜在 9フレーム×1,024トークン）で、長さによって増えない
  - 公式は「60秒の生成で誤差が積み重ならない」と書いている（Oracle-guided distillation）
  - → 長さの制約は「RAM にためること」と「処理時間（時間切れ）」だけ

### 2. 時間方向の状態（動きの連続性）— 公式 pipeline の仕組み（コードで確認）
- `prepare_params`（151行〜）：条件画像を VAE で潜在 `ref_img_latent` にし、`latent_motion_frames = ref_img_latent[:, :1]`（最初は画像そのもの）にする
- `generate`（236行〜）を呼ぶたびに
  1. 潜在の先頭に `latent_motion_frames`（前のチャンクの最後 5フレームを VAE で潜在にしたもの）を入れて、4ステップでノイズを除く
  2. **毎回 `ref_img_latent`（元の画像）を条件にする** → 顔立ちがずれにくい
  3. デコードした後、`match_and_blend_colors_torch` で**元の画像の色に合わせる**（color_correction_strength=1.0）→ 色が少しずつ変わっていくのを防ぐ
  4. 最後の 5フレームを VAE で潜在にし、次のチャンクの `latent_motion_frames` にする（315行）
- 乱数（`self.generator`）も同じオブジェクトを使い続ける
- **結論：同じプロセスの中で同じ pipeline を使い続ければ、チャンクの境目で状態はリセットされない**
  - 状態は「潜在 2フレーム分＋乱数」だけで、長さによって増えない
  - 逆に、途中でプロセスを分けると状態が切れる（再開を作る場合は、この状態を保存して戻す必要がある）

### 3. 1〜3分に必要な修正（設計。未実装）
1. **フレームを逐次ディスクに書く**（全フレームを RAM にためない）
   - ワーカーで `ffmpeg` を1本起動し、標準入力に rawvideo（rgb24・512×512・25fps）を流す（`-c:v libx264 -crf 16 -pix_fmt yuv420p -bf 0`）
   - チャンクごとに GPU のテンソル（float 0〜255）→ `clamp → uint8 → cpu → numpy` → `stdin.write` → すぐ捨てる
   - 書き出すフレーム数を数え、**音声から計算したフレーム数（round(秒×25)）に達したら、それ以上は書かない**（末尾の無音のつめ物の分は書かない）
   - RAM に置くのは常に1チャンク分（28フレーム×786KB＝約 22MB）だけ。3分でも RAM の使用量はほぼ一定
   - ffmpeg の stdin が詰まった場合（ディスクが遅いなど）に備えて、`wait()` と戻り値を確かめ、失敗なら status に記録する
2. **長さの扱い**
   - 長さは音声から自動で取る（`wave` で frames / rate）。固定の秒数で切り詰めない
   - 目標のフレーム数＝`round(音声の秒数 × 25)`
   - 生成するチャンク数＝`ceil(16kHz のサンプル数 / 17,920)`（1チャンク＝28フレーム＝1.12秒）。末尾は公式と同じく無音でつめ物をする → 生成フレーム数（チャンク数×28）は常に目標以上なので、**音声の末尾は欠けない**
   - 結合：映像（目標フレーム数ちょうど）＋元の WAV を `-t 音声の秒数` で AAC にし、映像と音声の差が 1フレーム（0.04秒）以内であることを ffprobe で確かめる（今の実装と同じ検査）
   - 音声と映像の対応：stream モードでは、窓の最後の 33フレーム＝「前のつなぎ 5＋新しい 28」に当たり、出力は `[5:]` なので、**映像のフレーム i は音声の i/25 秒に対応する**（遅れはない）。1チャンク目は窓の残りが無音
3. **上限の扱い**
   - `MAX_SECONDS` の固定をやめ、設定ファイル（`flashhead\tools\fh_limits.json`）に「検証済みの最大秒数」を置く（初期は短い音声の実測の後に 60 → 120 → 180 と更新していく）
   - それを超える音声は、GUI で「未検証の長さです」と表示し、ユーザーが明示的に確認したときだけ実行する（ハードな上限は設けないが、黙って実行はしない）
4. **時間切れ（run_stage）を長さから決める**：`timeout = 起動と読み込みの 120秒 ＋ 実測した 1チャンクの秒数 × チャンク数 × 1.5`（実測するまでは、1チャンク 3秒と仮定する）
5. **進捗と残り時間**：status.json に `chunk / chunks`、1チャンクの平均秒数、残りの見込み秒数、書いたフレーム数を入れる
6. **GPU のメモリ**：チャンクのループの中で中間のテンソルを毎回捨てる（`del`）。VRAM の使用量が増え続けていないかを run_stage の CSV で確かめる。増える場合だけ、N チャンクごとの `torch.cuda.empty_cache()` を検討する
7. **（任意・後で）途中の再開**：60秒ごとに区切りのファイル（`seg_000.mp4`…）に分けて書き、区切りの終わりに状態（`latent_motion_frames`・乱数の状態・音声の deque・チャンクの番号・書いたフレーム数）を `torch.save` する → 失敗したらその区切りから再開し、最後に concat で1本にする
   - 3分程度なら、まずは「1本の ffmpeg に流す方式」で十分。再開は実測で必要になってから作る

### 4. VRAM・RAM・その他の懸念点
- VRAM：チャンクの大きさは長さによらず一定。読み込み後に約 3.4GiB（実測）＋1チャンク分の推論・デコード（未実測。推定 2〜4GB）
  - **短い音声での実測で、1チャンクのピークを確定させる必要がある**
- RAM：逐次書き出しにすれば、ほぼ一定（ワーカーのプロセス＋ffmpeg）。音声の全体（3分＝16kHz の float32 で約 11.5MB）は問題ない
  - ただし Windows では、GPU の確保もコミットに計上されるので、ジョブのコミットは読み込みの 6.4GiB ＋推論分になる見込み
- 処理時間：Pro は RTX4090 で 10.8FPS → 3060 では 2〜4FPS と推定 → **1分で 6〜13分、3分で 19〜38分**（短い音声の実測で確定させる）
- 長時間の GPU 負荷：温度と、WDDM の TDR（1回の GPU 処理が長すぎると止まる）。ただし1ステップは短いので TDR の心配は小さい。nvidia-smi の温度も CSV に記録する
- ディスク：3分の 512×512・CRF16 で数十MB 程度（問題ない）
- 長さの品質：顔立ちは毎チャンク元の画像を条件にしているので保たれる見込み。ただし表情や頭の動きが同じパターンの繰り返しにならないか、色の揺れがないかは実測が必要

### 5. 動きの連続性の確かめ方
- 同じ pipeline・同じプロセスで、すべてのチャンクを順番に生成する（今の設計どおり）
- 数値：隣り合うフレームの差（顔の領域）を全フレームで計算し、**チャンクの境目（28フレームごと）の差が、チャンクの中の差の中央値と比べて大きくないか**を記録する（境目だけ急に動く＝リセットの兆候）
- 顔の一貫性：最初・中間・最後のフレームと条件画像の顔の領域を比べる（数値と目視）
- 目視：境目の前後を並べた確認画像を自動で作る

### 6. 最初に行う実機テスト（**それぞれ事前に条件を示して承認をもらう**。テストの回数を勝手に増やさない）
| 段階 | 音声 | 目的 | 合格の条件 |
|---|---|---|---|
| **T1** | 3〜5秒（日本語。外部で作った WAV または承認済みの Gemini 無料枠） | 推論が正しく動くか。1チャンクの時間・VRAM・RAM のピーク | MP4 が音声の長さどおり（差は1フレーム以内）、口・表情・頭が動く（目視）、エラーなし |
| T2 | 約 60秒 | 逐次書き出し（実装の後）、RAM が一定か、境目の連続性 | RAM・コミットが時間とともに増えない、境目の差に急な山がない |
| T3 | 約 120秒 | 長時間の安定（温度・VRAM） | 同上、所要時間が見込みの範囲 |
| T4 | 約 180秒 | 目標の上限 | 同上 → 「検証済みの最大秒数」を 180 に更新する |
- 共通の実行条件：開始前のコミット残量 19GiB 以上（常駐アプリを終了する）、run_stage（ジョブの上限 12GiB・空きRAM 2GiB・コミット残量 5GiB・VRAM 11,800MiB・時間切れは長さから計算）、1回ずつ、自動の再実行はしない
- 順番：**T1 は今のワーカー（20秒までの方式）でも実行できる**。T2 の前に逐次書き出しを実装し、接続テスト（推論なし）で 60秒の音声のときの RAM が一定であることを確かめる

### 7. GUI の変更点（設計）
- 「20秒まで」の表示をやめ、**音声の長さ・目標フレーム数・チャンク数・見込みの処理時間・出力のおおよその大きさ**を表示する
- 「検証済みの最大秒数」を超える場合は警告を出し、「未検証の長さで実行する」チェックを付けたときだけ開始できる
- 進捗：チャンク i / n、書いたフレーム数、1チャンクの平均時間、残り時間の見込み
- 停止ボタン：今の仕組み（STOP ファイル → run_stage がプロセスツリーごと停止）のまま。停止したときは、途中までの MP4 を「不完全」として残すか消すかを選べるようにする（初期は消す）
- 「動画を生成」ボタン：T1 の承認後に有効にする（AVATAR_ALLOW_GPU=1 で GUI を起動する方式）

## 37. 2026-10-30 22:11〜：T1 実機テスト（FlashHead Pro の初めての GPU 生成）【実測】
- 承認：ユーザーが「英語 4秒で今すぐ実行」を選んだ（日本語の音声が無いため。推論の動作と計測が目的）
- 入力：`inputs\avatar_bob.png`（use_face_crop=True・seed 42）＋ `runs\20261030-1200_integration_internal\inputs\internal_eng_4s.wav`（英語・4.0秒・社内テスト用）
- 実行：job_manager.start_job('fh_gpu')（AVATAR_ALLOW_GPU=1・--allow-gpu）を1回だけ実行。run_stage の制限は既定どおり。ジョブは `gui\jobs\20261030-221116_fh_gpu`
- 開始前：コミット残量 20.67GiB（Chrome は 14 プロセス起動したまま）、空きRAM 6.39GiB
- 結果：**exit 0・done**。出力は `outputs\t1\20261030-221116_fh_gpu.mp4`
  - 512×512、**映像 100 フレーム＝目標 100**、映像 4.00秒・音声 4.00秒、**差 0.00秒**
- 時間：合計 169.7秒
  - 入力の確認からモデルの読み込みまで 約 105秒（22:11:16→22:13:01）
  - **1チャンク（28フレーム）約 16〜17秒 → 約 1.7FPS**（推定の 2〜4FPS より遅い）
  - 結合 約 1秒
- 資源の最大：**VRAM 8,624MiB**、**ジョブのコミット 10.46GiB**（上限 12GiB まで余裕 1.5GiB）、コミット残量の最小 9.25GiB、空きRAM の最小 4.04GiB
- 目視（10フレームの一覧）：頭の傾き・まばたき・口の開閉・表情が変化する。顔立ちと背景は保たれている。**リップシンクの確認は、ユーザーが音声付きで再生する必要がある**
- 見込みの修正：1.7FPS なら **1分で約 15分、3分で約 45分**（＋読み込み 約 2分）。T2 以降の run_stage の時間切れは、この値をもとに計算する
- 注意：ジョブのコミット 10.46GiB は上限 12GiB に近い。T2（60秒）は逐次書き出しを実装した後に行う（フレームを RAM にためると上限を超えるおそれ）。チャンクのループで VRAM・コミットが増え続けないかを、T2 の CSV で確かめる

## 38. 2026-10-30 22:15〜：フレームの逐次書き出しを実装（推論なし・GPU 未使用で検証）【実測】
- 承認の範囲：逐次書き出しの実装・ワーカーへの組み込み・GPU を使わない接続テスト・メモリの確認・GUI との互換性の確認
- **実 GPU の推論はしていない**（60・120・180秒の実生成もしていない）

### 変更したファイル（元のファイルは `*.bak_20261030_pre_stream` として残した）
- **新規** `flashhead\tools\frame_sink.py`
  - FrameSink：ffmpeg を1本起動し、stdin に rawvideo（rgb24・512×512・25fps）を流す → libx264・CRF16・bf0・yuv420p
  - 目標のフレーム数で打ち切る（最終チャンクの余りは書かない）
  - パイプが詰まったら write がブロックし、ffmpeg が追いつくまで待つ
  - 書く前に ffmpeg が生きているかを確かめる（異常終了を検知）
  - close() で戻り値と書いたフレーム数を検査する
  - 既にあるファイルには書かない（-n も付ける）。stderr はファイルに書く
  - proc_mem_mib：Windows API でプロセスの private と working set を測る
- **新規** `flashhead\tools\fh_limits.json`
  - `gpu_verified_max_seconds`＝4.0（T1）
  - 実測値：1チャンク 16.5秒・起動 105秒・余裕 1.5倍
  - dry_run の上限は 1,200秒
- **新規** `flashhead\tools\verify_frames.py`：ダミーフレームに埋め込んだ通し番号を MP4 から読み戻し、重複・欠落・順序を検査する
- **新規** `flashhead\tools\test_frame_sink.py`：FrameSink の異常系のテスト
- **新規** `gui\tools\stream_dryrun_test.py`：job_manager 経由で接続テストを実行し、結果をまとめる（停止・故障の注入つき）
- `flashhead\tools\fh_worker.py`（作り直し）
  - 固定の `MAX_SECONDS=20` と `frames` リストを廃止した
  - dry_run と gpu で、同じ `run_chunks()`（チャンク → FrameSink → 参照を捨てる）を使う
  - **gpu**：make(i) の中で `run_pipeline(...)[5:]` → `uint8 → cpu → numpy`（T1 と同じ変換）を行い、`del v, emb` する
    - pipeline・deque・乱数は同じプロセスで使い続ける（時間方向の状態は今までどおり）
    - import・get_pipeline・get_base_data・audio_load の秒数をログに出す（次の GPU 実行で起動時間の内訳を実測するため）
  - **dry_run**：torch を読み込まない。チャンクごとに「16bit の通し番号を上端に入れたダミーフレーム」を作る。`dry_fail_at_chunk` で故障を注入できる（テスト用）
  - 結合：映像は `-c:v copy`（再エンコードなし）。音声は**切らずに** AAC 192k
    - `<out>\<job_id>.incomplete.mp4` に書く → ffprobe で7項目を検査する（フレーム数＝目標・512・25/1・h264・aac・**差が1フレーム未満**・音声が欠けていない）
    - 合格したときだけ `os.rename` で `<job_id>.mp4` にする（行き先があれば失敗するので、上書きしない）
  - 失敗：`result.json` に ok=false・error・incomplete_files を書き、status は failed
  - 途中のファイル名：`<job_dir>\_video_noaudio.partial.mp4`、`<out>\<job_id>.incomplete.mp4`
  - `worker_mem.csv`：チャンクごとに、ワーカーと ffmpeg の private・WS、VRAM の allocated・reserved を記録する
  - GPU の長さのゲート：検証済みの秒数を超え、かつ job.json に `allow_unverified_length=true` が無ければ失敗にする
- `gui\job_manager.py`
  - 固定の timeout 7,200秒をやめ、`plan_for(秒)` で決める
    - GPU：300＋チャンク数×16.5×1.5（60秒→1,636秒、180秒→4,284秒）
    - 接続テスト：120＋チャンク数×(dry＋1)×2
  - GPU で未検証の長さは、`allow_unverified_length` が無ければ開始前に拒否する（ジョブのフォルダは残さない）
  - read_status：完成品は「done・result.ok・ファイルがある」ときだけにした。failed・stopped・aborted のときは `incomplete_files` を出す
  - 履歴に「（不完全なファイルあり）」と表示する
- `gui\app.py`
  - 「20秒まで」の表示を削除した。音声を選ぶと、目標フレーム数・チャンク数・見込みの処理時間（実測から）・MP4 の大きさを出す。未検証の長さなら警告を出す
  - 進捗：書き出したフレーム数・1チャンクの平均・残り時間（実行中だけ）
  - 不完全なときは警告を出す。プレビューは result.ok のときだけ
  - 「動画を生成」は**無効のまま**

### 接続テストの結果（推論なし・GPU 未使用・ダミーフレーム・24kHz モノラルの WAV・dry 0.7秒/チャンク）【実測】
| 音声 | チャンク | 映像フレーム（目標） | 通し番号の検査 | 映像／音声の秒数 | 差 | ffmpeg | ジョブのコミット最大 | ワーカー private（最初の10%→最後の10%） | ffmpeg private（最大） | 所要時間 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4秒 | 4 | 100（100） | 0〜99 が順番どおり | 4.000 / 4.000 | 0.000 | 0 | 0.55GiB | 374.5→374.5MiB | 158MiB | 4.1秒 |
| 60秒 | 54 | 1,500（1,500） | 0〜1,499 が順番どおり | 60.000 / 60.000 | 0.000 | 0 | 0.55GiB | 374.4→374.6MiB | 158MiB | 41.9秒 |
| 180秒 | 161 | 4,500（4,500） | 0〜4,499 が順番どおり | 180.000 / 180.000 | 0.000 | 0 | 0.55GiB | 374.6→374.6MiB | 158MiB | 123.5秒 |
- 3本とも7項目の検査に合格し、完成品の名前になった。VRAM はデスクトップ分の約 980MiB だけ（torch を読み込んでいない）
- **ワーカーと ffmpeg のメモリは長さによらず一定**
  - ffmpeg は最初の数チャンクで約 158MiB まで上がり、そこで止まる（x264 の先読み）
  - 以前の方式なら、180秒で約 3.5GB を RAM にためていた
- 停止（180秒・20秒後に STOP）
  - **0.61秒で stopped**。ffmpeg のプロセスは残らなかった（tasklist で確認）
  - 出力フォルダには何も作られず、ジョブのフォルダに `_video_noaudio.partial.mp4` が残った
  - GUI と履歴では「完成していません／（不完全なファイルあり）」と表示する
- 故障の注入（60秒・11チャンク目で例外）
  - failed になり、result.json は ok=false・incomplete_files。出力フォルダには何も作られない
- FrameSink の異常系（test_frame_sink.py）：すべて OK
  - このテストが起動した ffmpeg を途中で kill → 次の write で検知した
  - 足りないまま close → エラーになった
  - 既にあるファイルへの書き込みを拒否し、元の中身は変わらなかった
  - float や大きさ違いのフレームを拒否した
  - 112 フレームを送って 100 で打ち切った（ffprobe で 100）
  - 注：ffmpeg.exe を名前で強制終了するテストは、他の作業を止めるおそれがあるので行わず、上の方法に変えた
- 長さのゲート（GPU は起動していない）
  - job_manager：60秒の fh_gpu を拒否した（ジョブのフォルダは残らない。テストではワーカーのパスを存在しないものに差し替えて行った）
  - ワーカー：mode=gpu・60秒・`--allow-gpu` なしで、長さのゲートが最初に失敗にした
- GUI
  - 関数を直接呼んで確かめた：on_audio（4・60・180秒の計画と警告）、接続テストの開始、実行中の表示、停止、履歴
  - 7869 番で起動して HTTP 200 を確認し、停止した
- 変更していないもの
  - FH-01〜03：`git apply --check -R` でパッチと完全に一致
  - FlashHead の venv：83個
- テストの出力：`outputs\stream_test\`（3本・合計 約 90MB）、`gui\jobs\20261030-2223*〜2229*`

### T1 の起動 105秒の内訳（既存のログ：status の時刻・job_stdout・job_memory.csv から。【実測】は時刻の差、【推定】は原因の割り当て）
| 区間 | 秒 | 根拠 |
|---|---|---|
| job_manager の前処理（画像チェック＝MediaPipe の別プロセス・ffmpeg で WAV に変換・SHA256・run_stage と Python の起動） | 約 5 | 【実測】job_id 22:11:16 → 最初の status 22:11:21。中の内訳は記録がない |
| ワーカーの import（numpy・torch・librosa・imageio・flash_head.inference → diffusers・transformers など） | 約 49 | 【実測】T0≈22:11:21.5 → 「モデルを読み込んでいます」22:12:10。この間 VRAM は変わらず（GPU を使っていない）、ジョブのコミットが 0→1.6GiB にゆっくり増えた。どのライブラリが重いかは【推定】（torch の CUDA DLL・transformers・diffusers）。ディスクのキャッシュが冷えていた、Defender の検査の影響の可能性も【推定】 |
| モデルの読み込み（get_pipeline：Pro・Wan VAE・wav2vec2）＋ get_base_data（顔の切り出し・条件画像の VAE エンコード） | 約 28 | 【実測】logger 22:12:10.5 → post_load_gate t=76.9（22:12:38）。VRAM は t≈50→75 で 900→5,531MiB。load_test では Pro 5.4秒・VAE 0.6秒だったので、残り約 20秒は CUDA の初期化・モデルの組み立て・wav2vec2・get_base_data と【推定】 |
| 音声の読み込み（librosa.load で 16kHz に変換）など、1チャンク目の開始まで | 約 23 | 【実測】22:12:38 → 「チャンク 1/4」22:13:01。VRAM は変わらず、CPU だけ。librosa の遅延 import・リサンプラー（numba など）の初回の準備と【推定】 |
| 初回の GPU 処理の準備 | 105秒の外 | 1チャンク目の最初のステップが 2.73秒（以降は 約 2.44秒）→ 初回のぶん 約 0.3秒【実測】。CUDA の初期化は、モデルの読み込みの区間に含まれる（分けて測っていない） |
- 1チャンク 約 16.2秒の内訳【実測】：ノイズ除去 4ステップ×約 2.45秒＝9.8秒、デコード 5.7秒、動きの潜在のエンコード 0.55秒、色の補正 0.03秒
- 今回、ワーカーに import・get_pipeline・get_base_data・audio_load の秒数ログを足した → 次の GPU 実行で実測で分けられる
- 改善の候補（**未実施**）
  - 音声を job_manager の ffmpeg で 16kHz・モノラルにしておき、librosa を使わない（約 20秒短くなる見込み。【推定】）
  - venv を Defender の除外に入れる（ユーザーの判断）

### T2 の前に残るリスク
1. **GPU 経路の逐次書き出しは実機で未実行**
   - 変換（`uint8 → cpu → numpy` の形 (28,512,512,3)）はコードで確認しただけ
   - 形が違えば FrameSink が拒否して failed になる（不完全なファイルは完成品として扱わない）
2. **VRAM が増え続けないか**
   - T1 の4チャンクで 8,486→8,624MiB と、少し増えていた
   - 54チャンクで増え続けるかは、worker_mem.csv の vram_alloc・reserved と run_stage の CSV で見る（上限 11,800MiB で自動停止）
3. **ジョブのコミット**
   - T1 は 10.46GiB。そのうちチャンクごとの増加（約 0.02GiB/チャンク）は frames リスト（28フレーム×786KB）のぶんと【推定】され、これは解消した。最後の imageio の書き出しのぶん（約 +0.2GiB）もなくなる見込み
   - ただし推論の中のピーク（約 10.2GiB）は変わらず、12GiB の上限まで 約 1.8GiB しかない
4. **時間と温度**：60秒は 約 17分、180秒は 約 46分の連続の GPU 負荷。温度は run_stage で記録していない（nvidia-smi の温度の記録は未実装）
5. **口・表情の質と、チャンクの境目の連続性**：長い動画では未確認。T1 のリップシンクも、ユーザーの目視の確認待ち
6. **常駐アプリ**：Chrome などでコミット残量が変わる。開始前の 19GiB 以上の確認は今までどおり必要

### 60秒の実 GPU テスト（T2）へ進める状態か
- **コードとテストの面では進められる**
  - 逐次書き出し・切り方・結合・検査・停止・失敗の扱いは、GPU なしで 60・180秒まで確認した
  - GPU 経路の変更は「frames リストへの追加 → FrameSink へ渡す」だけで、推論の部分は変えていない
- 実行の条件（承認が必要）
  - 音声は 60秒（日本語なら外部の WAV、無ければ英語などで代用）
  - job.json に `allow_unverified_length=true`（job_manager から直接実行する）
  - 開始前のコミット残量 19GiB 以上、1回だけ、自動の再実行はしない
  - 時間切れは 1,636秒
- 合格したら、`fh_limits.json` の `gpu_verified_max_seconds` を 60 に更新する

## 39. 2026-10-30 22:38〜：T1b 実機テスト（逐次書き出し方式で 4秒の GPU 生成を1回）【実測】
- 承認：T1b を最大1回。60・120・180秒の GPU 生成はしていない。自動の再実行なし。AVATAR_ALLOW_GPU は起動コマンドの1回だけで、ユーザーの環境変数には入れていない（確認済み）
- 入力：T1 と同じ
  - `inputs\avatar_bob.png`（比較用のサンプル。face crop=True・seed 42）
  - `internal_eng_4s.wav`
  - Pro、512×512、25fps、目標 100 フレーム
- 事前の確認
  - コミット残量 20.82GiB（Chrome は 14 プロセス起動）、空きRAM 8.09GiB、VRAM 使用 871MiB／12,288MiB、GPU 48.8℃
  - 重み 14 ファイルの SHA256 が一致、FH-01〜03 がパッチと一致（repo 9bc03de）
  - ffmpeg 8.1、出力先 `outputs\t1b` は新規、GPU のロックなし
- ジョブ：`gui\jobs\20261030-223842_fh_gpu`
  - run_stage の制限は既定どおり（ジョブ 12GiB・空きRAM 2GiB・コミット残量 5GiB・VRAM 11,800MiB）。時間切れは長さから計算して 399秒
  - nvidia-smi を1秒ごとに記録した（`nvidia_smi_1s.csv`）
- **結果：成功（exit 0・done・result.ok=true・7項目の検査にすべて合格）**
  - 出力：**`D:\Avatar-Pipeline\outputs\t1b\20261030-223842_fh_gpu.mp4`**（1.16MB）
  - 映像：H.264・512×512・25/1・**100 フレーム**・4.000秒
  - 音声：AAC・4.000秒、**差 0.000秒**
  - frame_sink は 100/100 を書き、ffmpeg は正常終了。ffmpeg_stream.log は空
- フレームの検査（`frame_analysis.py`。顔の中央部で、隣り合うフレームの差を計算）
  - 同じフレームの重複は 0
  - 境目（27→28・55→56・83→84）の差は 3.48・2.07・7.55。境目以外は中央値 2.66・95%点 8.37・最大 9.97 → **境目だけ飛び抜けてはいない**
  - T1 も同じ形（3.41・1.92・7.39）。83→84 が大きいのは動きの内容によるもの
  - 境目の前後 8 枚を並べた画像で、破綻がないことを目で確かめた
- T1 との画質の比較：PSNR 45.3dB・SSIM 0.989（ほぼ同じ）
  - 違いは主に、T1 が2回エンコードした（imageio の CRF16 → 結合で再エンコード）のに対し、T1b は1回だけ（結合は copy）だからと【推定】。ファイルは 0.82MB → 1.16MB
- GUI：refresh() で、プレビュー（outputs\t1b は許可パスの中）・保存先・履歴が表示された。ロックは解放された
- **リップシンクの正確さは、静止画や数値では判断していない**（ユーザーが再生して確かめる必要がある）

### 速度（T1 → T1b）
| 区間 | T1 | T1b | 根拠 |
|---|---|---|---|
| 全体（ジョブの作成〜完了） | 173秒 | **120秒** | 時刻 |
| 前処理（job_manager） | 約 5秒 | 約 2秒 | job.json の作成と更新の時刻 |
| ライブラリの読み込み | 約 49秒 | 38.6秒 | T1b はワーカーのログで【実測】 |
| モデルの読み込み（get_pipeline） | （下の行と合わせて）約 28秒 | 7.0秒 | 同上 |
| get_base_data（顔の切り出し・VAE） | — | 3.5秒 | 同上 |
| 音声の処理（librosa） | 約 23秒 | 3.7秒 | 同上 |
| チャンクのループ（4チャンク） | 約 67秒 | 62.4秒（16.3・15.3・15.3・15.3秒） | worker_mem.csv の時刻 |
| ノイズ除去 1ステップ／デコード | 2.45秒／5.7秒 | 2.33秒／5.44秒 | [generate] のログ |
| ffmpeg：逐次エンコードの終了待ち＋結合（copy） | 書き出しと再エンコード 約 1〜2秒 | 約 0.3秒＋0.3秒 | ログの t |
- 起動が速くなった主な理由は、T1 で一度読んだファイルが OS のキャッシュに残っていたことと、librosa の初回の準備が2回目には要らなかったことと【推定】
  - コードの変更による短縮は、結合の再エンコードをやめたぶん（1秒前後）だけ
  - **冷えた状態（再起動後など）では、T1 に近い 約 105秒の準備がかかる前提で見積もる**

### メモリ（T1 → T1b）
| 項目 | T1 | T1b |
|---|---|---|
| ジョブのコミットの最大 | 10.46GiB | **10.26GiB** |
| VRAM の最大（run_stage） | 8,624MiB | **8,486MiB** |
| 空きRAM の最小 | 4.04GiB（開始 6.39） | 6.41GiB（開始 8.07） |
| コミット残量の最小 | 9.25GiB | 10.48GiB |
| 読み込み後の確認 | 空きRAM 5.54・コミット残量 12.0・空き VRAM 4.72 | 6.6・11.8・4.72（条件 3・8・2 をすべて満たした） |
- **チャンクごとの VRAM（T1b）**
  - torch の allocated は 3,506MiB のまま一定
  - reserved は 7,474 → 7,476 → 7,476 → 7,476MiB
  - nvidia-smi は 8,469〜8,486MiB
  - → **T1 で見えた増加（8,486→8,624）は T1b では起きなかった**（T1 の増加は、最後の imageio の書き出しなどの影響と【推定】。4チャンクでは確定できないので、T2 でも記録する）
- ワーカーのプロセスの private は 10,316〜10,346MiB、ffmpeg は 114→157MiB。どちらもチャンクで増え続けてはいない
- 終了後：VRAM 882MiB（開始前 871）、コミット残量 20.82GiB（開始前と同じ）、空きRAM 8.14GiB。ffmpeg のプロセスなし → **解放された**
- GPU の温度：48.6℃ → 最高 70.3℃（使用率 100%・約 150〜159W）。終了 20秒後には 54.6℃

### 変更（T1b の後）
- fh_worker.py：完了時の status に target_frames を足した（GUI の「書き出し済み 100/?」の表示を直した。生成の処理は変わらない）

### 60秒の実 GPU テスト（T2）へ進めるか
- **進められる状態**
  - 逐次書き出しを GPU で実証した
  - VRAM・コミットはチャンクごとに増えず、上限まで余裕がある（VRAM 8.5／11.8GB、コミット 10.3／12GiB）
  - 音声とのずれ 0、境目に破綻なし
- 条件（承認が必要）
  - 60秒の音声
  - job.json に `allow_unverified_length=true`
  - 開始前のコミット残量 19GiB 以上
  - 1回だけ、時間切れ 1,636秒
  - 見込み：約 16〜17分（キャッシュが冷えていれば ＋1分程度）

### 未解決の問題
1. **リップシンクの目視**：T1・T1b とも未確認（ユーザーの再生確認待ち）。日本語の音声もまだ無い
2. **温度**：1分ほどで約 70℃。17分の連続負荷でどこまで上がるかは未確認。温度による自動停止は未実装（記録は nvidia-smi で可能）
3. **コミットの余裕**：ジョブの最大 10.26GiB と上限 12GiB の差は 約 1.7GiB
4. **GUI**：「未検証の長さで実行する」のチェックボックスと「動画を生成」ボタンは未実装・無効（T2 も job_manager から直接実行する）
5. **起動時間**：キャッシュが冷えているときは、ライブラリの読み込み・librosa で 約 70秒かかる。改善は未実施

## 40. 2026-10-30 22:45〜：GPU 温度の監視を追加（GPU 生成なし）。**T2 は事前条件がそろわないため保留**【実測】
### 変更したファイル（元のファイルは `*.bak_20261030_pre_temp` として残した）
- `tools\run_stage.py`（他の工程と共用なので、**新しい引数を付けたときだけ動く**）
  - nvidia-smi の読み取りを `memory.used,temperature.gpu` にした。パスは環境変数 `AVATAR_NVIDIA_SMI` で変えられる（テスト用）
  - `--max-gpu-temp-c` と `--gpu-temp-hold-s`：その温度以上が指定の秒数続いたら、TerminateJobObject でジョブ（ワーカーと ffmpeg）だけを止める
  - `--require-gpu-temp` と `--gpu-temp-missing-s`：温度を読めない状態が指定の秒数続いたら止める
  - memory.csv の最後の列に `gpu_temp_c` を足した（この CSV を読んでいるスクリプトは他にない）
  - result の peak に `gpu_temp_max_c`、limits に温度の設定を足した
- `gui\job_manager.py`
  - FlashHead のジョブ（接続テストと GPU）では、run_stage に**83℃・10秒・温度を15秒読めなければ停止**を渡す
  - fh_gpu は、開始前に nvidia-smi で温度を読めなければ**開始を拒否する**（ジョブのフォルダは残さない）
  - テスト用の `test_max_gpu_temp_c` は接続テスト（fh_dryrun）のときだけ効く
- `flashhead\tools\fh_worker.py`：worker_mem.csv の列を足した。チャンクごとに次を記録する
  - 推論 `gen_s`、ffmpeg への書き込み `write_s`
  - ワーカーと ffmpeg の private・WS
  - VRAM の allocated・reserved と nvidia-smi の使用量、GPU 温度
  - 空きRAM、コミット残量、ジョブのコミットの最大（自分が入っている run_stage のジョブを QueryInformationJobObject で読む）
- `gui\tools\stream_dryrun_test.py`：`--test-max-temp` を足した

### テスト（GPU 生成なし）【実測】
- 60秒の接続テスト（温度の監視あり）
  - done・1,500/1,500・通し番号 OK・差 0.000
  - 温度は 46.4〜46.7℃（GPU をほぼ使っていない状態）。job_memory.csv と worker_mem.csv の両方に記録された
  - ジョブのコミットの最大 0.54GiB
- 温度による停止（しきい値を 30℃ に下げたテスト）
  - **10.7秒で aborted**（「GPU 温度 47℃ ≥ 30℃ が 10秒続いた」）
  - ffmpeg は残らず、出力フォルダには何も作られない。履歴は「（不完全なファイルあり）」
- 温度を読めない場合（nvidia-smi のパスを存在しないものにしたテスト）
  - run_stage：**15.3秒で停止**（「GPU 温度を 15秒以上読めない」）
  - job_manager：fh_gpu の**開始を拒否**し、ジョブのフォルダは作られない（念のため、ワーカーのパスも存在しないものに差し替えて行った）
- 互換性：温度の引数なしの run_stage は exit 0・aborted なし（従来どおり）

### T2（60秒の実 GPU 生成）：**実行していない（保留）**
ユーザーの指示の事前条件のうち、そろっていないもの：
1. **T1b の品質確認**：`outputs\t1b\20261030-223842_fh_gpu.mp4` を、ユーザーが音声付きで確かめた結果が未着
2. **60秒の会話音声**：外部で作ったものが用意されていない（`inputs\` には avatar_bob.png だけ）。代わりの音声は作らない
3. **キャラクター画像**：T2 で使う画像の指定が未着（avatar_bob.png は固定のキャラクターではない）
- そろっているもの
  - 温度の監視（83℃・10秒）と、読めない場合に開始しない仕組み
  - 逐次書き出し、メモリの監視、チャンクごとの記録
- 実行するときの条件（変わらない）：コミット残量 19GiB 以上・空きRAM 3GiB 以上・重みの SHA256・FH-01〜03・`allow_unverified_length=true`・1回だけ・時間切れ 1,636秒
- 成功した場合だけ、`fh_limits.json` の `gpu_verified_max_seconds` を 60 にする

### 残っている要件（未実装。忘れないために記録）
- GUI の1画面化
- 虹色の太いアニメーションの枠線
- 進捗の表示の改善
- 「未検証の長さで実行する」チェックボックスと「動画を生成」ボタン（今は無効）

## 41. 2026-10-30 23:00：FlashHead の開発を中断（ユーザーの指示）。中断時点のまとめ
> 詳しい記録は 34〜40節にある。この節は再開のための要約。再開の手順は `RESTART_PROMPT.md` を読む

### 完了した作業
- 専用環境：`flashhead\.venv`（83個、torch 2.7.1+cu128）と SoulX-FlashHead @9bc03de に、FH-01〜03（Windows 対応・省メモリ読み込み）を入れた
- 重み（Pro・Wan VAE・wav2vec2）：SHA256 一致
- GUI（`gui\app.py`・127.0.0.1:7861）
  - 任意のキャラクター画像と外部の音声を入力できる
  - 接続テスト（推論なし）・停止・履歴が動く
  - 「動画を生成」は無効のまま
- FFmpeg の逐次書き出し（`frame_sink.py`）：全フレームを RAM にためない。目標フレーム数で打ち切る。検査に合格したときだけ `.incomplete.mp4` → `.mp4` にする
- GPU で検証済みの長さの管理（`fh_limits.json`。現在 4秒）と、長さから決める時間切れ
- GPU 温度の監視（run_stage）と、チャンクごとの記録（worker_mem.csv）

### 実 GPU 生成の検証結果（どちらも 4秒・avatar_bob.png・internal_eng_4s.wav（英語）・seed 42・face crop）
| | T1（旧方式） | T1b（逐次書き出し） |
|---|---|---|
| 出力 | `outputs\t1\20261030-221116_fh_gpu.mp4` | `outputs\t1b\20261030-223842_fh_gpu.mp4` |
| フレーム数・音声とのずれ | 100・0.000秒 | 100・0.000秒 |
| 全体の時間 | 173秒 | 120秒（主に OS のキャッシュの効果と推定） |
| 1チャンク（28フレーム） | 約 16.2秒 | 約 15.3秒（約 1.8FPS） |
| VRAM の最大 | 8,624MiB | 8,486MiB（チャンクごとに増えない） |
| ジョブのコミットの最大 | 10.46GiB | 10.26GiB |
| GPU 温度の最大 | （記録なし） | 70.3℃ |
- **リップシンクの品質は未検証**：ユーザーの目視の確認はまだ。日本語の会話音声での生成もまだしていない
- T2（60秒）・T3・T4 は**未実施**

### 安全基準（今の設定）
- 開始前
  - システムのコミット残量 19GiB 以上（常駐アプリを閉じる）
  - 空きRAM 3GiB 以上
  - 重みの SHA256 と FH-01〜03 の確認
  - 省メモリ読み込み（`FLASHHEAD_LOWMEM_LOAD=1`。job_manager が設定する）
- ワーカーの中
  - 読み込みの前：空きRAM 3GiB・コミット残量 12GiB
  - 読み込みの後：空きRAM 3GiB・コミット残量 8GiB・空き VRAM 2GiB
- run_stage（監視が強制終了する）
  - ジョブのコミットの上限 12GiB
  - 空きRAM 2GiB 未満
  - コミット残量 5GiB 未満
  - VRAM 11,800MiB 超
  - 時間切れ（長さから計算。60秒なら 1,636秒）
- **GPU 温度**
  - 83℃ 以上が 10秒続いたら停止
  - 温度を 15秒以上読めなければ停止
  - 開始前に温度を読めなければ fh_gpu を開始しない
- GPU 生成の鍵：`AVATAR_ALLOW_GPU=1`（その起動の1回だけ。恒久的には設定しない）と `--allow-gpu`
  - 検証済みの長さを超える場合は `allow_unverified_length=true` も必要
- 実行は1回ずつ。自動の再実行はしない。毎回ユーザーの承認が必要

### 主要なファイル
- 状況：`PROJECT_STATUS.md`（34〜41節が FlashHead）、`RESTART_PROMPT.md`
- ワーカー：`flashhead\tools\fh_worker.py`、`frame_sink.py`、`fh_limits.json`、`image_check.py`
- 検査：`verify_frames.py`、`test_frame_sink.py`、`load_test.py`
- GUI：`gui\app.py`、`gui\job_manager.py`、`gui\start_gui.cmd`、`gui\tools\stream_dryrun_test.py`
- 監視：`tools\run_stage.py`
- パッチと重み：`patches\flashhead\fh_01-03_windows_lowmem.patch`、`licenses\flashhead_weights_sha256.txt`、`flashhead\models\`
- ジョブの記録：`gui\jobs\<job_id>\`（job.json・status.json・result.json・settings.json・job_memory.csv・worker_mem.csv・job_stdout.txt。T1b には nvidia_smi_1s.csv もある）
- 出力：`outputs\t1\`、`outputs\t1b\`（GPU 生成）、`outputs\stream_test\`・`outputs\connection_test\`（推論なしのテスト）
- 変更前のバックアップ：`*.bak_20261030_pre_stream`、`*.bak_20261030_pre_temp`

### 未解決の問題
1. リップシンクの品質（会話音声・目視）が未検証
2. 長い生成（60秒以上）での温度・VRAM・顔の一貫性・境目の連続性が未検証
3. ジョブのコミットの余裕は 約 1.7GiB だけ（10.26／12GiB）
4. キャッシュが冷えているときは、起動に 約 105秒かかる（import と librosa）
5. GUI
   - 「動画を生成」ボタンと「未検証の長さで実行する」チェックボックスが無い
   - 1画面化・太い虹色のアニメーション枠線・進捗の表示の改善が未完了
6. 注：停止・異常停止したジョブの status.json は state=running のまま残る（ワーカーが強制終了されるため）。job_manager.read_status は job_result.json を見て、stopped・aborted として表示する（問題なし）

### 中断時の確認（23:00）
- 実行中のジョブなし（`.gpu.lock` なし）
- Avatar-Pipeline 関連の Python・ffmpeg・nvidia-smi のプロセスなし
- VRAM 967MiB（他のアプリのぶん）
- 生成動画・モデル・設定・ログは削除していない

### 次回の作業（順番。GPU 生成はそれぞれユーザーの承認が必要）
1. **優先1**：5〜10秒の実際の会話音声（外部で作ったもの）とユーザーが指定する画像で、リップシンクの品質を確かめる（T1 と同じ条件の GPU 生成を1回）
2. **優先2**：品質と安全条件を確かめたうえで、T2（60秒）を検討する
3. **優先3**：GUI を1画面に収め、太い虹色のラインが滑らかに流れるアニメーションを足す

## 42. 2026-10-31：GUI の1画面化・虹色のアニメーション・進捗表示の改善（GPU 生成なし）【実測】
> ユーザーの指示（案B）。今回、実 GPU 生成は承認されていない。T1c・T2・allow_unverified_length での生成・Gemini API・モデルや依存パッケージの変更・GPU 安全設定の緩和はしていない

### 変更したファイル（変更前のバックアップは `*.bak_20261031_pre_ui`）
- `gui\app.py`：画面の構成を作り直した（処理の部分は維持）
  - 左（①キャラクター画像・②外部の音声）：画像のプレビューの横に、キャラクター名と画像チェックを並べる。下に音声の読み込みと試聴、音声の情報
  - 右（③完成動画のプレビュー・④生成設定と操作・⑤処理状況・MP4 の保存先）
    - ④：seed・顔の切り出し・「未検証の長さで実行する」（無効）・ボタン1列（動画を生成〈無効〉／接続テスト／停止）・「設定の詳細」（出力フォルダ・モデル）は折りたたみ
    - ⑤：状態のバッジ・大きな進捗率・進捗バー・8項目（チャンク・書き出し済みフレーム・経過・残り・GPU 温度・VRAM・空きRAM・ジョブのコミット or コミット残量）
  - 下：「詳細ログ」（これまでの状況の文字表示＋ワーカー出力の末尾 12行）と「生成履歴」は折りたたみ
  - 監視の値：実行中は run_stage の `job_memory.csv` の最後の行、待機中はこの PC の今の値（空きRAM・コミット残量は GlobalMemoryStatusEx、VRAM・温度は nvidia-smi を5秒に1回）。待機中に コミット残量 19GiB 未満・空きRAM 3GiB 未満・温度を読めない とき注意を出す（表示だけ。判定は従来どおり job_manager・run_stage）
  - 1秒ごとの更新で、動画と履歴は変わったときだけ送る（再生中の動画が途切れない）
  - OS がダークモードでも白い表示に固定。`@property --rb-angle` は Gradio が css_paths から取り除くので、`head` に書いた
- `gui\static\app.css`：作り直した
  - 白の背景・文字 #1f2328。主要パネル2つの枠は 5px。conic-gradient の角度を回して、虹色（赤→オレンジ→黄→緑→水色→青→紫→赤）が枠線に沿って 6秒で1周する
  - 進捗バー：高さ 14px。長さ＝実際の進捗率（app.py が width を決める）。虹色の流れは別の CSS アニメーション（600px の模様を 6秒で横に流す。つなぎ目なし）
  - `prefers-reduced-motion: reduce` で枠とバーのアニメーションを止める（虹色の表示は残す）
  - 高さ 820px 以下の画面では、説明文を隠して余白を詰める。幅 1100px 未満では1列
- `tools\run_stage.py`：CSV に1行書くたびに `f.flush()` を足しただけ（GUI が実行中の値を読めるようにするため）。しきい値・判定・停止の処理は変えていない
- 変更なし：`gui\job_manager.py`、`flashhead\` 以下、`fh_limits.json`（gpu_verified_max_seconds は 4 のまま）、両方の venv（Gradio 5.50.0 のまま）

### 確認（GPU は推論に使っていない。入力は avatar_bob.png のコピーと ffmpeg で作ったサイン波 8秒・12秒）
- 実際のブラウザ（Chrome）で確認。テストのウィンドウの表示領域は 1904×899（1920×1080 のウィンドウ）と 1350×587（1366×768 のウィンドウ）
  - 1920×1080：主要パネルの下端 838px、折りたたみの見出しの下端 898px。表示領域 899px に全部収まる。横スクロールなし
  - 1366×768：画像・名前・画像チェック・音声プレーヤー・ボタン・処理状況（進捗バーと8項目）は、スクロールなしで見える。MP4 の保存先（下端 622px）と音声の情報（下端 662px）は、表示領域 587px より下に少しはみ出す。最大化したウィンドウ（表示領域 約 610〜640px【推定】）ならほぼ収まる見込み。横スクロールなし
- アニメーション：枠もバーも duration 6,000ms・playbackRate 1（Web Animations API で確認）。枠の角度は連続して変わる（75°→165°）。reduced-motion の規則は CSS に入っている（OS の設定を切り替えての確認はしていない）
- 進捗バーの長さは status.json の progress と一致（実行中 46%・75.5%、完了 100%）
- 既存の機能
  - 画像の入力と画像チェック（warn と表示、顔の切り出しが自動でオン）
  - 音声の読み込み・試聴・長さの表示（8秒・12秒で「未検証の長さ」の注意が出る）
  - 接続テスト（GUI から）：完了・プレビュー・保存先の表示。再生中の動画は4秒たっても途切れなかった
  - 停止（GUI から、12秒の音声の途中で）：「停止しました」・不完全なファイルは保存先に出ない
  - 詳細ログ・生成履歴：開いて表示を確認
  - 監視：実行中は run_stage の値（GPU 温度 55℃・VRAM・空きRAM・ジョブのコミット 0.54/12GiB）が1秒ごとに更新された
  - `gui\tools\stream_dryrun_test.py`（8秒）：done、200/200 フレーム、差 0.000秒、7項目の検査すべて合格、通し番号の欠落なし、履歴に載る
  - `jm.start_job('fh_gpu', {})`（AVATAR_ALLOW_GPU なし）：「未承認です」で拒否
  - 「動画を生成」ボタンと「未検証の長さで実行する」は無効のまま
- 終了時：GUI のサーバーは停止。`.gpu.lock` なし。Avatar-Pipeline 関連の Python・ffmpeg のプロセスなし。AVATAR_ALLOW_GPU は設定されていない
- 今日のテストジョブ：`gui\jobs\20261031-131240_fh_dryrun`・`-131716`（停止）・`-132131`・`-132302`（スクリプト）
- スクリーンショット：`outputs\ui_screenshots_20261031\`（01〜04 が 1920、05〜07 が 1366。01・02・04 は最終版の少し前の配置）
  - 最後に Chrome のウィンドウが最小化されたため、最終版の 1920 の画面写真は撮れていない（1920 の寸法は上の実測値）

### 未解決・注意
- Gradio 5.50 の DeprecationWarning（css_paths・js・theme を Blocks に渡す書き方は Gradio 6 で launch へ移る）。5.50 では問題なし
- 「動画を生成」ボタンの処理（クリック時の動作）はまだ作っていない。GPU 生成の承認をもらってから、`AVATAR_ALLOW_GPU=1` で起動したときだけ有効にする設計で作る
- 1366×768 で保存先・音声の情報が少しはみ出す（上記）

### 次回の作業（変わらず。GPU 生成はそれぞれ承認が必要）
1. 優先1：5〜10秒の実際の会話音声（外部で作ったもの）とユーザーが指定する画像で、リップシンクの品質を確かめる
2. 優先2：T2（60秒）を検討する
3. GUI の残り：「動画を生成」ボタンの処理（承認のあと）

## 43. 2026-10-31：T1c の準備（素材と安全条件の確認だけ。GPU 生成なし）【実測】
- 環境
  - `flashhead\.venv`：Python 3.10.21・83個・torch 2.7.1+cu128・diffusers 0.36.0・numpy 1.26.4
  - SoulX-FlashHead @9bc03de：FH-01〜03 は適用済み（`git apply --check -R` が通る）。ほかの変更なし
  - 重みの SHA256：14件すべて一致
  - `fh_limits.json`：gpu_verified_max_seconds = 4.0 のまま
- 素材
  - `inputs\`：avatar_bob.png だけ。**日本語の会話音声は、プロジェクトの中に1つも無い**
  - ほかの画像・音声は MuseTalk・LivePortrait・LTX のデモ素材だけ（台帳では internal_only・非商用、または実在の人物）。T1c の素材には使わない
- 音声の変換：自分で変換しなくてよい。job_manager が ffmpeg で PCM 16bit の WAV にし（サンプルレートと ch はそのまま）、ワーカーが librosa で 16kHz・モノラルにして使う。完成品には変換した WAV の音声を結合する
- 開始前の確認（15時台の値）
  - 空きRAM 5.98GiB（基準 3GiB 以上：OK）
  - **コミット残量 16.72GiB（基準 19GiB 以上：未達）**。常駐しているのは Chrome・ChatGPT・Gemini・Discord など
  - GPU 53℃・VRAM 1,128MiB（温度は読める）
  - `.gpu.lock` なし。関連プロセスなし
- 見込み（fh_limits の実測値から計算した推定）
  - 8秒：8チャンク・約 4分・時間切れ 498秒
  - 6秒：6チャンク・約 3分24秒・時間切れ 448秒
- 次：ユーザーが日本語の会話音声（6〜8秒）と画像を用意する → GPU 生成1回の承認と、allow_unverified_length の確認 → 常駐アプリを閉じてコミット残量 19GiB 以上 → 実行

## 44. 2026-10-31：T1c の日本語音声の確認（GPU 生成なし。元の MP3 は変更していない）【実測】
- 音声：`inputs\落ち着いた女性-2026-10-31-13-36-こんにちは。今日は音声と口の動きを確認します。少し間を置いて、もう一度話します。.mp3`
  - 133,397 bytes・SHA256 a0c067a7370170b2d7a4d22acaed4b416f74f961b1fca76dfd03f9167f93a0cd
  - MP3・44,100Hz・モノラル・128kbps・8.333秒（WAV にしても 8.333061秒＝367,488 サンプル）。デコードエラーなし
  - 平均 -22.2dB・最大 -3.7dB
  - 無音区間（-50dB・0.15秒以上）：0〜0.17・1.38〜1.72・4.20〜5.26・6.57〜6.95・8.02〜8.33 秒
  - 発話（推定）：0.17〜1.38「こんにちは」／1.72〜4.20「今日は音声と口の動きを確認します」／5.26〜6.57「少し間を置いて」／6.95〜8.02「もう一度話します」
- 推奨：**切り出さずに全体（8.333秒）を使う**。発話は 8.02秒で終わり、最後に 0.31秒の無音がある。1.06秒の間で、口が閉じるかも確かめられる
- 計画（job_manager.plan_for）：208 フレーム・8 チャンク（生成 224 → 余り 16 は書かない）・見込み 237秒（約 4分。T1b の実測に近ければ 2〜3分【推定】）・時間切れ 498秒・未検証の長さ（allow_unverified_length が必要）
- 比較用の画像：`inputs\avatar_bob.png`（SHA256 は台帳と一致。T1・T1b との比較用で、固定のキャラクターにはしない）
- 開始前の確認（この時点）：空きRAM 8.14GiB・**コミット残量 19.51GiB（基準 19 をわずかに超える）**・GPU 54℃・VRAM 1,016MiB・lock なし・関連プロセスなし。ChatGPT（1.1GiB）・Chrome（1.1GiB）・Discord（0.8GiB）が常駐中

## 45. 2026-10-31 13:44：T1c（日本語の会話音声 8.333秒）の実 GPU 生成 → **成功**【実測】
> ユーザーの明示的な承認：GPU 生成1回・未検証の長さ 8.333秒・このジョブに限り allow_unverified_length=True。実行は1回だけで、T2 には進んでいない
- 入力：画像 `inputs\avatar_bob.png`（T1・T1b との比較用。固定のキャラクターにはしない）、音声 44節の MP3（元のファイルは変更なし）。seed 42。顔の切り出しは画像チェックの推奨（warn・推奨あり）でオン
- 実行：`AVATAR_ALLOW_GPU=1` を、実行用のプロセス（scratchpad の run_t1c.py）にだけ付けて job_manager の fh_gpu を1回。ジョブ `gui\jobs\20261031-134405_fh_gpu`
- 開始前の確認（直前の値）：コミット残量 19.26GiB（基準 19 は満たす。目安の 21 には届かず。ChatGPT・Chrome・Discord が常駐。ユーザーのアプリなので閉じていない）・空きRAM 7.94GiB・GPU 54℃・VRAM 1,068MiB・SHA256 14件一致・FH-01〜03 適用済み・lock なし
- 結果
  - 出力：**`outputs\t1c\20261031-134405_fh_gpu.mp4`**（2,767,304 bytes）
  - 208/208 フレーム・映像 8.32秒・音声 8.333秒・差 0.013秒（1フレーム未満）・7項目の検査すべて合格（フレーム数・512・25fps・H.264・AAC・ずれ・音声の欠けなし）
  - 時間：全体 179.0秒（run_stage）、待ち時間込みで 182.5秒、生成ループ 123.2秒。最初のチャンクの書き出しは 69.9秒の時点。1チャンク 15.15〜16.19秒（約 1.8FPS）。見込みの 237秒より速い
  - 安全停止なし（exit 0）
- チャンクごと（worker_mem.csv）：VRAM の使用 8,664〜8,666MiB・allocated 3,506MiB で一定（増え続けない）・ワーカーの private 約 10.3GiB で一定・GPU 温度 71.2→75.6℃（チャンクごとに少しずつ上がる）・空きRAM 6.13〜6.20GiB・コミット残量 9.03〜9.10GiB
- 最大・最小（run_stage）：ジョブのコミット最大 **10.28GiB**（上限 12）・空きRAM最小 **6.10GiB**・システムのコミット最大 22.88GiB・コミット残量最小 **8.99GiB**（停止の基準 5）・VRAM最大 **8,666MiB**（上限 11,800）・GPU 最高 **76.5℃**（停止の基準 83）
- 終了後：空きRAM 8.07GiB・コミット残量 19.41GiB・VRAM 1,056MiB・GPU 56.7℃ → 解放された。lock なし・関連プロセスなし。AVATAR_ALLOW_GPU はユーザー・マシンの環境変数に無い
- チャンク境界（28・56・…・196 フレーム目）：前のフレームとの差は中央値 2.32 に対して 0.92〜3.14 倍。最大の差 7.85 は境界ではなく 139 フレーム目（チャンクの途中・発話3 の始まり付近）。境界の前後の画像（`outputs\t1c\analysis_chunk_boundaries.png`）でも、ずれや破綻は見えない。最初と最後のフレームのグレーの相関 0.994
- GUI：完成表示（完了・208/208・2:59）、保存先の表示、動画プレビュー（512×512・8.33秒）の再生を確認
- 口の開き（`flashhead\tools\lipsync_chart.py` を新しく作成。MediaPipe Face Mesh・CPU・ダウンロードなし。唇の内側の上下 ÷ 口の幅）。資料は `outputs\t1c\analysis\`（lipsync_chart.png・lipsync_metrics.json・mouth_open_per_frame.csv）
  | 区間（秒） | 内容 | 口の開き（平均／最大） |
  |---|---|---|
  | 0〜0.17 | 無音 | 0.085／0.170 |
  | 0.17〜1.38 | 発話1 | 0.165／0.315 |
  | 1.38〜1.72 | 短い無音 | **0.108／0.131（閉じきらない）** |
  | 1.72〜4.20 | 発話2 | 0.104／0.272 |
  | 4.20〜5.26 | 1.06秒の無音 | **0.032／0.047（閉じている）** |
  | 5.26〜6.57 | 発話3 | 0.082／0.139 |
  | 6.57〜6.95 | 短い無音 | 0.032／0.043 |
  | 6.95〜8.02 | 発話4 | 0.105／0.181 |
  | 8.02〜8.33 | 末尾の無音 | 0.033／0.039 |
  - 顔は 208 フレームすべてで検出。口の開きと音量（dB）の相関 0.47
  - 【所見・推定】長い無音と末尾では口が閉じる。1.38〜1.72 秒では半開きのまま（「こんにちは」の後の小さな音〈-50〜-60dB〉に反応している可能性）。冒頭の 0.1秒ほどで、音より少し早く口が開き始める
  - **これらは目安。リップシンクが正確だとは断定しない。最終判断はユーザーが動画を再生して行う（未完了）**
- `fh_limits.json`：技術検査に合格したので、gpu_verified_max_seconds を 4.0 → **8.333** に更新（履歴に T1c と「リップシンクの品質評価は未完了」を記録。変更前は `fh_limits.json.bak_20261031_pre_t1c`）。1チャンクの見込みの値（16.5秒）と起動の値（105秒）は変えていない
- 次：ユーザーが T1c を再生して、リップシンク・表情・頭の動き・人物の一貫性を評価する → 結果しだいで T2（60秒）を検討（別の承認が必要）

## 46. 2026-10-31：T1c のリップシンク改善の調査（コードとログの調査だけ。GPU・生成・コード修正・設定変更なし）【実測＋推定】
> 区別：音声と映像の**全体の長さが合うこと**（T1c は差 0.013秒で合格）と、**発音単位の口の動きが合うこと**は別。後者は再生での評価が必要で、まだ終わっていない

### 時間の対応の確認（結論：全体の時間のずれを生む不具合は見つからなかった）
1. 音声 → 映像フレーム：ワーカーは公式 `generate_video.py` の `audio_encode_mode='stream'` と同じ処理（8秒＝128,000サンプルの窓に 28 フレーム分＝17,920 サンプルずつ足す → 窓の最後の 33 フレーム〈167〜199〉の特徴量 → 先頭 5〈つなぎ〉を捨てて 28 フレーム）。新しく足した音声と書き出すフレームは 1対1 で対応する
2. 44.1kHz → 16kHz：librosa 0.11（soxr）。線形補間で写した波形との相互相関の最大は **0 サンプル（0ms）**。発話の始まりも 44.1k と 16k で一致（0.16・1.72 秒など）。完成品にはモデルに渡したものと同じ WAV を結合している
3. wav2vec2 → 25fps：畳み込みの出力は 20ms ごと（窓 128,000 → 399 個）。`align_corners=True` で 200 個に補間すると、k 番目＝元の 2k 番目（中心 0.04k＋0.0125 秒）。映像フレームとの差は 1フレームよりずっと小さい（約 -7.5〜+12.5ms）。各フレームは前後 2 フレーム（±80ms）の特徴量を使う
4. 33 フレーム単位：9 潜在フレーム（時間方向の圧縮 4）。先頭 2 潜在＝5 フレームは前のチャンクの最後の 5 フレーム（つなぎ）で上書きし、新しい 28 フレームを作る
5. 口と音のずれ（全体）：口の開きと音量（dB）の相互相関は **ずらし 0 フレームで最大（0.47）**。±1 フレームで 0.44〜0.46
- 公式との違い：FH-01（xfuser の import 保護）・FH-02（torch.compile なし）・FH-03（1テンソルずつ読み込む）・attention は SDPA（公式は FlashAttention があれば使う）・書き出しは libx264 crf16。どれも時間の対応には影響しない【推定】。プログラムの不具合は見つからなかった

### 指摘された症状の原因
- **発話の始まりで口が早く動く（0.17秒）**：音の始まりに対して、口の開きが半分に達するのは -10ms。開き始めは約 2 フレーム（80ms）前。各フレームが ±2 フレームの音声を見るモデルの設計どおりで、人の発話でも口は音より先に動く（予備的な調音）。ずれの不具合ではない。ただし 0.20〜0.28 秒に 1回大きく開いてすぐ閉じる動き（0.24 → 0.05）がある → 再生での確認が必要
  - ほかの発話の始まり（1.72・5.26・6.95 秒）は、半分に達するのが +170〜+220ms 後。ただし「し」「も」「きょ」など口が狭い・閉じる子音で始まるので、この指標だけでは遅れとは言えない
- **短い無音（1.38〜1.72）で半開き**：-40dB では 0.82〜1.72 秒が無音と判定されるが、実際は 0.82〜1.38 秒に -50〜-60dB の小さな音（「こんにちは」の後の息や余韻）がある。口はこの区間ずっと 0.10〜0.15 のまま。本当の無音（-65dB 以下・4.20〜5.26 秒）では 0.03 まで閉じる。原因は「入力音声の小さな音」と「0.34秒＝約 2 潜在フレームしかなく、次の『きょう』の先読み（±80ms）と重なる時間の細かさの限界」の組み合わせ【推定】。チャンク 1 の始まり（28 フレーム目＝1.12秒）もこの区間にあり、つなぎの 5 フレーム（半開き）を引き継ぐことも一因の可能性【推定】

### 改善の候補
**A：プログラムの修正で改善できる可能性が高い**
- A1 先読みのある音声のエンコード
  - 根拠：stream 方式は、窓の終わり＝そのチャンクの新しい 28 フレームの終わり。そのため後ろの文脈（未来の音声）が無い。wav2vec2 は双方向のモデルで、±2 の参照も窓の端で切り詰められる（各チャンクの最後の 2 フレームは同じ特徴量を重ねて使う）。公式も、音声全体を一度にエンコードする `once` 方式を用意している
  - 方法：ファイルからの生成（リアルタイムでない）なので、窓の後ろに 1〜2 秒の先読みを足し、使う位置を前にずらす。全体を一度に読む `once` は 3分の音声で VRAM が増えるおそれがあるので、メモリが一定になる「先読みつきの窓」を推奨
  - GPU テスト：1回（T1c と同じ入力・seed 42 で比べる）
  - 期待できる効果：中くらい。チャンクの終わり（1.12秒ごと）の子音と口の形がよくなる可能性。全体の時間の対応は変わらない
- （修正すべき不具合は見つからなかった。全体の音声をずらす修正は不要で、してはいけない）

**B：設定や入力画像・音声の変更で試せる**
- B1 入力音声の小さな音（-50〜-60dB の余韻・息）を整える。ノイズゲート、余韻を切る、TTS で句読点の間をはっきり無音にする。公式にもコメントアウトされた loudness_norm がある
  - GPU テスト 1回。期待：短い無音で口が閉じる。効果は小〜中
- B2 顔の切り出しの比率（facecrop の face_ratio・既定 2.0・推奨範囲 1.5〜3.0）を 1.5〜1.8 にして、口の画素を増やす
  - 今のワーカーからは指定できないので、小さなコードの修正が必要
  - GPU テスト 1〜2回。期待：口の形がはっきりする。効果は小〜中
- B3 参照画像：口を閉じた自然な正面顔（今の画像は笑顔で口角が上がっている）
  - GPU テスト 1〜2回。期待：小〜中（キャラクターによる）
- B4 seed を変える：Pro は 4 ステップで毎回ノイズを足し直すので、動きの細部が変わる
  - 決まった傾向の改善は見込みにくい。GPU テスト 2〜3回。期待：小
- B5 音声の話す速さ：少しゆっくり、はっきりした発音の TTS にする。期待：小〜中

**C：SoulX-FlashHead Pro 自体の限界と考えられる**
- C1 音声エンコーダは wav2vec2-base-960h（英語の LibriSpeech 960時間で学習）。日本語の音の区別は粗い可能性。替えるには再学習が必要
- C2 時間の細かさ：25fps・時間方向の圧縮 4（1 潜在＝160ms）・音声の参照 ±80ms。日本語の 1 拍（約 120〜150ms）に近いので、速い口の形の変化や短い無音はならされる
- C3 Pro は 4 ステップに蒸留したモデル。口の開き方の大きさや個々の音の形の正確さには限界がある【推定】
- C4 チャンクのつなぎ（前の 5 フレームを引き継ぐ）は連続性のための設計。口の状態が次のチャンクの最初に引き継がれる

### 次にすること（どれも GPU 生成なので、それぞれ承認が必要）
1. ユーザーが T1c を再生して評価する（数値は目安）
2. 推奨の順番：A1（先読み・コード修正 → 接続テスト → GPU 1回）→ B1（音声の余韻を整える・GPU 1回）→ 必要なら B2（face_ratio）。同じ入力・seed で 1つずつ変えて比べる

## 47. 2026-10-31：改善の第1段階：音声の余韻の軽い調整と、A1（先読み）の要否の CPU 確認（GPU・生成・本体の修正なし）【実測＋推定】
- 元の MP3 は変更していない。資料と音声は `outputs\t1c\audio_comparison\`
  - `original_decoded_44k.wav`：job_manager と同じ変換の元の音声（比較の基準）
  - `adjusted_light_tail_-12dB_44k.wav`：調整した音声
  - `audio_adjust_check.json`
  - `audio_comparison_chart.png`・`audio_comparison_zoom_0.6-1.9s.png`
  - `a1_context_check.json`・`a1_context_check.png`
- 新しいスクリプト（CPU だけ）：`flashhead\tools\audio_tail_adjust.py`、`flashhead\tools\a1_context_check.py`（wav2vec2 だけを CPU で動かす。本体のコードは読み込むだけ）

### 元の音声の再解析
- 有声（pyin）は 0.72秒で終わる。0.74〜0.82秒は「は」の息の減衰（-33〜-45dB）
- 0.82〜1.38秒は -48〜-64dB の無声の音。0.96〜1.20秒は高い音が多く平坦（重心 2〜2.8kHz）→ 吸気と推定
- 1.38〜1.68秒は -55〜-70dB の低い音。**1.70秒に「きょう」の k の破裂（-64 → -37dB）**
- 長い無音の雑音の床は -80dB。0.82〜1.68秒は床より 20〜30dB 大きい

### 調整の内容（軽い調整・1種類）
- ゲインを掛けるだけ（時間方向の処理なし）
- 保護：-45dBFS を超える窓か有声の窓を発話とし、前 100ms・後 150ms を保護する。保護の中はビット単位で元のまま
- 保護の外で -48dBFS より小さい音だけ、最大 -12dB（完全な無音にはしない）。30ms のなめらかな傾き
- 1回目は境目でゲインが 6dB 段差になったので、弱める範囲を内側に縮めて直した → 1サンプルあたり最大 0.009dB
- 変更した区間：**0.985〜1.604秒**（息と短い無音）・6.565〜6.78秒・8.18〜8.32秒だけ。0.82〜0.98秒の「は」の息と、1.70秒の k は保護されて変わらない

### 検査
- サンプル数は同じ（367,488・8.333061秒）
- 相互相関のずれは 0 サンプル
- -40dB での発話の開始・終了 14 か所はすべて同じ
- 最大音量は -3.7dBFS で同じ
- 発話の区間の平均の音量は変化なし（発話2 -27.3・発話3 -32.9・発話4 -24.1dB）
- 無音の区間：0.82〜1.38 -55.4 → -63.5dB、1.38〜1.72 -62.4 → -69.8dB、6.57〜6.95 -71.7 → -77.5dB、長い無音は変化なし
- 不自然な音切れ：数値の上では無い（段差なし・語頭と語尾は保護）。耳での確認はユーザーに頼む

### 調整の効果の見込み（wav2vec2 の特徴量・CPU。リップシンクそのものではない）
- 「無音らしさ」（本当の無音〈4.40〜5.10秒〉の特徴量とのコサイン類似度。長い無音は 0.80、発話は 0.14）
  | 区間 | 元 | 調整後 |
  |---|---|---|
  | 0.98〜1.38 | 0.445 | 0.491 |
  | 1.38〜1.72 | 0.566 | 0.609 |
  | 6.57〜6.95 | 0.531 | 0.542 |
  | 0.82〜0.98（保護） | 0.262 | 0.248 |
- 少し無音に近づくが、長い無音（0.80）にはほど遠い。0.82〜0.98秒の「は」の息は発話に近い特徴のまま（保護したので当然）
- → 効果は**小さい**見込み【推定】。**標準の処理には採用しない**。GPU で1回比べる候補として残す

### A1（先読み）の CPU 確認
- stream 方式と once 方式の特徴量の一致度（±2 の参照を含む）は、全体で 0.853
- チャンクの中の位置による違いがはっきりある：位置 0〜20 は 0.88 前後、位置 23 から下がり、**26 は 0.70、27 は 0.63**。チャンクの終わりで、後ろの文脈が無いことの影響が実際にあることを確認した
- チャンクの終わり（位置 23〜27）の時刻：0.92〜1.12・2.04〜2.24・3.16〜3.36・4.28〜4.48・5.40〜5.60・6.52〜6.72・7.64〜7.84 秒
- 観察された問題との一致は**一部だけ**
  - 0.82〜1.72 秒の半開き：チャンクの終わり 0.92〜1.12 を含む（一致度 0.72）。ただし 1.38〜1.72 はチャンク 1 の途中（位置 10〜16）
  - 0.20〜0.28 秒の開き閉じ：チャンク 0 の始まり（位置 5〜7）。チャンクの終わりではない
  - 発話の始まり 1.72・5.26・6.95 秒：チャンクの途中（位置 15・19・6）
  - ほかのチャンクの終わり（4.28〜4.48 の長い無音、6.52〜6.72 の短い無音2）では、口は閉じていて問題は見られない
- → **A1 の優先度は下げる**（1番目 → 調整した音声の比較の後）。チャンクの終わりの特徴量の劣化は実際にあるが、今回見えた問題の主な原因とは言えない【推定】

### 次の GPU 比較テストの価値
- 調整した音声での T1c 比較（同じ画像・seed 42・顔の切り出しオン・8.333秒）：**1回なら試す価値はある**。期待は小さく、主に 0.98〜1.72 秒の半開きの変化を見る。ただし、まずユーザーが元の T1c を再生して、この半開きが気になるかを判断してからがよい
- 実行には GPU 生成1回の承認が必要。8.333秒は検証済みなので、allow_unverified_length は不要

## 48. 2026-10-31：GUI の「動画を生成」を実装（GPU 推論・生成なし。リハーサル〈推論なし〉でテスト）【実測】
> 方針（ユーザー）：B1（音声の調整）と A1（先読み）の GPU 比較は当面保留。元の T1c 音声を標準の入力とする。GUI だけで生成できる構成を優先する
- 変更：`gui\job_manager.py`・`gui\app.py`・`gui\static\app.css`（追記だけ）。新規：`gui\start_gui_gpu.cmd`。変更前は `*.bak_20261031_pre_gpubtn`
- 変更なし：`fh_worker.py`・`run_stage.py`・`fh_limits.json`（8.333秒のまま）・モデル・両方の venv・GUI のデザイン（1画面・虹色）

### GPU 実行許可の扱い（3段階を維持）
- 1段目（どちらか）
  - (a) 従来どおり：そのプロセスだけの `AVATAR_ALLOW_GPU=1`（Claude Code・コマンドライン）
  - (b) 新規：**GUI の1回限りの許可（permit）**
    - GUI を GPU 生成モードで起動したときだけ使える（`start_gui_gpu.cmd` → `app.py --gpu-enabled`。起動引数だけで、環境変数は使わない。ウィンドウを閉じれば消える）
    - 「動画を生成」を押すと、バックエンド（`issue_gpu_permit`）が入力と開始前の条件を検査する。合格したら、乱数の ID をメモリの中だけに発行する
    - permit は、画像・音声の SHA256・長さ・seed・顔の切り出し・出力先・未検証の確認に結び付く。期限は 180秒で、**1回使うと（不合格でも）消える**
    - 確認パネルの「この内容で1回だけ生成する」で `start_job(permit=…)` を呼ぶ。入力が変わっていれば拒否する
    - `prepare_gpu`・`confirm_gpu`・`cancel_gpu` は `api_name=False` で API に出さない。通常モードでボタンを無理に有効にして押しても、バックエンドが拒否する（確認済み）
- 2段目：ワーカーへの `--allow-gpu`（従来どおり。リハーサルでは付けない）
- 3段目：検証済み（8.333秒）を超える音声は、GUI の「未検証の長さで実行する」にチェックして発行した permit に限り、allow_unverified_length=True。ワーカーも従来どおり確認する。検証済みの長さはコードからは更新しない
- **開始前の検査をバックエンドで強制**（`gpu_precheck`。permit の発行時と開始時の2回。環境変数の経路にも適用 → 従来より厳しい）
  - 実行中のジョブが無い
  - コミット残量 19GiB 以上・空きRAM 3GiB 以上
  - GPU 温度を読める・80℃ 未満
  - 空き VRAM 8,192MiB 以上
  - FH-01〜03 が適用済み
  - 重みの SHA256（初回は全体を計算〈約 16秒〉。合格したらファイルの大きさと更新時刻を `gui\jobs\.weights_sha256_ok.json` に記録し、変わらなければ再計算しない）
- 実行中の監視（run_stage の 12GiB・空きRAM 2GiB・コミット残量 5GiB・VRAM 11,800MiB・83℃ が10秒・温度を15秒読めない）と、ワーカーの読み込みの前後の検査は変更なし。同時に1ジョブ・自動の再実行なし
- リハーサルモード（`app.py --gpu-rehearsal`）：同じ許可と検査を通したうえで、ワーカーを推論なし（dry_run）で動かす。ジョブ名は `_fh_gpu_rehearsal`。GUI のテスト用

### GUI
- バッジ：通常＝「GPU 生成：無効（通常モード）」／GPU 生成モード＝「このウィンドウの間だけ・1回ごとに確認」／リハーサル
- 未検証のチェックボックス：GPU 生成モードで、検証済みを超える音声のときだけ押せる（音声を変えるとオフに戻る）
- 確認パネル：音声の長さ・フレーム数・チャンク数・見込みの時間・時間切れ・seed・顔の切り出し・未検証かどうか・開始前の8項目（✅/⛔ と今の値）
- 生成中の表示：進捗率・チャンク・書き出し済みフレーム・経過・残り・GPU 温度・VRAM・空きRAM・**コミット残量**（新規。run_stage の記録＋コミットの上限から計算）。ジョブのコミットは下の行に表示
- 停止ボタンは従来どおり

### テスト（GPU 推論なし）
- バックエンド（`scratchpad\test_gpu_permit.py`、リハーサル）：拒否すべき 18 件はすべて拒否され、成功の経路も OK だった
  - 拒否した 18 件：通常モードでの発行／通常モードでの permit 付き開始／環境変数・permit なし／音声なし／未検証で確認なし／コミット 15GiB／空きRAM 2.5GiB／温度を読めない／81℃／空き VRAM 7,288MiB／偽の permit／確認後に seed を変更／同じ permit の再使用／期限切れ／発行後に条件が悪化／使用済み permit の再利用／実行中に次の発行
  - 成功：実際の値（コミット残量 19.19GiB）で 8.333秒のリハーサルが完了（208 フレーム・mode=dry_run・--allow-gpu なし・source=gui_permit）。12秒（未検証を確認）は開始 → 停止 → stopped・完成品なし
  - 条件の不合格は、このテストの中だけで sys_mem・gpu_status を差し替えて作った
- ブラウザ（Chrome）
  - リハーサルモード（7861・実際の値）：12秒で未検証の確認なし → 拒否のメッセージ。確認あり → コミット残量 18.55GiB で開始前の条件により拒否
  - テスト用のハーネス（7862・リハーサル・このプロセスの中だけコミット残量 +3GiB）：確認パネル → 開始 → 生成中の8項目 → 完了 → プレビュー（8.33秒・512px）、「やめる」でパネルが閉じる、生成中の停止 → 停止しました・完成品なし
  - 通常モード：ボタン無効・API に生成の処理が無い。ボタンを無理に有効にして押しても、バックエンドが拒否した
  - 写真：`outputs\ui_screenshots_20261031_gpubtn\`（01 未検証の拒否／02 開始前の条件の拒否／03 確認パネル／04 生成中）
- 既存の `stream_dryrun_test.py`：done・200/200・検査すべて合格
- 終了時：GUI は停止・lock なし・関連プロセスなし・AVATAR_ALLOW_GPU はユーザー・マシンの環境変数に無い
- リハーサルのジョブ：`gui\jobs\20261031-141242/141253/141547/141618_fh_gpu_rehearsal`。出力は `outputs\gui_gpu_rehearsal\`・`outputs\`

### 注意・残り
- 確認パネルを開いている間は、パネルの分だけ処理状況が下にずれる（1920×1080 で一時的にスクロールが必要）
- リハーサルの状況の文はワーカーの dry_run の文（「接続テスト」）のまま。種類の欄には「GPU 生成のリハーサル」と出る
- **GUI からの実際の GPU 生成は未実施**（承認が必要）。手順：常駐アプリを閉じてコミット残量 19GiB 以上 → `gui\start_gui_gpu.cmd` で起動 → 画像と音声 → 「動画を生成」→ 確認 →「この内容で1回だけ生成する」

## 49. 2026-10-31：GUI-G1（GUI からの初回の実 GPU 生成）の準備 → **開始前の条件が未達のため、生成していない**【実測】
- 承認（ユーザー）：T1c と同じ入力の GPU 生成を最大1回。最後の「この内容で1回だけ生成する」はユーザーが押す（Claude Code は押さない）。ユーザーのアプリは無断で閉じない
- 準備
  - `gui\start_gui_gpu.cmd` で起動（プロセスは `app.py --gpu-enabled`。バッジ「GPU 生成モード」）
  - 画像と音声を読み込んだ。ファイルは `gui\tmp\ui_test` のコピーで、SHA256 は元と同じ（avatar_bob.png・T1c の MP3）
  - 顔の切り出しは画像チェックで自動オン（T1c と同じ）・seed 42・8.333秒（検証済み）・出力先 `outputs\gui_g1`（新規。T1c は上書きしない）
- 開始前の検査
  - GUI を起動する前：8項目すべて合格（コミット残量 19.27GiB）
  - GUI を起動した後：**コミット残量 18.43〜18.6GiB（19GiB 未満）** → 「動画を生成」で確認画面は出なかった（バックエンドが拒否。permit は発行されていない）
  - ほかの項目は合格（空きRAM 7.7GiB・53℃・空き VRAM 11,149MiB・FH-01〜03）
  - GUI 自身（Python 約 0.8GiB）の分だけ残量が下がった。常駐：ChatGPT 約 0.7〜1.1GiB・Chrome 約 0.4〜1.2GiB・Discord 約 0.4〜0.8GiB
- 生成していない。ユーザーがアプリを閉じて GUI で操作するのを待つ（GUI は起動したまま。読み取りだけの見守りを動かしている）

## 50. 2026-10-31 14:26：GUI-G1 の結果 → **ワーカーが開始直後に拒否（GPU 推論なし）。長さの測り方が一致していない不具合**【実測】
- ユーザーがアプリを閉じて、GUI で「動画を生成」→「この内容で1回だけ生成する」を押した（Claude Code は押していない）
- ジョブ `gui\jobs\20261031-142654_fh_gpu`
  - 許可：source=gui_permit・rehearsal なし・`--allow-gpu` あり
  - 開始前の検査は8項目すべて合格（コミット残量 19.38GiB・空きRAM 8.01GiB・54℃・空き VRAM 11,446MiB・FH-01〜03・SHA256）
  - 出力先 `outputs\gui_g1`・seed 42・顔の切り出しオン
- 結果：**failed**（0.5秒）
  - ワーカーの判定：「8.3秒は GPU 生成で未検証の長さです（検証済みは 8秒まで）」
  - run_stage：exit 2・VRAM 最大 0MiB・ジョブのコミット最大 0.01GiB → モデルは読み込んでいない（GPU 推論なし）
  - 出力ファイルなし・lock なし
  - GUI は「失敗」と理由を表示。確認パネルは閉じ、permit は使用済みで消えている → 次のジョブには引き継がれない
- 原因（不具合）
  - job_manager（GUI の計画・permit・未検証の判定）は、**元の MP3 を ffprobe で測った 8.333000秒**を使う
  - ワーカーは、**変換後の WAV のサンプル数の 8.333061秒**（367,488 / 44,100）を使う
  - `fh_limits.json` の gpu_verified_max_seconds = 8.333 は、45節で T1c の値を3桁に丸めて記録したもの（記録の誤り。T1c で実際に処理したのは 8.333061秒の WAV）
  - 結果として、GUI は「検証済み」と表示し、ワーカーは「未検証」として拒否した。ワーカーが安全側で止まったので、実害はない
  - 表示の「検証済みは 8秒まで」も、ワーカーの文の書式（:.0f）による丸め
- 修正案（コードの修正。まだしていない・承認が必要）
  1. job_manager の長さの判定（計画・permit・未検証）を、ワーカーと同じ「変換後の WAV のサンプル数」にそろえる
  2. 検証済みの判定をフレーム数でも行う（T1c は 208 フレーム）。秒の小数の丸めに左右されないようにする。`fh_limits.json` には T1c の正確な値（8.333061秒・208 フレーム）を記録する（検証済みの長さの記録を直すので、承認が必要）
  3. ワーカーとGUIの文の秒数を3桁で表示する
- GUI-G1 のやり直しは、修正とユーザーの承認のあと。自動の再実行はしていない

## 51. 2026-10-31 14:35：GUI-G1 の失敗の原因を直した（長さの判定を GUI とワーカーで統一。GPU 推論・生成なし）【実測】
### 原因（確定）
1. `fh_limits.json`：gpu_verified_max_seconds = **8.333**（45節で T1c の値を3桁に丸めて記録した誤り）
2. GUI（app.on_audio）と job_manager（plan_for・issue_gpu_permit・start_job）：**元の MP3 を ffprobe で測った 8.333000秒**で判定 → 検証済み
3. fh_worker：**変換後の WAV のサンプル数の 8.333061秒**（367,488 / 44,100）で `8.333061 > 8.333 + 1e-6` → 未検証として拒否。run_stage.py には長さの判定は無い
4. 判定の中に整数への切り捨ては無い。ただし表示が丸められていた（ワーカーの `{verified:.0f}` で「検証済みは 8秒まで」・`{audio_sec:.1f}` で「8.3秒」、job_manager の開始時の文 `:.0f`、GUI のヒント `:.0f`）
5. 設定のキャッシュは無い（plan_for は毎回ファイルを読む。起動時のヒントの文だけ固定だった）
6. 失敗したジョブ 20261031-142654_fh_gpu：ワーカーは入力の検査（0.01秒）で終了。run_stage は exit 2・0.5秒・VRAM 最大 0MiB・ジョブのコミット 0.01GiB → **モデルの読み込み前。GPU は使っていない**
### 修正（必要最小限。安全条件は変えていない。変更前は `*.bak_20261031_pre_lenfix`）
- 新規 `flashhead\tools\fh_length.py`（標準ライブラリだけ）：長さ＝PCM の WAV のサンプル数 ÷ サンプルレート、判定＝目標のフレーム数（round(秒×25)）が検証済みのフレーム数を超えるか。GUI・job_manager・ワーカーの全部がこれを使う
- `fh_limits.json`：gpu_verified_max_seconds を **8.333061**（T1c の実測）に訂正し、**gpu_verified_max_frames = 208** を追加（判定の根拠）。T1c の履歴の注記を訂正。検証済みの範囲は T1c の 208 フレームのまま（引き上げていない）
- `fh_worker.py`：長さの判定を fh_length に置き換え（文は 3桁＋フレーム数）。settings.json の length に正確な秒数・フレーム数を記録
- `gui\job_manager.py`：`audio_seconds()` を追加（start_job と同じ ffmpeg の変換で WAV にして測る。変換した WAV は `gui\tmp\length_check\` に SHA256 ごとに置き、元のファイルは変更しない）。plan_for・issue_gpu_permit・permit の照合・start_job の判定をこれにそろえた。job.json の audio_source に wav_seconds を記録
- `gui\app.py`：音声の表示と未検証のチェックボックスを audio_seconds にそろえた。表示は 3桁の秒＋フレーム数。起動時に固定だったヒントの秒数の表示はやめた
- 注：フレーム数で判定するので、208 フレームになる音声（約 8.34秒未満）は検証済みとして扱う（例：8.339秒）。209 フレーム以上は未検証
### CPU テスト（GPU なし）
- GUI とワーカーの判定の一致（`scratchpad\test_length_agree.py`。ワーカーは mode=gpu・--allow-gpu なしで直接起動 → 長さに合格すると torch を読み込む前に「--allow-gpu が必要」で止まる）：7ケースすべて一致
  | ケース | 秒（WAV） | フレーム | 判定 |
  |---|---|---|---|
  | T1c の元の MP3 | 8.333061 | 208 | 検証済み（GUI・ワーカーとも受付） |
  | T1c のジョブの WAV | 8.333061 | 208 | 検証済み |
  | 8.339秒 | 8.339002 | 208 | 検証済み |
  | 8.36秒 | 8.360000 | 209 | 未検証 → 確認なしは両方で拒否 |
  | 9秒 | 9.0 | 225 | 未検証 → 確認なしは拒否、確認ありは受付 |
  | 12秒 | 12.0 | 300 | 未検証 → 拒否 |
- permit のテスト（`scratchpad\test_gpu_permit.py`・リハーサル・実際の値でコミット残量 19.93GiB）：拒否すべき 18 件はすべて拒否、成功・停止も OK。T1c の MP3 は検証済みとして 208 フレームのリハーサルが完了（job.json：wav_seconds 8.333061・unverified False・verified_max_frames 208）
- 元の MP3 の SHA256 は変わらない（a0c067a7…）。通常モードの GUI は GPU_MODE=False で読み込める（表示「8.333秒・208 フレーム」）
- 終了時：GUI は停止（変更前のコードで動いていたため）・lock なし・関連プロセスなし
### GUI-G1 の再実行
- 再実行できる状態。手順：常駐アプリを閉じてコミット残量 19GiB 以上 → `gui\start_gui_gpu.cmd` を起動し直す（変更を読み込むため）→ 画像・音声・出力先 `outputs\gui_g1` → 「動画を生成」→ 確認 →「この内容で1回だけ生成する」（ユーザーが押す）
- 前回の承認（最大1回）は生成まで進まなかった。再実行してよいかはユーザーが決める

## 52. 2026-10-31 14:39：**GUI-G1 成功**（GUI からの初めての実 GPU 生成。ユーザーが最終ボタンを押した）【実測】
- 承認：GUI-G1 の GPU 生成を新たに最大1回。最後の「この内容で1回だけ生成する」はユーザーが押した（Claude Code は押していない）。未検証の長さの許可は使っていない
- 起動：修正後の `gui\start_gui_gpu.cmd`（`app.py --gpu-enabled`）
- 起動後の開始前の8条件はすべて合格
  - 確認画面の時点：コミット残量 19.31GiB（目安の 20GiB には届かず。正式な基準 19 は満たす）・空きRAM 8.13GiB・54℃・空き VRAM 11,445MiB・FH-01〜03・SHA256
  - ユーザーのアプリは閉じていない
- ジョブ `gui\jobs\20261031-143923_fh_gpu`
  - 許可：source=gui_permit・`--allow-gpu` あり・allow_unverified_length=False
  - 入力：avatar_bob.png・T1c の MP3（WAV 8.333061秒・208 フレーム＝検証済み）・seed 42・顔の切り出しオン
- 結果：**done**
  - 出力：**`outputs\gui_g1\20261031-143923_fh_gpu.mp4`**（2,767,304 bytes）
  - 208/208 フレーム・映像 8.32秒・音声 8.333秒・差 0.013秒・7項目の検査すべて合格
- T1c との比較
  | | T1c（Claude Code から） | GUI-G1（GUI から） |
  |---|---|---|
  | 全体（run_stage） | 179.0秒 | **173.1秒** |
  | 生成ループ | 123.2秒 | 130.2秒 |
  | 1チャンク | 15.15〜16.19秒 | 15.99〜17.11秒 |
  | 最初のチャンクの書き出し | 69.9秒の時点 | 57.7秒の時点（import 11.9秒・読み込み 21.7秒） |
  | VRAM の最大 | 8,666MiB | **8,436MiB** |
  | ジョブのコミットの最大 | 10.28GiB | **10.29GiB** |
  | 空きRAM の最小 | 6.10GiB | 6.34GiB |
  | コミット残量の最小 | 8.99GiB | 9.18GiB |
  | GPU の最高温度 | 76.5℃ | 76.1℃ |
  | 出力 | — | **T1c と SHA256 が同じ（a74f79eb…）・208 フレームすべてピクセル差 0** |
  - 同じ入力・seed 42 で、映像はまったく同じ（再現性がある）。映像の品質は T1c と同じ（リップシンクの評価は、T1c と共通で未完了）
  - チャンクごと（worker_mem.csv）：VRAM 8,434〜8,436MiB・allocated 3,506MiB で一定・ワーカーの private 約 10.3GiB で一定・温度 69.5〜76.3℃・空きRAM 6.36〜6.46GiB・コミット残量 9.20〜9.27GiB。安全停止なし（exit 0）
- GUI
  - 完了後の表示：完了・8/8・208/208・2:53・保存先を表示
  - プレビュー：8.33秒・512px・再生できる
  - 生成中の表示は、ユーザーが画面で見た（Claude Code は終了後の表示と status.json・worker_mem.csv の更新で確認）
- 終了後：VRAM 817〜820MiB・空きRAM 8.15GiB・コミット残量 19.41GiB → 解放された。worker・run_stage・ffmpeg のプロセスなし・lock なし
- 実行許可：確認パネルは閉じ、permit は使用済みで消えた。ジョブは1つだけで、次のジョブには引き継がれていない。もう一度生成するには、新しい検査と確認が必要
- 注：見守りのスクリプト（scratchpad）は、ジョブのフォルダができてからロックができるまでの一瞬を「終了」と誤って判定した。GUI と安全機構の問題ではない。その後、ロックが消えるまで待って確認した
- GUI（GPU 生成モード）は起動したまま。T2 には進んでいない

## 53. 2026-10-31：GUI-G1 の動画で音声が聞こえない件の調査（修正・再生成なし）【実測】
- 対象：`D:\Avatar-Pipeline\outputs\gui_g1\20261031-143923_fh_gpu.mp4`（このフォルダの唯一の MP4。2,767,304 bytes・14:42:17）
- MP4 自体は正常
  - ストリーム：映像 h264 High 512×512・208 フレーム・8.320秒。**音声 aac LC・44,100Hz・モノラル・8.333秒・約 172kbps・default=1**
  - 音声のデコード：エラー 0
  - 音量：平均 -22.2dB・最大 -3.7dB（元の MP3 と同じ）。無音の区間（-50dB）も 46節と同じ位置
  - 元の WAV（ジョブの inputs\audio.wav）との比較：ずれ 0 サンプル・相関 1.0000
  - ジョブのログ：plan → stream_done（208）→ done・aac=true・audio_not_cut=true。ffmpeg の書き出しのログは空（エラーなし）
  - → 結合処理（ffmpeg）に問題は無い
- GUI のコード：app.py・app.css に mute・autoplay・volume の設定は無い
- **原因：Claude Code が GUI-G1 の完了後に再生を確かめたとき、ユーザーと同じブラウザのタブで `video.muted = true; video.play()` を実行したため**（自動再生を通すためのミュート。52節の確認作業）
  - タブの動画要素：muted=true、defaultMuted=false、muted 属性なし（＝スクリプトで設定された状態）、volume=1。音声はデコードされている（webkitAudioDecodedByteCount 482,504）
  - Gradio 5.50 の動画プレーヤーは「再生・時間・シーク・全画面・ダウンロード」だけで、**ミュートの解除や音量のボタンが無い** → 画面からは元に戻せなかった
- 対処（コードの修正は不要）：ブラウザのタブを再読み込み（F5）すると、動画要素が作り直されて muted=false に戻る。ダウンロードした MP4 を別のプレーヤーで再生しても音声は出る
- 今後：確認のためにユーザーのタブで再生するときは、ミュートしない（自動再生が拒否されたら、再生できることだけを確かめる）か、自分用の別のタブで行う。必要なら、GUI に「音声つきで再生」の案内を出すことを検討（今回は修正しない）

## 54. 2026-10-31：完成動画の音声のミュート・解除ボタンを追加（GPU 推論・再生成なし）【実測】
- 背景：53節。Gradio 5.50 の動画プレーヤーには、ミュートや音量のボタンが無い
- 変更（変更前は `*.bak_20261031_pre_mute`）
  - `gui\app.py`
    - 動画プレビューの下に `gr.HTML` のボタン `#av-mute-btn`（「🔊 音声オン」／「🔇 ミュート中」）と説明を追加
    - `HEAD` に、ブラウザの中だけで動くスクリプトを追加。サーバーとのやり取りは無い
  - `gui\static\app.css`：ボタンの見た目（音声オン＝緑の枠、ミュート中＝赤の枠と淡い赤の背景、高さ 40px・丸い形）、幅 600px 以下で全幅・高さ 44px、高さ 820px 以下で説明を隠す
- スクリプトの仕組み
  - 対象は `#out-video video` だけ（入力音声の試聴 `#char-audio` には触れない）
  - ボタンの表示は、いつも実際の video 要素の状態から作る：`muted` または `volume===0` なら「ミュート中」
  - クリック：ミュート中なら `muted=false`（volume が 0 なら 1 に戻す。video 要素の音量だけで、ブラウザ・Windows の音量には触れない）、そうでなければ `muted=true`
  - video 要素の volumechange・loadedmetadata・loadstart・emptied・play と、MutationObserver（要素の作り直し・src の変更）、2秒ごとの確認で、表示を実際の状態に合わせる
  - 1回目の実装は、ボタンの文字の書き換えを MutationObserver が拾ってまた書き換える**無限ループでタブが固まった**（テスト用のタブだけ）→ 値が変わったときだけ書き換え、ボタン自身の中の変化は無視し、走査は 1フレームに1回にまとめて解決
- テスト（専用のタブ・テスト用の GUI：通常モード 7862、GPU 生成モードの画面＝リハーサル 7863。ユーザーのタブ〈7861〉と GUI には触れていない）。完成動画は `outputs\gui_g1\20261031-143923_fh_gpu.mp4`
  1. 初期：「🔊 音声オン」・muted=false・volume=1
  2. 実際のマウス操作で動画を再生 → 音声つきで再生（muted=false・再生位置が進む・音声のデコード量が増える）。耳で聞く確認はユーザーに頼む
  3. 再生中にボタン → 「🔇 ミュート中」・muted=true・再生は続く。もう一度 → 「🔊 音声オン」・muted=false・再生は続く（1.31→1.92秒）
  4. 一時停止中のミュート → 再開してもミュートのまま、表示も一致。解除できる
  5. シーク（5秒・1秒）の前後で、状態と表示が一致
  6. ページの再読み込みの後も動く
  7. 動画の差し替え（src を別の MP4 に変更）と video 要素の作り直し（ミュートあり・なし）の後も、表示が実際の状態に合い、ボタンで切り替えられる
  8. volume=0 を外から設定 → 「ミュート中」と表示 → 押すと volume=1・音声オン。muted=true を外から設定（53節の問題の再現）→ 「ミュート中」と表示 → 押すと解除できる
  9. 入力音声の試聴（`#char-audio audio`）は、ミュートの切り替えの前後とも muted=false・volume=1 のまま
  10. 通常モードと GPU 生成モード（リハーサル）の両方で動く。390px 幅（テスト用のタブの中の iframe）でもボタンは 201×44px・横スクロールなし。1904×951 で1画面に収まる（パネルの下端 820px・折りたたみ 880px）
  11. 元の MP4 の SHA256 は変わらない（a74f79eb…）
  - 注：スクリプトからのクリック（ユーザーの操作でない）では、ブラウザの自動再生の制限により、音声つきの再生が止められることがある。実際のマウス操作では問題ない
  - 写真：`outputs\ui_screenshots_20261031_mute\`
- ユーザーの 7861 の GUI は変更前のコードのまま動いている → 新しいボタンを使うには `start_gui_gpu.cmd`（または `start_gui.cmd`）を起動し直す

## 55. 2026-10-31 15:00：ミュートボタンが 7861 の GUI に表示されない件 → 原因は古い GUI のプロセス（コードの問題ではない）【実測】
### 原因（確定）
- 7861 番で待ち受けていたのは PID 24008（`python.exe app.py --gpu-enabled`）。**2026-10-31 14:37:33 に Claude Code が GUI-G1 のために裏で起動した GUI**（親は Claude のシェル → cmd 21744 → venv の python 19832 → 24008）
- ミュートボタンを加えた app.py の更新は 14:52:42。このプロセスは更新前のコードのまま動いていた
- 裏で起動したので、**コンソールのウィンドウが無く、ユーザーには閉じられなかった**。ユーザーが `start_gui_gpu.cmd` で起動し直しても、7861 番が使用中のため新しい GUI は起動できずに終わり、ブラウザには古い GUI が表示され続けた（Gradio の起動エラーの画面はすぐ閉じる）
- ユーザーのタブの DOM（読み取りだけ）：`#av-mute-btn` も `#av-mute` も、head のスクリプトも無い → 古いコードが配信されていた
- app.py のボタンの定義・配置（③ 完成動画の列、動画の直下）・CSS・通常モードと GPU 生成モードの条件には問題なし（どちらのモードでも同じ場所に出る）
### 修正（変更前は `*.bak_20261031_pre_portcheck`）
- `gui\app.py`
  - 起動時に、ポートがすでに使われていれば「前の GUI が動いています。そのウィンドウを閉じてから起動して」と表示して終了する（exit 3）。確認済み：7861 が使用中のとき、4秒で終了し、既存の GUI には影響なし
  - ミュートボタン：動画があるかの判定を直した。Chrome は src を外しても currentSrc に前の URL が残るので、`src` の属性・プロパティ・source 要素と networkState で判定する（動画が無いときはボタンを無効にし、「完成動画がありません」と表示）
- `gui\start_gui.cmd`・`gui\start_gui_gpu.cmd`：エラーで終わったときは `pause` して、メッセージを読めるようにした
### ブラウザでの確認（専用のタブ・テスト用の GUI 7862 通常モード／7863 GPU 生成モードの画面〈リハーサル〉。ユーザーのタブと 7861 には触れていない）
- ボタンは「③ 完成動画のプレビュー」の直下（動画から 23px 下）に、150×40px で表示される（両方のモード）
- 実際のマウス操作：再生中に「🔊 音声オン」→「🔇 ミュート中」（muted=true・再生は続く）→「🔊 音声オン」（muted=false・再生は続く）。ボタンの表示・aria-pressed と video.muted は常に一致
- 動画の差し替え（src の変更）の後も切り替えられる。動画が無いときはボタンが無効になり、押しても何も起きない
- 入力音声の試聴（#char-audio audio）は、切り替えの前後とも muted=false・volume=1
- 無限ループなし：3秒間でボタンの書き換え 0 回・描画 91 回。ボタンのスクリプトによる JavaScript のエラーなし（コンソールには Gradio 内部の通信中断の AbortError が1件。ページの移動による）
- 元の MP4 の SHA256 は変わらない
- 写真：`outputs\ui_screenshots_20261031_mute\03〜05`
### ユーザーの操作
- 7861 の古い GUI（PID 24008。親 19832・21744）を終了してから、`gui\start_gui_gpu.cmd`（または `start_gui.cmd`）で起動し、ブラウザを再読み込みする
- ウィンドウが無いので、タスクマネージャーの「詳細」で、コマンドラインに `app.py --gpu-enabled` を含む python.exe（PID 24008・19832）を終了する。または、Claude Code に「7861 の古い GUI を止めて」と指示する（許可があれば止める）
- 今後、ユーザーが使う GUI は Claude Code が裏で起動せず、ユーザーが cmd で起動する（ウィンドウがあれば、いつでも閉じられる）

## 56. 2026-10-31 15:20：OpenAI Codex への引き継ぎ資料を作成（GPU・生成・コードの変更なし）
- 引き継ぎ資料：**`D:\Avatar-Pipeline\CODEX_HANDOFF.md`**（目的・実装状況・フォルダ構成・環境・FH-01〜03・GPU 生成の仕組み・GUI の起動・モードの違い・GPU 許可・安全基準・T1〜GUI-G1 の結果・リップシンク・不具合・次の作業・テスト・禁止事項・バックアップ・60/120/180秒の計画）
- CPU テストの写しを `tests_cpu\` に置いた（Claude の一時フォルダにしか無かったため）：test_gpu_permit.py・test_length_agree.py・gui_rehearsal_harness.py・run_t1c.py（GPU を使う参考用）・README.md
- 作成時の状態：**7861 は今も古い GUI（PID 24008・14:37 起動。app.py の更新は 15:04）**。ミュートボタンのコードは動く（55節）が、ユーザーの画面にはまだ出ていない → ユーザーが古い GUI を止めて起動し直すまで未解決（CODEX_HANDOFF 13章-1）
- 注意：`secrets\README.txt` に API キーのような文字列が平文で書かれている（内容は記録しない）。ユーザーに無効化・削除の検討を伝えた

## 57. 2026-10-31 15:40：古い GUI の停止・GUI-G1 の結果の整理・機密情報の確認・T2 の準備状況（GPU・生成なし）【実測】
> 方針（ユーザー）：週の利用制限までは Claude Code で続ける。`CODEX_HANDOFF.md` は制限のあとの引き継ぎ資料として残す
### 古い GUI の停止（ユーザーが承認）
- 停止前の確認：7861 の持ち主は PID 24008（`D:\Avatar-Pipeline\python\...\python.exe app.py --gpu-enabled`・14:37:33 起動）← 19832（`gui\.venv\Scripts\python.exe app.py --gpu-enabled`）← 21744（`cmd /c ...\start_gui_gpu.cmd`）← Claude のシェル。子プロセスなし。`.gpu.lock` なし・fh_worker・run_stage・ffmpeg なし・GPU の計算プロセスに Avatar-Pipeline のものなし
- PID 24008 だけを停止 → 19832・21744 は自動で終了。**7861 は解放された**（待ち受けなし・app.py の python なし）。新しい GUI は起動していない（ユーザーが `start_gui_gpu.cmd` で起動し、ミュートボタンが出るかをユーザーが確認する）
### GUI-G1 の結果の整理（記録の読み取りだけ）
| | GUI-G1（20261031-143923） | T1c（20261031-134405） |
|---|---|---|
| 全体（run_stage） | 173.1秒 | 179.0秒 |
| ワーカー全体 / 生成ループ | 171.2秒 / 130.2秒 | 177.4秒 / 123.2秒 |
| import / 読み込み / base_data / 音声 | 11.9 / 21.7 / 3.8 / 2.9 秒 | 36.4 / 10.0 / 3.5 / 3.6 秒 |
| 1チャンク（平均・範囲） | 16.16秒（15.99〜17.11） | 15.30秒（15.15〜16.19） |
| VRAM の使用の最大（nvidia-smi・ほかのアプリの分を含む） | 8,436MiB | 8,666MiB |
| VRAM allocated / reserved（torch） | 3,506 / 7,474〜7,476MiB | 3,506 / 7,474〜7,476MiB（同じ） |
| ジョブのコミットの最大 | 10.29GiB | 10.28GiB |
| システムのコミットの最大 / コミット残量の最小 | 22.70GiB / 9.18GiB | 22.88GiB / 8.99GiB |
| 空きRAM の最小（開始前） | 6.34GiB（8.19） | 6.10GiB（7.92） |
| GPU 最高温度 | 76.06℃ | 76.5℃ |
| チャンクごとの温度 | 71.2・69.5・71.3・73.3・73.0・73.8・74.3・76.3 | 71.2・70.2・71.6・72.6・73.6・74.3・74.9・75.6 |
| ワーカーの private（チャンクごと） | 10,317〜10,352MiB で一定 | 10,316〜10,350MiB で一定 |
| 出力 | T1c と SHA256 が同じ | — |
- 終了後（52節）：VRAM 817〜820MiB・空きRAM 8.15GiB・コミット残量 19.41GiB → 解放。プロセス・ロックなし
- 差の解釈【推定】：VRAM の使用の差（約 230MiB）は torch の allocated・reserved が同じなので、ほかのアプリの VRAM の差。生成ループが約 7秒長いのは1チャンクあたり約 0.9秒の差（温度・ほかのアプリの負荷の差の可能性）。全体が短いのは import が OS のキャッシュで速かったため
### 機密情報（secrets\README.txt）
- 文字列は表示・転記していない（長さ 53・"AQ" で始まる形式であることだけ確認）
- **git**：D:\Avatar-Pipeline は git の管理下になく（`.git` なし・親にもリポジトリなし）、**追跡・履歴への混入は無い**。`.gitignore` には `secrets/` がある（将来 git にしても除外される）
- ほかの場所：D:\Avatar-Pipeline（venv・cache・python・models を除く）で含むのは README.txt の1件だけ。Claude のメモリ 0件。Obsidian の vault（git・Google ドライブ同期）の作業ツリー 0件・git の履歴 0件
- **ただし、2026-10-31 のこの Claude Code のセッションで、引き継ぎ資料を作るときに README を読み、文字列がツールの出力に表示された** → このセッションのログ（`%USERPROFILE%\.claude\projects\G---------Obsidian\0604c447-….jsonl`）に2回含まれ、会話として Anthropic の API にも送られた
- ユーザーの環境変数（ユーザー・マシン）に GEMINI_API_KEY は無い
- → 実際のキーの可能性があるので、無効化・再発行を推奨（手順は報告に記載）。削除・変更・外部送信はしていない（承認なし）
### T2（60秒）の準備状況（GPU 生成はしていない）
- 計画（plan_for）：1,500 フレーム・54 チャンク・見込み 996秒（約 17分）・時間切れ 1,636秒・MP4 約 21MB・未検証（208 フレームを超える）
- 揃っているもの：GUI の GPU 生成モード・未検証の確認・開始前の検査・監視（すべて実装済み・GUI-G1 で実機確認）。GPU なしの接続テストは 60・180秒で合格済み（38節）。ディスクの空き：C 62.6GB・D 268.8GB。開始前の検査の今の値はすべて合格（コミット残量 19.96GiB・空きRAM 9.86GiB・53℃・空き VRAM 11,450MiB）
- **足りないもの：60秒の日本語の会話音声**（inputs には 8.333秒の MP3 だけ）。画像の指定。リップシンクの評価（ユーザー）。GPU 生成1回の承認と、未検証の長さの確認
- 想定するリスクと見ること
  1. **温度**：8 チャンクで 71→76℃。54 チャンクで上がり続けると 83℃・10秒の停止に当たる可能性【推定】→ 室温・ケースの風通しを確保。worker_mem.csv と GUI の温度を見る。開始時は 80℃ 未満
  2. **コミット残量**：開始時 19GiB ぎりぎりのことが多い。実行中の最小は約 9GiB（停止の基準 5GiB）。長時間のあいだに常駐アプリが増えると減る → 実行中はほかのアプリを開かない
  3. **メモリの増え方**：8 チャンクでは一定だった。54 チャンクで worker の private・VRAM・ffmpeg のメモリが増え続けないかを worker_mem.csv で確認（増え続けたら停止）
  4. **時間**：約 17分・時間切れ 1,636秒（見込みの約 1.6 倍）
  5. **品質**：チャンクの境目（28 フレームごと）・顔の一貫性・長時間での表情のずれ・音声の末尾
  6. 停止しても不完全なファイルは完成品と区別される（`.incomplete.mp4`）。自動の再実行なし
- 事前にすること：60秒の音声と画像をユーザーが用意する → 開始前の検査（GUI の確認画面）→ ユーザーが GPU 生成モードの GUI で「未検証の長さで実行する」にチェック →「この内容で1回だけ生成する」を押す

## 58. 2026-10-31 15:22：T2（60秒）の準備（読み取りだけ。GPU・生成なし）【実測】
- 時刻の訂正：56節（「15:20」）と 57節（「15:40」）の時刻は、時計を確かめずに書いた誤り。実際はどちらも 15:19 より前（このとき時計は 15:21）
- ミュート機能：ユーザーが確認して完了（ユーザーの報告）。7861 の GUI は PID 22788（親 21352・cmd 7088 = ユーザーが `start_gui_gpu.cmd` で 15:19 に起動。app.py の更新 15:04 より後 → ミュートのコードを含む）
### 環境（読み取りだけ）
- 開始前の検査はすべて合格：実行中のジョブなし・コミット残量 19.31GiB・空きRAM 9.41GiB・54℃・空き VRAM 11,394MiB・FH-01〜03・SHA256 一致。lock なし
- `fh_limits.json`：検証済み 208 フレーム（8.333061秒）・1チャンク 16.5秒・起動 105秒・時間切れの係数 1.5
- 監視（job_manager → run_stage）：ジョブのコミット 12GiB（Windows の Job Object の上限としても設定）・空きRAM 2GiB・コミット残量 5GiB・VRAM 11,800MiB・83℃ が 10秒・温度を 15秒読めない・0.5秒ごとに確認
### 素材
- `inputs\`：avatar_bob.png と 8.333秒の MP3 だけ → **60秒前後の日本語の会話音声が無い**
- 画像の候補：`inputs\avatar_bob.png`（素材台帳で ai_generated・commercial_ok=yes・allowed_use=internal_only。T1c・GUI-G1 と比べられる）。ほかの画像はデモ素材・実在の人物で使わない（43節）。キャラクターを固定しない方針なので、別の画像があればユーザーが指定する
### 計画（plan_for。ワーカーと同じ計算）
| 音声 | フレーム | チャンク | 見込み（設定値から） | 時間切れ | MP4 |
|---|---|---|---|---|---|
| 55秒 | 1,375 | 50 | 930秒 | 1,537秒 | 約 19MB |
| 60秒 | 1,500 | 54 | 996秒（約 17分） | 1,636秒 | 約 21MB |
| 60.48秒まで | 1,512 | 54 | 996秒 | 1,636秒 | — |
| 65秒 | 1,625 | 59 | 1,078秒 | 1,760秒 | 約 23MB |
- 実測からの推定【推定】：GUI-G1 の起動 約 40秒（キャッシュあり）＋ 54 × 16.2秒 ≈ 915秒（約 15分）。キャッシュが冷えていると起動に 70〜105秒 → 約 16〜17分
### 推奨する 60秒の音声
- 長さ 55〜60.4秒（60.48秒以下なら 54 チャンク）。1本の連続した日本語の会話（1人の話者）
- 形式：WAV（PCM 16bit または 24bit）を推奨。MP3・M4A・FLAC・AAC・OGG・Opus も可（job_manager が PCM の WAV に変換。長さは変換後の WAV のサンプル数で判定）
- サンプルレート 16kHz 以上（44.1kHz・48kHz でよい。モデルは中で 16kHz・モノラルにする）。モノラル推奨（ステレオも可。完成品には元の ch のまま結合）
- 内容：BGM・効果音・強い残響なし。音割れなし（最大 -1dBFS 以下）。平均の音量は T1c（-22dB）前後。発話の途中に、短い間（0.3〜0.5秒）と長めの間（1秒程度）を数か所入れると、口の閉じ方も確かめられる。最初と最後の無音は 0.5秒程度まで
- 権利：作成した外部サービスの規約で、商用利用と AI での加工が許されていること（素材台帳 `materials\MATERIALS_REGISTER.json` に記録する）
- 置き場所：`D:\Avatar-Pipeline\inputs\`（例：`t2_ja_60s.wav`）。元のファイルは変更しない
### 未検証の長さの確認（実装済み・48・51節で確認）
- GPU 生成モードの GUI で、検証済み（208 フレーム）を超える音声のときだけ「未検証の長さで実行する」が押せる。チェックしないで「動画を生成」を押すと拒否。チェックして発行された permit のジョブだけ allow_unverified_length=True。ワーカーも同じ判定（fh_length.py）で確認する。確認画面に「未検証の長さです（1,500 フレーム ＞ 208 フレーム）」と出る
### 途中で止まったとき
- 停止ボタン・監視の停止：run_stage が Job Object ごと（ワーカー・ffmpeg）を即時終了。自動の再実行なし
- 出力：ジョブのフォルダの `_video_noaudio.partial.mp4`（映像だけ・途中まで）。結合の途中なら出力先に `<job_id>.incomplete.mp4`。**完成品の名前（.mp4）にはならない**（7項目の検査に合格したときだけ改名）
- GUI：状態は「停止しました」（ユーザーの停止）または「安全停止」（監視による停止）と理由、不完全なファイルの名前を表示。status.json は running のまま残るが、job_manager は job_result.json の aborted を見て表示する（41節の注）
- 記録：job_result.json（aborted の理由・peak）・job_memory.csv（0.5秒ごと）・worker_mem.csv（チャンクごと）・job_stdout.txt
### 終了後の確認（GPU 不要）
- 技術検査：result.json の7項目（フレーム数＝round(秒×25)・512・25fps・h264・aac・差 1フレーム未満・音声の欠けなし）
- メモリの解放：nvidia-smi の VRAM が 1GB 前後に戻る・`jm.sys_mem()` のコミット残量が開始前に戻る・fh_worker・run_stage・ffmpeg のプロセスが無い・`.gpu.lock` が無い
- 長時間の傾向：worker_mem.csv で worker_private_mib・vram_used・ffmpeg のメモリ・コミット残量・温度が、チャンクごとに増え続けていないか
- 境目と一貫性：28 フレームごとの境目の前後のフレームの差（45節と同じ方法）・最初と最後の顔の比較・`flashhead\tools\lipsync_chart.py` で口の開き（目安）。最終の判断はユーザーが再生して行う
### 主なリスク（長時間）
1. GPU 温度：8 チャンクで 71→76℃（GUI-G1）。54 チャンクで上がり続けると 83℃・10秒で停止する可能性【推定】
2. コミット残量：開始時に 19GiB ぎりぎりのことが多い。実行中にほかのアプリを開くと 5GiB の停止に近づく
3. メモリの増加：8 チャンクでは一定。54 チャンクでも一定かは未確認（音声の配列は 60秒で約 4MB だけ。wav2vec2 の窓は 8秒で一定）
4. PC のスリープ・画面のロックで処理が止まる可能性（Windows の設定は変えない。ユーザーが実行中にスリープしないよう注意する）
5. 品質：チャンクの境目（53 か所）・長時間での顔のずれ・表情の単調さ・音声の末尾
- 監視は GUI-G1・T1c で正常に動いた（停止の条件には届かなかった）。停止の動作は GPU なしで確認済み（38・40節：停止・故障・温度のしきい値を下げた停止）→ 長時間でも同じく働く見込み【推定】
### T2 を始めるために足りないもの
- 60秒前後の日本語の会話音声（上の仕様）→ ユーザーが外部で作り、`inputs\` に置く
- 使う画像の決定（avatar_bob.png で比べるか、別の画像か）
- T1c（＝GUI-G1）のリップシンクの評価（任意だが、T2 の前が望ましい）
- GPU 生成1回の明示的な承認と、未検証の長さ（約 60秒・1,500 フレーム）の確認

## 59. 2026-10-31 15:35：T2 の音声（60秒）と開始前の条件の確認（GPU・生成なし）【実測】
- 入力：音声 `inputs\t2_ja_60s.wav`（ユーザーが用意・15:31）・画像 `inputs\avatar_bob.png`（ユーザーが指定。比較用で、固定のキャラクターにはしない）
### 音声
- 5,292,044 bytes・SHA256 e6248f42506b7ca35850040a320905f64a02363684235b54003bd7f0180fade5
- WAV・PCM 16bit（s16le）・44,100Hz・モノラル・**60.000000秒**（2,646,000 サンプル）。デコードエラー 0
- 音量：平均 -19.1dB・最大 -0.7dB（音割れなし。サンプルの段差 0.5 超は 0 件）
- 冒頭：0〜0.05秒は無音、0.10秒から立ち上がり、有声は 0.12秒から → 語頭は切れていない
- 末尾：59.4〜59.5秒 -26dB → 59.5〜60.0秒は無音（-57〜-83dB）→ 語尾は切れていない
- 10ms で -60dB 未満から -15dB 超へ急に上がる所（不自然なつなぎ目）は 0 件
- 無音（-50dB・0.15秒以上）：23 か所。長さは 0.16〜1.00秒（最長 35.10〜36.10秒）。0.5〜0.9秒の間が多く、0.16〜0.31秒の短い間もある → 口の開閉を確かめやすい
### 計画（job_manager の audio_seconds → plan_for。ワーカーと同じ測り方・判定）
- 判定に使う長さ 60.000秒 → **1,500 フレーム・54 チャンク**（生成 1,512 → 余り 12 は書かない）・未検証（208 フレームを超える）
- 見込み 996秒（約 17分、設定値から）・時間切れ 1,636秒・MP4 約 21MB。GUI-G1 の実測からは約 15〜17分【推定】
### 開始前の条件（15:35 の時点）
- すべて合格：実行中のジョブなし・コミット残量 19.38GiB・空きRAM 9.32GiB・54℃・空き VRAM 11,374MiB・FH-01〜03・SHA256 一致
- 監視（変更なし）：ジョブのコミット 12GiB・空きRAM 2GiB・コミット残量 5GiB・VRAM 11,800MiB・83℃ が 10秒・温度を 15秒読めない・時間切れ 1,636秒・自動の再実行なし
### 未検証の長さの確認（バックエンドで確認。ユーザーの GUI・タブには触れていない）
- 別のプロセスで job_manager の関数だけを呼んだ（permit は発行して、使わずに取り消した。ジョブ・出力・ロックは作られていない）
  - 通常モード → 拒否
  - GPU 生成モード（リハーサル）で確認なし → 拒否「GPU 生成で未検証の長さです：60.000秒（1500 フレーム）・検証済みは 208 フレーム（8.333秒）まで」
  - 確認あり → permit 発行（計画 1,500・54・unverified=True）→ 取り消し
- ユーザーの GUI（7861・PID 22788）は同じコードで動いているので、画面の「未検証の長さで実行する」は、この音声を選ぶと押せる状態になる【推定：48節で同じ動作を画面で確認済み】
### 残り
- 音声の権利（作成したサービス・規約の商用利用と AI 加工の可否）を素材台帳に記録する（未記録）
- T2 の GPU 生成1回の明示的な承認と、未検証の長さ（60.000秒・1,500 フレーム）の確認
- 出力先の案：`outputs\t2`（まだ無い）

## 60. 2026-10-31 15:36：T2 の実行直前の最終確認（GPU・生成なし。最後のボタンはユーザーが押す）【実測】
- 入力：画像 `inputs\avatar_bob.png`（SHA256 7df6b8b5…）・音声 `inputs\t2_ja_60s.wav`（SHA256 e6248f42…）とも存在
- 音声：判定に使う長さ 60.000000秒 → 1,500 フレーム・54 チャンク・未検証・見込み 996秒・時間切れ 1,636秒
- GUI：7861 は PID 22788（親 21352 → cmd 7088。ユーザーが `start_gui_gpu.cmd` で 15:19 に起動。app.py の更新 15:04 より後）。app.py の python はこの2つだけ（古い GUI は残っていない）。HTTP 200・GPU 生成モードのバッジとミュートボタンを含むページを返す（ページの取得だけで、ユーザーのタブには触れていない）
- 実行中のジョブなし（lock なし・fh_worker・run_stage なし）。`outputs\t2` はまだ無い
- 開始前の8項目：すべて合格（コミット残量 **19.18GiB**〈余裕 0.18GiB〉・空きRAM 9.20GiB・54℃・空き VRAM 11,327MiB・FH-01〜03・SHA256・実行中のジョブなし・80℃ 未満）
- 未検証の長さの確認：別のプロセスでバックエンドを確認。確認なし → 拒否、確認あり → permit 発行 → 使わずに取り消し（ジョブ・ロックは作られない）
- 生成は始めていない。ユーザーが GUI で始めたときのために、読み取りだけの見守り（`scratchpad\watch_t2.py`。job.json と job_result.json と current_job で判定）を動かしている

## 61. 2026-10-31 15:41〜15:56：**T2（日本語 60秒）成功**【実測】
> ユーザーの明示的な承認：T2 の GPU 生成を最大1回・未検証の長さ 60.000秒／1,500 フレーム・このジョブだけ allow_unverified_length=True・GUI の操作を Claude Code が行うこと。2回目・T3・T4 は行っていない
### 実行の経路
- ブラウザの拡張機能に接続できず、ブラウザでの自動操作は使えなかった → GUI の「動画を生成」「この内容で1回だけ生成する」が内部で呼ぶ**同じバックエンドの経路**を使った（`scratchpad\run_t2.py`）
  - GPU 生成モードと同じ GPU_SESSION（rehearsal なし・環境変数 AVATAR_ALLOW_GPU は使わない）→ `issue_gpu_permit`（入力・未検証の確認・開始前の8項目）→ 1回限りの permit → `start_job(permit)`（permit を使って消し、開始時に8項目を再検査）→ run_stage（独立したプロセス）の監視の下でワーカー
  - 迂回はしていない（GUI と同じ検査・同じ permit）。permit はジョブの開始で消えた（残り 0）
- 開始前の8項目（15:41）：すべて合格・コミット残量 **21.73GiB**・空きRAM 10.32GiB・54℃・空き VRAM 11,416MiB・FH-01〜03・SHA256
### 結果
- ジョブ `gui\jobs\20261031-154143_fh_gpu`（source=gui_permit・`--allow-gpu`・allow_unverified_length=True・seed 42・顔の切り出しオン）
- 出力：**`D:\Avatar-Pipeline\outputs\t2\20261031-154143_fh_gpu.mp4`**（19,756,396 bytes）
- **1,500/1,500 フレーム・54 チャンク・映像 60.000秒・音声 60.000秒（AAC・44.1kHz・モノラル）・差 0.000秒・7項目の検査すべて合格・デコードエラー 0・安全停止なし（exit 0）**
- 時間：run_stage 全体 **876.2秒（14.6分）**・ワーカー 874.3秒・生成ループ 830.3秒・最初のチャンクの書き出しは 57.9秒の時点（import 13.4・読み込み 23.1・base_data 3.5・音声 1.7秒）・1チャンク 平均 15.29秒（15.12〜16.03）。見込み 996秒より短い
- 最大・最小（run_stage）：VRAM 最大 **8,490MiB**・ジョブのコミット最大 **10.29GiB**・システムのコミット最大 20.45GiB・空きRAM 最小 **8.41GiB**・コミット残量 最小 **11.42GiB**・GPU 最高 **82.09℃**
- チャンクごと（worker_mem.csv・54行。最初の 1/6 → 最後の 1/6 の平均）
  - VRAM 使用 8,486 → 8,485MiB（8,474〜8,487）・allocated 3,506 で一定・reserved 7,474〜7,476 → **増え続けない**
  - ワーカーの private 10,338 → 10,344MiB（10,320〜10,354）・ffmpeg の private 150 → 157MiB（最大 158）・ジョブのコミット 10.11GiB で一定
  - 空きRAM 8.46 → 9.04GiB・コミット残量 11.48 → 11.50GiB
  - **温度**：72, 70, 72, 73, 73, 76, 74, 77, 76, 76, 78, 76, 76, 78, 74, 78, 78, 77, 79, 79, 79, 79, 80, 80, 80, 81, 80, 81, 80, 81, 80, 82, 81, 81, 80, 81, 80, 81, 81, 81, 82, 81, 81, 81, 81, 81, 81, 80, 81, 81, 81, 81, 82, 81 ℃
  - job_memory.csv（0.5秒ごと 1,717行）：80℃ 以上は 343.5秒の時点から約 452秒、82℃ 以上は 14 サンプル（連続は最長 2 サンプル＝約 1秒）、83℃ 以上は 0 → 約 25 チャンク目から **80〜82℃ で横ばい**（停止の基準 83℃・10秒まで約 1℃ の余裕）
- 終了後：VRAM 884MiB・GPU 56℃・空きRAM 10.63GiB・コミット残量 21.83GiB → 解放。fh_worker・run_stage・ffmpeg のプロセスなし・lock なし・current_job なし。GPU の計算プロセスに Avatar-Pipeline の python なし
- GUI：完成品を GUI のサーバーが配信できる（`/gradio_api/file=…` に HTTP 200・video/mp4・19,756,396 bytes）。ブラウザの画面での再生は、ブラウザの自動操作が使えないため未確認（ユーザーの確認待ち）
### 音声・つなぎ目・顔（CPU）
- 音声：完成品の音声と元の WAV はずれ 0 サンプル・相関 1.0000。先頭（0.10〜0.40秒）・末尾（59.0〜59.5秒・59.5〜60.0秒）の音量も同じ → 先頭・末尾の欠けなし
- つなぎ目（53 か所）：前のフレームとの差の平均 3.32（チャンクの内側 2.73）・最大 10.62（308 フレーム目。内側の 99 パーセンタイル 8.17 を超えたのはここだけ）。全体の最大 13.23 はチャンクの内側（311 フレーム目）。308 の前後はまばたき（目を閉じる瞬間）と小さな頭の動きの重なりで、画像では大きな破綻は見えない（`outputs\t2\analysis\top4_chunk_boundaries.png`）
- 顔の一貫性：0・15・30・45秒・末尾の比較画像（`outputs\t2\analysis\compare_0_15_30_45_end.png`）では同じ人物に見える。最初のフレームとのグレーの相関 0.990・0.901（30秒：正面寄りで少し大きく写る）・0.992・0.977
- 口の開き（`outputs\t2\analysis\lipsync_chart.png`・目安）：1,500 フレームすべてで顔を検出。音量との相関 0.32。長めの無音（1.0・0.92・0.74秒）で 0.027〜0.035（閉じている）、発話で 0.09〜0.16
### T1c・GUI-G1 との比較
| | T1c | GUI-G1 | T2 |
|---|---|---|---|
| 長さ・フレーム・チャンク | 8.333秒・208・8 | 同左 | **60.000秒・1,500・54** |
| 全体 | 179.0秒 | 173.1秒 | 876.2秒 |
| 1チャンク平均 | 15.30秒 | 16.16秒 | 15.29秒 |
| VRAM 最大 | 8,666MiB | 8,436MiB | 8,490MiB（allocated 3,506 は3回とも同じ） |
| ジョブのコミット最大 | 10.28GiB | 10.29GiB | 10.29GiB |
| 空きRAM 最小 | 6.10GiB | 6.34GiB | 8.41GiB |
| コミット残量 最小 | 8.99GiB | 9.18GiB | 11.42GiB |
| GPU 最高 | 76.5℃ | 76.1℃ | **82.09℃** |
- メモリは長さによらず一定（54 チャンクでも増えない）。温度は長時間で 80〜82℃ に達して横ばい
### 確認できたこと・未確認のこと
- 確認できた：技術検査7項目・フレーム数・音声の一致と欠けなし・メモリが増え続けないこと・温度は 83℃ 未満・つなぎ目に大きな破綻なし（画像と数値）・顔の一貫性（静止画）・終了後の解放・プロセスとロックの終了・GUI のサーバーからの配信
- 未確認：**リップシンク・表情・頭の動きの自然さ（ユーザーが再生して判断）**・GUI の画面での再生（ユーザー）・長時間（120・180秒）での温度
### 注意・提案
- **温度の余裕は約 1℃**。T3（120秒）・T4（180秒）は、室温やファンの状態によって 83℃・10秒の停止に当たる可能性が高い【推定】。T3 の前に、冷却（室温・ケースの風通し・ファン）を確認する必要がある。基準は変えない
- 検証済みの長さ（208 フレーム）は更新していない。T2 は技術検査に合格したので、ユーザーの承認があれば `fh_limits.json` を 1,500 フレーム（60.000秒）に更新できる（リップシンクの評価は未完了と注記する）

## 62. 2026-10-31：検証済みの長さを T2 の 1,500 フレーム（60.000秒）に更新（ユーザー承認。GPU・生成なし）【実測】
- 根拠：61節の T2（技術検査7項目すべて合格・1,500/1,500 フレーム・差 0.000秒・安全停止なし）
- `flashhead\tools\fh_limits.json`：gpu_verified_max_frames 208 → **1500**、gpu_verified_max_seconds 8.333061 → **60.0**、履歴に T2 を追加（変更前は `fh_limits.json.bak_20261031_pre_t2`）。安全基準・実測の係数（1チャンク 16.5秒・起動 105秒・時間切れ 1.5）は変えていない
- **検証されたのは「T2 の入力条件で 60秒の動画を技術的に生成できたこと」だけ。未確認**：発音単位のリップシンク精度・60秒間の表情と頭の動きの自然さ・別の画像や音声での品質・120秒以上の連続生成
- **温度の注意**：T2 の GPU 最高 82.09℃。停止の基準（83℃ が 10秒続く）との差は約 0.9℃ と小さい。約 25 チャンク目から 80〜82℃ で横ばいだった
- CPU テスト（新規 `tests_cpu\test_length_t2.py`。ワーカーは mode=gpu・--allow-gpu なしで直接起動し、torch を読む前に止まる）：8ケースすべて job_manager とワーカーの判定が一致
  | ケース | 秒（WAV） | フレーム | 判定 |
  |---|---|---|---|
  | T1c の MP3 | 8.333061 | 208 | 検証済み |
  | T2 の WAV | 60.000000 | 1,500 | 検証済み |
  | 60.019秒 | 60.019002 | 1,500 | 検証済み（1,500 フレームに収まる） |
  | 60.04秒 | 60.040000 | 1,501 | 未検証 → 確認なしは両方で拒否、確認ありは受付 |
  | 65秒 | 65.0 | 1,625 | 未検証 → 確認なしは拒否 |
  | 120秒 | 120.0 | 3,000 | 未検証 → 確認なしは拒否、確認ありは受付 |
  - permit（GPU 生成モードと同じ・リハーサル・ジョブは作らない）：T2 の 60秒は確認なしで発行（unverified=False）、65秒は確認なしで拒否・確認ありで発行。発行したものはすべて取り消し → ジョブ・ロック・permit は残っていない
  - 入力の音声（t2_ja_60s.wav・T1c の MP3）の SHA256 は変わらない
- 動いている GUI（7861・PID 22788）は、判定のたびに fh_limits.json を読むので、起動し直さなくても新しい上限（1,500 フレーム）で判定・表示する（app.py・job_manager に設定のキャッシュは無い）

## 63. 2026-10-31 16:35：120秒（T3）の実現可能性の調査（調査だけ。GPU・生成・コードや設定の変更なし）【実測＋推定】
> 区別：【確定】＝コード・実測で確認、【推定】＝実測からの見込み、【未検証】＝120秒を実際に生成しないと分からないこと
### 1. 技術的な実現可能性（コード）
- 長さの扱い【確定】：音声全体を1回読み込み（120秒・16kHz で約 192万サンプル＝float32 で約 7.3MB）、28 フレーム＝17,920 サンプルずつ処理する。wav2vec2 の窓は 8秒の固定（`fh_worker.py:291` の `deque(maxlen=8秒)`）で、長さによらない。チャンクの状態として持つのは、前の5フレームの潜在（latent_motion_frames）だけ（公式 pipeline と同じ）→ 長さに比例して増えるのは音声の配列だけ
- フレーム数・チャンク数・端数【確定】：120.000秒 → 3,000 フレーム・108 チャンク（生成 3,024 → 余り 24 は書かない）。`job_manager.py:44` と `fh_worker.py:115-118` で同じ式。CPU の接続テスト（`stream_dryrun_test.py`、tone_120.wav、ジョブ 20261031-163513_fh_dryrun）で **3,000/3,000・108 チャンク・差 0.000秒・通し番号の欠落なし**・ワーカーの private 374.8MiB で一定・ffmpeg 最大 158MiB（38節の 180秒の接続テストも合格済み）
- 時間のずれの蓄積【確定】：1チャンク＝ちょうど 28 フレーム＝17,920 サンプルで、整数で切るので蓄積しない。目標フレーム数で打ち切り、元の WAV を結合する（T2 も差 0.000秒）
- FFmpeg の逐次書き出し【確定】：stdin へ rawvideo を流すだけで、長さの制限なし。終了処理の待ちは 600秒（`frame_sink.py:66`）
- 長さの判定・実行の許可【確定】：GUI・job_manager・ワーカーが fh_length.py の同じ判定（1,500 フレームまで検証済み）。120秒は未検証 → 確認なしは拒否、確認ありで受付（62節の `tests_cpu\test_length_t2.py` で確認済み）
- Windows の制限【確定】：run_stage がジョブのコミットを 12GiB に制限（Job Object の JobMemoryLimit＋監視）。それ以外にプロセスの制限は設定していない
- 監視の安定性【確定＋推定】：run_stage は 0.5秒ごとに確認し、CSV に1行ずつ書く（T2 で 1,717行）。120秒なら約 3,500行で問題ない【推定】
- 小さなコードの注意【確定・低リスク】：チャンク数は `fh_worker.py:116-118`（round(秒×16000) から計算）と `:285-288`（librosa で読んだ配列の長さから計算）の2か所で求め、`assert` で一致を確かめている。長さが 17,920 サンプルの倍数ちょうど（±1）の音声では、librosa の長さの丸めで一致しない可能性がある【推定】。この場合はモデルの読み込みの後に安全側（失敗）で止まる。120.000秒（107.14 チャンク分）は該当しない
### 2. メモリ（T2 のチャンクごとの記録・54 チャンク）
- 観測【確定】：チャンクあたりの傾き（10 チャンク目以降）は、VRAM +0.001MiB・ワーカーの private +0.037MiB・ffmpeg の private +0.015MiB・ジョブのコミット 0（10.11GiB で一定）・コミット残量 +0.0004GiB。VRAM の allocated 3,506MiB・reserved 7,474〜7,476MiB で一定
- 108 チャンクの見込み【推定】：同じ傾きなら、追加の増加は数 MiB 程度。60秒→120秒で新しく増えるのは、音声の配列（+3.7MB）と ffmpeg の mp4 の索引（数十 KB）程度 → ジョブのコミットの上限 12GiB（T2 は最大 10.29GiB、余裕 約 1.7GiB）に届く可能性は低い
- 【未検証】108 チャンク連続での実際の推移
### 3. GPU 温度（最重要）
- 観測【確定】（T2・job_memory.csv・0.5秒ごと。生成は 42〜872秒）
  - 100秒ごとの平均：64.4（0〜100秒）→ 74.6 → 77.3 → 79.2 → 80.5 → 80.9（500〜600秒）→ 80.9 → 81.1 → 80.7℃
  - 上昇の速さ：60〜300秒 +1.97℃/分 → 300〜500秒 +0.81℃/分 → **500〜700秒 +0.02℃/分 → 700〜876秒 −0.35℃/分**
  - 500〜860秒：平均 81.0℃・標準偏差 0.62・最大 82.09℃・82℃ 以上は 706 サンプル中 14（連続は最長 約 1秒）・83℃ 以上 0
  - 9 チャンクずつの平均：73.7・76.9・79.7・80.9・80.8・80.9℃
- 判断：約 500秒（約 25 チャンク目）から約 6分間、**上昇が止まって横ばい**だった。この室温・負荷では、平衡に近い状態に達していたと判断できる【確定（ログ上）】。ただし、6分間だけの観測で、室温・時間帯が変われば平衡の温度も変わる【推定】
- 120秒でのリスク【推定】：同じ条件なら、残り約 14分も 81℃ 前後で推移する可能性が高い。停止の条件は「83℃ 以上が 10秒続く」で、T2 の揺れ（最大で平均より約 +1.1℃）だけでは届きにくい。しかし**平衡の温度が室温などで 1〜2℃ 上がると、83℃・10秒に届く可能性が高い**。余裕は約 2℃（平均）・約 0.9℃（最大）しかない
- 【未検証】GPU Boost の温度目標（RTX 3060 は 83℃ 付近でクロックを下げる仕組みがあると言われる）が実際に働くか。T2 では 1チャンクの時間（15.1〜16.0秒）は最後まで遅くならず、クロック低下の兆候は見えなかった
- 安全にできる冷却の改善（基準は変えない・分解・電圧・オーバークロックはしない）：
  1. 室温を下げる（エアコン・涼しい時間帯）。室温はそのまま平衡の温度に効く【推定】
  2. ケースの吸気・排気口をふさがない。壁から離す。側面・前面の周りに物を置かない
  3. 電源を切ってから、ケースのフィルター・ファンのほこりを掃除する（分解しない範囲）
  4. 開始前に GPU を待機の温度（50℃台）まで冷ます。直前に GPU を使う作業をしない
  5. 生成中は、GPU を使うアプリ（ゲーム・動画の再生・ブラウザの重いページ）を開かない
  6. （任意・ユーザーの判断）ファンの回転を上げる設定。新しいソフトの導入やドライバーの設定の変更になるので、Claude は行わない
### 4. タイムアウト
- 計算式【確定】：`job_manager.plan_for`（`job_manager.py:47`）＝ 300 ＋ チャンク数 × 16.5 × 1.5 → **120秒（108 チャンク）は 2,973秒**。run_stage が経過時間で止める（`run_stage.py:153`）
- GUI：待ち時間の制限なし（ジョブは独立したプロセス。1秒ごとに状態を読むだけ）。ワーカー全体の時間切れは無い（nvidia-smi の呼び出しだけ 5秒）。FFmpeg：終了処理の待ち 600秒（`frame_sink.py:66`）
- 見込み【推定】：T2 の実測（1チャンク 15.29秒・起動 42秒）から、起動 40〜105秒 ＋ 108 × 15.3秒（≈1,652秒）＋ 結合 数秒 ≈ **1,700〜1,770秒（約 28〜30分）**。設定値からの見込みは 1,887秒。時間切れ 2,973秒まで約 1,200秒の余裕 → 時間切れで止まる可能性は低い
- 意図しない停止の条件：温度（3）・コミット残量 5GiB 未満（実行中にほかのアプリを開く）・空きRAM 2GiB 未満・PC のスリープ・停止ボタン
- **コードの変更は不要**【確定】
### 5. 映像の品質
- 60秒の実績で言えること【確定】：つなぎ目 53 か所で大きな破綻なし（最大は 308 フレーム目のまばたき）。冒頭・15・30・45・60秒で同じ人物に見える。長めの無音では口が閉じる。音声と映像の差 0
- 120秒を作らないと分からないこと【未検証】：60秒以降の顔立ちのずれ（チャンクを重ねると、前の5フレームから次を作るので、少しずつのずれが積み重なる可能性）・表情や頭の動きの単調さ・長時間での大きな頭の動き・リップシンク（発音単位は 60秒でも未評価）
### 6. CPU での確認（今回）
- 120秒の接続テスト（推論なし）：3,000 フレーム・108 チャンク・端数 24・逐次書き出し・結合・7項目の検査・通し番号 → すべて合格（出力 `outputs\stream_test\20261031-163513_fh_dryrun.mp4`）
- 時間切れの計算・長さの判定・未検証の確認：plan_for（2,973秒・unverified=True）と 62節の CPU テストの結果で確認
### 7. 結論
- A：現状のコードで技術的に可能【確定（CPU の 120秒テスト）＋推定（GPU）】
- B：メモリは足りる見込み（VRAM 8.5GB で一定・ジョブのコミット 10.3GiB・上限 12GiB）【推定】
- C：**温度での途中停止は、条件しだいで起こりうる**。今の室温のままなら完走する可能性は高いが、余裕は 1〜2℃【推定】
- D：時間切れ・長さの制限で止まる心配は低い（時間切れ 2,973秒・見込み 約 1,700〜1,900秒）【確定＋推定】
- E：処理時間 約 28〜32分【推定】
- F：冷却（3 の 1〜5）・開始時のコミット残量 19GiB 以上（できれば 20GiB 以上）・スリープしない・実行中にほかのアプリを開かない・GPU 生成1回の承認と未検証（3,000 フレーム）の確認
- G：コードの変更なしで実行できる
- H：120秒×1本 と 60秒×2本の比較
  | | 120秒×1本 | 60秒×2本＋結合 |
  |---|---|---|
  | 承認 | 未検証の確認が必要 | 検証済み（1,500 フレーム）の範囲 |
  | 温度 | 約 30分連続。平衡の温度が上がると停止の可能性 | 1本 約 15分。間に冷却の時間を取れる |
  | 停止したとき | 全部やり直し（自動の再実行なし） | 失敗した半分だけやり直せる |
  | つなぎ目 | 途中に切れ目なし（チャンクの境目だけ） | 60秒地点で動きがつながらない（2本目は参照画像から始まる）。音声は無音の所で分ける必要がある。2本目の最初は 8秒の音声の窓が空から始まる |
  | 手間 | 1回 | 音声の分割・2回の生成・結合（映像はコピーで結合し、音声は元の WAV を付け直すのが安全） |
- **実行可能性：中**（コード・メモリは高いが、温度の余裕が小さい）
- **現在の安全性：高**（83℃・10秒・メモリ・時間切れの監視が独立して働き、途中で止まっても完成品と区別される）
- **現時点の推奨：条件付き推奨**（冷却の改善と、開始前に GPU が冷えていること・コミット残量に余裕があることを条件に、1回だけ。温度で止まったら自動で再実行せず、冷却を見直すか 60秒×2本に切り替える）

## 64. 2026-10-31：GitHub での共同開発の準備（ローカルだけ。リポジトリの作成・push・外部への送信はしていない）【実測】
### 分類
- A（共有する・49 ファイル・合計 564.2KB・最大でも 1MB 未満）：文書（README.md・docs/SETUP.md・docs/DEVELOPMENT.md・docs/THIRD_PARTY.md・PROJECT_STATUS.md・RESTART_PROMPT.md・CODEX_HANDOFF.md・env.sh）、GUI（gui/app.py・job_manager.py・static/app.css・start_gui*.cmd・requirements.in・requirements.lock.txt・tools/*.py）、flashhead/env_check.py・flashhead/tools/*.py・fh_limits.json、tools/run_stage.py、patches/flashhead/（FH-01〜03 のパッチ・依存関係のロック）、licenses/flashhead_weights_sha256.txt・MODEL_LICENSES.json、materials/MATERIALS_REGISTER.json、tests_cpu/*.py・README.md、.gitignore・.gitattributes
- B（大きいので除外）：flashhead/models（約 6.9GB）・motion-ltx（37GB）・musetalk（11.6GB）・liveportrait（7.6GB）・cache（14GB）・outputs・runs・tts の生成物・gui/jobs・gui/tmp・tools/uv.exe・tests_cpu/lentest*
- C（機密・権利で除外）：secrets/（README.txt を含む）・inputs/（avatar_bob.png は internal_only、T1c・T2 の音声は権利の記録なし）・生成動画
- D（各 PC で作り直す）：gui/.venv・flashhead/.venv・python/（uv の CPython）・cache/・flashhead/SoulX-FlashHead（公式リポジトリ。パッチで再現）
- 以前の実験（liveportrait・musetalk・motion-ltx・tts・runs・LivePortrait 用のパッチなど）は、現在の方式では使わないので含めない（必要なら、ライセンスを確認してから別途追加）。`*.bak_*` も含めない
### 機密情報の検査（値は表示していない）
- 共有する 49 ファイルで、secrets\README.txt の値・Google（AIza・AQ.）・OpenAI・GitHub・Hugging Face・AWS のキー・Bearer・秘密鍵・password=・キーの代入・メールアドレス：**すべて 0 件**
- 個人情報に近いもの：PROJECT_STATUS.md に Windows のユーザー名を含むパスが1か所（Claude のログのパス）。非公開リポジトリなので残している（消すかはユーザーが判断）
### Git の準備
- `.gitignore` を **allowlist 方式**に作り直した（既定ですべて除外し、共有するものだけを許可。許可した場所の中でも secrets・.venv・__pycache__・*.bak_*・ログ・重み〈safetensors・pth・bin など〉・動画・音声・画像・zip・exe は除外）。以前の内容は `.gitignore.bak_20261031_pre_github`
- `.gitattributes`：改行の扱い（.cmd は CRLF・.py/.md/.sh は LF・**.patch は変換しない**）
- `git init -b main` をローカルで実行（`D:\Avatar-Pipeline\.git`）。**まだ何も add・commit していない**。`git add --dry-run -A` で 49 ファイルだけが対象になることを確認。`git check-ignore` で secrets・重み・動画・音声・画像・バックアップ・venv・ジョブ・公式リポジトリ・uv.exe・cache・python が除外されることを確認
- git の user.name・user.email は未設定。GitHub CLI（gh 2.93.0）はある
### 第三者の利用条件
- 含めるのは独自コードと、SoulX-FlashHead（Apache-2.0）への差分のパッチだけ。Apache-2.0 は改変物の再配布を認める（docs/THIRD_PARTY.md に由来を明記）。重み（Apache-2.0）・公式リポジトリは含めず、取得方法を docs/SETUP.md に記載。Model_Lite・VAE_LTX（研究目的のみ）は取得しない
### 新しい文書
- README.md（概要・フォルダ・最初に読むもの）・docs/SETUP.md（uv・2つの venv・公式リポジトリとパッチ・重みの取得先と配置・SHA256・確認）・docs/DEVELOPMENT.md（禁止事項・安全基準・テスト・記録のルール・ブランチ運用）・docs/THIRD_PARTY.md
- 注：SETUP.md の新しい PC での一括の再現と、`huggingface-cli download --include` の組み合わせは未実行（元の PC での手順の記録から作成）
### アップロードに必要な操作（ユーザーの確認のあと）
1. git の名前とメール（GitHub の noreply を推奨）を設定 → 2. `gh auth login` → 3. `git add -A`（allowlist で 49 ファイルだけ）→ `git status` で確認 → 4. 最初のコミット → 5. `gh repo create avatar-pipeline --private --source . --remote origin --push`

## 65. 2026-10-31：GitHub の非公開リポジトリに初回登録（ユーザー承認。GPU・生成なし）【実測】
- リポジトリ：**https://github.com/kwonq10/avatar-pipeline**（**PRIVATE**・既定のブランチ main）。作成前に同名のリポジトリが無いことを確認。強制 push はしていない
- 認証：`gh auth status` で kwonq10 にログイン済み（トークンは表示・保存していない）
- git の設定（このリポジトリのローカルだけ）：user.name=kwonq10・user.email=GitHub の noreply アドレス（API で得たアカウントの ID と GitHub の決まった形式から作成）
- 登録前の変更：PROJECT_STATUS.md の Windows のユーザー名を含むパス1か所を `%USERPROFILE%` の表現に置き換えた
- 最終検査（ステージした内容そのものを検査）：49 ファイル・567.6KB。secrets・venv・重み・動画・音声・画像・キャッシュ・ログ・バックアップ・公式リポジトリは 0 件。機密情報の検査は、64節の検査項目の説明文の「password=」1件だけで、実際の値は無し。`git diff --cached --check` の指摘は Windows 用パッチの中の行末の空白だけ（元のソースの空白。消すとパッチが当たらないので変更しない）。登録したパッチは作業中のファイルとバイト単位で同じで、SoulX-FlashHead に対して適用済みと判定される
- 初回コミット：**b56a53ce75b51321b12d7e68ba9bf47b17fe835b**「Initial commit: Avatar-Pipeline Windows GUI and FlashHead Pro integration」（49 ファイル）
- push：`gh repo create avatar-pipeline --private --source=. --remote=origin --push` → main を新規作成・origin/main を追跡
- 確認：visibility=PRIVATE・ローカルの HEAD とリモートの main が同じ（b56a53c）・origin = https://github.com/kwonq10/avatar-pipeline.git・リモートのファイル 49（567.6KB）がローカルのコミットと同じ・除外すべき種類のファイルは 0・CODEX_HANDOFF.md・README.md（docs/SETUP.md へのリンクあり）・docs・gui・flashhead/tools・パッチがある
- 改行：.gitattributes により、リポジトリ内は .py・.md・.sh が LF、.cmd が CRLF。ローカルの作業ファイルは次に git が書き換えるまで元の改行のまま（動作には影響しない）
- **この節（65節）と RESTART_PROMPT・CODEX_HANDOFF への追記は、ローカルの変更として残している（まだコミット・push していない。次のコミットはユーザーの承認のあと）**
