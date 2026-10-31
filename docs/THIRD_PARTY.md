# 第三者のコード・モデルと利用条件

2026-10-31 時点の確認です。商用で使う前に、各ライセンスの原文をあらためて確認してください（`licenses/MODEL_LICENSES.json` も参照）。

## このリポジトリに含めるもの
| 内容 | 由来 | 扱い |
|---|---|---|
| `patches/flashhead/fh_01-03_windows_lowmem.patch` | SoulX-FlashHead（Apache-2.0）への差分。元のコードの一部の行を含む | Apache-2.0 は改変物の再配布を認める。パッチは SoulX-FlashHead のコードに対する変更であることをここに明記する（元のライセンス：https://github.com/Soul-AILab/SoulX-FlashHead/blob/main/LICENSE） |
| `patches/flashhead/*.txt`・`*.in` | Python パッケージの名前と版の一覧 | パッケージ本体は含まない |
| `licenses/flashhead_weights_sha256.txt` | 重みのファイル名と SHA256 | 重み本体は含まない |
| そのほか（gui・flashhead/tools・tools/run_stage.py・tests_cpu・docs） | 独自のコード・文書 | 独自 |

## このリポジトリに含めないもの（各自で取得する）
| 内容 | 取得元 | ライセンス | 理由 |
|---|---|---|---|
| SoulX-FlashHead（コード） | https://github.com/Soul-AILab/SoulX-FlashHead @9bc03de | Apache-2.0 | 公式から取得し、パッチを当てる方が更新を追いやすい |
| SoulX-FlashHead-1_3B の Model_Pro・VAE_Wan | Hugging Face `Soul-AILab/SoulX-FlashHead-1_3B` | Apache-2.0（モデルカードの表記） | 約 6.5GB と大きい |
| wav2vec2-base-960h | Hugging Face `facebook/wav2vec2-base-960h` | Apache-2.0 | 大きい |
| **Model_Lite・VAE_LTX** | 同上 | VAE_LTX は LTX-Video 0.9 の RAIL-M（研究目的のみ） | **商用不可のため取得も使用もしない** |
| Python パッケージ | PyPI・PyTorch の索引 | 各パッケージ | 仮想環境は各 PC で作る |
| 以前の実験（LivePortrait・MuseTalk・LTX-Video の clone・重み・独自の修正） | 各公式 | 各ライセンス（InsightFace・一部の顔検出などは非商用・権利未確認があった） | 現在の方式では使わない。必要になったら、ライセンスを確認してから別途追加する |
| 入力素材（`inputs/`）・生成動画（`outputs/`） | ユーザー | 素材ごと（`materials/MATERIALS_REGISTER.json`） | 肖像・音声の権利と容量のため |
| `secrets/` | — | — | 機密情報 |
