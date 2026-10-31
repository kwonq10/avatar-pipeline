# キャラクター画像の事前チェック（FlashHead 専用 venv で動かす。torch もモデルも使わない。CPU の MediaPipe だけ）
#
# FlashHead Pro の入力の作り方を、公式コードと同じ手順で再現して確かめる：
#   - use_face_crop=False（既定）：flash_head/utils/utils.py resize_and_centercrop → 短い辺を 512 に合わせて**中央を正方形に切る**
#   - use_face_crop=True：flash_head/utils/facecrop.py process_image → MediaPipe（model_selection=1、信頼度 0）で
#       **最初に見つかった顔（boxes[0]）**を中心に、幅×2.0 の正方形（上 55%・下 45%）を切り、512×512 にする。
#       画像の端で切り取りが削られると正方形でなくなり、512×512 に引き伸ばされて**縦横比が歪む**。顔が無いと全体の中央切りに戻る
# 結果：JSON（level = ok / warn / ng、理由の一覧、推奨の use_face_crop）と、モデルに入る 512×512 のプレビュー画像 2 枚
#   python image_check.py <画像> <出力フォルダ>
import json, os, sys, math
import numpy as np
from PIL import Image

TARGET = 512

def center_crop_box(w, h):
    s = max(TARGET / w, TARGET / h)
    fw, fh = math.ceil(w * s), math.ceil(h * s)
    # 元画像の座標での切り取り範囲（中央の正方形）
    side = min(w, h)
    return ((w - side) / 2, (h - side) / 2, (w + side) / 2, (h + side) / 2), (fw, fh)

def face_crop_box(face, w, h, ratio=2.0):
    x1, y1, x2, y2 = face
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw = (x2 - x1) * ratio
    raw = (cx - nw * 0.5, cy - nw * 0.55, cx + nw * 0.5, cy + nw * 0.45)
    clipped = (int(max(0, raw[0])), int(max(0, raw[1])), int(min(w, raw[2])), int(min(h, raw[3])))
    return raw, clipped

def check(path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    reasons, level = [], "ok"
    def bump(l):
        nonlocal level
        order = {"ok": 0, "warn": 1, "ng": 2}
        if order[l] > order[level]: level = l
    try:
        im = Image.open(path); fmt = im.format; im = im.convert("RGB")
    except Exception as e:
        return {"level": "ng", "reasons": [f"画像を開けません：{e}"]}
    w, h = im.size
    info = {"format": fmt, "size": [w, h], "aspect": round(w / h, 3)}
    short = min(w, h)
    if short < 256:
        bump("ng"); reasons.append(f"画像が小さすぎます（短い辺 {short}px。256px 以上が必要、512px 以上を推奨）")
    elif short < 512:
        bump("warn"); reasons.append(f"短い辺が {short}px で、512px に拡大されます（ぼやける可能性があります）")

    import mediapipe as mp
    arr = np.array(im)
    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.0) as fd:   # 公式と同じ設定
        det = fd.process(arr).detections or []
    faces = []
    for d in det:
        b = d.location_data.relative_bounding_box
        faces.append({"box": [b.xmin * w, b.ymin * h, (b.xmin + b.width) * w, (b.ymin + b.height) * h], "score": round(float(d.score[0]), 3)})
    confident = [f for f in faces if f["score"] >= 0.5]
    info["faces_detected"] = len(faces); info["faces_confident"] = len(confident)

    # 中央切り（既定）のプレビュー
    cbox, _ = center_crop_box(w, h)
    im.crop(tuple(int(round(v)) for v in cbox)).resize((TARGET, TARGET), Image.BICUBIC).save(os.path.join(out_dir, "preview_center_crop.png"))
    rec_face_crop = False
    if not confident:
        bump("ng")
        reasons.append("人の顔を検出できません。FlashHead Pro は実写の人物の顔を中心にしたモデルです"
                       "（アニメ・イラスト・動物・横顔・顔が小さい画像は**動作未確認**です）")
    else:
        if len(confident) > 1:
            bump("warn"); reasons.append(f"顔が {len(confident)} 個見つかりました。1人だけが写った画像を推奨します（顔の切り出しでは最初に見つかった顔を使います）")
        f0 = faces[0]                                     # 公式の process_image は boxes[0] を使う
        x1, y1, x2, y2 = f0["box"]
        fw_, fh_ = x2 - x1, y2 - y1
        info["face0"] = {"box": [round(v) for v in f0["box"]], "score": f0["score"], "width_ratio": round(fw_ / w, 3)}
        # 中央切りのとき、顔が正方形の中に収まるか
        inside = x1 >= cbox[0] and x2 <= cbox[2] and y1 >= cbox[1] and y2 <= cbox[3]
        face_in_square = fw_ / (cbox[2] - cbox[0])
        info["center_crop"] = {"box": [round(v) for v in cbox], "face_inside": inside, "face_width_in_square": round(face_in_square, 3)}
        if not inside:
            bump("warn"); rec_face_crop = True
            reasons.append("既定の中央切りでは顔の一部が画面の外に出ます → 「顔を検出して切り出す」をオンにするか、顔が中央に来るように画像を切り抜いてください")
        elif (y1 - cbox[1]) < 0.35 * fh_:
            # MediaPipe の顔の枠はおでこ〜あご。上の余白が顔の高さの 35% 未満だと、頭の上（髪）が切れる
            bump("warn"); rec_face_crop = True
            reasons.append("既定の中央切りでは頭の上（髪）が切れます → 「顔を検出して切り出す」をオンにすると頭全体が入ります（プレビューで確認してください）")
        elif face_in_square < 0.15:
            bump("warn"); rec_face_crop = True
            reasons.append(f"中央切りでは顔が小さすぎます（正方形の幅の {face_in_square:.0%}）→ 「顔を検出して切り出す」を推奨します")
        # 顔の切り出しのとき、正方形が保たれるか（端で削られると歪む）
        raw, cl = face_crop_box(f0["box"], w, h)
        cw, ch = cl[2] - cl[0], cl[3] - cl[1]
        sq = min(cw, ch) / max(cw, ch) if max(cw, ch) else 0
        info["face_crop"] = {"box": list(cl), "square_ratio": round(sq, 3), "clipped": tuple(map(int, raw)) != cl}
        im.crop(cl).resize((TARGET, TARGET), Image.BICUBIC).save(os.path.join(out_dir, "preview_face_crop.png"))
        if sq < 0.9:
            reasons.append(f"顔が画像の端に近く、「顔を検出して切り出す」では切り取り範囲が正方形にならず、512×512 にすると顔が歪みます（縦横比 {sq:.2f}）"
                           "→ 顔のまわりに余白がある画像を推奨します")
            if rec_face_crop:
                # 中央切りも顔切りも問題がある → どちらも勧められない
                bump("warn"); rec_face_crop = False
                reasons.append("この画像は、どちらの切り方でもうまく入力できない可能性があります。顔を中心に余白をとって切り抜いた画像を用意してください")
        if fw_ / w < 0.08 and fw_ / h < 0.08:
            bump("warn"); reasons.append("顔が画像全体に対してとても小さく写っています")
    if level == "ok":
        reasons.append("問題は見つかりませんでした（実写の人物の顔を検出。中央切りで顔が収まります）")
    res = {"level": level, "reasons": reasons, "recommend_use_face_crop": rec_face_crop, "info": info,
           "previews": {k: os.path.join(out_dir, f) for k, f in (("center_crop", "preview_center_crop.png"), ("face_crop", "preview_face_crop.png"))
                        if os.path.exists(os.path.join(out_dir, f))},
           "note": "顔検出は MediaPipe（実写の人物向け）。アニメ・イラスト・動物は動作未確認"}
    return res

if __name__ == "__main__":
    r = check(sys.argv[1], sys.argv[2])
    json.dump(r, open(os.path.join(sys.argv[2], "image_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False))
