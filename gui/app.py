# キャラクター音声同期動画 GUI（Gradio 5.50、GUI 専用 venv）
# 任意のキャラクター画像（性別などは問わない）を、外部で作った音声に合わせて FlashHead Pro でしゃべらせる汎用ツール。
# - GUI のプロセスは torch もモデルも読み込まない。画像のチェックと生成は FlashHead 専用 venv の別プロセスで行う
#   （生成は run_stage.py の監視の下：ジョブの上限・空きRAM・コミット残量・VRAM・GPU 温度・停止ファイル）
# - 音声の作成元（外部サービス・録音など）には依存しない。API キーは不要
# - 公開しない：127.0.0.1 のみ、share なし
# - 今の承認範囲：接続テスト（推論なし）まで。「動画を生成」ボタンと「未検証の長さで実行する」は無効
# - 画面：1920×1080 で主要な操作がスクロールなしで収まる1画面構成（左＝入力、右＝出力・操作・進捗）。詳細は折りたたみ
# 起動：gui\start_gui.cmd
import os, sys, json, time, html, datetime, ctypes, subprocess
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import argparse
import gradio as gr
import job_manager as jm

# GPU 生成モード：起動引数でだけ有効にする（環境変数は使わない・恒久的な設定にしない）
#   --gpu-enabled   … 「動画を生成」を使える。押すたびにバックエンドが検査して、そのジョブ1回だけの許可（permit）を発行する
#   --gpu-rehearsal … 上と同じ許可・検査を通すが、ワーカーは推論なし（GPU 未使用）で動く。GUI のダミーテスト用
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--gpu-enabled", action="store_true")
_ap.add_argument("--gpu-rehearsal", action="store_true")
_args, _ = _ap.parse_known_args()
jm.GPU_SESSION.update(enabled=bool(_args.gpu_enabled or _args.gpu_rehearsal), rehearsal=bool(_args.gpu_rehearsal))
GPU_MODE = jm.GPU_SESSION["enabled"]; REHEARSAL = jm.GPU_SESSION["rehearsal"]

PREVIEW_ROOTS = [os.path.abspath(jm.JOBS), os.path.abspath(jm.DEFAULT_OUT), os.path.abspath(os.path.join(jm.GUI, "tmp"))]
LEVEL_LABEL = {"ok": "✅ 使えます", "warn": "⚠️ 使えますが注意があります", "ng": "⛔ この画像は使えません"}
STATE_LABEL = {"queued": "起動中", "running": "実行中", "done": "完了", "failed": "失敗", "stopped": "停止しました",
               "aborted": "安全停止", "（ジョブなし）": "待機中"}
# 開始前の確認の目安（RESTART_PROMPT の事前確認と同じ値。表示だけで、判定は job_manager・run_stage が行う）
PRE_MIN_HEADROOM_GIB, PRE_MIN_AVAIL_GIB = 19.0, 3.0

def previewable(path):
    if not path or not os.path.isfile(path):
        return False
    ap = os.path.abspath(path)
    for r in PREVIEW_ROOTS:
        try:
            if os.path.commonpath([ap, r]) == r:
                return True
        except ValueError:
            pass
    return False

def on_image(path):
    """画像を選んだら、FlashHead の入力の作り方どおりにチェックし、モデルに入る 512×512 のプレビューを出す"""
    if not path:
        return "キャラクター画像が選ばれていません（必須）", [], gr.update()
    r = jm.check_image(path)
    i = r.get("info") or {}
    lines = [f"**{LEVEL_LABEL.get(r.get('level'), r.get('level'))}**　大きさ {i.get('size')}・縦横比 {i.get('aspect')}・検出した顔 {i.get('faces_confident', 0)} 個"]
    lines += [f"- {x}" for x in r.get("reasons", [])]
    gal = []
    pv = r.get("previews") or {}
    if pv.get("center_crop"): gal.append((pv["center_crop"], "顔の切り出しオフ（中央を切る）"))
    if pv.get("face_crop"): gal.append((pv["face_crop"], "顔の切り出しオン"))
    rec = r.get("recommend_use_face_crop")
    return "\n".join(lines), gal, (gr.update(value=True) if rec else gr.update())

def on_audio(path):
    """音声の情報と、「未検証の長さで実行する」の有効・無効（GPU 生成モードで、検証済みを超える長さのときだけ押せる。毎回オフに戻す）"""
    if not path:
        return ("音声ファイルが選ばれていません（必須）。作成元は問いません（外部サービス・録音など）。API キーは不要です。",
                gr.update(value=False, interactive=False))
    unv = jm.plan_for(jm.audio_seconds(path), "fh_gpu")["unverified"]      # ワーカーと同じ測り方・同じ判定
    return _audio_text(path), gr.update(value=False, interactive=bool(GPU_MODE and unv))

def _audio_text(path):
    a = jm.audio_info(path)
    sec = jm.audio_seconds(path)                                         # 判定と同じ長さ（変換後の WAV）。表示は 3桁に丸めるだけ
    note = ""
    if sec < 0.5: note = " ⛔ 短すぎます（0.5秒以上）"
    pl = jm.plan_for(sec, "fh_gpu")
    m, s_ = divmod(pl["est_sec"], 60)
    warn = (f"\n⚠️ GPU 生成では**未検証の長さ**です（検証済みは {pl['verified_max_frames']} フレーム＝{pl['verified_max_seconds']:.3f}秒まで。「未検証の長さで実行する」の確認が必要）"
            if pl["unverified"] else "")
    return (f"**{html.escape(os.path.basename(path))}**：{sec:.3f}秒・{a['sample_rate']}Hz・{a['channels']}ch・{a['codec']}{note}\n"
            f"{pl['target_frames']} フレーム・{pl['chunks']} チャンク・GPU 生成の見込み 約 {m}分{s_:02d}秒・MP4 約 {pl['est_mp4_mib']}MB{warn}")

# ---------- 監視の値（表示用） ----------
class _MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

def sys_mem():
    """空きRAM・コミット残量（GiB）。run_stage.py の mem() と同じ定義"""
    m = _MEMSTAT(); m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30, m.ullAvailPageFile / 2**30

_gpu_cache = {"t": 0.0, "v": (None, None)}
def gpu_now(max_age=5.0):
    """待機中だけ使う：nvidia-smi で VRAM 使用量（MiB）と温度（℃）。5秒ごとに1回だけ読む（実行中は run_stage の記録を使う）"""
    if time.time() - _gpu_cache["t"] < max_age:
        return _gpu_cache["v"]
    v = (None, None)
    try:
        out = subprocess.run([os.environ.get("AVATAR_NVIDIA_SMI", "nvidia-smi"), "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip().splitlines()
        f = [x.strip() for x in out[0].split(",")]
        v = (float(f[0]), float(f[1]))
    except Exception:
        pass
    _gpu_cache.update(t=time.time(), v=v)
    return v

def live_monitor(job_id):
    """実行中のジョブ：run_stage.py が1秒ごとに書く job_memory.csv の最後の行（空きRAM・コミット・VRAM・温度）"""
    fp = os.path.join(jm.JOBS, job_id, "job_memory.csv")
    try:
        with open(fp, "rb") as f:
            f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 2048))
            lines = f.read().decode("utf-8", "replace").strip().splitlines()
        head = open(fp, encoding="utf-8").readline().strip().split(",")
        row = dict(zip(head, lines[-1].split(","))) if len(lines) >= 2 else {}
        if "t_s" in row and row.get("t_s") != "t_s":
            return row
    except Exception:
        pass
    return {}

def _f(x, fmt="{:.1f}"):
    try:
        return fmt.format(float(x))
    except Exception:
        return "—"

def _hms(sec):
    if sec is None:
        return "—"
    sec = int(max(0, sec)); m, s = divmod(sec, 60); h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def progress_bar(p):
    """進捗バー：長さ（width）は実際のジョブの進捗率。虹色の流れは CSS のアニメーションで、進捗とは別に動く"""
    p = max(0.0, min(1.0, float(p or 0)))
    return (f'<div class="rb-progress" role="progressbar" aria-label="ジョブの進捗" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{p * 100:.0f}">'
            f'<div class="track"><div class="fill" style="width:{p * 100:.1f}%"></div></div></div>')

def status_panel(target, st, running):
    state = st.get("state", "（ジョブなし）") if target else "（ジョブなし）"
    label = STATE_LABEL.get(state, state)
    prog = float(st.get("progress", 0) or 0) if target else 0.0
    msg = st.get("message", "") if target else IDLE_MSG
    job = {}
    if target:
        try: job = json.load(open(os.path.join(jm.JOBS, target, "job.json"), encoding="utf-8"))
        except Exception: job = {}
    mon = st.get("monitor") or {}
    # 経過・残り
    elapsed = None
    if running and job.get("created"):
        try: elapsed = (datetime.datetime.now() - datetime.datetime.fromisoformat(job["created"])).total_seconds()
        except Exception: pass
    elif mon.get("elapsed_s") is not None:
        elapsed = float(mon["elapsed_s"])
    remain, remain_note = None, ""
    if running:
        if st.get("eta_s"):
            remain = float(st["eta_s"])
        elif (job.get("plan_estimate") or {}).get("est_sec") and elapsed is not None:
            remain, remain_note = max(0.0, job["plan_estimate"]["est_sec"] - elapsed), "（見込み）"
    elif target and state == "done":
        remain = 0
    # 監視の値：実行中は run_stage の記録、待機中はこの PC の今の値
    if running:
        lm = live_monitor(target)
        avail, headroom = lm.get("avail_gib"), None
        vram, temp = lm.get("vram_used_mib"), lm.get("gpu_temp_c")
        jcommit = lm.get("job_commit_peak_gib")
        try: headroom = jm.sys_mem()["commit_limit_gib"] - float(lm.get("sys_commit_gib"))   # run_stage の mem() と同じ定義
        except Exception: headroom = None
        src = "監視（run_stage・1秒ごと）"
    else:
        a, h = sys_mem(); v, t = gpu_now()
        avail, headroom, vram, temp, jcommit = a, h, v, t, None
        src = "この PC の今の値（待機中・5秒ごと）"
    chunks = st.get("chunks") or ((job.get("plan_estimate") or {}).get("chunks") if target else None)
    tf = st.get("target_frames") or ((job.get("plan_estimate") or {}).get("target_frames") if target else None)
    cells = [
        ("チャンク", f"{st.get('chunk', 0) if target else 0} / {chunks or '—'}"),
        ("書き出し済みフレーム", f"{st.get('frames_written', 0) if target else 0} / {tf or '—'}"),
        ("経過", _hms(elapsed)),
        ("残り" + remain_note, _hms(remain)),
        ("GPU 温度", (_f(temp, "{:.0f}") + "℃") if temp not in (None, "") else "読めません"),
        ("VRAM", (_f(vram, "{:,.0f}") + " MiB") if vram not in (None, "") else "—"),
        ("空きRAM", _f(avail, "{:.1f}") + " GiB"),
        ("コミット残量", _f(headroom, "{:.1f}") + " GiB"),
    ]
    warn = []
    if not running:
        if headroom is not None and headroom < PRE_MIN_HEADROOM_GIB: warn.append(f"コミット残量が {PRE_MIN_HEADROOM_GIB:.0f}GiB 未満（GPU 生成の前に常駐アプリを閉じる）")
        if avail is not None and avail < PRE_MIN_AVAIL_GIB: warn.append(f"空きRAM が {PRE_MIN_AVAIL_GIB:.0f}GiB 未満")
        if temp is None: warn.append("GPU 温度を読めません（GPU 生成は開始されません）")
    kind = {"fh_dryrun": "接続テスト（推論なし）", "fh_gpu": "GPU 生成", "dummy": "ダミー"}.get(job.get("kind"), job.get("kind") or "")
    if job.get("kind") == "fh_gpu" and job.get("mode") == "dry_run":
        kind = "GPU 生成のリハーサル（推論なし・GPU 未使用）"
    if running and jcommit not in (None, ""):
        kind += f"・ジョブのコミット {_f(jcommit, '{:.2f}')} / 12 GiB"
    grid = "".join(f'<div class="cell"><span>{html.escape(k)}</span><b>{html.escape(str(v))}</b></div>' for k, v in cells)
    w = ("".join(f'<li>{html.escape(x)}</li>' for x in warn))
    return (f'<div class="st-panel st-{html.escape(state if target else "idle")}">'
            f'<div class="st-head"><span class="st-title">⑤ 処理状況</span><span class="st-badge">{html.escape(label)}</span>'
            f'<span class="st-msg">{html.escape(str(msg))}</span><span class="st-pct">{prog * 100:.0f}<small>%</small></span></div>'
            f'{progress_bar(prog)}'
            f'<div class="st-sub">{html.escape(target or "ジョブなし")}{("・" + html.escape(kind)) if kind else ""}</div>'
            f'<div class="st-grid">{grid}</div>'
            f'<div class="st-src">{html.escape(src)}</div>'
            + (f'<ul class="st-warn">{w}</ul>' if w else "") + '</div>')

def detail_log(target, st):
    """詳細ログ（折りたたみの中）：これまでの状況の文字表示と、ワーカーの出力の末尾"""
    state = st.get("state", "（ジョブなし）")
    mon = st.get("monitor") or {}; peak = mon.get("peak") or {}
    ch = (st.get("settings") or {}).get("character") or {}
    lines = [f"ジョブ：{target or '—'}", f"状態：{state}", f"内容：{st.get('message', '')}"]
    if st.get("frames_written") is not None:
        eta = f"・残り約 {st['eta_s']}秒" if st.get("eta_s") and state == "running" else ""
        avg = f"・1チャンク平均 {st['sec_per_chunk']}秒" if st.get("sec_per_chunk") else ""
        lines.append(f"書き出し済み：{st['frames_written']}/{st.get('target_frames', '?')} フレーム{avg}{eta}")
    if st.get("incomplete_files"):
        lines.append("⚠️ 完成していません（不完全なファイル：" + "、".join(os.path.basename(x) for x in st["incomplete_files"]) + "）")
    if ch:
        lines.append(f"キャラクター：{ch.get('name') or '（名前なし）'}（{ch.get('image_original_name')}、画像チェック {ch.get('image_check', {}).get('level')}）")
    if peak:
        lines.append(f"監視（run_stage）：空きRAM最小 {peak.get('avail_min_gib')}GiB・ジョブのコミット最大 {peak.get('job_commit_peak_gib')}GiB・"
                     f"コミット残量最小 {peak.get('commit_headroom_min_gib')}GiB・VRAM最大 {peak.get('vram_max_mib')}MiB・"
                     f"GPU 最高 {peak.get('gpu_temp_max_c')}℃・{mon.get('elapsed_s')}秒")
    res = st.get("result") or {}
    plan = (st.get("settings") or {}).get("plan") or {}
    if res:
        if res.get("ok") is False:
            lines.append(f"結果：失敗（{res.get('error', '')}）")
        else:
            lines.append(f"結果：映像 {res.get('video_frames')}フレーム／{res.get('video_seconds')}秒・音声 {res.get('audio_seconds_out')}秒・"
                         f"差 {res.get('av_diff_seconds')}秒・1フレーム未満={res.get('av_within_1_frame')}")
    elif plan:
        lines.append(f"計画：チャンク {plan.get('chunks')}・生成 {plan.get('generated_frames')}フレーム → 書き出すのは {plan.get('target_frames')}フレーム（余りは書かない）")
    if target:
        fp = os.path.join(jm.JOBS, target, "job_stdout.txt")
        try:
            tail = open(fp, encoding="utf-8", errors="replace").read().splitlines()[-12:]
            if tail:
                lines += ["", "--- ワーカーの出力（末尾）---", *tail]
        except Exception:
            pass
    return "\n".join(lines)

def refresh(last=None):
    """1秒ごとの更新。動画と履歴は、変わったときだけ送る（再生中の動画や履歴の表示を毎秒リセットしないため）"""
    last = dict(last or {})
    j = jm.current_job()
    hist = jm.history()
    target = j or (hist[0][0] if hist else None)
    st = jm.read_status(target) if target else {}
    state = st.get("state")
    out = st.get("output") if (state == "done" and (st.get("result") or {}).get("ok")) else None
    shown = out if previewable(out) else None
    path_text = (out + ("" if shown else "（このフォルダはプレビューの対象外です。ファイルを直接開いてください）")) if out else "（まだありません）"
    v_upd = gr.update() if last.get("video", "-") == shown else shown
    h_key = json.dumps(hist, ensure_ascii=False, default=str)
    h_upd = gr.update() if last.get("hist") == h_key else hist
    last.update(video=shown, hist=h_key)
    return (status_panel(target, st, j is not None), detail_log(target, st), v_upd, path_text,
            h_upd, gr.update(interactive=j is None), gr.update(interactive=j is not None), gr.update(interactive=bool(GPU_MODE and j is None)), last)

def start_test(img, aud, name, out_dir, seed, face_crop, last):
    try:
        jid = jm.start_job("fh_dryrun", {"image": img, "audio": aud, "character_name": name, "output_dir": (out_dir or "").strip(),
                                          "seed": int(seed), "use_face_crop": bool(face_crop)})
        gr.Info(f"接続テストを開始しました：{jid}")
    except Exception as e:
        gr.Warning(str(e))
    return refresh(last)

def stop(last):
    j = jm.stop_job()
    gr.Info(f"停止を要求しました：{j}" if j else "実行中のジョブはありません")
    return refresh(last)

# ---------- 動画を生成（GPU 生成モードだけ） ----------
# 1回目のクリック：バックエンドが入力と開始前の条件を検査し、合格したら「このジョブ1回だけ」の許可（permit）を発行して確認パネルを出す
# 2回目のクリック（確認パネルの「この内容で1回だけ生成する」）：permit を使ってジョブを開始する（permit はその場で消える）
# どちらの処理も API からは呼べない（api_name=False）。permit が無い・期限切れ・入力が変わった場合はバックエンドが拒否する
def _payload(img, aud, name, out_dir, seed, face_crop, allow_unv):
    return {"image": img, "audio": aud, "character_name": name, "output_dir": (out_dir or "").strip() or jm.DEFAULT_OUT,
            "seed": int(seed if seed is not None else 42), "use_face_crop": bool(face_crop), "allow_unverified_length": bool(allow_unv)}

def prepare_gpu(img, aud, name, out_dir, seed, face_crop, allow_unv, old_permit):
    jm.revoke_gpu_permit(old_permit)
    hide = (gr.update(visible=False), "", None)
    if not GPU_MODE:
        gr.Warning("この GUI は GPU 生成モードで起動していません（start_gui_gpu.cmd で起動してください）")
        return hide
    try:
        pid, pl, pre = jm.issue_gpu_permit(_payload(img, aud, name, out_dir, seed, face_crop, allow_unv))
    except Exception as e:
        gr.Warning(str(e))
        return hide
    m, s_ = divmod(pl["est_sec"], 60)
    rows = "\n".join(f"| {n} | {'✅' if ok else '⛔'} | {v} |" for n, ok, v in pre["items"])
    unv = (f"\n\n⚠️ **未検証の長さです**（{pl['seconds']:.3f}秒＝{pl['target_frames']} フレーム ＞ 検証済み {pl['verified_max_frames']} フレーム＝{pl['verified_max_seconds']:.3f}秒）。"
           "あなたが確認したので、**このジョブに限り** allow_unverified_length=True で実行します。" if pl["unverified"] else
           f"\n\n検証済みの長さ（{pl['verified_max_frames']} フレーム＝{pl['verified_max_seconds']:.3f}秒まで）の範囲です。")
    mode = "**リハーサル**：同じ許可と検査を通しますが、推論はしません（GPU 未使用）。" if REHEARSAL else "**FlashHead Pro で GPU 生成を1回だけ行います。**"
    md = (f"#### 生成の確認\n{mode}\n\n音声 {pl['seconds']:.3f}秒・{pl['target_frames']} フレーム・{pl['chunks']} チャンク・"
          f"見込み 約 {m}分{s_:02d}秒・時間切れ {pl['timeout_s']}秒・seed {int(seed or 42)}・顔の切り出し {'オン' if face_crop else 'オフ'}{unv}\n\n"
          f"| 開始前の条件 | | 今の値 |\n|---|---|---|\n{rows}\n\n"
          f"この確認は {jm.PERMIT_TTL_S}秒で無効になります。入力を変えた場合は、もう一度「動画を生成」を押してください。")
    return gr.update(visible=True), md, pid

def confirm_gpu(permit, img, aud, name, out_dir, seed, face_crop, allow_unv, last):
    try:
        jid = jm.start_job("fh_gpu", _payload(img, aud, name, out_dir, seed, face_crop, allow_unv), permit=permit)
        gr.Info(("リハーサルを開始しました：" if REHEARSAL else "GPU 生成を開始しました：") + jid)
    except Exception as e:
        gr.Warning(str(e))
    return (gr.update(visible=False), "", None) + tuple(refresh(last))

def cancel_gpu(permit):
    jm.revoke_gpu_permit(permit)
    gr.Info("生成をやめました（許可は取り消しました）")
    return gr.update(visible=False), "", None

if not GPU_MODE:
    IDLE_MSG = "接続テスト（推論なし）を実行できます。「動画を生成」は GPU 生成モード（start_gui_gpu.cmd）で起動したときだけ使えます"
elif REHEARSAL:
    IDLE_MSG = "リハーサルモード：「動画を生成」は許可と検査だけを本番どおりに行い、推論はしません（GPU 未使用）"
else:
    IDLE_MSG = "「動画を生成」を押すと、開始前の条件を確かめて確認画面を出します（生成はそのジョブ1回だけ）"
BADGE = ('<span class="gpu-badge gpu-off">GPU 生成：無効（通常モード）</span>' if not GPU_MODE else
         '<span class="gpu-badge gpu-reh">GPU 生成：リハーサル（GPU 未使用）</span>' if REHEARSAL else
         '<span class="gpu-badge gpu-on">GPU 生成モード（このウィンドウの間だけ・1回ごとに確認）</span>')

CSS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "app.css")
# 白を基調にするため、OS がダークモードでも明るい表示に固定する
FORCE_LIGHT = "() => { document.body.classList.remove('dark'); document.documentElement.classList.remove('dark'); }"
# 枠の虹色を線に沿って回すための角度。Gradio は css_paths の @property を取り除くので、head に直接書く
HEAD = '<style>@property --rb-angle { syntax: "<angle>"; inherits: false; initial-value: 0deg; }</style>'
# 完成動画の音声のミュート・解除（Gradio 5.50 の動画プレーヤーには音量・ミュートのボタンが無いため）
# - 対象は完成動画のプレビュー（#out-video の video 要素）だけ。入力音声の試聴（#char-audio）には触れない
# - ボタンの表示は、いつも実際の video 要素の状態（muted、または volume が 0）から作る。表示だけを変えることはしない
# - video 要素が作り直されても（読み込み・生成の完了・差し替え）、MutationObserver で見つけて状態を表示し直す
# - 解除するとき volume が 0 なら 1 に戻す（video 要素の音量だけ。ブラウザ・Windows の音量には触れない）
# - サーバーとのやり取りは無い（ブラウザの中だけで動く）
HEAD += """<script>
(() => {
  const SEL = '#out-video video';
  const isMuted = (v) => v.muted || v.volume === 0;
  // 値が変わったときだけ書き換える（書き換えが MutationObserver を呼び、また書き換える無限ループを防ぐ）
  const setIf = (el, k, val) => { if (el[k] !== val) el[k] = val; };
  const setAttrIf = (el, k, val) => { if (el.getAttribute(k) !== val) el.setAttribute(k, val); };
  function render() {
    const btn = document.getElementById('av-mute-btn');
    if (!btn) return;
    const v = document.querySelector(SEL);
    // 動画があるか：src（属性・プロパティ）か source 要素があり、読み込めない状態（NETWORK_NO_SOURCE）でないこと
    // （Chrome は src を外しても currentSrc に前の URL が残るので、currentSrc は使わない）
    const has = !!(v && (v.getAttribute('src') || v.src || v.querySelector('source[src]')) && v.networkState !== 3);
    const m = has && isMuted(v);
    setIf(btn, 'disabled', !has);
    setIf(btn, 'textContent', m ? '🔇 ミュート中' : '🔊 音声オン');
    if (btn.classList.contains('muted') !== m) btn.classList.toggle('muted', m);
    setAttrIf(btn, 'aria-pressed', String(m));
    setAttrIf(btn, 'title', !has ? '完成動画がありません' : m ? 'クリックで音声を戻します' : 'クリックでミュートします');
  }
  function bind(v) {
    if (v.__avMuteBound) return;
    v.__avMuteBound = true;
    ['volumechange', 'loadedmetadata', 'loadstart', 'emptied', 'play'].forEach((e) => v.addEventListener(e, render));
  }
  document.addEventListener('click', (ev) => {
    const btn = ev.target && ev.target.closest ? ev.target.closest('#av-mute-btn') : null;
    if (!btn || btn.disabled) return;
    const v = document.querySelector(SEL);
    if (!v) return;
    if (isMuted(v)) { v.muted = false; if (v.volume === 0) v.volume = 1; } else { v.muted = true; }
    render();
  });
  const scan = () => { document.querySelectorAll(SEL).forEach(bind); render(); };
  let queued = false;
  const queue = () => { if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; scan(); }); } };   // 1フレームに1回だけ
  const start = () => {
    new MutationObserver((recs) => {
      // ボタン自身（#av-mute）の中の変化は無視する。それ以外（動画の作り直しなど）で走査する
      if (recs.some((r) => !(r.target.closest && r.target.closest('#av-mute')) && !(r.target.parentElement && r.target.parentElement.closest('#av-mute')))) queue();
    }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });
    scan(); setInterval(scan, 2000);
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
</script>"""

with gr.Blocks(title="キャラクター音声同期動画（FlashHead Pro）", css_paths=[CSS_FILE], js=FORCE_LIGHT, head=HEAD, fill_width=True,
               theme=gr.themes.Default(), elem_id="app-root") as demo:
    with gr.Row(elem_id="app-title", equal_height=True):
        gr.HTML('<div class="title-line"><h1>キャラクター音声同期動画 <span>SoulX-FlashHead Pro・ローカル</span></h1>'
                '<p>キャラクター画像と外部の音声から、音声と同じ長さの動画（512×512・25fps）を作ります。画面を開いただけではモデルを読み込みません。</p>'
                + BADGE + '</div>')
    with gr.Row(elem_id="main-row", equal_height=True):
        # ---- 左：入力（キャラクター画像・名前・音声）----
        with gr.Column(scale=4, min_width=360, elem_classes="rb-panel", elem_id="panel-input"):
            with gr.Column(elem_classes="rb-inner"):
                gr.Markdown("### ① キャラクター画像（必須）", elem_classes="sec-head")
                with gr.Row(elem_id="img-row", equal_height=False):
                    # 画像（左）と、名前・画像チェック（右）を横に並べて高さを節約する
                    img = gr.Image(type="filepath", label="画像（PNG / JPG）・プレビュー", elem_id="char-image", show_label=True, scale=1, min_width=170)
                    with gr.Column(scale=1, min_width=170, elem_id="img-side"):
                        char_name = gr.Textbox(label="キャラクター名（任意）", placeholder="例：解説役A", elem_id="char-name", max_lines=1)
                        img_check = gr.Markdown("キャラクター画像が選ばれていません（必須）", elem_id="img-check")
                with gr.Accordion("モデルに入る画像（512×512）を見る", open=False, elem_id="acc-preview"):
                    gr.Markdown("顔検出は実写の人物向け（MediaPipe）です。アニメ・イラスト・動物は**動作未確認**です。", elem_classes="hint")
                    gal = gr.Gallery(label="モデルに入る画像", columns=2, height=200, allow_preview=True, elem_id="crop-gallery")
                gr.Markdown("### ② 外部の音声（必須）", elem_classes="sec-head")
                aud = gr.Audio(type="filepath", label="音声ファイル・試聴（WAV・MP3・FLAC・M4A など）", sources=["upload"], elem_id="char-audio")
                aud_info = gr.Markdown("音声ファイルが選ばれていません（必須）。作成元は問いません（外部サービス・録音など）。API キーは不要です。", elem_id="aud-info")
        # ---- 右：出力（動画・設定・操作・進捗・保存先）----
        with gr.Column(scale=7, min_width=520, elem_classes="rb-panel", elem_id="panel-output"):
            with gr.Column(elem_classes="rb-inner"):
                with gr.Row(elem_id="out-top", equal_height=False):
                    with gr.Column(scale=5, min_width=300, elem_id="col-video"):
                        gr.Markdown("### ③ 完成動画のプレビュー", elem_classes="sec-head")
                        video = gr.Video(label="再生・ダウンロード", interactive=False, elem_id="out-video")
                        gr.HTML('<div class="av-mute-wrap"><button id="av-mute-btn" type="button" aria-pressed="false" disabled>🔊 音声オン</button>'
                                '<span class="av-mute-note">完成動画の音声だけを切り替えます（MP4 は変更しません）</span></div>', elem_id="av-mute")
                    with gr.Column(scale=4, min_width=280, elem_id="col-settings"):
                        gr.Markdown("### ④ 生成設定と操作", elem_classes="sec-head")
                        with gr.Row(elem_id="set-row"):
                            seed = gr.Number(label="seed", value=42, precision=0, elem_id="seed", min_width=90, scale=1)
                            face_crop = gr.Checkbox(label="顔を検出して切り出す", value=False, elem_id="face-crop", scale=2,
                                                    info="画像チェックで勧められたら自動でオン")
                        allow_unv = gr.Checkbox(label="未検証の長さで実行する（検証済みを超える音声のときだけ）", value=False, interactive=False, elem_id="allow-unverified")
                        with gr.Row(elem_id="btn-row", equal_height=True):
                            gen_btn = gr.Button("▶ 動画を生成", interactive=False, elem_id="btn-generate", scale=5, min_width=110)
                            test_btn = gr.Button("接続テスト", elem_id="btn-test", scale=4, min_width=90)
                            stop_btn = gr.Button("■ 停止", interactive=False, elem_id="btn-stop", scale=2, min_width=64)
                        gr.Markdown(("「動画を生成」は GPU 生成モード（start_gui_gpu.cmd）で起動したときだけ使えます。" if not GPU_MODE else
                                     "「動画を生成」→ 条件の確認 →「この内容で1回だけ生成する」で開始します。") + "「接続テスト」は推論なし・GPU 未使用です。",
                                    elem_classes="hint", elem_id="btn-hint")
                        with gr.Group(visible=False, elem_id="gpu-confirm") as confirm_box:
                            confirm_md = gr.Markdown("", elem_id="gpu-confirm-text")
                            with gr.Row(elem_id="gpu-confirm-btns"):
                                confirm_btn = gr.Button("この内容で1回だけ生成する", elem_id="btn-confirm-gpu", scale=3)
                                cancel_btn = gr.Button("やめる", elem_id="btn-cancel-gpu", scale=1)
                        with gr.Accordion("設定の詳細（出力フォルダ・モデル）", open=False, elem_id="acc-settings"):
                            out_dir = gr.Textbox(label="出力フォルダ（MP4 の保存先）", value=jm.DEFAULT_OUT, max_lines=1, elem_id="out-dir")
                            gr.Textbox(label="モデル", value="SoulX-FlashHead Pro（固定。Lite は商用不可のため選べません）", interactive=False, max_lines=1)
                            gr.Markdown("接続テストは推論せず、番号入りのダミー映像を本番と同じ逐次書き出しで作り、選んだ音声と結合します。"
                                        "GPU で検証済みの長さは、音声を選ぶと表示されます（判定は毎回 fh_limits.json を読み、フレーム数で行います）。", elem_classes="hint")
                status_html = gr.HTML(status_panel(None, {}, False), elem_id="status-panel")
                with gr.Row(elem_id="out-path-row", equal_height=True):
                    gr.HTML('<div class="out-path-label">MP4 の保存先</div>', elem_id="out-path-label")
                    out_path = gr.Textbox(show_label=False, container=False, interactive=False, max_lines=1, elem_id="out-path", scale=8)
    # ---- 下：詳細ログと生成履歴（通常は閉じる）----
    with gr.Row(elem_id="bottom-row"):
        with gr.Column(scale=1, min_width=360):
            with gr.Accordion("詳細ログ", open=False, elem_id="log-acc"):
                status = gr.Textbox(label="処理状況（詳細）", lines=10, max_lines=20, interactive=False, elem_id="status-box", show_copy_button=True)
        with gr.Column(scale=1, min_width=360):
            with gr.Accordion("生成履歴", open=False, elem_id="history-acc"):
                hist = gr.Dataframe(headers=["ジョブ", "種類", "キャラクター", "状態", "進捗", "秒", "VRAM最大MiB", "空きRAM最小GiB", "出力"],
                                    interactive=False, wrap=False, elem_id="history-table")

    img.change(on_image, img, [img_check, gal, face_crop])
    aud.change(on_audio, aud, [aud_info, allow_unv])
    last = gr.State({})
    permit = gr.State(None)                                              # 1回限りの許可の ID（このブラウザのセッションだけ）
    outs = [status_html, status, video, out_path, hist, test_btn, stop_btn, gen_btn, last]
    gen_btn.click(prepare_gpu, [img, aud, char_name, out_dir, seed, face_crop, allow_unv, permit], [confirm_box, confirm_md, permit], api_name=False)
    confirm_btn.click(confirm_gpu, [permit, img, aud, char_name, out_dir, seed, face_crop, allow_unv, last],
                      [confirm_box, confirm_md, permit] + outs, api_name=False)
    cancel_btn.click(cancel_gpu, permit, [confirm_box, confirm_md, permit], api_name=False)
    test_btn.click(start_test, [img, aud, char_name, out_dir, seed, face_crop, last], outs)
    stop_btn.click(stop, last, outs)
    gr.Timer(1.0).tick(refresh, last, outs, show_progress="hidden")
    demo.load(refresh, None, outs, show_progress="hidden")

def _port_in_use(port):
    """そのポートで、すでに別のプログラム（前に起動した GUI など）が待ち受けているか"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0

if __name__ == "__main__":
    _port = int(os.environ.get("AVATAR_GUI_PORT", "7861"))
    if _port_in_use(_port):
        # 前の GUI が動いたままだと、新しい GUI は起動できず、ブラウザには古い GUI が表示され続ける（2026-10-31 の事例）
        print(f"\n[エラー] ポート {_port} は、すでに別の GUI が使っています。新しい GUI は起動しませんでした。\n"
              f"  ブラウザの http://127.0.0.1:{_port} に表示されているのは、前に起動した GUI です（古いコードの可能性があります）。\n"
              "  前の GUI のコンソールのウィンドウを閉じてから、もう一度起動してください。\n"
              "  ウィンドウが見当たらない場合は、タスクマネージャーで、コマンドラインに app.py を含む python.exe を終了してください。\n", flush=True)
        sys.exit(3)
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=_port,
                                                   share=False, inbrowser=False, allowed_paths=PREVIEW_ROOTS, show_api=True)
