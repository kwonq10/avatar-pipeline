"""CPU-only video motion measurements; no torch, engine imports or downloads.

Frames are zero-based. Differences at frame n compare n-1 -> n.
PnP angles are estimates using a generic face and uncalibrated camera.
Output directory must be new, preventing accidental overwrite of prior analysis.
T2-specific: 25 fps, 1500 frames, 28 newly written frames per chunk.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

T2_FPS = 25
T2_FRAMES = 1500
CHUNK_FRAMES = 28
EVENT_FIELDS = ['start_frame', 'end_frame', 'start_s', 'end_s', 'duration_s',
                'peak_frame', 'peak_move_px', 'peak_phase', 'max_angle_step_deg',
                'blink_overlap']


def validate_t2_metadata(fps, frames):
    if not np.isfinite(fps) or abs(fps-T2_FPS) > 1e-6 or frames != T2_FRAMES:
        raise ValueError('This tool expects T2: 25 fps / 1500 frames / 28-frame chunks')


def new_output_dir(path, inputs=()):
    """Reserve a new directory outside input directories; never reuse or delete."""
    path = Path(path).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if path == source or path in source.parents or (source.is_dir() and source in path.parents):
            raise ValueError('Output must be separate from all input files/directories')
    path.mkdir(parents=True, exist_ok=False)
    return path


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def pose(points, width, height):
    # Generic model in image-like coordinates (x right, y down, z away).
    model = np.array([(0, 0, 0), (0, 330, 65), (-225, -170, 135),
                      (225, -170, 135), (-150, 150, 125), (150, 150, 125)], float)
    image = points[[1, 152, 33, 263, 61, 291]].astype(float)
    camera = np.array([[width, 0, width/2], [0, width, height/2], [0, 0, 1]], float)
    ok, rv, tv = cv2.solvePnP(model, image, camera, np.zeros(4),
                            np.zeros((3, 1)), np.array([[0.], [0.], [1200.]]),
                            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or tv[2, 0] <= 0:
        raise RuntimeError('Invalid PnP pose: failed solve or face behind camera')
    rot = cv2.Rodrigues(rv)[0]
    angles = cv2.RQDecomp3x3(rot)[0]
    projected = cv2.projectPoints(model, rv, tv, camera, np.zeros(4))[0].reshape(-1, 2)
    values = [*angles, float(np.sqrt(np.mean(np.sum((projected-image)**2, axis=1))))]
    if not np.isfinite(values).all():
        raise RuntimeError('Non-finite pose')
    return values


def geometry(p, width, height):
    dist = lambda a, b: float(np.linalg.norm(p[a]-p[b]))
    center = p[[33, 263, 168, 234, 454]].mean(axis=0)  # excludes eyelids and lips
    ear_l = (dist(160, 144)+dist(158, 153))/(2*dist(33, 133))
    ear_r = (dist(385, 380)+dist(387, 373))/(2*dist(362, 263))
    pitch, yaw, roll, err = pose(p, width, height)
    eye_vector = p[263]-p[33]
    return dict(cx=center[0], cy=center[1], face_width=dist(234, 454),
                eye_span=dist(33, 263), eye_roll_deg=np.degrees(np.arctan2(eye_vector[1], eye_vector[0])),
                pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll, pnp_error_px=err,
                ear_left=ear_l, ear_right=ear_r, ear=(ear_l+ear_r)/2,
                mar=dist(13, 14)/dist(61, 291))


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    fields = list(fieldnames) if fieldnames is not None else (list(rows[0]) if rows else [])
    if not fields:
        raise ValueError('CSV column names are required for empty data')
    with open(path, 'x', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sheet(video, indices, rows, path, cols=5):
    cap = cv2.VideoCapture(str(video))
    canvas = Image.new('RGB', (cols*256, ((len(indices)+cols-1)//cols)*290), 'white')
    draw = ImageDraw.Draw(canvas)
    for j, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if not ok:
            raise RuntimeError(f'Cannot read frame {idx}')
        x, y = (j % cols)*256, (j//cols)*290
        canvas.paste(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((256, 256)), (x, y))
        r = rows[idx]
        draw.text((x+3, y+258), f"f{idx} {r['t_s']:.2f}s phase={idx%28}", fill='black')
        draw.text((x+3, y+273), f"move={r['move_px']:.2f}px EAR={r['ear']:.3f}", fill='black')
    cap.release()
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('video', type=Path)
    ap.add_argument('out', type=Path)
    a = ap.parse_args()
    before = sha(a.video)
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(2)
    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        validate_t2_metadata(fps, expected)
        a.out = new_output_dir(a.out, [a.video])
    except Exception:
        cap.release()
        raise
    rows, previous, prev_points = [], None, None
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
            refine_landmarks=False, min_detection_confidence=0.3) as fm:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            i = len(rows)
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            result = fm.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:
                raise RuntimeError(f'Face missing at {i}; do not interpolate silently')
            h, w = fr.shape[:2]
            p = np.array([(v.x*w, v.y*h) for v in result.multi_face_landmarks[0].landmark])
            r = dict(frame=i, t_s=i/fps, chunk=i//CHUNK_FRAMES, phase=i%CHUNK_FRAMES, boundary=int(i>0 and i%CHUNK_FRAMES==0), **geometry(p, w, h))
            r.update(move_px=0., angle_step_deg=0., scale_step_pct=0., image_mad=0., stable_mad=0., aligned_stable_mad=0.)
            if previous is not None:
                old = rows[-1]
                r['move_px'] = float(np.hypot(r['cx']-old['cx'], r['cy']-old['cy']))
                r['angle_step_deg'] = float(np.linalg.norm([(r[k]-old[k]+180)%360-180 for k in ('pitch_deg', 'yaw_deg', 'roll_deg')]))
                r['scale_step_pct'] = 100*(r['face_width']/old['face_width']-1)
                diff = cv2.absdiff(gray, previous)
                r['image_mad'] = float(diff.mean())
                mask = np.zeros_like(gray)
                cv2.fillConvexPoly(mask, cv2.convexHull(p.astype(np.int32)), 255)
                # Exclude moving eyelids, eyebrows and mouth from stable-face comparison.
                for ids in ([33, 133, 159, 145, 70, 105], [362, 263, 386, 374, 300, 334], [61, 291, 13, 14, 0, 17]):
                    q = np.vstack([p[ids], prev_points[ids]])
                    lo = np.floor(q.min(axis=0)-12).astype(int); hi = np.ceil(q.max(axis=0)+12).astype(int)
                    cv2.rectangle(mask, tuple(lo), tuple(hi), 0, -1)
                r['stable_mad'] = float(diff[mask>0].mean())
                ids = [33, 263, 168, 234, 454, 6, 197]
                mat, _ = cv2.estimateAffinePartial2D(prev_points[ids], p[ids], method=cv2.LMEDS)
                if mat is None:
                    raise RuntimeError(f'Alignment failed at {i}')
                warped = cv2.warpAffine(previous, mat, (w, h))
                r['aligned_stable_mad'] = float(cv2.absdiff(gray, warped)[mask>0].mean())
            rows.append(r)
            previous, prev_points = gray, p
            if i % 250 == 0:
                print(f'CPU frames: {i}', flush=True)
    cap.release()
    if len(rows) != expected or sha(a.video) != before:
        raise RuntimeError('Frame count or input integrity failure')
    data = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    # Candidate is a measurement threshold, not a judgement of unnaturalness.
    move_cut = float(np.percentile(data['move_px'][1:], 95))
    angle_cut = float(np.percentile(data['angle_step_deg'][1:], 95))
    blink_cut = float(np.median(data['ear'])*0.55)
    for r in rows:
        r['blink_candidate'] = int(r['ear'] < blink_cut)
        r['motion_candidate'] = int(r['move_px'] > move_cut or r['angle_step_deg'] > angle_cut)
    write_csv(a.out/'per_frame.csv', rows)
    write_csv(a.out/'candidates.csv', [r for r in rows if r['motion_candidate'] or r['blink_candidate']], fieldnames=list(rows[0]))
    # Group consecutive candidate steps, allowing at most two below-threshold frames.
    indices = [r['frame'] for r in rows if r['motion_candidate']]
    groups = []
    for idx in indices:
        if not groups or idx-groups[-1][-1] > 3:
            groups.append([])
        groups[-1].append(idx)
    events = []
    for group in groups:
        start, end = group[0], group[-1]
        peak = max(group, key=lambda i: rows[i]['move_px'])
        events.append(dict(start_frame=start, end_frame=end, start_s=(start-1)/fps,
                           end_s=end/fps, duration_s=(end-start+1)/fps, peak_frame=peak,
                           peak_move_px=rows[peak]['move_px'], peak_phase=peak%28,
                           max_angle_step_deg=max(rows[i]['angle_step_deg'] for i in group),
                           blink_overlap=int(any(rows[i]['blink_candidate'] for i in range(start, end+1)))))
    events.sort(key=lambda r:r['peak_move_px'], reverse=True)
    write_csv(a.out/'events.csv', events, fieldnames=EVENT_FIELDS)
    boundary = [r for r in rows if r['boundary']]
    write_csv(a.out/'boundaries.csv', boundary)
    summary = dict(input=str(a.video), sha256=before, fps=fps, frames=len(rows),
                   thresholds=dict(move_p95_px=move_cut, angle_p95_deg=angle_cut, blink_ear=blink_cut),
                   landmarks='MediaPipe FaceMesh CPU; uncalibrated generic-face PnP estimates',
                   indexing='zero-based, step n compares n-1 to n; phase=n%28',
                   top_events=events[:12], target=[rows[i] for i in [300, 305, 307, 308, 309, 310, 311, 312, 315, 320, 325]],
                   boundary_comparison={})
    for key in ['move_px', 'angle_step_deg', 'image_mad', 'stable_mad', 'aligned_stable_mad']:
        summary['boundary_comparison'][key] = {}
        for label, sample in [('boundary', boundary), ('interior', [r for r in rows[1:] if not r['boundary']])]:
            v = [r[key] for r in sample]
            summary['boundary_comparison'][key][label] = dict(n=len(v), mean=float(np.mean(v)), p95=float(np.percentile(v,95)), max=float(max(v)))
    (a.out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for label, limits in [('full', (0,len(rows)/fps)), ('12s', (12,13))]:
        fig, axes = plt.subplots(6, 1, figsize=(15,14), sharex=True)
        specs = [('cx','cy'), ('pitch_deg','yaw_deg','eye_roll_deg'), ('face_width','eye_span'), ('ear','mar'), ('move_px','angle_step_deg'), ('image_mad','stable_mad','aligned_stable_mad')]
        for ax, keys in zip(axes, specs):
            for key in keys:
                ax.plot(data['t_s'],data[key],label=key,linewidth=1)
            for r in boundary:
                ax.axvline(r['t_s'],color='gray',alpha=.2)
            ax.legend(loc='upper right',fontsize=8); ax.grid(alpha=.2); ax.set_xlim(*limits)
        axes[-1].set_xlabel('Seconds; gray lines = chunk boundary')
        fig.suptitle('CPU motion diagnostics: PnP angles are estimates, not calibrated head pose')
        fig.tight_layout(); fig.savefig(a.out/f'motion_{label}.png',dpi=120); plt.close(fig)
    sheet(a.video, list(range(300,326)), rows, a.out/'frames_12_13.png')
    selected = [e['peak_frame'] for e in events if not 290 <= e['peak_frame'] <= 335][:4]
    sheet(a.video, [i for peak in [311]+selected for i in range(peak-2,peak+3) if 0<=i<len(rows)], rows, a.out/'other_motion.png')
    print(json.dumps(dict(frames=len(rows), sha256=before, thresholds=summary['thresholds'], top_events=events[:8]),indent=2),flush=True)


if __name__ == '__main__':
    main()
