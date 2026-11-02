"""Optional CPU-only rigid stabilization experiment; never imported by GUI/engine.

Uses source-hash-matched t2_motion_analysis CSV. New output directory required.
No facial warping, time interpolation, GPU, model changes or downloads.
REFERENCE ONLY, NOT ADOPTED: background motion/edge/blur side effects outweighed
the partial improvement. Not a GUI feature or a recommended production filter.
T2 only: 25 fps, 1500 frames, 512x512, 44100 Hz audio, 28-frame output chunks.
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from t2_motion_analysis import sha, geometry, write_csv, new_output_dir, validate_t2_metadata


def run(args):
    return subprocess.run(args, check=True, capture_output=True).stdout


def smooth(x, sigma):
    radius = int(np.ceil(3*sigma))
    k = np.exp(-np.arange(-radius,radius+1,dtype=float)**2/(2*sigma*sigma)); k /= k.sum()
    return np.apply_along_axis(lambda y: np.convolve(np.pad(y,(radius,radius),mode='edge'),k,mode='valid'),0,x)


def transforms(center, angle, strength):
    center = np.asarray(center, dtype=float)
    angle = np.asarray(angle, dtype=float)
    if center.ndim != 2 or center.shape[1] != 2 or len(center) == 0 or angle.shape != (len(center),):
        raise ValueError('Expected nonempty Nx2 centers and N angles')
    if not np.isfinite(center).all() or not np.isfinite(angle).all():
        raise ValueError('Motion data must be finite')
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Strength must be finite and between 0 and 1')
    # Offline symmetric 0.12s smoothing: no persistent lock to a reference pose.
    angle = np.degrees(np.unwrap(np.radians(angle)))
    speed = np.r_[0,np.linalg.norm(np.diff(center,axis=0),axis=1)]
    turn = np.r_[0,np.abs(np.diff(angle))]
    gate = smooth(np.maximum(np.clip((speed-2)/4,0,1),np.clip((turn-1)/3,0,1)),2)
    delta = (smooth(center,3)-center)*strength*gate[:,None]
    limit = 12*strength
    if strength == 0:
        return np.tile(np.array([[1.,0,0],[0,1.,0]]),(len(center),1,1)),gate
    length = np.linalg.norm(delta,axis=1)
    delta *= (limit*np.tanh(length/limit)/np.maximum(length,1e-10))[:,None]
    rotation = 2.5*strength*np.tanh((smooth(angle,3)-angle)*gate/2.5)
    theta = np.radians(rotation)
    matrices = []
    for c,d,t in zip(center,delta,theta):
        r = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        matrices.append(np.column_stack([r,c+d-r@c]))
    return np.array(matrices),gate


def encode(frames, audio, dest, fps):
    h,w = frames[0].shape[:2]
    cmd=['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}',
         '-r',str(fps),'-i','pipe:0','-i',str(audio),'-map','0:v','-map','1:a',
         '-c:v','libx264','-threads','2','-preset','medium','-crf','16','-pix_fmt','yuv420p',
         '-c:a','copy','-movflags','+faststart',str(dest)]
    # stderr is bounded for these small clips; consume after closing stdin.
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for frame in frames:
            p.stdin.write(frame.tobytes())
        p.stdin.close(); err=p.stderr.read(); rc=p.wait()
        if rc:
            raise RuntimeError(err.decode('utf-8','replace'))
    finally:
        if p.poll() is None:
            p.kill(); p.wait()


def measure(path):
    cap=cv2.VideoCapture(str(path)); rows=[]; prev=None; bg=[]; blur=[]
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,
            refine_landmarks=False,min_detection_confidence=.3) as mesh:
        while True:
            ok,fr=cap.read()
            if not ok: break
            gray=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
            result=mesh.process(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:
                raise RuntimeError('Missing face in rendered comparison')
            p=np.array([(v.x*512,v.y*512) for v in result.multi_face_landmarks[0].landmark])
            rows.append(geometry(p,512,512))
            # Repeated fixed background boxes (plant/shelf left and curtain right).
            mask=np.zeros_like(gray); mask[40:330,10:75]=255; mask[40:330,455:502]=255
            if prev is not None:
                pts=cv2.goodFeaturesToTrack(prev,100,.01,5,mask=mask)
                if pts is None: raise RuntimeError('No background features')
                nxt,status,_=cv2.calcOpticalFlowPyrLK(prev,gray,pts,None)
                back,status2,_=cv2.calcOpticalFlowPyrLK(gray,prev,nxt,None)
                keep=(status.ravel()>0)&(status2.ravel()>0)&(np.linalg.norm(back-pts,axis=2).ravel()<1)
                if keep.sum()<5: raise RuntimeError('Insufficient background tracking')
                bg.append(float(np.median(np.linalg.norm(nxt[keep]-pts[keep],axis=2))))
            else: bg.append(0.)
            # Diagnostic only: not a perceptual sharpness score, subject pose affects it.
            blur.append(float(cv2.Laplacian(gray[120:330,160:350],cv2.CV_64F).var()))
            prev=gray
    cap.release()
    c=np.array([[r['cx'],r['cy']] for r in rows]); a=np.unwrap(np.radians([r['eye_roll_deg'] for r in rows]))*180/np.pi
    move=np.linalg.norm(np.diff(c,axis=0),axis=1); turn=np.abs(np.diff(a))
    accel=np.linalg.norm(np.diff(c,n=2,axis=0),axis=1)
    return rows, dict(move_max_px=float(move.max()),move_p95_px=float(np.percentile(move,95)),
                     roll_step_max_deg=float(turn.max()),roll_step_p95_deg=float(np.percentile(turn,95)),
                     accel_p95_px_frame2=float(np.percentile(accel,95)),
                     background_p95_px=float(np.percentile(bg[1:],95)),background_max_px=max(bg),
                     sharpness_proxy=float(np.median(blur))),bg


def audio_pcm(path):
    return np.frombuffer(run(['ffmpeg','-v','error','-i',str(path),'-vn','-f','f32le','-acodec','pcm_f32le','-']),dtype=np.float32)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('video',type=Path); ap.add_argument('analysis',type=Path); ap.add_argument('out',type=Path)
    ap.add_argument('--weak',type=float,default=.35); ap.add_argument('--medium',type=float,default=.65)
    a=ap.parse_args()
    if not 0<a.weak<a.medium<=1: ap.error('Require 0 < weak < medium <= 1')
    source_hash=sha(a.video)
    summary=json.loads((a.analysis/'summary.json').read_text(encoding='utf-8'))
    if summary['sha256']!=source_hash: raise RuntimeError('Analysis belongs to another source')
    table=list(csv.DictReader(open(a.analysis/'per_frame.csv',encoding='utf-8-sig')))
    fps=summary['fps']
    validate_t2_metadata(fps, len(table))
    cap = cv2.VideoCapture(str(a.video))
    try:
        validate_t2_metadata(cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) != (512, 512):
            raise ValueError('This reference trial expects a 512x512 source')
    finally:
        cap.release()
    probe = json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(a.video)]))
    audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
    if len(audio_streams) != 1 or int(audio_streams[0]['sample_rate']) != 44100:
        raise ValueError('This reference trial expects one 44100 Hz audio stream')
    center=np.array([[float(r['cx']),float(r['cy'])] for r in table]); angle=np.array([float(r['eye_roll_deg']) for r in table])
    assert np.isfinite(center).all() and np.isfinite(angle).all()
    a.out = new_output_dir(a.out, [a.video, a.analysis])
    cv2.ocl.setUseOpenCL(False); cv2.setNumThreads(2)
    settings=[('A_original',0.),('B_weak',a.weak),('C_medium',a.medium)]
    windows=[('12s',290,340),('45s',1115,1155),('49s',1225,1265)]
    results=[]; validations=[]; transforms_rows=[]
    for name,start,end in windows:
        n=end-start; audio=a.out/f'{name}_audio.m4a'; wav=a.out/f'{name}_source_audio.wav'
        sample_start=start*44100//25; sample_end=end*44100//25
        run(['ffmpeg','-v','error','-n','-i',str(a.video),'-vn','-af',f'atrim=start_sample={sample_start}:end_sample={sample_end},asetpts=PTS-STARTPTS','-c:a','pcm_s16le',str(wav)])
        run(['ffmpeg','-v','error','-n','-i',str(wav),'-c:a','aac','-b:a','192k',str(audio)])
        cap=cv2.VideoCapture(str(a.video)); cap.set(cv2.CAP_PROP_POS_FRAMES,start); frames=[]
        for _ in range(n):
            ok,fr=cap.read()
            if not ok: raise RuntimeError('Source decode failed')
            frames.append(fr)
        cap.release(); original_measure=None; source_pcm=audio_pcm(wav); baseline_audio=None; measured={}; rendered={}
        for variant,strength in settings:
            matrices,gate=transforms(center,angle,strength)
            output=[]; exposed=[]; corner_speed=[]; old_corners=None
            corners=np.array([[0,0,1],[511,0,1],[0,511,1],[511,511,1]],float)
            for i,fr in enumerate(frames,start):
                mat=matrices[i]
                transformed=fr if strength==0 else cv2.warpAffine(fr,mat,(512,512),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                output.append(transformed)
                valid=cv2.warpAffine(np.full((512,512),255,np.uint8),mat,(512,512),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
                exposed.append(float(np.mean(valid==0)*100))
                cc=corners@mat.T
                corner_speed.append(0. if old_corners is None else float(np.linalg.norm(cc-old_corners,axis=1).max()))
                old_corners=cc
                transforms_rows.append(dict(window=name,variant=variant,source_frame=i,gate=float(gate[i]),
                    m00=mat[0,0],m01=mat[0,1],m02=mat[0,2],m10=mat[1,0],m11=mat[1,1],m12=mat[1,2],edge_filled_pct=exposed[-1]))
            dest=a.out/f'{name}_{variant}.mp4'; encode(output,audio,dest,fps)
            rows,metrics,bg=measure(dest); measured[variant]=rows; rendered[variant]=output
            write_csv(a.out/f'{name}_{variant}_landmarks.csv',[dict(source_frame=start+i,**r) for i,r in enumerate(rows)])
            if original_measure is None: original_measure=rows
            metrics.update(window=name,variant=variant,strength=strength,frames=len(rows),
                edge_filled_max_pct=max(exposed),induced_corner_step_max_px=max(corner_speed),
                ear_mean_abs_difference=float(np.mean([abs(r['ear']-o['ear']) for r,o in zip(rows,original_measure)])),
                mar_mean_abs_difference=float(np.mean([abs(r['mar']-o['mar']) for r,o in zip(rows,original_measure)])))
            results.append(metrics)
            probe=json.loads(run(['ffprobe','-v','error','-show_streams','-of','json',str(dest)]))
            v=next(s for s in probe['streams'] if s['codec_type']=='video'); au=next(s for s in probe['streams'] if s['codec_type']=='audio')
            assert int(v['nb_frames'])==n and v['r_frame_rate']=='25/1'
            assert abs(float(v['duration'])-n/fps)<1e-6 and abs(float(au['duration'])-n/fps)<1/44100
            assert float(v.get('start_time',0))==0 and float(au.get('start_time',0))==0
            run(['ffmpeg','-v','error','-i',str(dest),'-f','null','-'])
            pcm=audio_pcm(dest)
            if baseline_audio is None: baseline_audio=pcm
            assert np.array_equal(pcm,baseline_audio)
            nn=min(len(source_pcm),len(pcm)); corr=float(np.corrcoef(source_pcm[:nn],pcm[:nn])[0,1])
            # AAC may add codec padding after the declared duration; compare valid source samples.
            correlations=[float(np.corrcoef(source_pcm[80:nn-80],pcm[80+lag:nn-80+lag])[0,1]) for lag in range(-40,41)]
            lag=int(np.argmax(correlations)-40)
            assert corr>.99 and abs(lag)<=1
            validations.append(dict(file=dest.name,frames=n,duration_s=n/fps,source_start_frame=start,
                                    audio_zero_lag_corr=corr,audio_best_lag_samples=lag,audio_identical_ABC=True,decode_ok=True))
            print(name,variant,metrics,flush=True)
        # Simultaneous comparison, one original audio track. Labels in top margin, no face overlay.
        combined=[]
        for k in range(n):
            canvas=np.full((548,1536,3),255,np.uint8)
            for col,(variant,_) in enumerate(settings):
                canvas[36:,col*512:(col+1)*512]=rendered[variant][k]
                cv2.putText(canvas,variant,(col*512+16,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),1,cv2.LINE_AA)
            combined.append(canvas)
        encode(combined,audio,a.out/f'{name}_ABC_comparison.mp4',fps)
        for offset in ([18,21,25,28] if name=='12s' else [14,16,19,24]):
            Image.fromarray(cv2.cvtColor(combined[offset],cv2.COLOR_BGR2RGB)).save(a.out/f'{name}_frame_{start+offset}.png')
        fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
        times=np.arange(start,end)/fps
        for variant,rows in measured.items():
            c=np.array([[r['cx'],r['cy']] for r in rows]); roll=np.array([r['eye_roll_deg'] for r in rows])
            axes[0].plot(times[1:],np.linalg.norm(np.diff(c,axis=0),axis=1),label=variant)
            axes[1].plot(times[1:],np.abs(np.diff(roll)),label=variant)
            axes[2].plot(times[2:],np.linalg.norm(np.diff(c,n=2,axis=0),axis=1),label=variant)
        for ax,label in zip(axes,['Center step px','In-plane roll step deg','Center acceleration px/frame^2']):
            ax.set_ylabel(label); ax.legend(); ax.grid(alpha=.2)
        axes[-1].set_xlabel('Source seconds'); fig.tight_layout(); fig.savefig(a.out/f'{name}_metrics.png'); plt.close(fig)
    assert sha(a.video)==source_hash
    write_csv(a.out/'results.csv',results); write_csv(a.out/'transforms.csv',transforms_rows)
    (a.out/'validation.json').write_text(json.dumps(dict(source_sha256=source_hash,source_unchanged=True,clips=validations),indent=2),encoding='utf-8')
    (a.out/'settings.json').write_text(json.dumps(dict(weak=a.weak,medium=a.medium,fps=fps,windows=windows,
        method='local zero-phase Gaussian high-pass correction gated by speed/roll, bounded rigid transform; no scaling',
        border='replicate (artificial edge pixels quantified)',engine_integration=False),indent=2),encoding='utf-8')


if __name__=='__main__': main()
