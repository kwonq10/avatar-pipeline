"""CPU numeric tests; optional validation of EXISTING clips (never encodes video)."""
import argparse
import csv
import json
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'flashhead/tools'))
from cpu_stabilization_trial import transforms, run, audio_pcm
from t2_motion_analysis import new_output_dir, sha, write_csv


class StabilizationTests(unittest.TestCase):
    def setUp(self):
        self.center=np.tile([250.,250.],(100,1)); self.angle=np.full(100,10.)
        self.identity=np.tile(np.array([[1.,0,0],[0,1.,0]]),(100,1,1))

    def test_stationary_and_zero_strength(self):
        for strength in [0,.35,.65,1]:
            mat,_=transforms(self.center,self.angle,strength)
            np.testing.assert_allclose(mat,self.identity,atol=1e-12)
        moving=self.center+np.arange(100)[:,None]*[3.,2.]
        mat,_=transforms(moving,self.angle,0)
        np.testing.assert_allclose(mat,self.identity)

    def test_slow_motion_is_preserved(self):
        linear=self.center+np.arange(100)[:,None]*np.array([.3,.2])
        mat,_=transforms(linear,self.angle,.65)
        np.testing.assert_allclose(mat,self.identity,atol=1e-12)

    def test_rigidity_translation_and_rotation_caps(self):
        self.center[50:]+=[20.,10.]; self.angle[50:]+=15
        original=self.center.copy(); angles=self.angle.copy()
        points=np.array([[1.,1.],[20.,40.],[30.,10.]])
        for strength in [.35,.65,1.]:
            mat,_=transforms(self.center,self.angle,strength)
            for c,m in zip(self.center,mat):
                r=m[:,:2]
                np.testing.assert_allclose(r.T@r,np.eye(2),atol=1e-12)
                self.assertLess(abs(np.linalg.det(r)-1),1e-12)
                self.assertLessEqual(np.linalg.norm(r@c+m[:,2]-c),12*strength+1e-8)
                self.assertLessEqual(abs(np.degrees(np.arctan2(r[1,0],r[0,0]))),2.5*strength+1e-8)
                q=points@r.T+m[:,2]
                np.testing.assert_allclose(np.linalg.norm(q[1:]-q[0],axis=1),np.linalg.norm(points[1:]-points[0],axis=1))
            self.assertGreater(np.max(np.abs(mat-self.identity)),0)
        np.testing.assert_array_equal(self.center,original)
        np.testing.assert_array_equal(self.angle,angles)

    def test_invalid_or_empty_motion(self):
        cases=[(np.empty((0,2)),np.empty(0),.5),
               (np.zeros((3,3)),np.zeros(3),.5),(np.zeros((3,2)),np.zeros(2),.5),
               (np.full((3,2),np.nan),np.zeros(3),.5),
               (np.zeros((3,2)),np.full(3,np.inf),.5)]
        cases += [(self.center,self.angle,s) for s in [-1,1.1,np.nan,np.inf]]
        for center,angle,strength in cases:
            with self.subTest(strength=strength),self.assertRaises(ValueError):
                transforms(center,angle,strength)

    def test_roll_wrap_and_single_frame(self):
        angle=np.array([179.,180.,-179.,-178.])
        mat,gate=transforms(np.tile([250.,250.],(4,1)),angle,.65)
        self.assertTrue(np.isfinite(mat).all())
        self.assertTrue(np.all((gate>=0)&(gate<=1)))
        mat,_=transforms(np.array([[250.,250.]]),np.array([10.]),.65)
        np.testing.assert_allclose(mat[0],self.identity[0],atol=1e-12)


def validate_existing(comparisons, output):
    names=[]
    for prefix in ['12s','45s','49s']:
        names += [comparisons/f'{prefix}_{v}.mp4' for v in ['A_original','B_weak','C_medium','ABC_comparison']]
        names += [comparisons/f'{prefix}_{v}_landmarks.csv' for v in ['A_original','B_weak','C_medium']]
    before={p:sha(p) for p in names}
    output=new_output_dir(output,[comparisons])
    quality=[]; files=[]
    for prefix,count in [('12s',50),('45s',40),('49s',40)]:
        with (comparisons/f'{prefix}_A_original_landmarks.csv').open(encoding='utf-8-sig',newline='') as f:
            baseline=list(csv.DictReader(f))
        assert len(baseline)==count
        audio=audio_pcm(comparisons/f'{prefix}_A_original.mp4')
        for variant in ['A_original','B_weak','C_medium','ABC_comparison']:
            path=comparisons/f'{prefix}_{variant}.mp4'
            probe=json.loads(run(['ffprobe','-v','error','-show_streams','-of','json',str(path)]))
            v=next(s for s in probe['streams'] if s['codec_type']=='video')
            a=next(s for s in probe['streams'] if s['codec_type']=='audio')
            assert int(v['nb_frames'])==count and v['r_frame_rate']=='25/1'
            assert abs(float(v['duration'])-count/25)<1e-6
            assert abs(float(a['duration'])-count/25)<1/44100
            assert float(v.get('start_time',0))==0 and float(a.get('start_time',0))==0
            assert np.array_equal(audio,audio_pcm(path))
            run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'])
            files.append(path.name)
            if variant=='ABC_comparison': continue
            with (comparisons/f'{prefix}_{variant}_landmarks.csv').open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            assert len(rows)==count
            q=dict(window=prefix,variant=variant)
            for key in ['ear','mar']:
                x=np.array([float(r[key]) for r in baseline]); y=np.array([float(r[key]) for r in rows])
                assert np.isfinite(x).all() and np.isfinite(y).all()
                q[key+'_corr']=float(np.corrcoef(x,y)[0,1])
                q[key+'_max_abs_difference']=float(np.max(np.abs(x-y)))
            q['min_ear_frame_shift']=int(np.argmin([float(r['ear']) for r in rows])-np.argmin([float(r['ear']) for r in baseline]))
            quality.append(q)
    assert all(sha(p)==digest for p,digest in before.items())
    write_csv(output/'expression_quality.csv',quality)
    result=dict(files=files,decode_ok=True,all_ABC_audio_identical=True,
                counts_and_fps_ok=True,inputs_unchanged=True,video_generation=False)
    with (output/'final_file_validation.json').open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comparisons',type=Path); ap.add_argument('--out',type=Path)
    a=ap.parse_args()
    if (a.comparisons is None)!=(a.out is None): ap.error('Use --comparisons and --out together')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(StabilizationTests))
    if not result.wasSuccessful(): raise SystemExit(1)
    if a.comparisons is not None: validate_existing(a.comparisons,a.out)
    else: print('Existing clip validation NOT RUN (no asset arguments)')


if __name__=='__main__': main()
