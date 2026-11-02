"""CPU tests: no-argument numerical tests; optional read-only local T2 validation."""
import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'flashhead/tools'))
import t2_motion_analysis as m


class MotionTests(unittest.TestCase):
    def test_known_three_axis_pose(self):
        model = np.array([(0,0,0),(0,330,65),(-225,-170,135),(225,-170,135),(-150,150,125),(150,150,125)], float)
        camera = np.array([[512,0,256],[0,512,256],[0,0,1]], float)
        for axis in range(3):
            with self.subTest(axis=axis):
                rv = np.zeros(3); rv[axis] = np.deg2rad(10)
                points = np.zeros((468,2))
                points[[1,152,33,263,61,291]] = cv2.projectPoints(model,rv,np.array([0.,0.,1400.]),camera,np.zeros(4))[0].reshape(-1,2)
                angles = m.pose(points,512,512)
                self.assertLess(abs(angles[axis]-10), 1e-4)
                self.assertLess(angles[3], 1e-4)

    def test_t2_metadata(self):
        m.validate_t2_metadata(25.,1500)
        self.assertEqual(m.CHUNK_FRAMES,28)
        for fps, frames in [(0,1500),(float('nan'),1500),(30,1500),(25,0),(25,1499)]:
            with self.subTest(fps=fps, frames=frames), self.assertRaises(ValueError):
                m.validate_t2_metadata(fps,frames)

    def test_empty_csv_has_header_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'empty.csv'
            m.write_csv(path, [], fieldnames=m.EVENT_FIELDS)
            with path.open(encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames,m.EVENT_FIELDS)
                self.assertEqual(list(reader),[])
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                m.write_csv(path,[{'value':1}])
            self.assertEqual(path.read_bytes(),before)

    def test_empty_csv_needs_schema_without_creating_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'missing.csv'
            with self.assertRaises(ValueError): m.write_csv(path,[])
            self.assertFalse(path.exists())

    def test_output_directory_separation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'input'; source.mkdir()
            sample = source/'data.csv'; sample.write_bytes(b'original input')
            before = m.sha(sample)
            for invalid in [source,source/'nested',sample,root]:
                with self.subTest(path=invalid), self.assertRaises(ValueError):
                    m.new_output_dir(invalid,[source,sample])
            out = m.new_output_dir(root/'new-output',[source,sample])
            m.write_csv(out/'result.csv',[{'value':1}])
            with self.assertRaises(FileExistsError):
                m.new_output_dir(out,[source])
            self.assertEqual(m.sha(sample),before)


def validate_local(video, analysis, mouth_csv, output):
    inputs = [video,analysis/'summary.json',analysis/'per_frame.csv',mouth_csv]
    before = {p:m.sha(p) for p in inputs}
    summary = json.loads((analysis/'summary.json').read_text(encoding='utf-8'))
    if summary['sha256'] != before[video]: raise ValueError('Source does not match analysis hash')
    with (analysis/'per_frame.csv').open(encoding='utf-8-sig',newline='') as f:
        rows = list(csv.DictReader(f))
    with mouth_csv.open(encoding='utf-8-sig',newline='') as f:
        existing = list(csv.DictReader(f))
    m.validate_t2_metadata(summary['fps'],len(rows))
    assert len(existing)==1500
    assert all(np.isfinite(float(v)) for r in rows for v in r.values())
    assert sum(int(r['boundary']) for r in rows)==53
    assert float(rows[308]['t_s'])==12.32 and int(rows[311]['phase'])==3
    delta = max(abs(float(r['mar'])-float(e['mouth_open'])) for r,e in zip(rows,existing))
    assert delta<.000051
    cap = cv2.VideoCapture(str(video))
    try:
        m.validate_t2_metadata(cap.get(cv2.CAP_PROP_FPS),cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert (cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))==(512,512)
        output = m.new_output_dir(output,[video,analysis,mouth_csv])
        checks = []
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=False,min_detection_confidence=.3) as mesh:
            for i in list(range(300,326))+list(range(1127,1139)):
                cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,frame=cap.read(); assert ok
                result=mesh.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); assert result.multi_face_landmarks
                points=np.array([(v.x*512,v.y*512) for v in result.multi_face_landmarks[0].landmark])
                g=m.geometry(points,512,512)
                checks.append(dict(frame=i,**g,center_difference_px=float(np.hypot(g['cx']-float(rows[i]['cx']),g['cy']-float(rows[i]['cy'])))))
    finally:
        cap.release()
    assert all(m.sha(p)==digest for p,digest in before.items())
    m.write_csv(output/'static_validation.csv',checks)
    result=dict(finite_frames=1500,boundaries=53,independent_static_frames=len(checks),
                max_static_center_difference_px=max(r['center_difference_px'] for r in checks),
                existing_mar_max_difference=delta,input_sha256=before[video],inputs_unchanged=True)
    with (output/'validation.json').open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video',type=Path); ap.add_argument('--analysis',type=Path)
    ap.add_argument('--mouth-csv',type=Path); ap.add_argument('--out',type=Path)
    a=ap.parse_args()
    supplied=[a.video,a.analysis,a.mouth_csv,a.out]
    if any(v is not None for v in supplied) and not all(v is not None for v in supplied):
        ap.error('Local validation needs --video, --analysis, --mouth-csv and --out together')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MotionTests))
    if not result.wasSuccessful(): raise SystemExit(1)
    if a.video is not None: validate_local(a.video,a.analysis,a.mouth_csv,a.out)
    else: print('Local T2 validation NOT RUN (no asset arguments)')


if __name__=='__main__': main()
