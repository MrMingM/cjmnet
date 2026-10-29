"""Synthetic standard-library fixtures only; no real data or scientific runs."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import diagnose as d


def gt(oid,x):
    return dict(object_id=oid,corners=[[x,0,0]]*8)


def models():
    result={}
    for v in ('B0','B1'):
        frames=[];cached={};stats={};aps={}
        for w in ('clean','fog','snow'):
            correct=(v=='B0') if w=='fog' else (v=='B1') if w=='snow' else True
            pair=[dict(object_id='car',prediction_index=0,iou=.8)] if correct else []
            counts=dict(tp=int(correct),fp=int(not correct),fn=int(not correct),tp_flags=[int(correct)],fp_flags=[int(not correct)])
            frames.append(dict(condition=w,index=0,frame_key=['scene','1'],fixed_gt_count=1,
                protocols={'F':dict(skipped=False,matched_pairs={t:copy.deepcopy(pair) for t in ('0.3','0.5','0.7')},
                                   thresholds={t:copy.deepcopy(counts) for t in ('0.3','0.5','0.7')})}))
            cached[(w,0)]=dict(prediction_count=1,scores=[.8],prediction_boxes=[[[0,0,0]]*8],official_gt=[gt('car',40)])
            stats[w]=dict(gt=1,tp=[int(correct)],fp=[int(not correct)],score=[.8])
            aps[w]={'AP':{'F':dict(ap30=float(correct),ap50=float(correct),ap70=float(correct))}}
        result[v]=dict(frames=frames,cached=cached,stats=stats,summary=dict(results=aps),
            evaluation_contract=dict(frame_keys=[['scene','1']]),selected=dict(checkpoint='fake',epoch=20,checkpoint_sha256='fake'))
    return result


class Tests(unittest.TestCase):
    def test_relabel_does_not_change_object_ids(self):
        row=d.relabel(dict(object_id='B1',b1_tp=1,AP={'B1':.4,'B0':.3}))
        self.assertEqual(row,dict(object_id='B1',b2_tp=1,AP={'B2':.4,'B0':.3}))

    def test_neighborhood_proxy_excludes_self_and_missing_is_not_zero(self):
        geometry={'a':gt('a',0),'b':gt('b',3),'c':gt('c',6)}
        metadata={k:dict(points_sum_across_cavs=n) for k,n in (('a',2),('b',3),('c',0))}
        p=d.neighborhood_proxy('a',geometry,metadata)
        self.assertEqual(p['radii']['2']['known_observed_other_targets'],0)
        self.assertEqual(p['radii']['5']['known_observed_other_targets'],1)
        self.assertEqual(p['nearest_observed_other_gt']['object_id'],'b')
        metadata['b']['points_sum_across_cavs']=None
        self.assertEqual(d.neighborhood_proxy('a',geometry,metadata)['proxy_5m_bin'],'missing')

    def test_changes_counts_csv_and_report(self):
        aliases=models();actual={'B0':aliases['B0'],'B2':aliases['B1']}
        targets=d.enrich(actual,None,d.relabel(d.core.analyze_targets(aliases,None)))
        self.assertEqual(targets['results']['fog']['counts']['lost'],1)
        self.assertEqual(targets['results']['snow']['counts']['recovered'],1)
        t=targets['results']['fog']['target_changes'][0]
        self.assertIsNone(t['point_observation_per_cav'])
        self.assertEqual(t['neighborhood']['actual_bev_eight_neighbor_status'],'NOT_CACHED')
        self.assertIn('predicted_corners',t['prediction_evidence']['B0']['0.7'])
        with tempfile.TemporaryDirectory() as folder:
            d.csv_targets(Path(folder),targets)
            self.assertIn('b2_score07',(Path(folder)/'TARGET_CHANGES.csv').read_text(encoding='utf-8-sig'))
        text=d.markdown(targets,actual,[])
        self.assertIn('NOT_CACHED',text);self.assertIn('无新AP/IoU计算',text)

    def test_saved_match_missing_is_rejected(self):
        aliases=models();f=aliases['B0']['frames'][0]
        del f['protocols']['F']['matched_pairs']['0.7']
        with self.assertRaises(ValueError):d.core.analyze_targets(aliases,None)


if __name__=='__main__':unittest.main()
