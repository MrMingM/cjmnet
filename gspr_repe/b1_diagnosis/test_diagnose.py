"""Standard-library fixtures only; no real checkpoints, Torch or datasets."""
import copy
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('b1_diagnose',Path(__file__).with_name('diagnose.py'))
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


def fixed_gt():
    return [dict(object_id='car',corners=[[40,0,0]]*8)]


def fake_frame(weather, correct):
    cache=dict(prediction_count=1,scores=[.8],official_gt=fixed_gt())
    pairs=[dict(object_id='car',prediction_index=0,iou=.8)] if correct else []
    count=dict(tp=int(correct),fp=int(not correct),fn=int(not correct),tp_flags=[int(correct)],fp_flags=[int(not correct)])
    frame=dict(condition=weather,index=0,frame_key=['scene','0001'],fixed_gt_count=1,
               protocols={'F':dict(skipped=False,matched_pairs={t:copy.deepcopy(pairs) for t in ('0.3','0.5','0.7')},
                                  thresholds={t:copy.deepcopy(count) for t in ('0.3','0.5','0.7')})})
    return frame,cache


def fake_models():
    result={}
    for v in ('B0','B1'):
        frames=[];cached={};stats={};aps={}
        for w in d.WEATHERS:
            correct=(v=='B0') if w=='fog' else (v=='B1') if w=='snow' else True
            f,c=fake_frame(w,correct);frames.append(f);cached[(w,0)]=c
            stats[w]=dict(gt=1,tp=[int(correct)],fp=[int(not correct)],score=[.8])
            aps[w]={'AP':{'F':dict(ap30=float(correct),ap50=float(correct),ap70=float(correct))}}
        result[v]=dict(evaluation_contract=dict(frame_keys=[['scene','0001']]),frames=frames,cached=cached,
                       stats=stats,summary=dict(results=aps),selected={'checkpoint':'fixture'})
    return result


def complete_fixture(root):
    """829 tiny synthetic frames, fake weight bytes; no scientific execution."""
    def put(path,row):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(row),encoding='utf-8')
        return d.digest(path)
    def seal(row):
        row['identity']=d.identity(row)
        return row
    folder=root/'B1';evaluation=folder/'final_evaluation'
    weight=folder/'net_epoch20.pth';weight.parent.mkdir(parents=True)
    weight.write_bytes(b'fake fixture bytes, not a torch checkpoint')
    sources=dict(files={},dependencies={},pointcloud_tools={})
    config=folder/'config.yaml';config.write_text('fixture')
    contract=seal(dict(variant='B1',epochs=20,smoke_only=False,sources=sources,config_sha256=d.digest(config),
                       frozen_gt='fixed-gt',input_identity='inputs'))
    contract_sha=put(folder/'contract.json',contract)
    rows=[]
    for epoch in range(1,21):
        ap=folder/'epochs'/('%02d'%epoch)/'clean.json'
        ap_sha=put(ap,dict(status='complete',full_validation=True,frame_count=829,conditions=['clean'],
                           checkpoint_sha256=d.digest(weight),results={'clean':{'AP':{'O':{'ap70':epoch/100}}}}))
        row=dict(status='complete',epoch=epoch,contract=contract['identity'],AP70=epoch/100,
                 checkpoint=str(weight),checkpoint_sha256=d.digest(weight),clean_summary=str(ap),clean_summary_sha256=ap_sha)
        put(folder/'epochs'/('%02d'%epoch)/'complete.json',row);rows.append(row)
    put(folder/'history.json',{'epochs':rows})
    selected=seal(dict(status='complete',weather_used=False,epoch=20,checkpoint=str(weight),
                       checkpoint_sha256=d.digest(weight),clean_O_AP70=.2,training_contract_sha256=contract_sha))
    put(folder/'selected.json',selected)
    ec=seal(dict(checkpoint_sha256=d.digest(weight),source_hashes=dict(variant='B1',sources=sources),
        associated_d1d_training=dict(training_contract_sha256=contract_sha),
        protocols={'F':{'name':'OFFICIAL_CLEAN_FIXED_GT'}},frame_keys=[['scene',str(i)] for i in range(829)],
        frozen_d1c_gt_identity='fixed-gt',input_manifest_identity='inputs'))
    put(evaluation/'contract.json',ec);put(evaluation/'protocol.json',ec)
    frames=[];gtrows=[];predrows=[]
    for w in d.WEATHERS:
        for i,key in enumerate(ec['frame_keys']):
            f,c=fake_frame(w,True)
            c.update(contract_identity=ec['identity'],condition=w,index=i,frame_key=key,
                     prediction_boxes=[[[0,0,0]]*8],prediction_is_none=False)
            c['record_identity']=d.identity(c)
            path=evaluation/'predictions'/w/('%06d.json'%i);sha=put(path,c)
            f.update(index=i,frame_key=key,prediction_cache_sha256=sha)
            f['protocols']['F']['prediction_cache_sha256']=sha
            frames.append(f)
            gtrows.append(dict(condition=w,index=i,frame_key=key,fixed_gt_identity=d.identity(fixed_gt()),
                               official_gt_identity=d.identity(fixed_gt()),fixed_ids=['car']))
            predrows.append(dict(condition=w,index=i,path=str(path),sha256=sha))
    gt=seal(dict(contract_identity=ec['identity'],definition=ec['protocols'],frames=gtrows,fixed_total_per_condition=829))
    pm=seal(dict(contract_identity=ec['identity'],files=predrows))
    put(evaluation/'gt_manifest.json',gt);put(evaluation/'prediction_manifest.json',pm)
    (evaluation/'per_frame_metrics.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames),encoding='utf-8')
    for w in d.WEATHERS:
        put(evaluation/('stat_'+w+'_F.json'),{'0.7':dict(tp=[1]*829,fp=[0]*829,score=[.8]*829,gt=829)})
    output_hashes={p.name:d.digest(p) for p in evaluation.iterdir() if p.is_file()}
    summary=dict(status='complete',full_validation=True,frame_count=829,conditions=list(d.WEATHERS),
        checkpoint_sha256=d.digest(weight),model_variant='B1',contract_identity=ec['identity'],
        gt_identity=gt['identity'],frozen_d1c_gt_identity='fixed-gt',prediction_cache_identity=pm['identity'],output_hashes=output_hashes)
    put(evaluation/'summary.json',summary)
    for stage,paths in (('B1_train',[folder/'selected.json',folder/'contract.json',folder/'history.json']),
                        ('B1_evaluate',[evaluation/'summary.json'])):
        put(root/'stages'/stage/'complete.json',dict(status='complete',stage=stage,artifacts={str(p):d.digest(p) for p in paths}))
    return evaluation


class DiagnoseTests(unittest.TestCase):
    def test_report_distinguishes_coefficient_from_causality(self):
        targets=d.analyze_targets(fake_models(),None)
        gate=dict(selected_epoch=20,source_initial_gate=-5,initial_mixing_ratio=d.sigmoid(-5),
                  final=dict(mixing_ratio=.1,raw_gate=-2.19722458),trajectory=[],limitations=[])
        text=d.report(gate,targets,['点数记录缺失'])
        for phrase in ('不能说只执行了','尚未验证','目标净丢失和误检增加同时存在','本次不执行','点数记录缺失'):
            self.assertIn(phrase,text)

    def test_no_model_training_inference_or_gpu_calls(self):
        source=Path(__file__).with_name('diagnose.py').read_text(encoding='utf-8')
        tree=ast.parse(source)
        forbidden={'backward','step','cuda','generate','inference_intermediate_fusion','build_dataset','create_model','save','load_state_dict'}
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):self.assertFalse((node.module or '').startswith(('opencood','models','auditlib')))
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,forbidden)

    def test_full_cache_verification_and_missing_cache_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);evaluation=complete_fixture(root)
            model=d.load_model_evidence(root,'B1',d.Reader())
            self.assertEqual(len(model['frames']),2487)
            (evaluation/'predictions/fog/000005.json').unlink()
            with self.assertRaises(FileNotFoundError):d.load_model_evidence(root,'B1',d.Reader())

    def test_saved_trace_rejects_missing_duplicate_and_bad_counts(self):
        f,c=fake_frame('fog',True)
        self.assertEqual(set(d.trace_matches(f,'0.7',c,{'car'})),{'car'})
        for mutation in ('missing','duplicate','counts','flags'):
            g=copy.deepcopy(f)
            if mutation=='missing':del g['protocols']['F']['matched_pairs']['0.7']
            elif mutation=='duplicate':g['protocols']['F']['matched_pairs']['0.7']*=2
            elif mutation=='counts':g['protocols']['F']['thresholds']['0.7']['fp']=1
            else:g['protocols']['F']['thresholds']['0.7']['tp_flags']=[]
            with self.assertRaises(ValueError):d.trace_matches(g,'0.7',c,{'car'})

    def test_losses_recoveries_and_fp_net(self):
        targets=d.analyze_targets(fake_models(),None)
        fog=targets['results']['fog'];snow=targets['results']['snow']
        self.assertEqual((fog['counts']['lost'],fog['counts']['recovered'],fog['counts']['tp_net'],fog['counts']['fp_net']),(1,0,-1,1))
        self.assertEqual((snow['counts']['lost'],snow['counts']['recovered']),(0,1))
        self.assertEqual(fog['examples'][0]['object_id'],'car')
        self.assertIsNone(fog['examples'][0]['points_sum_across_cavs'])
        self.assertIn('missing',{r['bin'] for r in fog['strata']})

    def test_strata_denominators_and_counts(self):
        current=dict(cavs=[dict(before_point_filter_ids=['car'],official_points_in_box=[3]),
                           dict(before_point_filter_ids=['car'],official_points_in_box=[2])])
        meta=d.target_metadata(fixed_gt()[0],current)
        self.assertEqual((meta['point_bin'],meta['coverage_bin'],meta['points_sum_across_cavs']),('1-5','2+',5))
        evidence=[dict(conditions={w:current for w in d.WEATHERS})]
        row=d.analyze_targets(fake_models(),evidence)['results']['fog']
        self.assertTrue(all(s['loss_rate_among_b0_correct']==1 for s in row['strata']))

    def test_gate_cpu_load_scalar_and_extreme_values(self):
        class Tensor:
            class device:type='cpu'
            def numel(self):return 1
            def item(self):return -5.
        class Torch:
            def load(self,path,**kw):
                self.kw=kw
                return {'repe_a.gate':Tensor()}
        torch=Torch();row=d.gate_from_state('fixture',torch)
        self.assertEqual(torch.kw,dict(map_location='cpu',weights_only=True))
        self.assertAlmostEqual(row['mixing_ratio'],.0066928509242848554)
        self.assertEqual(d.sigmoid(-1000),0)
        self.assertEqual(d.sigmoid(1000),1)

    def test_input_hash_and_change_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'sample.json';p.write_text('{}')
            r=d.Reader();r.json(p)
            p.write_text('{"changed":true}')
            with self.assertRaises(ValueError):r.unchanged()
            with self.assertRaises(ValueError):r.track(p,'wrong')

    def test_manifest_hashes_are_checkpoint_bound_but_geometry_must_match(self):
        models={}
        ckeys=('settings','policy','train_manifest_sha256','train_manifest_identity','pair_identity',
               'config_sha256','input_identity','frozen_gt','sources')
        ekeys=('frame_keys','conditions','protocols','inference_seed','official_config_sha256',
               'official_source_hashes','frozen_d1c_gt_identity','input_manifest_identity','d1b')
        for v in ('B0','B1'):
            models[v]=dict(contract={k:'same' for k in ckeys},evaluation_contract={k:'same' for k in ekeys},
                summary=dict(gt_identity=v,frozen_d1c_gt_identity='same'),
                gt=dict(frames=[dict(frame_key=['scene','1'],fixed_ids=['car'])],definition={'F':dict(name='OFFICIAL_CLEAN_FIXED_GT')}))
        self.assertEqual(d.verify_pair(models)['status'],'passed')
        models['B1']['gt']['frames'][0]['fixed_ids']=['wrong']
        with self.assertRaises(ValueError):d.verify_pair(models)


if __name__=='__main__':unittest.main()
