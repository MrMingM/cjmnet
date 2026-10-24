"""Migration contract tests: AST and fabricated path dictionaries only, no scientific imports."""
import ast
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parent


class MigrationStaticTests(unittest.TestCase):
    def source(self,name):
        return (ROOT/name).read_text(encoding='utf-8-sig')

    def test_cuda_dispatch_before_legacy_prepare(self):
        tree=ast.parse(self.source('training.py'))
        fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='prepare')
        self.assertIsInstance(fn.body[0],ast.If)
        self.assertIn('cuda_prepare',ast.unparse(fn.body[0]))

    def test_dataset_keys_uses_official_cav_ego_order(self):
        tree=ast.parse(self.source('cuda_evaluation.py'))
        node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='dataset_keys')
        scope={'Path':Path}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<isolated key parser>','exec'),scope)
        class Dataset:
            scenario_database={0:{'-1':{'ego':False,'0001':{'yaml':'/d/scene/-1/0001.yaml'}},
                '200':{'ego':True,'0001':{'yaml':'/d/scene/200/0001.yaml'},'0002':{'yaml':'/d/scene/200/0002.yaml'}}}}
        self.assertEqual(scope['dataset_keys'](Dataset()),[['scene','0001'],['scene','0002']])

    def test_extraction_hash_keeps_original_compact_json(self):
        import hashlib,json
        node=next(x for x in ast.parse(self.source('cuda_runtime.py')).body if isinstance(x,ast.FunctionDef) and x.name=='extraction_identity')
        scope={'hashlib':hashlib,'json':json}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<original extraction hash>','exec'),scope)
        value={'split':'validate','frame_keys':[['scene','0001']]}
        expected=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        self.assertEqual(scope['extraction_identity'](value),expected)
        self.assertNotEqual(scope['extraction_identity'](value),hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest())

    def test_no_package_installation_in_runner(self):
        for name in ('run_all.sh','cuda_runtime.py','cuda_evaluation.py'):
            text=self.source(name)
            for forbidden in ('pip install','apt-get install','conda install'):
                self.assertNotIn(forbidden,text)

    def test_cuda_visibility_preserves_platform(self):
        text=self.source('run_all.sh')
        cuda=text.split('cuda)',1)[1].split('rocm)',1)[0]
        self.assertNotIn('unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES',cuda)
        self.assertIn('unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES',cuda)
        self.assertIn('${CUDA_VISIBLE_DEVICES+x}',cuda)
        self.assertIn('${GPU+x}',cuda)

    def test_new_gt_uses_original_observer_and_match(self):
        self.assertIn("'observe_gt.py'",self.source('cuda_runtime.py'))
        text=self.source('cuda_evaluation.py')
        self.assertIn('.official_match_trace',text)
        self.assertIn('inference_utils.inference_intermediate_fusion(',text)
        self.assertIn('eval_utils_opv2v.calculate_ap(',text)
        self.assertIn('model.load_state_dict(state,strict=True)',text)

    def test_transferred_data_schema(self):
        text=self.source('cuda_runtime.py')
        self.assertIn("record['files'].items()",text)
        self.assertNotIn("pair_manifest['files']",text)
        self.assertIn("row['view']",text)
        self.assertIn("plan['protocol']['frame_keys']",text)
        self.assertIn("payload_hashes.get((scene,cav,name))",text)

    def test_gpu_allocation_does_not_change_training_batch(self):
        text=self.source('training.py')
        self.assertIn('batch_size=8',text)
        self.assertNotIn('except torch.cuda.OutOfMemoryError',text)
        self.assertNotIn('batch_size=4',text)

    def test_posix_script_is_lf(self):
        raw=(ROOT/'run_all.sh').read_bytes()
        self.assertNotIn(b'\r',raw)
        self.assertFalse(raw.startswith(b'\xef\xbb\xbf'))
        for bad in (b'[[',b'pipefail',b'#!/bin/bash'):
            self.assertNotIn(bad,raw)


if __name__=='__main__':
    unittest.main()
