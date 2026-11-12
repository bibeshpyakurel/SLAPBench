"""Exact runtime-transition review contracts; synthetic metadata only."""

import contextlib
import copy
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import precise_native_runtime_review as runtime  # noqa: E402
import score_precise_native_restored as entry  # noqa: E402
from precise_matching.data import InputError  # noqa: E402
from precise_matching.fixed_pairs import canonical_digest  # noqa: E402
from precise_matching.training import digest, write_json  # noqa: E402


def cores():
    before = {'code_commit':'a'*40,'packages':{'torch':'synthetic'},'source_sha256':{'code/synthetic.py':'synthetic'},
              'input_sha256':{'bundle':'b','config':'c','review':'r'},'model_identity':{'arm':'adapted_dinov2'},
              'hardware':{'device':'cuda','cuda_device':'NVIDIA GeForce RTX 4080 SUPER','driver_version':'595.99.02',
                          'platform':'Linux-7.0.0-42-generic-x86_64-with-glibc2.43','driver_metadata_source':'proc_driver_nvidia'}}
    after=copy.deepcopy(before);after['hardware'].update(platform='Linux-7.0.0-43-generic-x86_64-with-glibc2.43',
                                                        driver_metadata_source='nvidia-smi')
    return before,after


class RuntimeTransitionTests(unittest.TestCase):
    def test_only_exact_kernel_discovery_pair_is_eligible(self):
        before,after=cores();self.assertEqual(set(runtime.transition_delta(before,after)),{'platform','driver_metadata_source'})
        for key,value in (('driver_version','different'),('cuda_device','another GPU'),('platform','Linux-other')):
            changed=copy.deepcopy(after);changed['hardware'][key]=value
            with self.subTest(key=key),self.assertRaises(InputError):runtime.transition_delta(before,changed)
        changed=copy.deepcopy(after);changed['packages']['torch']='different'
        with self.assertRaises(InputError):runtime.transition_delta(before,changed)

    def fixture(self,root):
        private=root/'local/precise';run=private/'runs/synthetic/adapted-training';run.mkdir(parents=True)
        before,after=cores();write_json(run/'run_record.json',{'provenance':before})
        checkpoint=run/'final.pt';checkpoint.write_bytes(b'SYNTHETIC NOT MODEL WEIGHTS')
        selected={'checkpoint':'final.pt','training_complete':True,'epochs_completed':3,
                  'checkpoint_rule':'fixed_final_epoch','test_scores_accessed':False}
        write_json(run/'selection.json',selected)
        verification=private/'verification.json';write_json(verification,{'passed':True,'checkpoint_sha256':digest(checkpoint),
                         'selection_sha256':digest(run/'selection.json'),'counts':{'changed_frozen_tensors':0}})
        proposal=root/'docs/notes/proposal.md';proposal.parent.mkdir(parents=True);proposal.write_text('SYNTHETIC ONLY')
        human=private/'human.json';write_json(human,{'response':'SYNTHETIC APPROVAL ONLY','proposal_sha256':digest(proposal)})
        paths={'training_record':run/'run_record.json','selection':run/'selection.json','checkpoint':checkpoint,
               'verification':verification,'proposal':proposal,'human_record':human}
        sources={**before['source_sha256'],**{n:'synthetic' for n in runtime.BRIDGE_PATHS}}
        review={'scope':runtime.SCOPE,'execution_authorized':True,'authorized_phases':['calibrate','test'],
                'frozen_calibration_retry_authorized':True,'training_provenance_sha256':canonical_digest(before),
                'scoring_provenance_sha256':canonical_digest(after),'source_sha256':sources,
                'input_sha256':{**before['input_sha256'],**{k:digest(p) for k,p in paths.items()}},
                **{k:str(paths[k]) for k in ('verification','proposal','human_record')}}
        path=private/'review.json';write_json(path,review)
        return run,before,after,sources,path,review

    def test_actual_review_and_preserved_evidence_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(runtime,'ROOT',Path(tmp)):
            run,before,after,sources,path,review=self.fixture(Path(tmp))
            summary,_,_=runtime.inspect_runtime_review(path,run,after,before['input_sha256'],sources)
            self.assertEqual(summary['training_execution_source_commit'],'a'*40)
            for change in ({'execution_authorized':False},{'frozen_calibration_retry_authorized':False},
                           {'authorized_phases':['train','calibrate','test']},{'input_sha256':{}},{'source_sha256':{}}):
                path.write_text(json.dumps({**review,**change}))
                with self.subTest(change=change),self.assertRaises(InputError):
                    runtime.inspect_runtime_review(path,run,after,before['input_sha256'],sources)
            path.write_text(json.dumps(review));(run/'final.pt').write_bytes(b'ALTERED')
            with self.assertRaises(InputError):runtime.inspect_runtime_review(path,run,after,before['input_sha256'],sources)

    def test_saved_scoring_reports_actual_hardware_and_separate_source(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(runtime,'ROOT',Path(tmp)):
            run,before,after,sources,path,_=self.fixture(Path(tmp))
            summary,_,_=runtime.inspect_runtime_review(path,run,after,before['input_sha256'],sources)
            scored=copy.deepcopy(after);scored.update(code_commit='b'*40,source_sha256=sources,scoring_runtime_transition=summary)
            scored['model_identity']={'arm':'frozen_dinov2'}
            result=runtime.inspect_scoring_transition(path,run,scored)
            self.assertEqual(result[0]['delta']['platform']['scoring'],after['hardware']['platform'])
            self.assertEqual(json.loads((run/'run_record.json').read_text())['provenance'],before)
            changed=copy.deepcopy(scored);changed['hardware']['platform']=before['hardware']['platform']
            with self.assertRaises(InputError):runtime.inspect_scoring_transition(path,run,changed)

    def test_one_output_and_no_training_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'cal';review={'outputs':{'calibrate_frozen_dinov2':str(output)}}
            self.assertEqual(entry.checked_output(review,'calibrate','frozen_dinov2',output),output)
            with self.assertRaises(InputError):entry.checked_output(review,'calibrate','adapted_dinov2',output)
            output.with_name('cal-failure.json').write_text('synthetic failure')
            with self.assertRaises(InputError):entry.checked_output(review,'calibrate','frozen_dinov2',output)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:entry.main(['train'])
        self.assertEqual(error.exception.code,2)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Runs in matcher CPU CI')
    def test_both_calibration_arms_use_original_state_and_unchanged_phase_functions(self):
        from dataclasses import asdict
        import torch
        from precise_matching.fixed_pairs import FixedConfig
        from test_precise_training_group_holdout import pair_fixture
        with tempfile.TemporaryDirectory() as tmp,patch.object(runtime,'ROOT',Path(tmp)), \
             patch.object(entry,'ROOT',Path(tmp)),patch.object(entry.native,'ROOT',Path(tmp)):
            root=Path(tmp);run,before,after,_,path,review=self.fixture(root)
            for name in [*before['source_sha256'],*runtime.BRIDGE_PATHS]:
                f=root/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('# synthetic source only\n')
            original={k:digest(root/k) for k in before['source_sha256']}
            sources={**original,**{k:digest(root/k) for k in runtime.BRIDGE_PATHS}}
            before['source_sha256']=original;after['source_sha256']=original
            cfg=FixedConfig(batch_pairs=1,accumulation=32,warmup_steps=0)
            record={'config':asdict(cfg),'provenance':before};(run/'run_record.json').write_text(json.dumps(record))
            selected=json.loads((run/'selection.json').read_text());selected['history']=[{'epoch':n,'pairs_seen':6} for n in (1,2,3)]
            (run/'selection.json').write_text(json.dumps(selected))
            verification=Path(review['verification']);v=json.loads(verification.read_text());v['selection_sha256']=digest(run/'selection.json');verification.write_text(json.dumps(v))
            review.update(training_provenance_sha256=canonical_digest(before),scoring_provenance_sha256=canonical_digest(after),source_sha256=sources)
            review['input_sha256'].update(training_record=digest(run/'run_record.json'),selection=digest(run/'selection.json'),verification=digest(verification))
            review['outputs']={f'calibrate_{arm}':str(run.parent/arm) for arm in ('frozen_dinov2','adapted_dinov2')}
            path.write_text(json.dumps(review));original_record=(run/'run_record.json').read_bytes()
            ordinary=root/'local/precise/ordinary-review.json';write_json(ordinary,{})
            training,testing=pair_fixture();images={k:k for p in training+testing for k in (p.image_a,p.image_b)}
            bundle={'scope':'numeric_group_disjoint_fixed_pairs','images':{k:{'sha256':'raw','decoded_sha256':'decoded'} for k in images}}
            calls=[];models=[]
            class Model:
                def load_state_dict(self,state,strict):self.loaded=state;assert strict
            def load_model(*a,**kw):
                m=Model();models.append((m,kw));return m
            def calibrate(model,pairs,load,config,output,provenance,**kw):
                kw['integrity']();self.assertEqual(pairs,training);self.assertEqual(config,cfg)
                self.assertEqual(provenance['hardware'],after['hardware']);self.assertEqual(provenance['code_commit'],'b'*40)
                self.assertEqual(provenance['source_sha256'],sources)
                load(training[0].image_a)
                with self.assertRaises(InputError):load(testing[0].image_a)
                output.mkdir();calls.append(provenance)
            common=[v for k,p in [('bundle',root/'local/precise/bundle.json'),('config',root/'local/precise/config.json'),
                    ('review',ordinary),('runtime-review',path),('completed-training-run',run),('source-root',root/'images'),
                    ('weights',root/'local/precise/weights')] for v in ('--'+k,str(p))]
            patches=[patch.object(entry,'reviewed_shared_inputs',return_value=(cfg,before['input_sha256'],lambda:None)),
                patch.object(entry,'load_bundle',return_value=(bundle,training,testing,images)),
                patch.object(entry.native,'validate_native_policy'),patch.object(entry.native,'source_record',return_value=original),
                patch.object(entry.native,'git_output',side_effect=lambda *a,**kw:'' if a[0]=='status' else 'b'*40),
                patch.object(entry,'current_core',return_value=after),patch.object(entry.native,'load_encoder',side_effect=load_model),
                patch.object(entry.native,'prepare_image',return_value='synthetic pixel token'),patch.object(entry.native,'image_tensor',side_effect=lambda x:x),
                patch.object(entry,'calibrate_fixed',side_effect=calibrate),patch.object(torch.cuda,'is_available',return_value=True),
                patch.object(torch,'use_deterministic_algorithms'),patch.object(torch,'autocast',return_value=contextlib.nullcontext()),
                patch('torch.nn.attention.sdpa_kernel',return_value=contextlib.nullcontext()),contextlib.redirect_stdout(io.StringIO())]
            with contextlib.ExitStack() as stack:
                for context in patches:stack.enter_context(context)
                checked=stack.enter_context(patch.object(entry,'checked_checkpoint',return_value={
                    'epochs_completed':3,'history':selected['history'],'model':{'original':'state'}}))
                for arm in ('frozen_dinov2','adapted_dinov2'):
                    out=Path(review['outputs']['calibrate_'+arm])
                    result=entry.main(['calibrate',*common,'--arm',arm,'--output',str(out)])
                    diagnostic=out.with_name(out.name+'-failure.json')
                    self.assertEqual(result,0,diagnostic.read_text() if diagnostic.exists() else 'no synthetic diagnostic')
            self.assertEqual(len(calls),2);checked.assert_called_once()
            self.assertEqual(checked.call_args.args[2],record)
            self.assertFalse(hasattr(models[0][0],'loaded'))
            self.assertEqual(models[1][0].loaded,{'original':'state'})
            self.assertEqual((run/'run_record.json').read_bytes(),original_record)


if __name__=='__main__':unittest.main()
