"""Synthetic interruption/review contracts; no private pixels or real retries."""

import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import check_precise_native_resume as review  # noqa: E402
from precise_matching.data import InputError  # noqa: E402
from precise_matching.fixed_pairs import FixedConfig, GROUP_SCOPE  # noqa: E402
from precise_matching.native_large import WEIGHTS_SHA256  # noqa: E402
from precise_matching.training import digest, write_json  # noqa: E402
from report_precise_group_holdout import check_final_training  # noqa: E402


class ResumeReviewTests(unittest.TestCase):
    def fixture(self, root):
        run = root / 'local/precise/runs/synthetic/adapted-training'
        (run / 'attempt-0001').mkdir(parents=True)
        record = {'config':asdict(FixedConfig(batch_pairs=1,accumulation=32,warmup_steps=0)),
                  'provenance':{'code_commit':'a'*40,'model_profile':'dinov2_large_native_v1',
                               'benchmark_scope':GROUP_SCOPE,'weights_sha256':WEIGHTS_SHA256,
                               'input_sha256':{'bundle':'b','config':'c','review':'r'}}}
        write_json(run/'run_record.json',record)
        write_json(run/'attempt-0001/attempt.json',{'start_epoch':0,'resume_sha256':None})
        checkpoint = run/'attempt-0001/epoch-0002.pt';checkpoint.write_bytes(b'SYNTHETIC NOT MODEL WEIGHTS')
        checkpoint.with_suffix('.sha256').write_text(digest(checkpoint)+'\n')
        console = run.parent/'training-console.log';console.write_text('synthetic interrupted progress\n')
        proposal=root/'docs/notes/resume.md';proposal.parent.mkdir(parents=True);proposal.write_text('SYNTHETIC PROPOSAL ONLY')
        meta=root/'local/precise/metadata';meta.mkdir()
        human=meta/'human.json';write_json(human,{'response':'APPROVED SYNTHETIC FIXTURE ONLY','proposal_sha256':digest(proposal)})
        inspection=meta/'inspection.json'
        history=[{'epoch':n,'pairs_seen':6} for n in (1,2)]
        write_json(inspection,{'passed':True,'epochs_completed':2,'epochs_remaining':1,'training_complete':False,
            'counts':{'changed_frozen_tensors':0},'execution_source_commit':'a'*40,'history':history,
            'input_sha256':{'record':digest(run/'run_record.json'),'checkpoint':digest(checkpoint),'console':digest(console)},
            'last_reported_progress':{'epoch':3,'pairs_seen':2},'crash_root_cause':'unconfirmed'})
        evidence={'training_record':run/'run_record.json','resume_checkpoint':checkpoint,'interruption_verification':inspection,
                  'proposal':proposal,'review_record':human}
        config={'scope':'one_explicit_native_epoch_boundary_resume','execution_authorized':True,'max_new_attempts':1,
            'resume_from_epoch':2,'execution_source_commit':'a'*40,'resume_checkpoint':'attempt-0001/epoch-0002.pt',
            'interruption_verification':str(inspection),'proposal':str(proposal),'review_record':str(human),
            'input_sha256':{**record['provenance']['input_sha256'],**{k:digest(p) for k,p in evidence.items()}}}
        path=meta/'review.json';write_json(path,config)
        return run,record,path,config,history

    def test_review_binds_actual_response_proposal_and_preserved_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(review,'ROOT',Path(tmp)):
            run,record,path,config,_=self.fixture(Path(tmp))
            review.inspect_resume_review(path,run,record)
            for update in ({'execution_authorized':False},{'max_new_attempts':2},{'resume_from_epoch':1},
                           {'input_sha256':{}},{'execution_source_commit':'z'*40}):
                path.write_text(json.dumps({**config,**update}))
                with self.subTest(update=update),self.assertRaises(InputError):review.inspect_resume_review(path,run,record)
            path.write_text(json.dumps(config))
            (run/'attempt-0001/epoch-0002.pt').write_bytes(b'ALTERED')
            with self.assertRaises(InputError):review.inspect_resume_review(path,run,record)

    def test_report_rejects_extra_attempts_and_changed_preserved_history(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(review,'ROOT',Path(tmp)):
            run,record,path,config,history=self.fixture(Path(tmp))
            second=run/'attempt-0002';second.mkdir()
            write_json(second/'attempt.json',{'start_epoch':2,'resume_sha256':config['input_sha256']['resume_checkpoint'],
                                             'started_at_utc':'2026-11-10T01:00:00+00:00'})
            selected={'checkpoint':'attempt-0002/epoch-0003.pt','history':history+[{'epoch':3,'pairs_seen':6}],
                      'finished_at_utc':'2026-11-10T01:01:00+00:00'}
            write_json(second/'result.json',selected)
            result=review.check_resume_chain(path,run,record,selected)
            self.assertEqual(result['resume_attempt_elapsed_seconds'],60)
            self.assertTrue(result['training_wall_interval_includes_interruption_downtime'])
            changed=copy.deepcopy(selected);changed['history'][0]['pairs_seen']=5
            (second/'result.json').write_text(json.dumps(changed))
            with self.assertRaises(InputError):review.check_resume_chain(path,run,record,changed)
            (second/'result.json').write_text(json.dumps(selected));(run/'attempt-0003').mkdir()
            with self.assertRaises(InputError):review.check_resume_chain(path,run,record,selected)

    def test_final_report_requires_explicit_review_and_retains_full_final_checks(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(review,'ROOT',Path(tmp)):
            run,record,path,config,history=self.fixture(Path(tmp))
            second=run/'attempt-0002';second.mkdir()
            write_json(second/'attempt.json',{'start_epoch':2,'resume_sha256':config['input_sha256']['resume_checkpoint'],
                                             'started_at_utc':'2026-11-10T01:00:00+00:00'})
            checkpoint=second/'epoch-0003.pt';checkpoint.write_bytes(b'SYNTHETIC FINAL NOT MODEL WEIGHTS')
            selected={'checkpoint':'attempt-0002/epoch-0003.pt','history':history+[{'epoch':3,'pairs_seen':6}],
                      'finished_at_utc':'2026-11-10T01:01:00+00:00','checkpoint_rule':'fixed_final_epoch',
                      'test_scores_accessed':False,'training_complete':True,'epochs_completed':3}
            write_json(run/'selection.json',selected);write_json(second/'result.json',selected)
            identity={'checkpoint_sha256':digest(checkpoint),'selection_sha256':digest(run/'selection.json')}
            cfg=FixedConfig(**record['config'])
            verification={'passed':True,'base_weights_sha256':WEIGHTS_SHA256,**identity,
                          'epochs_completed':3,'optimizer_updates_completed':cfg.steps(6),
                          'counts':{'changed_frozen_tensors':0},'history':selected['history']}
            args=(run,verification,cfg,[None]*6,identity,'b','c')
            kwargs={'expected_base_sha':WEIGHTS_SHA256,'expected_profile':'dinov2_large_native_v1'}
            with self.assertRaises(InputError):check_final_training(*args,**kwargs)
            check_final_training(*args,**kwargs,resume_review=path)
            verification['counts']['changed_frozen_tensors']=1
            with self.assertRaises(InputError):check_final_training(*args,**kwargs,resume_review=path)


@unittest.skipUnless(importlib.util.find_spec('torch'),'Runs in matcher CPU CI')
class InterruptedScheduleTests(unittest.TestCase):
    def test_only_complete_finite_unfinished_epoch_boundary_is_eligible(self):
        import numpy as np
        import random
        import torch
        from verify_precise_native_interruption import interrupted_schedule
        cfg=FixedConfig(batch_pairs=1,accumulation=32,warmup_steps=0)
        saved={'epochs_completed':2,'history':[{'epoch':n,'pairs_seen':6} for n in (1,2)],
               'scheduler':{'last_epoch':2},'optimizer':{'state':{0:{'step':torch.tensor(2.),
                   'exp_avg':torch.ones(1),'exp_avg_sq':torch.ones(1)}}},
               'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
               'cuda_rng':[torch.zeros(4,dtype=torch.uint8)]}
        result=interrupted_schedule(saved,cfg,6)
        self.assertEqual(result['epochs_remaining'],1)
        for change in ({'epochs_completed':3},{'scheduler':{'last_epoch':1}}, {'cuda_rng':[]},
                       {'history':[{'epoch':1,'pairs_seen':5},{'epoch':2,'pairs_seen':6}]}):
            with self.subTest(change=change),self.assertRaises(InputError):interrupted_schedule({**saved,**change},cfg,6)
        bad=copy.deepcopy(saved);bad['optimizer']['state'][0]['exp_avg'][0]=float('nan')
        with self.assertRaises(InputError):interrupted_schedule(bad,cfg,6)


if __name__=='__main__':unittest.main()
