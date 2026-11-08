"""Sealed native report composition; synthetic source/outcome fixtures only."""

import contextlib
import copy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import report_precise_native_large as report  # noqa: E402
from audit_precise_qwen_results import summary_check  # noqa: E402
from compare_precise_qwen_ab import Decision, operating_summary  # noqa: E402
from precise_matching.data import InputError  # noqa: E402
from precise_matching.fixed_pairs import FixedConfig, GROUP_SCOPE, canonical_digest  # noqa: E402
from precise_matching.fixed_evaluation import fixed_summary  # noqa: E402
from precise_matching.evaluation import Score, select_cutoff, write_scores  # noqa: E402
from precise_matching.training import digest, write_json  # noqa: E402
from test_precise_training_group_holdout import pair_fixture  # noqa: E402
import run_precise_native_large as entry  # noqa: E402
import render_precise_native_report as renderer  # noqa: E402


class NativeReportTests(unittest.TestCase):
    def test_render_rows_keep_class_denominators_and_qwen_failure_accuracy(self):
        # Tiny aggregate fixture: one failed genuine and one failed impostor.
        summary = {'attempts':{'genuine':4,'impostor':4},'failures':{'genuine':1,'impostor':1},
                   'all_attempts_failure_as_reject':{'genuine_comparisons':4,'impostor_comparisons':4,
                       'genuine_accepts':2,'false_accepts':1,'tar':.5,'far':.25},
                   'descriptive_valid_score_discrimination':None}
        dino = {arm:{'threshold':{'cutoff':.7},'results':{'pooled':summary,
                    'by_slap_type':{hand:summary for hand in renderer.HANDS}}} for arm in renderer.ARMS}
        qwen = {kind:{'operating_point':summary,'by_slap_type':{hand:summary for hand in renderer.HANDS},
                     'confusion_matrix':{'TP':2,'TN':2,'FP':1,'FN':1},'accuracy_all_attempts':.5}
                for kind in ('base','finetuned')}
        data = {'dino':dino,'prior_small_same_cohort':dino,'qwen':qwen,
                'sdk_descriptive':{arm:{'pooled':{'discrimination':None,'valid':6}}
                                   for arm in ('verifinger_fusion','verifinger_sum')}}
        rows = renderer.operating_rows(data)
        self.assertEqual(len(rows),24)
        self.assertEqual(rows[0]['accuracy'],5/8)  # DINO failure-as-reject.
        self.assertEqual(rows[-1]['accuracy'],4/8)  # Qwen failure-as-incorrect.
        self.assertEqual(rows[-1]['TN'],2)
        self.assertEqual(rows[-1]['attempts'],8)
        self.assertIsNone(rows[-1]['threshold'])
        self.assertEqual(len(renderer.continuous_curves(data)),6)
        self.assertTrue(all('Qwen' not in name for name, *_ in renderer.continuous_curves(data)))
        corrupted = copy.deepcopy(data);corrupted['qwen']['base']['accuracy_all_attempts']=1
        with self.assertRaises(ValueError):renderer.operating_rows(corrupted)

    def test_committed_runtime_source_hash_is_verified(self):
        p = {"code_commit": "a" * 40, "source_sha256": {"code/synthetic.py": hashlib.sha256(b"synthetic").hexdigest()}}
        with patch.object(report.subprocess, "check_output", return_value=b"synthetic"):
            report.verify_runtime_source(p)
        with patch.object(report.subprocess, "check_output", return_value=b"changed"), self.assertRaises(InputError):
            report.verify_runtime_source(p)

    def test_sealed_report_binds_native_base_and_same_small_qwen_membership(self):
        training, testing = pair_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);private = root / "local/precise";private.mkdir(parents=True)
            sdk, qwen, runs = private / "sdk", private / "qwen", private / "runs"
            sdk.mkdir();qwen.mkdir();(qwen / "training-source").mkdir()
            files = {k: private / (k + ".json") for k in ("bundle","config","review","checkpoint_verification","cohort_audit","small_aggregate","qwen_aggregate")}
            prompt = {"system_prompt":"synthetic system","user_prompt":"synthetic prompt"}
            write_json(sdk / report.TRAIN_JSON,prompt);write_json(qwen / "training-source" / report.TRAIN_JSON,prompt)
            source_hash = {report.TRAIN_JSON:digest(sdk / report.TRAIN_JSON)}
            for role,name in report.LISTS.items():
                phase,label = role.split('_');pairs = training if phase=='train' else testing
                (sdk/name).write_text(''.join(f"{p.image_a} {p.image_b} 0 | 1 2 sum=3 nfing=2\n" for p in pairs if p.label==label))
                source_hash[name]=digest(sdk/name)
            write_json(files['cohort_audit'],{'new_train_test_overlap':{'pairs':0,'images':0,'numeric_groups':0}})
            keys={k for p in training+testing for k in (p.image_a,p.image_b)}
            write_json(files['bundle'],{'scope':GROUP_SCOPE,'labels':'as_supplied','training':[asdict(p) for p in training],
                'testing':[asdict(p) for p in testing],'images':{k:{'sha256':'raw-'+k,'decoded_sha256':'decoded-'+k} for k in keys},
                'source_sha256':{**source_hash,'audit':digest(files['cohort_audit'])}})
            cfg=FixedConfig(batch_pairs=1,accumulation=32,warmup_steps=0);write_json(files['config'],asdict(cfg))
            hashes={k:digest(files[k]) for k in ('bundle','config')}
            write_json(files['review'],{'input_sha256':hashes});hashes['review']=digest(files['review'])
            train=runs/'adapted-training';(train/'attempt-0001').mkdir(parents=True)
            write_json(train/'attempt-0001/attempt.json',{'start_epoch':0,'resume_sha256':None,'started_at_utc':'2026-11-08T01:00:00+00:00'})
            checkpoint=train/'final.pt';checkpoint.write_bytes(b'SYNTHETIC CHECKPOINT ONLY')
            history=[{'epoch':i,'pairs_seen':len(training)} for i in (1,2,3)]
            selection={'checkpoint':'final.pt','training_complete':True,'epochs_completed':3,'checkpoint_rule':'fixed_final_epoch',
                'test_scores_accessed':False,'history':history,'finished_at_utc':'2026-11-08T01:00:30+00:00'}
            write_json(train/'selection.json',selection)
            identity={'checkpoint_sha256':digest(checkpoint),'selection_sha256':digest(train/'selection.json')}
            write_json(files['checkpoint_verification'],{'passed':True,'base_weights_sha256':report.WEIGHTS_SHA256,**identity,
                'epochs_completed':3,'history':history,'optimizer_updates_completed':cfg.steps(len(training)),
                'counts':{'changed_frozen_tensors':0}})
            small_dino={}
            for arm in report.ARMS:
                short=arm.split('_')[0];cal=runs/(short+'-calibration');test=runs/(short+'-test');cal.mkdir();test.mkdir()
                model={'arm':arm,**(identity if short=='adapted' else {})}
                provenance={'benchmark_scope':GROUP_SCOPE,'model_profile':'dinov2_large_native_v1','model':report.MODEL_ID,
                    'model_revision':report.REVISION,'weights_sha256':report.WEIGHTS_SHA256,'input_sha256':hashes,
                    'source_sha256':{'code/synthetic.py':'synthetic'},'code_commit':'a'*40,'packages':{},'hardware':{},'model_identity':model,
                    'image_preprocessing':{'original_width_height':list(report.ORIGINAL_SIZE),'padded_width_height':list(report.PADDED_SIZE),
                                           'resampling':False,'padding':'centered_white'},'attention':'sdpa_forced_flash_only','inference_precision':'bf16'}
                if short=='adapted':write_json(train/'run_record.json',{'config':asdict(cfg),'provenance':provenance})
                train_scores=[Score(p.pair_id,'ok',.9 if p.label=='genuine' else .2) for p in training]
                test_scores=[Score(p.pair_id,'ok',.8 if p.label=='genuine' else .1) for p in testing]
                write_scores(cal/'scores.csv',train_scores)
                write_json(cal/'run_record.json',{'phase':'training_calibration','config':asdict(cfg),
                    'pairs_sha256':canonical_digest([asdict(p) for p in training]),'provenance':provenance})
                cutoff=select_cutoff([.2]*3,cfg.target_far);cutoff['selected_on']='training_impostors_resubstitution'
                write_json(cal/'selection.json',{'selection_frozen':True,'test_scores_accessed':False,'checkpoint_rule':cfg.checkpoint_rule,
                    'threshold_rule':cfg.threshold_rule,'model_identity':model,'threshold':cutoff,'scores_sha256':digest(cal/'scores.csv'),
                    'run_record_sha256':digest(cal/'run_record.json')})
                test_sha=canonical_digest([asdict(p) for p in testing]);selection_sha=digest(cal/'selection.json')
                write_json(cal/'test_claim.json',{'output':str(test),'selection_sha256':selection_sha,'testing_pairs_sha256':test_sha})
                write_json(test/'run_record.json',{'phase':'held_out_pair_test','config':asdict(cfg),'provenance':provenance,
                    'testing_pairs_sha256':test_sha,'selection_sha256':selection_sha})
                write_scores(test/'scores.csv',test_scores)
                results=fixed_summary(testing,test_scores,.2)
                write_json(test/'aggregate.json',{'results':results,'threshold':cutoff})
                write_json(test/'completion.json',{'complete':True,'scores_sha256':digest(test/'scores.csv'),'aggregate_sha256':digest(test/'aggregate.json')})
                small_dino[arm]={'results':results}
            write_json(files['small_aggregate'],{'benchmark_scope':GROUP_SCOPE,'input_sha256':{'bundle':hashes['bundle']},'dino':small_dino})
            qwen_report={'scope':'source_verified_common_pair_decisions_run_provenance_incomplete','benchmark_scope':GROUP_SCOPE,
                'outcome_membership_verified':True,'qwen':{},'checkpoint_evidence':{},'source_run_claims':{},'provenance_limits':['synthetic']}
            for kind in ('base','finetuned'):
                decisions={p.pair_id:Decision(p.pair_id,'ok',p.label=='genuine') for p in testing}
                summary={**prompt,'prompt_sha256':hashlib.sha256(json.dumps(list(prompt.values()),ensure_ascii=False).encode()).hexdigest(),
                    'genuine_counts':{'attempted':3,'predicted_a':3,'predicted_b':0,'invalid':0,'errors':0},
                    'impostor_counts':{'attempted':3,'predicted_a':0,'predicted_b':3,'invalid':0,'errors':0},
                    'confusion_matrix':{'TP':3,'FN':0,'FP':0,'TN':3},'metrics':{'overall_accuracy_all_attempts_percent':100}}
                write_json(qwen/f'summary_{kind}_matched.json',summary)
                for label in ('genuine','impostor'):
                    (qwen/f'res_test_{label}_2100_nonzero_{kind}_matched.txt').write_text(''.join(
                        f"{p.image_a} {p.image_b} {'A' if label=='genuine' else 'B'}\n" for p in testing if p.label==label))
                qwen_report['qwen'][kind]={**summary_check(summary,testing,decisions,prompt),
                    'by_slap_type':{hand:operating_summary([p for p in testing if p.hand==hand],decisions) for hand in ('left','right','thumbs')}}
            qwen_report['source_sha256']={str(p.relative_to(qwen)):digest(p) for p in qwen.rglob('*') if p.is_file()}
            write_json(files['qwen_aggregate'],qwen_report)
            output=root/'reports/native.json';receipt=private/'native-receipt.json'
            args=[v for k,p in files.items() for v in ('--'+k.replace('_','-'),str(p))]
            args += ['--sdk-intake',str(sdk),'--qwen-intake',str(qwen),'--run-root',str(runs),'--output',str(output),
                     '--private-receipt',str(receipt),'--study-date','2026-11-08']
            # The exact native schedule/review gate has separate CPU/real synthetic
            # coverage. This tiny saved-outcome fixture only tests composition.
            with patch.object(report,'ROOT',root),patch.object(entry,'validate_native_policy'), \
                 patch.object(report,'verify_runtime_source'),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(report.main(args),0)
            result=json.loads(output.read_text())
            self.assertEqual(result['model_profile'],'dinov2_large_native_v1')
            self.assertEqual(result['training_elapsed_seconds'],30)
            self.assertIsNone(result['binary_qwen_roc_auc_eer'])
            self.assertEqual(result['qwen']['finetuned']['accuracy_all_attempts'],1)
            self.assertNotIn('synthetic_a_',output.read_text())
            with patch.object(report,'ROOT',root),contextlib.redirect_stderr(io.StringIO()):self.assertEqual(report.main(args),1)


if __name__=='__main__':unittest.main()
