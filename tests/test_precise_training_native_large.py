"""Native-large entry gates and phase dispatch; synthetic files/models only."""

import contextlib
from dataclasses import asdict, replace
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import run_precise_native_large as entry  # noqa: E402
from precise_matching.data import InputError  # noqa: E402
from precise_matching.fixed_pairs import FixedConfig, GROUP_SCOPE, REVIEW_FLAGS  # noqa: E402
from precise_matching.training import digest, write_json  # noqa: E402
from test_precise_training_group_holdout import pair_fixture  # noqa: E402


def config():return FixedConfig(batch_pairs=1, accumulation=32)

def profile_review():return {"model_profile": "dinov2_large_native_v1", "native_pixels_reviewed": True,
                              "weights_sha256": entry.WEIGHTS_SHA256}


class NativePolicyTests(unittest.TestCase):
    def test_exact_model_native_pixels_and_schedule_review_are_required(self):
        cfg = config();cfg.validate(37194)
        entry.validate_native_policy({"scope": GROUP_SCOPE}, cfg, profile_review())
        for update in ({"model_profile": "small"}, {"native_pixels_reviewed": False}, {"weights_sha256": "wrong"}):
            with self.subTest(update=update), self.assertRaises(InputError):
                entry.validate_native_policy({"scope": GROUP_SCOPE}, cfg, {**profile_review(), **update})
        with self.assertRaises(InputError):entry.validate_native_policy({"scope": GROUP_SCOPE}, replace(cfg, batch_pairs=2), profile_review())
        with self.assertRaises(InputError):entry.validate_native_policy({"scope": "shared_seen_image_fixed_pairs"}, cfg, profile_review())


@unittest.skipUnless(importlib.util.find_spec("torch"), "Runs in matcher CPU CI")
class NativeEntryTests(unittest.TestCase):
    def test_training_and_both_arm_scoring_dispatch_bind_native_profile(self):
        import torch
        training, testing = pair_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); private = root / "local/precise";private.mkdir(parents=True)
            images = root / "images";images.mkdir()
            source = root / "code/synthetic.py";source.parent.mkdir();source.write_text("# synthetic source")
            lock = root / "code/precise_matching/environment/requirements-linux-x86_64-py312-cu124.lock"
            lock.parent.mkdir(parents=True);lock.write_text("synthetic lock")
            mapping = {}
            for key in {k for p in training + testing for k in (p.image_a,p.image_b)}:
                gray = Image.new("L", (1600,1500), 123);path = images / key;gray.save(path)
                mapping[key] = {"path": key,"sha256":digest(path),"decoded_sha256":hashlib.sha256(str(gray.size).encode()+gray.tobytes()).hexdigest()}
            bundle, cfg, review = [private / name for name in ("bundle.json","config.json","review.json")]
            write_json(bundle,{"schema_version":1,"scope":GROUP_SCOPE,"labels":"as_supplied","images":mapping,
                               "training":[asdict(p) for p in training],"testing":[asdict(p) for p in testing]})
            write_json(cfg,asdict(config()))
            record = {flag:True for flag in REVIEW_FLAGS};record.update(profile_review())
            record.update(scope=GROUP_SCOPE,authorized_phases=["train","calibrate","test"],review_record="Synthetic fixture only, no actual authority",
                          input_sha256={"bundle":digest(bundle),"config":digest(cfg)})
            write_json(review,record)
            common=[v for key,path in (("bundle",bundle),("config",cfg),("review",review),("source-root",images),("weights",private/"weights")) for v in ("--"+key,str(path))]
            events=[];run=private/"runs/train"

            def training_phase(model,pairs,load,config,output,provenance,**kw):
                output.mkdir(parents=True)
                write_json(output/"run_record.json",{"config":asdict(config),"provenance":provenance})
                history=[{"epoch":i,"pairs_seen":len(pairs)} for i in (1,2,3)]
                checkpoint=output/"final.pt"
                torch.save({"record":{"config":asdict(config),"provenance":provenance},"model":model.state_dict(),"epochs_completed":3,"history":history},checkpoint)
                checkpoint.with_suffix('.sha256').write_text(digest(checkpoint))
                write_json(output/"selection.json",{"training_complete":True,"epochs_completed":3,"checkpoint_rule":"fixed_final_epoch",
                    "test_scores_accessed":False,"checkpoint":"final.pt","checkpoint_sha256":digest(checkpoint),"history":history})
                events.append("train")

            def scoring_phase(model,*args,**kwargs):
                # Shared scoring helpers are covered separately; this checks native
                # loader pixels, declared numeric protocol and phase selection.
                pairs,load,config=args[:3]
                provenance=args[-1]
                self.assertEqual(tuple(load(pairs[0].image_a).shape),(3,1512,1610))
                self.assertEqual(provenance["model_profile"],"dinov2_large_native_v1")
                self.assertEqual(provenance["code_commit"],"synthetic-source")
                self.assertEqual(provenance["inference_precision"],"bf16")
                self.assertEqual(provenance["attention"],"sdpa_forced_flash_only")
                events.append(provenance["model_identity"]["arm"])

            def test_phase(model,train,test,load,cfg,cal,output,provenance,**kwargs):
                self.assertEqual(kwargs["scope"],GROUP_SCOPE)
                self.assertEqual(tuple(load(test[0].image_a).shape),(3,1512,1610))
                events.append("test-"+provenance["model_identity"]["arm"])

            # A tiny fixture cannot supply 100 warmup updates. Schedule arithmetic
            # is tested with the real37194count above and by shared trainer tests.
            with patch.object(entry,"ROOT",root),patch.object(entry,"git_output",side_effect=lambda *a,**kw:"synthetic-source" if a[0]=="log" else "unrelated-head"), \
                 patch.object(entry,"source_record",return_value={"code/synthetic.py":digest(source)}), \
                 patch.object(entry,"package_versions",return_value={}),patch.object(FixedConfig,"validate"), \
                 patch.object(entry,"load_encoder",side_effect=lambda *a,**kw:torch.nn.Linear(1,1)), \
                 patch.object(entry,"cuda_driver_metadata",return_value={"driver_version":"synthetic"}), \
                 patch.object(torch.cuda,"get_device_name",return_value="synthetic CUDA"), \
                 patch.object(torch,"autocast",side_effect=lambda *a,**kw:contextlib.nullcontext()) as autocast, \
                 patch.object(entry,"train_fixed",side_effect=training_phase), \
                 patch.object(entry,"calibrate_fixed",side_effect=scoring_phase),patch.object(entry,"test_fixed",side_effect=test_phase), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entry.main(["train",*common,"--output",str(run)]),0)
                for arm in ("frozen_dinov2","adapted_dinov2"):
                    extra=["--arm",arm]+(["--training-run",str(run)] if arm=="adapted_dinov2" else [])
                    cal=private/('runs/cal-'+arm)
                    self.assertEqual(entry.main(["calibrate",*common,*extra,"--output",str(cal)]),0)
                    self.assertEqual(entry.main(["test",*common,*extra,"--calibration",str(cal),"--output",str(private/('runs/test-'+arm))]),0)
                self.assertEqual(autocast.call_count,4)
            self.assertEqual(len(events),5)


if __name__=="__main__":unittest.main()
