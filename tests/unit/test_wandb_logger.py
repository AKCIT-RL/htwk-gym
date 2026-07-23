from pathlib import Path
import sys

MIMICKIT = Path(__file__).resolve().parents[2] / "MimicKit" / "mimickit"
sys.path.insert(0, str(MIMICKIT))

from util import wandb_logger


class FakeArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.files = []

    def add_file(self, filename, name):
        self.files.append((filename, name))


def test_wandb_logger_records_metrics_config_and_final_model(tmp_path, monkeypatch):
    calls = {"define": [], "log": [], "artifacts": [], "finished": 0}
    fake_run = type("Run", (), {"id": "run-id"})()

    def fake_init(**kwargs):
        calls["init"] = kwargs
        monkeypatch.setattr(wandb_logger.wandb, "run", fake_run)
        return fake_run

    monkeypatch.setenv("WANDB_PROJECT", "g1-amp")
    monkeypatch.setenv("WANDB_ENTITY", "g1-research-team")
    monkeypatch.setenv("WANDB_NAME", "wave-seed1")
    monkeypatch.setenv("WANDB_GROUP", "wave-overfit")
    monkeypatch.setenv("WANDB_TAGS", "g1,official,seed-1")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(wandb_logger.wandb, "run", None)
    monkeypatch.setattr(wandb_logger.wandb, "init", fake_init)
    monkeypatch.setattr(wandb_logger.wandb, "define_metric", lambda *args, **kwargs: calls["define"].append((args, kwargs)))
    monkeypatch.setattr(wandb_logger.wandb, "log", lambda data, step=None: calls["log"].append((data, step)))
    monkeypatch.setattr(wandb_logger.wandb, "Artifact", FakeArtifact)
    monkeypatch.setattr(wandb_logger.wandb, "log_artifact", lambda artifact, aliases: calls["artifacts"].append((artifact, aliases)))
    monkeypatch.setattr(wandb_logger.wandb, "finish", lambda: calls.__setitem__("finished", calls["finished"] + 1))

    output_dir = tmp_path / "amp_g1_wave_official_50m_seed1"
    log = wandb_logger.WandbLogger("mimickit", {"disc_grad_penalty": 10})
    log.set_step_key("Samples")
    log.configure_output_file(str(output_dir / "log.txt"))
    log.log("Samples", 8192, collection="1_Info")
    log.log("Test_Episode_Length", 43.5, collection="0_Main")
    log.log("Disc_Demo_Acc", 1.0)
    log.write_log()

    model_file = output_dir / "model.pt"
    model_file.write_bytes(b"checkpoint")
    log.log_model(str(model_file))
    log.finish()

    assert calls["init"]["project"] == "g1-amp"
    assert calls["init"]["entity"] == "g1-research-team"
    assert calls["init"]["name"] == "wave-seed1"
    assert calls["init"]["group"] == "wave-overfit"
    assert calls["init"]["tags"] == ["g1", "official", "seed-1"]
    assert calls["init"]["mode"] == "offline"
    assert calls["init"]["config"] == {"disc_grad_penalty": 10}
    assert calls["define"] == [(("1_Info/Samples",), {}), (("*",), {"step_metric": "1_Info/Samples"})]
    assert calls["log"] == [({
        "1_Info/Samples": 8192,
        "0_Main/Test_Episode_Length": 43.5,
        "Misc/Disc_Demo_Acc": 1.0,
    }, 8192)]
    assert calls["artifacts"][0][0].files == [(str(model_file), "model.pt")]
    assert calls["artifacts"][0][1] == ["final"]
    assert calls["finished"] == 1
