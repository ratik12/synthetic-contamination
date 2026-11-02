"""Stage 02: generate synthetic twins (resumable). Needs a GPU for the real backend."""
import _boot  # noqa
from src.generation import assign_twins, run_generation
from src.io_utils import data_dir, pick_device, read_jsonl, stage_main, write_json, write_jsonl


def run(cfg, opts):
    out = data_dir(cfg)
    human = read_jsonl(out / "human.jsonl")
    gens = cfg["generation"]["generators"]
    pairs = assign_twins(human, gens, cfg["data"]["twins_S4"], cfg["seed"])
    if "only" in opts:                                  # e.g. --extra only=qwen2.5-1.5b  (run one generator per Colab session)
        pairs = [p for p in pairs if p["generator"] == opts["only"]]
        gens = [g for g in gens if g["name"] == opts["only"]]
        if not gens:
            raise ValueError(f"unknown generator {opts['only']}")
        cfg = {**cfg, "generation": {**cfg["generation"], "generators": gens}}
    if "limit" in opts:                                 # smoke test: first N pairs per generator, kept (resumable) if good
        n = int(opts["limit"])
        pairs = [p for g in gens for p in [q for q in pairs if q["generator"] == g["name"]][:n]]
        print(f"SMOKE MODE: only the first {n} passages per generator")
    else:
        write_jsonl(out / "pairs.jsonl", pairs)
    mock = cfg["generation"]["backend"] == "mock"
    device = "cpu" if mock else pick_device("auto")
    if not mock and device != "cuda" and opts.get("allow_cpu") != "1":
        raise RuntimeError("No GPU found. Generation with 1.5B-3.8B models on CPU would take many hours. "
                           "Use a Colab/Kaggle GPU runtime (or pass --extra allow_cpu=1 to override).")
    print(f"{len(pairs)} twins to generate on {device}")
    summary = run_generation(cfg, human, pairs, out, device)
    if "limit" not in opts:
        write_json(out / f"generation_summary_{opts.get('only', 'all')}.json", summary)
    print(summary)
    return str(summary)


if __name__ == "__main__":
    stage_main("02_generate", run)
