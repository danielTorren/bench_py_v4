"""
Run the original NetLogo model headless for a fixed list of seeds.

The model file is copied into a work folder together with the CGE CSV files
(NetLogo reads them from the model's own folder).  One change is made to the
copy: the `debug` procedure writes every agent to `debug.csv` on every tick
(about 10 MB per run, even with `debugfiles` off).  Its body is replaced by
`ask turtles [ ]`.  `ask` still shuffles the agents with the model RNG, so the
random number stream, and therefore every result, is unchanged.  A BehaviorSpace
experiment is written next to it and run with NetLogo_Console.

Usage
-----
    python validation/run_netlogo.py --runs 100
    python validation/run_netlogo.py --runs 5 --cases ES --learning "Slow dynamics"

Output: validation/netlogo_out/netlogo_table.csv (BehaviorSpace table format).
"""

import argparse
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "netlogo" / "BENCH_ v04_ B-NLD.ESP.nlogox"
DATA = ROOT / "data"
DEFAULT_NETLOGO = Path(r"C:\Program Files\NetLogo 7.0.4\NetLogo_Console.exe")

EXPERIMENT = "python-port-validation"

METRICS = (
    ["year", "a1"]
    + [f"number.group{g}" for g in range(1, 6)]
    + [f"group{g}.a1" for g in range(1, 6)]
    + [f"number.dwage{c}" for c in range(1, 4)]
    + [f"dwage{c}.a1" for c in range(1, 4)]
)


def _enum(variable: str, values: list[str]) -> str:
    vals = "".join(f'<value value="{escape(v, {chr(34): "&quot;"})}"></value>' for v in values)
    return f'<enumeratedValueSet variable="{variable}">{vals}</enumeratedValueSet>'


def _q(s: str) -> str:
    """NetLogo string literal."""
    return f'"{s}"'


def write_experiment(path: Path, cases: list[str], learnings: list[str],
                     first_seed: int, runs: int) -> None:
    # Widget settings as in the saved 2023-01-27 experiments, except:
    #   Generate-seed? = false with Seed-for-random = first_seed .. first_seed+runs-1
    #   35 ticks (2016-2050, the model's own stop condition) instead of 34.
    constants = [
        _enum("Generate-seed?", ["false"]),
        _enum("Data", [_q("Empirical-survey")]),
        _enum("Scenario", [_q("Ref.SSP2")]),
        _enum("debugfiles", ["false"]),
        _enum("Memory", ["true"]),
        _enum("Investment", ["true"]),
        _enum("Conservation", ["false"]),
        _enum("Switching", ["false"]),
        _enum("NAT", ["false"]),
        _enum("TPB", ["false"]),
        _enum("COM", ["true"]),
        _enum("I1.cost", ["3000"]),
        _enum("case-study", [_q(c) for c in cases]),
        _enum("Learning", [_q(lv) for lv in learnings]),
        (f'<steppedValueSet variable="Seed-for-random" first="{first_seed}" '
         f'step="1" last="{first_seed + runs - 1}"></steppedValueSet>'),
    ]
    metrics = "".join(f"<metric>{m}</metric>" for m in METRICS)
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<experiments>\n"
        f'  <experiment name="{EXPERIMENT}" repetitions="1" sequentialRunOrder="true" '
        'runMetricsEveryStep="true" timeLimit="35">\n'
        "    <setup>setup</setup>\n"
        "    <go>go</go>\n"
        f"    <metrics>{metrics}</metrics>\n"
        f"    <constants>{''.join(constants)}</constants>\n"
        "  </experiment>\n"
        "</experiments>\n"
    )
    path.write_text(xml, encoding="utf-8")


def _without_debug_output(src: str) -> str:
    start = src.index("To debug\n")
    end = src.index("\nend", start)
    body = src[start:end]
    if 'file-open "debug.csv"' not in body:
        raise SystemExit("Unexpected `debug` procedure; not patching.")
    return src[:start] + "To debug\n  ask turtles [ ]" + src[end:]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--first-seed", type=int, default=1)
    p.add_argument("--cases", nargs="+", default=["ES", "NL"])
    p.add_argument("--learning", nargs="+",
                   default=["No learning", "Slow dynamics", "Informative"])
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--netlogo", type=Path, default=DEFAULT_NETLOGO)
    p.add_argument("--out-dir", type=Path, default=ROOT / "validation" / "netlogo_out")
    args = p.parse_args()

    if not MODEL.exists():
        raise SystemExit(f"NetLogo model not found: {MODEL}")

    work = args.out_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    model = work / MODEL.name
    model.write_text(_without_debug_output(MODEL.read_text(encoding="utf-8")),
                     encoding="utf-8", newline="")
    for csv in DATA.glob("cge-*.csv"):
        shutil.copy2(csv, work / csv.name)

    setup = work / "experiment.xml"
    table = work / "netlogo_table.csv"
    write_experiment(setup, args.cases, args.learning, args.first_seed, args.runs)

    cmd = [str(args.netlogo), "--headless",
           "--model", str(model),
           "--setup-file", str(setup),
           "--experiment", EXPERIMENT,
           "--table", str(table)]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    # NetLogo resolves the CSV files against the current directory.
    subprocess.run(cmd, check=True, cwd=work)
    print(f"Table written: {table}")


if __name__ == "__main__":
    main()
