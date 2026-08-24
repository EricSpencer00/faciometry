"""The command line.

argparse rather than a framework, because the CLI has five verbs and adding a
dependency to get decorators would put a third-party package between a user
and `vitruve catalogue`, which is a command that has to work on a machine with
nothing installed.

The verbs divide on one line: `fetch-weights` opens a socket and nothing else
does. That division is the offline guarantee, and it is asserted in
`tests/integration/test_offline.py` rather than promised here.
"""

from __future__ import annotations

import argparse
import sys

from .. import __version__
from ..core.spec import Evidence, View
from ..models.licensing import Tier
from .exits import Exit
from .licenses import tier_from_string

ANALYSIS_TIERS = ("permissive", "copyleft", "noncommercial")
ALL_TIERS = tuple(t.name.lower() for t in Tier)

DESCRIPTION = """\
Facial morphometrics with intervals. Measurements are reported against their
own error, and any measurement that moves further under a ten-degree head
rotation than it moves between different people is withheld with its reason.
No aggregate score is produced.
"""

EPILOG = """\
examples:
  vitruve catalogue                        every measurement and how well each survives a photograph
  vitruve catalogue --id gonial_angle_l    everything known about one measurement
  vitruve licenses --tier copyleft         what the AGPL tier obliges you to
  vitruve doctor                           device, weights, versions
  vitruve fetch-weights                    the only command that uses the network
  vitruve analyze front.jpg --profile side.jpg --out report/
  vitruve analyze a.jpg b.jpg c.jpg --out report/   pool three captures of one face

exit codes:
  0  ran; some measurements may have been withheld, which is a result
  1  unexpected failure
  2  bad input
  3  the photograph did not clear the quality gate
  4  a backend exceeded the permitted license tier
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vitruve",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"vitruve {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="command")

    analyze = sub.add_parser(
        "analyze",
        help="measure a face from a photograph",
        description="Measure a face. Nothing is downloaded; run `vitruve fetch-weights` first.",
    )
    analyze.add_argument(
        "frontal",
        nargs="+",
        help="frontal photograph. Several may be given, and they are pooled "
        "before measuring, which is the largest reduction available for the "
        "landmark term. Every one must be the same person in the same session, "
        "holding the same pose: captures that disagree by more than the models' "
        "own noise are reported as a quality issue and are not pooled.",
    )
    analyze.add_argument("--profile", help="profile photograph, for the sagittal measurements")
    analyze.add_argument(
        "--license-tier",
        choices=ANALYSIS_TIERS,
        default="permissive",
        help="highest license tier a backend may carry (default: permissive)",
    )
    analyze.add_argument(
        "--declared-sex",
        choices=("male", "female"),
        help="declared by the subject, used only to pick a normative stratum and an "
        "interpupillary prior. Never inferred.",
    )
    analyze.add_argument(
        "--declared-ancestry",
        help="declared by the subject, used only to pick a normative stratum. Never inferred.",
    )
    analyze.add_argument(
        "--ruler-mm",
        type=float,
        help="a known length in millimetres, applied against the pupil span measured "
        "in the photograph. Give your own interpupillary distance from a "
        "pupillometer or a spectacle prescription and the population scale prior, "
        "worth about 5.5%% under every millimetre value, is replaced by it.",
    )
    analyze.add_argument("--out", help="directory to write report.json, report.txt and report.html into")
    analyze.add_argument("--json", action="store_true", help="print JSON instead of text")
    analyze.add_argument(
        "--pdf",
        action="store_true",
        help="also write report.pdf, the typeset report (needs pip install 'vitruve[pdf]')",
    )
    analyze.add_argument("--seed", type=int, default=0, help="Monte-Carlo seed (default: 0)")

    catalogue = sub.add_parser(
        "catalogue",
        help="every measurement, its evidence tier and its pose sensitivity",
    )
    catalogue.add_argument("--view", choices=("frontal", "profile"))
    catalogue.add_argument(
        "--evidence", choices=tuple(e.value for e in Evidence), help="filter by evidence tier"
    )
    catalogue.add_argument("--id", dest="spec_id", help="print everything about one measurement")
    catalogue.add_argument("--json", action="store_true")

    licenses = sub.add_parser("licenses", help="what a license tier obliges you to")
    licenses.add_argument("--tier", choices=ALL_TIERS, default="permissive")

    fetch = sub.add_parser(
        "fetch-weights",
        help="download and hash-verify model weights (the only networked command)",
    )
    fetch.add_argument("--tier", choices=ANALYSIS_TIERS, default="permissive")
    fetch.add_argument(
        "--force", action="store_true", help="re-download even if the cached digest matches"
    )

    doctor = sub.add_parser("doctor", help="device, weights, versions")
    doctor.add_argument("--tier", choices=ANALYSIS_TIERS, default="permissive")

    serve = sub.add_parser("serve", help="local HTTP API and web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8731)
    serve.add_argument(
        "--allow-remote",
        action="store_true",
        help="permit binding a non-loopback address. This puts a service that "
        "receives photographs of faces on the network.",
    )
    serve.add_argument(
        "--store",
        action="store_true",
        help="write uploaded images to disk. Off by default, and nothing is written "
        "without it.",
    )
    serve.add_argument("--license-tier", choices=ANALYSIS_TIERS, default="permissive")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return Exit.OK

    if args.command == "catalogue":
        from . import catalogue as catalogue_cmd

        return catalogue_cmd.run(
            view=View(args.view) if args.view else None,
            evidence=Evidence(args.evidence) if args.evidence else None,
            spec_id=args.spec_id,
            as_json=args.json,
        )

    if args.command == "licenses":
        from . import licenses as licenses_cmd

        return licenses_cmd.run(tier_from_string(args.tier))

    if args.command == "doctor":
        from . import doctor as doctor_cmd

        return doctor_cmd.run(tier_from_string(args.tier))

    if args.command == "fetch-weights":
        from . import weights as weights_cmd

        return weights_cmd.run(tier_from_string(args.tier), force=args.force)

    if args.command == "analyze":
        from . import analyze as analyze_cmd

        return analyze_cmd.run(
            frontal=args.frontal,
            profile=args.profile,
            license_tier=tier_from_string(args.license_tier),
            declared_sex=args.declared_sex,
            declared_ancestry=args.declared_ancestry,
            ruler_mm=args.ruler_mm,
            out=args.out,
            as_json=args.json,
            seed=args.seed,
            pdf=args.pdf,
        )

    if args.command == "serve":
        from ..api.serve import RemoteBindRefused
        from ..api.serve import serve as serve_cmd

        try:
            return serve_cmd(
                host=args.host,
                port=args.port,
                allow_remote=args.allow_remote,
                store=args.store,
                license_tier=tier_from_string(args.license_tier),
            )
        except RemoteBindRefused as exc:
            print(f"vitruve serve: {exc}", file=sys.stderr)
            return Exit.BAD_INPUT
        except ImportError as exc:
            print(
                f"vitruve serve: {exc}. Install the server with "
                "`pip install 'vitruve[api]'`.",
                file=sys.stderr,
            )
            return Exit.ERROR

    parser.error(f"unknown command {args.command!r}")
    return Exit.BAD_INPUT  # pragma: no cover - parser.error exits


def entrypoint() -> None:  # pragma: no cover - console_script wrapper
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
