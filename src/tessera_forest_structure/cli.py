"""Command-line interface for inspecting and validating the study release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .demo import run_demo, write_demo_result
from .repository import ROOT, input_status, verify_repository, workflow, workflows_by_id


def list_workflows() -> None:
    for item in workflows_by_id().values():
        print(f"{item['id']:<28} {item['title']}")


def show_workflow(identifier: str) -> None:
    print(json.dumps(workflow(identifier), indent=2))


def show_status(identifier: str) -> None:
    item = workflow(identifier)
    print(item["title"])
    for record in input_status(identifier):
        mark = "available" if record["available"] else "missing"
        print(f"  {mark:<9} {record['name']}: {record['path']} ({record['access']})")


def show_plan(identifier: str) -> None:
    item = workflow(identifier)
    print(item["title"])
    print(f"Parameters: {item['parameters']}")
    print("Inputs:")
    for record in item["inputs"]:
        print(f"  {record['path']} ({record['access']})")
    print("Outputs:")
    for path in item["outputs"]:
        print(f"  {path}")


def verify() -> None:
    checked, errors = verify_repository()
    if errors:
        print("Repository verification failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print(f"verified {checked} required paths and artifact checksums")


def demo(output: Path | None) -> None:
    source = ROOT / "examples" / "demo_data" / "reference_transfer.csv"
    write_demo_result(run_demo(source), output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and validate the Tessera forest-structure study."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List supported scientific workflows")
    show = subparsers.add_parser("show", help="Show one workflow record")
    show.add_argument("workflow")
    status = subparsers.add_parser("status", help="Check one workflow's inputs")
    status.add_argument("workflow")
    plan = subparsers.add_parser("plan", help="Show the frozen execution plan")
    plan.add_argument("workflow")
    subparsers.add_parser("verify", help="Validate paths and checksums")
    demo_parser = subparsers.add_parser("demo", help="Run the synthetic transfer example")
    demo_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "list":
        list_workflows()
    elif args.command == "show":
        show_workflow(args.workflow)
    elif args.command == "status":
        show_status(args.workflow)
    elif args.command == "plan":
        show_plan(args.workflow)
    elif args.command == "demo":
        demo(args.output)
    else:
        verify()
