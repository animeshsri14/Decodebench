"""CLI entry point - decodebench command."""
from __future__ import annotations
import argparse
import csv
import sys
import os
import importlib.util
from decodebench.verdict import DEFAULT_BYTE_THRESHOLD

def parse_comma_list(s: str) -> list[int]:
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid comma-separated list: '{s}'")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decodebench",
        description="DecodeBench: compare launch overhead with an analytic eliminable-byte estimate in LLM decode fusion."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- demo ---
    demo_p = sub.add_parser("demo", help="Run a bundled Llama-decode demo.")
    demo_p.add_argument("name", choices=["f1", "f2", "f4"], help="Demo name.")
    demo_p.add_argument("--dim", type=int, choices=[2048, 4096], default=4096, help="Hidden dimension.")
    demo_p.add_argument("--batch", type=int, default=1, help="Batch size.")
    demo_p.add_argument("--trials", type=int, default=30, help="Number of timing trials.")
    demo_p.add_argument("--byte-threshold", type=float, default=DEFAULT_BYTE_THRESHOLD, help="Byte ratio threshold.")
    demo_p.add_argument("--csv", type=str, help="Output path for trial CSV.")
    demo_p.add_argument("--dry-run", action="store_true", help="Print plan and exit.")

    # --- profile ---
    prof_p = sub.add_parser("profile", help="Profile a user-defined Sequence chain.")
    prof_p.add_argument("target", help="Target sequence wrapper (path/to/module.py:build_fn).")
    prof_p.add_argument("--trials", type=int, default=30, help="Number of timing trials.")
    prof_p.add_argument("--byte-threshold", type=float, default=DEFAULT_BYTE_THRESHOLD, help="Byte ratio threshold.")
    prof_p.add_argument("--csv", type=str, help="Output path for trial CSV.")
    prof_p.add_argument("--dry-run", action="store_true", help="Print plan and exit.")

    # --- sweep ---
    sweep_p = sub.add_parser("sweep", help="Sweep batch sizes and hidden dims for a demo.")
    sweep_p.add_argument("name", choices=["f1", "f2", "f4"], help="Demo name.")
    sweep_p.add_argument("--batch", type=parse_comma_list, default=[1, 2, 4, 8], help="Comma-separated batch sizes to sweep.")
    sweep_p.add_argument("--dim", type=int, choices=[2048, 4096], default=4096, help="Hidden dimension.")
    sweep_p.add_argument("--trials", type=int, default=30, help="Number of timing trials.")
    sweep_p.add_argument("--csv", type=str, help="Output path for sweep summary CSV.")
    sweep_p.add_argument("--dry-run", action="store_true", help="Print plan and exit.")

    return parser

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        if args.dry_run:
            print(f"[dry-run] demo {args.name} with dim={args.dim}, batch={args.batch}, trials={args.trials}, threshold={args.byte_threshold}")
            return 0
        
        # Lazy imports for GPU functionality
        import torch
        if not torch.cuda.is_available():
            print("Error: GPU/CUDA is not available.", file=sys.stderr)
            return 1
        
        from decodebench.demos.llama_decode import build_demo
        seq, inputs, replicas = build_demo(args.name, dim=args.dim, batch=args.batch)
        report = seq.profile(inputs, trials=args.trials, byte_threshold=args.byte_threshold, input_replicas=replicas)
        try:
            print(report.render())
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        if args.csv:
            report.to_csv(args.csv)
            print(f"Saved trials to {args.csv}")

    elif args.command == "profile":
        if ":" not in args.target:
            print("Error: Target must be in the format 'path/to/module.py:build_fn'", file=sys.stderr)
            return 2
        
        module_path, build_fn_name = args.target.split(":", 1)

        if args.dry_run:
            print(f"[dry-run] profile target {module_path}:{build_fn_name} with trials={args.trials}, threshold={args.byte_threshold}")
            return 0
        
        if not os.path.exists(module_path):
            print(f"Error: Module path '{module_path}' does not exist.", file=sys.stderr)
            return 1

        # Load user module
        try:
            spec = importlib.util.spec_from_file_location("user_module", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module from {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            build_fn = getattr(module, build_fn_name)
        except Exception as e:
            print(f"Error loading target: {e}", file=sys.stderr)
            return 1

        import torch
        if not torch.cuda.is_available():
            print("Error: GPU/CUDA is not available.", file=sys.stderr)
            return 1

        seq, inputs = build_fn()
        report = seq.profile(inputs, trials=args.trials, byte_threshold=args.byte_threshold)
        try:
            print(report.render())
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        if args.csv:
            report.to_csv(args.csv)
            print(f"Saved trials to {args.csv}")

    elif args.command == "sweep":
        if args.dry_run:
            print(f"[dry-run] sweep demo {args.name} with batch={args.batch}, dim={args.dim}, trials={args.trials}")
            return 0

        import torch
        if not torch.cuda.is_available():
            print("Error: GPU/CUDA is not available.", file=sys.stderr)
            return 1

        from decodebench.demos.llama_decode import build_demo
        
        sweep_rows = []
        # Header for sweep CSV:
        # demo,dim,batch,trials,t_stream_us,t_graph_us,delta_launch_us,b_bytes_est_us,bound,total_bytes,eliminable_bytes,ci_lo,ci_hi
        
        for b in args.batch:
            print(f"Sweeping {args.name} dim={args.dim} batch={b}...")
            seq, inputs, replicas = build_demo(args.name, dim=args.dim, batch=b)
            report = seq.profile(inputs, trials=args.trials, input_replicas=replicas)
            try:
                v = report.verdict()
            except RuntimeError as e:
                print(f"  WARN: {e}", file=sys.stderr)
                continue

            ci_lo = v.delta_launch_ci[0] if v.delta_launch_ci else 0.0
            ci_hi = v.delta_launch_ci[1] if v.delta_launch_ci else 0.0

            print(f"  Stream: {v.t_stream:.2f} us | Graph: {v.t_graph:.2f} us | Bound: {v.bound}")

            sweep_rows.append([
                args.name, args.dim, b, args.trials,
                f"{v.t_stream:.2f}", f"{v.t_graph:.2f}", f"{v.delta_launch:.2f}", f"{v.b_bytes_est:.2f}",
                v.bound, v.total_bytes, v.eliminable_bytes, f"{ci_lo:.2f}", f"{ci_hi:.2f}"
            ])
                
        if args.csv:
            with open(args.csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "demo", "dim", "batch", "trials", "t_stream_us", "t_graph_us",
                    "delta_launch_us", "b_bytes_est_us", "bound", "total_bytes",
                    "eliminable_bytes", "ci_lo", "ci_hi"
                ])
                w.writerows(sweep_rows)
            print(f"Saved sweep results to {args.csv}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
