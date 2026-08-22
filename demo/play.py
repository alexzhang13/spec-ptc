"""Console demo player (headless twin of the TUI): prints the event stream."""

from __future__ import annotations

import argparse
import time

from rich.console import Console

from demo.scenarios import CATALOG, get_scenario
from spec_ptc.contracts.events import EventBus
from spec_ptc.runtime.engines import MockLM
from spec_ptc.runtime.harness import Harness


def attach_printer(bus: EventBus, console: Console, t0: float, verbose: bool) -> None:
    def on_ev(ev):
        dt = ev.t - t0
        d = ev.data
        if ev.kind == "dispatch":
            icon = "◆ peek    " if d.get("source") == "peek" else "● dispatch"
            console.print(
                f"[yellow]{dt:7.2f}s  {icon} #{d['seq']:<3} "
                f"{d['tool']}({d.get('preview', '')})[/yellow]"
            )
        elif ev.kind == "adopt":
            console.print(
                f"[dim yellow]{dt:7.2f}s  ◇ adopt    #{d['seq']:<3} {d['tool']}[/dim yellow]"
            )
        elif ev.kind == "ready":
            console.print(
                f"[green]{dt:7.2f}s  ✔ ready    #{d['seq']:<3} ({d.get('ms', 0):.0f}ms)[/green]"
            )
        elif ev.kind == "claim_hit":
            console.print(
                f"[bold green]{dt:7.2f}s  ✚ claim    #{d.get('seq', '?'):<3} "
                f"{d['tool']} (ready={d.get('already_ready')})[/bold green]"
            )
        elif ev.kind == "claim_miss":
            console.print(f"[red]{dt:7.2f}s  ○ miss     {d['tool']}[/red]")
        elif ev.kind == "evict":
            console.print(
                f"[red]{dt:7.2f}s  ✖ evict    #{d.get('seq', '?')} ({d.get('reason')})[/red]"
            )
        elif ev.kind == "shadow_stop":
            console.print(f"[magenta]{dt:7.2f}s  ▲ shadow stop: {d.get('reason')}[/magenta]")
        elif ev.kind == "stmt_closed" and verbose:
            src = d["src"].split("\n")[0][:60]
            console.print(f"[dim]{dt:7.2f}s  » stmt closed: {src}[/dim]")
        elif ev.kind in ("stream_begin", "stream_end", "exec_begin", "exec_end"):
            console.print(f"[blue]{dt:7.2f}s  ── {ev.kind} ──[/blue]")

    bus.subscribe(on_ev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="oolong-mood-agg")
    ap.add_argument("--mode", default="spec")
    ap.add_argument("--live", action="store_true", help="vLLM sub-LM, scripted main")
    ap.add_argument("--full-live", action="store_true", help="vLLM writes the code too")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    console = Console()
    if args.list:
        for s in CATALOG:
            console.print(f"[bold]{s.name:26s}[/bold] [{s.category}] {s.description}")
        return

    t0 = time.perf_counter()
    if args.full_live:
        from demo.live import engine_from_env, run_live

        bus = EventBus()
        attach_printer(bus, console, t0, args.verbose)
        final, h, _ = run_live(
            args.scenario, args.mode, bus=bus, engine=engine_from_env(bus=bus)
        )
        console.print(f"\n[bold]final answer:[/bold] {str(final)[:400]}")
    else:
        sc = get_scenario(args.scenario)
        bus = EventBus()
        attach_printer(bus, console, t0, args.verbose)
        if args.live:
            from demo.live import HybridEngine, engine_from_env

            eng = HybridEngine(sc.timing, engine_from_env(bus=bus))
        else:
            eng = MockLM(sc.timing, bus=bus)
        h = Harness(eng, args.mode, bus=bus, context=sc.context)
        final = None
        for turn in sc.turns:
            out = h.run_turn(eng.stream_main(turn))
            final = out.final_answer or final
        h.launcher.shutdown()
        console.print(f"\n[bold]final answer:[/bold] {str(final)[:400]}")

    wall = time.perf_counter() - t0
    specs = h.store.all
    claimed = sum(1 for s in specs if s.state == "claimed")
    serial = sum((s.t_ready - s.t_dispatch) for s in specs if s.t_ready) if specs else 0
    console.print(
        f"[bold]wall {wall:.2f}s · {len(specs)} speculated · {claimed} claimed · "
        f"serial-call-time {serial:.1f}s · "
        f"saved ≈ {max(0, serial - wall):.1f}s[/bold]"
    )


if __name__ == "__main__":
    main()
