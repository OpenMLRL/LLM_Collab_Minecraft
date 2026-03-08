from __future__ import annotations

from typing import Any, Mapping, Sequence

from LLM_Collab_Minecraft.bridge_build.utils.bridge_builder import BridgeState, TaskSpec


def _agent_markers(state: BridgeState) -> dict[tuple[int, int], str]:
    markers: dict[tuple[int, int], str] = {}
    for idx, pos in enumerate(state.agent_positions):
        x, z = int(pos[0]), int(pos[1])
        mark = "A" if idx == 0 else ("B" if idx == 1 else str(idx))
        key = (x, z)
        if key in markers and markers[key] != mark:
            markers[key] = "X"
        else:
            markers[key] = mark
    return markers


def render_ascii_map(task: TaskSpec, state: BridgeState) -> str:
    """Render current bridge state as an ASCII map.

    Legend:
    - `.` water
    - `#` land
    - `S`/`T` anchors
    - `Y`/`N` pillars
    - `*` filled block on non-anchor/non-pillar cells
    - `A`/`B` agent positions (`X` if overlapped)
    """
    markers = _agent_markers(state)
    filled = {(int(x), int(z)) for (x, z), block in state.filled.items() if str(block).strip()}

    lines = ["   " + "".join(str(x % 10) for x in range(task.width))]
    for z, row in enumerate(task.rows_topdown):
        cells = []
        for x, base in enumerate(row):
            pos = (int(x), int(z))
            ch = base
            if pos in filled and base not in ("S", "T", "Y", "N"):
                ch = "*"
            if pos in markers:
                ch = markers[pos]
            cells.append(ch)
        lines.append(f"{z:02d} " + "".join(cells))
    return "\n".join(lines)


def print_turn_debug(
    *,
    task: TaskSpec,
    state: BridgeState,
    turn_idx: int,
    reward: float,
    metrics: Mapping[str, Any],
    agent_outputs: Sequence[str],
) -> None:
    connected = bool(metrics.get("connected", False))
    gap_st = metrics.get("gap_st")
    max_gap_st = int(metrics.get("max_gap_st", 0))
    n_adj = int(metrics.get("n_adjacent_count", 0))
    y_uncovered = int(metrics.get("y_uncovered_count", 0))
    probe_cnt = int(metrics.get("num_valid_probes", 0))
    comm_tokens = int(metrics.get("comm_tokens", 0))

    print(
        "[bridge_build debug] "
        f"task={task.task_id} turn={int(turn_idx)} reward={float(reward):.4f} "
        f"connected={connected} gap_ST={gap_st}/{max_gap_st} N_adj={n_adj} Y_uncovered={y_uncovered} "
        f"probe={probe_cnt} comm_tokens={comm_tokens}",
        flush=True,
    )

    for idx, raw in enumerate(agent_outputs):
        label = "A" if idx == 0 else ("B" if idx == 1 else str(idx))
        print(f"[bridge_build debug] agent_{label}_output:", flush=True)
        txt = str(raw or "").rstrip()
        print(txt if txt else "(empty)", flush=True)

    print("[bridge_build debug] current_ascii_map:", flush=True)
    print(render_ascii_map(task, state), flush=True)
