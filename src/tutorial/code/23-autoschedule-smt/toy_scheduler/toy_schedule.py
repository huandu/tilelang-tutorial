#!/usr/bin/env python3
"""A最小可复现的昇腾软件流水调度玩具模型（纯 Python，无第三方依赖）。

本文件把 `_vendor/tilelang-v0.1.15/tilelang/ascend/transform/z3_scheduler.py`
里 `z3_schedule_loop_python` 的模型缩到能一眼看懂的规模：

* 每个任务 = (name, latency, ii, pipe)；
* 依赖 = `data_deps (u, v, distance, latency)`，`distance < 0` 表示"自动多缓冲"；
* 资源冲突 = `resource_deps (i, j)`，同一硬件流水不能同时发射；
* 片上容量 = `memory_groups`，每组的 `num_stages` 份物理版本不能超容量；
* 目标 = 最小化启动间隔 II（start = k * II + r）。

它和 z3_scheduler.py 的差别只有两处（都在正文里说明过）：

1. 求解器换成回溯搜索，而不是 Z3；对 4~5 个任务的玩具实例是精确解；
2. 玩具模型不带 owner-exclusion（共享同一物理 ring 的两个 owner）约束。

用法：
    python3 toy_schedule.py demo            # 默认示例，打印调度结果
    python3 toy_schedule.py sensitivity     # 延迟估计不准会怎样
    python3 toy_schedule.py --print-smt     # 打印等价的 Z3 编码（伪代码）
    python3 toy_schedule.py --z3            # 装了 z3-solver 时用 Z3 复算并比对
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# 1. 任务与资源标签
# --------------------------------------------------------------------------

# 与 src/ascend/transform/ascend_pipe.h 的 ResourcePipe 位掩码一致
MTE1, MTE2, MTE3, CUBE, VECTOR, FIXPIPE, SCALAR = 1, 2, 4, 8, 16, 32, 64
PIPE_NAME = {
    MTE1: "MTE1",
    MTE2: "MTE2",
    MTE3: "MTE3",
    CUBE: "Cube",
    VECTOR: "Vector",
    FIXPIPE: "Fixpipe",
    SCALAR: "Scalar",
}


@dataclass
class Task:
    name: str
    latency: int  # 发射到结果可用的周期数
    ii: int  # 同一硬件流水两次发射之间的最小间隔
    pipe: int  # ResourcePipe 位掩码


@dataclass
class Problem:
    tasks: list[Task]
    data_deps: list[tuple[int, int, int, int]]  # (u, v, distance, latency)
    resource_deps: list[tuple[int, int]]  # 同一流水上不能并发的无序任务对
    buffer_sizes: list[int] = field(default_factory=list)  # 自动多缓冲的字节数
    memory_groups: list[list[int]] = field(default_factory=list)  # [cap, buf_idx...]
    num_stages: int = 2  # 每个自动 buffer 允许的最大版本数
    max_k: int = 1  # 允许的 k 上界（1 = 最多两级 stage）

    @property
    def n(self) -> int:
        return len(self.tasks)


# --------------------------------------------------------------------------
# 2. 约束检查（逐条对应 z3_scheduler.py 里的 solver.add(...)）
# --------------------------------------------------------------------------


def check_partial(problem: Problem, ii: int, assign: dict[int, tuple[int, int]],
                  bufvar: list[int]) -> bool:
    """检查已赋值任务之间的全部约束；返回 False 表示可以剪枝。"""

    def start(idx: int) -> int:
        k, r = assign[idx]
        return k * ii + r

    for u, v, distance, latency in problem.data_deps:
        if u not in assign or v not in assign:
            continue
        if distance >= 0:
            # start_v - start_u >= latency - II * distance
            if start(v) - start(u) < latency - ii * distance:
                return False
        else:
            # 负 distance 编码自动 buffer：start_v - start_u >= latency - II * buf
            b = bufvar[-distance - 1]
            if start(v) - start(u) < latency - ii * b:
                return False

    for i, j in problem.resource_deps:
        if i not in assign or j not in assign:
            continue
        _ki, ri = assign[i]
        _kj, rj = assign[j]
        # 取 (r_j - r_i) mod II 作为模序；两个方向只有一种等价写法
        d = (rj - ri) % ii
        if not (d >= problem.tasks[i].ii and (ii - d) % ii >= problem.tasks[j].ii):
            return False
    return True


def capacity_ok(problem: Problem, bufvar: list[int]) -> bool:
    for group in problem.memory_groups:
        cap, members = group[0], group[1:]
        usage = sum(bufvar[b] * problem.buffer_sizes[b] for b in members)
        if usage > cap:
            return False
    return True


# --------------------------------------------------------------------------
# 3. 回溯搜索：给定 II 与 buffer 版本，判断是否可行
# --------------------------------------------------------------------------


def feasible(problem: Problem, ii: int, bufvar: list[int], want_model: bool = False):
    """返回 None（不可行）或 start_times 列表。"""
    n = problem.n
    # k 取 [0, max_k + 1] 再在最后按最小 start 归一化，等价于 C++ 里
    # `start_times[i] - offset * ii` 的做法（见 auto_schedule.cc:1117-1131）。
    domains = [
        [(k, r) for k in range(problem.max_k + 2) for r in range(ii)] for _ in range(n)
    ]
    assign: dict[int, tuple[int, int]] = {}
    # 模序对全体 r 的平移不变（资源约束只看 (r_j - r_i) mod II，数据约束只看
    # start 之差），所以可以固定 r_0 = 0 而不丢解；这正是把搜索空间砍到
    # (2 * ii)^(n-1) 的关键。
    domains[0] = [(0, 0)]
    order = list(range(n))

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        for k, r in domains[idx]:
            assign[idx] = (k, r)
            if check_partial(problem, ii, assign, bufvar) and dfs(pos + 1):
                return True
            del assign[idx]
        return False

    if not dfs(0):
        return None
    if not want_model:
        return []
    starts = [assign[i][0] * ii + assign[i][1] for i in range(n)]
    offset = min(starts)
    # 归一化到从 0 开始：等价于 C++ 里 `start_times[i] - offset * ii`
    return [s - offset for s in starts]


def solve(problem: Problem, verbose: bool = False):
    """线性扫描 II，返回 (ii, start_times, buffer_versions)。"""
    lower = max(t.ii for t in problem.tasks)
    upper = sum(t.latency for t in problem.tasks) + 1
    num_buffers = len(problem.buffer_sizes)
    combos = list(itertools.product(range(1, problem.num_stages + 1), repeat=num_buffers))
    combos.sort(key=sum)  # 同 II 下优先用更少的内存
    for ii in range(lower, upper + 1):
        for bufvar in combos:
            if not capacity_ok(problem, list(bufvar)):
                continue
            starts = feasible(problem, ii, list(bufvar), want_model=True)
            if starts is not None:
                if verbose:
                    print(f"  first feasible II = {ii}, buffer versions = {list(bufvar)}")
                return ii, starts, list(bufvar)
    raise RuntimeError("no feasible schedule found within the search bound")


# --------------------------------------------------------------------------
# 4. 玩具实例：一个 5 任务的 GEMM 流水
# --------------------------------------------------------------------------


def gemm_like_problem(mte2_latency: int = 40) -> Problem:
    """把 example_gemm.py 的内循环缩成 4 个任务。

    真实延迟量级见 src/ascend/transform/estimate_latency.h（例如
    mte2_gm_to_l1_base_latency=190、l1_to_l0a_bandwidth=256）。这里为了手算
    方便把数字等比缩小，模型结构不变。
    """
    tasks = [
        Task("loadA(MTE2 GM->L1)", mte2_latency, 10, MTE2),
        Task("l1->l0(MTE1)", 15, 3, MTE1),
        Task("cube(Cube)", 30, 8, CUBE),
        Task("l0c->gm(Fixpipe)", 25, 6, FIXPIPE),
    ]
    data_deps = [
        # 同迭代真依赖：A 就绪 -> MTE1；MTE1 -> Cube；Cube -> Fixpipe
        (0, 1, 0, mte2_latency),
        (1, 2, 0, 15),
        (2, 3, 0, 30),
        # 跨迭代 WAR：下一轮的 loadA 覆盖 L1 前，上一轮的 MTE1 必须读完。
        # distance = -(buf_id + 1)，即交给调度器决定 ring 宽度。
        (1, 0, -1, 15),
    ]
    resource_deps = [(2, 1)]  # Cube 与 MTE1 争 L0 端口
    return Problem(
        tasks=tasks,
        data_deps=data_deps,
        resource_deps=resource_deps,
        buffer_sizes=[1000],  # L1 中 A 一份的字节数（示意）
        memory_groups=[[4096, 0]],  # L1 容量与成员
        num_stages=2,
        max_k=3,
    )


def describe(problem: Problem, ii: int, starts: list[int], bufvar: list[int]) -> None:
    print(f"  II = {ii}（相邻两轮迭代的最小启动间隔）")
    print(f"  buffer versions = {bufvar}  "
          f"(容量占用 {sum(b * s for b, s in zip(bufvar, problem.buffer_sizes))}"
          f" / {problem.memory_groups[0][0]})")
    print("  start(k, r) 按开始时间排序：")
    for idx in sorted(range(problem.n), key=lambda i: starts[i]):
        t = problem.tasks[idx]
        k = starts[idx] // ii
        r = starts[idx] % ii
        print(f"    [{starts[idx]:3d}] {t.name:<26s} k={k} r={r:2d} "
              f"latency={t.latency:2d} ii={t.ii:2d} pipe={PIPE_NAME[t.pipe]}")
    makespan = max(starts[i] + problem.tasks[i].latency for i in range(problem.n))
    print(f"  makespan（单轮）= {makespan} cycles")


# --------------------------------------------------------------------------
# 5. 模式：demo / sensitivity / print-smt / z3
# --------------------------------------------------------------------------


def run_demo() -> None:
    problem = gemm_like_problem()
    ii, starts, bufvar = solve(problem, verbose=True)
    describe(problem, ii, starts, bufvar)


def run_sensitivity() -> None:
    """延迟估计不准会怎样：真值 40，估计值 16。"""
    true_problem = gemm_like_problem(mte2_latency=40)
    est_problem = gemm_like_problem(mte2_latency=16)

    print("== 真值 ==")
    ii_t, starts_t, buf_t = solve(true_problem)
    describe(true_problem, ii_t, starts_t, buf_t)

    print("\n== EstimateLatency 低估 MTE2（40 -> 16）后调度器看到的世界 ==")
    ii_e, starts_e, buf_e = solve(est_problem)
    describe(est_problem, ii_e, starts_e, buf_e)

    print("\n== 用真值模型复核这份调度 ==")
    violations = []
    for u, v, distance, latency in true_problem.data_deps:
        if distance < 0:
            continue
        slack = starts_e[v] - starts_e[u] - latency
        if slack < 0:
            violations.append((u, v, slack))
    if violations:
        for u, v, slack in violations:
            print(f"  违约：{true_problem.tasks[v].name} 比 "
                  f"{true_problem.tasks[u].name} 早了 {-slack} cycles "
                  f"（数据未就绪）")
    else:
        print("  没有违约——低估延迟只是让 II 更小，但安排恰好仍然合法。")
    print(f"\n  结论：估计值 {est_problem.tasks[0].latency} 让调度器选了 II={ii_e}，"
          f"真值是 II={ii_t}。模型误差直接变成时间表误差。")


def print_smt() -> None:
    """把同一套模型写成 Z3 的约束，便于与 z3_scheduler.py 对照。"""
    problem = gemm_like_problem()
    n = problem.n
    b = len(problem.buffer_sizes)
    print("# 以下编码与 tilelang/ascend/transform/z3_scheduler.py 的")
    print("# z3_schedule_loop_python 一一对应（去掉 owner-exclusion）。")
    print('ctx    = z3.Context()          # 每次求解用私有 context，保证可复现')
    print('solver = z3.Solver(ctx=ctx); solver.set("rlimit", 1000000)')
    print("k      = [z3.Int('k_' + str(i), ctx) for i in range(" + str(n) + ")]")
    print("r      = [z3.Int('r_' + str(i), ctx) for i in range(" + str(n) + ")]")
    print("buf    = [z3.Int('buf_' + str(i), ctx) for i in range(" + str(b) + ")]"
          "  # 1 <= buf <= num_stages")
    print("start  = [k[i] * II + r[i] for i in range(" + str(n) + ")]")
    print("# 模域：0 <= r[i] < II，k[i] >= 0")
    for u, v, distance, latency in problem.data_deps:
        if distance >= 0:
            rhs = str(latency) if distance == 0 else f"{latency} - II * {distance}"
            print(f"solver.add(start[{v}] - start[{u}] >= {rhs})")
        else:
            print(f"solver.add(start[{v}] - start[{u}] >= "
                  f"{latency} - II * buf[{-distance - 1}])")
    for i, j in problem.resource_deps:
        print(f"# 资源对 ({i},{j})：二选一建立模序，并保护绕回")
        print(f"o = z3.Bool('O_{i}_{j}', ctx)")
        print(f"solver.add(z3.Implies(o, r[{j}] - r[{i}] >= {problem.tasks[i].ii}))")
        print(f"solver.add(z3.Implies(o, r[{i}] - r[{j}] + II >= {problem.tasks[j].ii}))")
        print(f"solver.add(z3.Implies(z3.Not(o), r[{i}] - r[{j}] >= {problem.tasks[j].ii}))")
        print(f"solver.add(z3.Implies(z3.Not(o), r[{j}] - r[{i}] + II >= {problem.tasks[i].ii}))")
    for cap, *members in problem.memory_groups:
        terms = " + ".join(f"buf[{m}] * {problem.buffer_sizes[m]}" for m in members)
        print(f"solver.add({terms} <= {cap})   # 片上容量")
    print("# 目标：对 II 做二分，取满足约束的最小 II")


def run_z3() -> None:
    try:
        import z3  # type: ignore
    except ImportError:
        print("[skip] 本机没有 z3-solver；--print-smt 输出的编码等价，"
              "但不执行。")
        return
    problem = gemm_like_problem()
    py_ii, py_starts, py_buf = solve(problem)
    print(f"[python 回溯] II={py_ii} starts={py_starts} buf={py_buf}")
    # 用与 z3_scheduler.py 相同的约束复算同一个 II，验证两条路是否一致。
    ctx = z3.Context()
    solver = z3.Solver(ctx=ctx)
    solver.set("rlimit", 1000000)
    n, b = problem.n, len(problem.buffer_sizes)
    k = [z3.Int(f"k_{i}", ctx) for i in range(n)]
    r = [z3.Int(f"r_{i}", ctx) for i in range(n)]
    buf = [z3.Int(f"buf_{i}", ctx) for i in range(b)]
    ii = py_ii
    for i in range(n):
        solver.add(k[i] >= 0, r[i] >= 0, r[i] < ii)
    for i in range(b):
        solver.add(buf[i] >= 1, buf[i] <= problem.num_stages)
    starts = [k[i] * ii + r[i] for i in range(n)]
    for u, v, distance, latency in problem.data_deps:
        if distance >= 0:
            solver.add(starts[v] - starts[u] >= latency - ii * distance)
        else:
            solver.add(starts[v] - starts[u] >= latency - ii * buf[-distance - 1])
    for i, j in problem.resource_deps:
        o = z3.Bool(f"O_{i}_{j}", ctx)
        solver.add(z3.Implies(o, r[j] - r[i] >= problem.tasks[i].ii))
        solver.add(z3.Implies(o, r[i] - r[j] + ii >= problem.tasks[j].ii))
        solver.add(z3.Implies(z3.Not(o), r[i] - r[j] >= problem.tasks[j].ii))
        solver.add(z3.Implies(z3.Not(o), r[j] - r[i] + ii >= problem.tasks[i].ii))
    for cap, *members in problem.memory_groups:
        solver.add(z3.Sum([buf[m] * problem.buffer_sizes[m] for m in members]) <= cap)
    print(f"[z3] II={ii} 可行？ {solver.check() == z3.sat}  "
          f"(z3 {z3.get_version_string()})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="demo",
                        choices=["demo", "sensitivity", "print-smt", "z3"])
    parser.add_argument("--print-smt", action="store_true",
                        help="打印等价的 Z3 约束")
    parser.add_argument("--z3", action="store_true",
                        help="本机装有 z3-solver 时复算并比对")
    args = parser.parse_args()
    if args.print_smt or args.mode == "print-smt":
        print_smt()
    elif args.z3 or args.mode == "z3":
        run_z3()
    elif args.mode == "sensitivity":
        run_sensitivity()
    else:
        run_demo()


if __name__ == "__main__":
    main()
