"""Assemble and compress compressed-operator tiles in worker processes.

Tile assembly is Python-heavy, so threads contend for the interpreter lock. Each
worker process holds its own copy of the coefficient oracle and a kernel-table
state, assembles whole tiles and returns them already compressed; the caller
stores results in tile order exactly as the in-process path does.
"""
from contextlib import contextmanager
import multiprocessing as mp
import os
import pickle
import sys

MIB = 1024**2
# Interpreter, libraries, kernel tables and one tile's assembly workspace.
WORKER_BYTES = 384*MIB
MAX_WORKERS = 8
# Smaller operators finish before worker start-up would pay for itself.
MIN_TILES = 256
COUNTERS = ('calls', 'entries', 'dropped_routes')

_WORKER = None


@contextmanager
def _without_main_module():
    """Start workers without re-importing the caller's __main__.

    Workers need only ghost_backend; importing a GUI entry point is wasted
    start-up, and an unguarded script would run its solve again in every worker.
    """
    main = sys.modules.get('__main__')
    saved = {name: getattr(main, name) for name in ('__spec__', '__file__') if hasattr(main, name)}
    try:
        if main is not None:
            main.__spec__ = None
            if '__file__' in saved:
                del main.__file__
        yield
    finally:
        for name, value in saved.items():
            setattr(main, name, value)


def _ignore_checkpoint():
    pass


def _compressor(operator):
    """Picklable stand-in exposing only StreamedOperator.compress_tile."""
    from ghost_backend.compressed.operator import StreamedOperator
    shell = StreamedOperator.__new__(StreamedOperator)
    shell.__dict__.update(groups=operator.groups, group_bounds=operator.group_bounds,
                          tolerance=operator.tolerance, compression=operator.compression,
                          checkpoint=_ignore_checkpoint)
    return shell


def _sources(oracle):
    return list(getattr(oracle, 'oracles', [oracle]))


def prepare(oracle, operators):
    """(worker count, worker payload) for this operator, or (0, None) to stay in process."""
    from ghost_backend.execution.options import _ASSEMBLY_ALLOCATION, current_options, environment_value
    tiles = len(operators[0].groups)**2
    if (not getattr(oracle, 'process_tiles', False) or tiles < MIN_TILES or
            mp.current_process().daemon or current_options() is None):
        return 0, None
    configured = environment_value('GHOST_TILE_PROCESSES', '').strip()
    if configured:
        count = int(configured)
    else:
        allocation = _ASSEMBLY_ALLOCATION.get()
        # A desktop solve owns the host; a scheduled solve keeps to its reservation.
        count = min(MAX_WORKERS, (os.cpu_count() or 1)//2) if allocation is None else int(allocation)
    if count < 2:
        return 0, None
    payload = pickle.dumps((oracle, [_compressor(op) for op in operators], dict(current_options(), assembly_threads=1)),
                           protocol=pickle.HIGHEST_PROTOCOL)
    from ghost_backend.twod.solver import _solve_memory_limit_gb
    from ghost_backend.compressed.runtime import storage_budget
    spare = int(_solve_memory_limit_gb()*1024**3) - storage_budget()
    count = min(count, max(0, spare)//(WORKER_BYTES + 2*len(payload)))
    return (int(count), payload) if count >= 2 else (0, None)


def _initialize(payload):
    global _WORKER
    from ghost_backend.execution.thread_control import threadpool_limits
    from ghost_backend.execution.cpu import CPUState
    import ghost_backend.twod.assembly.native.far as far
    import scipy.linalg  # noqa: F401 -- SciPy's own BLAS must be loaded before limiting
    oracle, compressors, options = pickle.loads(payload)
    # Workers are the unit of parallelism: one BLAS thread and unsplit far blocks.
    limits = threadpool_limits(1)
    far.SPLIT_WORKERS = 1
    _WORKER = oracle, compressors, options, CPUState(), limits


def _tile(task):
    from ghost_backend.execution.cpu import _STATE
    from ghost_backend.execution.options import execution_scope
    i, j = task
    oracle, compressors, options, state, _ = _WORKER
    groups = compressors[0].groups
    sources = _sources(oracle)
    before = [[getattr(o, name) for name in COUNTERS] for o in sources]
    with execution_scope(options, assembly_threads=1), _STATE.override(state):
        values = oracle.get_with_error(groups[i], groups[j])
    if len(compressors) == 1:
        values = [values]
    counters = [([getattr(o, name)-old for name, old in zip(COUNTERS, previous)], o.max_entries)
                for o, previous in zip(sources, before)]
    return [shell.compress_tile(i, j, raw, tail) for shell, (raw, tail) in zip(compressors, values)], counters


def compressed_tiles(oracle, operators, workers, payload, checkpoint):
    """Yield per-tile compress_tile results, one per operator, in assembly order.

    The parent oracle's query counters are advanced as the workers report them.
    If a worker process dies, the remaining tiles are assembled in this process.
    """
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    groups = operators[0].groups
    tiles = [(i, j) for j in range(len(groups)) for i in range(len(groups))]
    sources = _sources(oracle)
    done = 0
    executor = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
                                   initializer=_initialize, initargs=(payload,))
    try:
        with _without_main_module():
            results = executor.map(_tile, tiles, chunksize=2)
        for compressed, counters in results:
            checkpoint()
            for source, (deltas, largest) in zip(sources, counters):
                for name, delta in zip(COUNTERS, deltas):
                    setattr(source, name, getattr(source, name)+delta)
                source.max_entries = max(source.max_entries, largest)
            done += 1
            yield compressed
    except BrokenProcessPool:
        pass
    finally:
        for process in list(getattr(executor, '_processes', {}).values()):
            if process.is_alive():
                process.terminate()
        executor.shutdown(wait=True, cancel_futures=True)
    for i, j in tiles[done:]:
        checkpoint()
        values = oracle.get_with_error(groups[i], groups[j])
        if len(operators) == 1:
            values = [values]
        yield [operator.compress_tile(i, j, raw, tail) for operator, (raw, tail) in zip(operators, values)]
