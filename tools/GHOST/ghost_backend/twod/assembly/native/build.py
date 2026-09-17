"""Build the optional table evaluator and far-block quadrature with a C99 compiler."""
import os
from pathlib import Path
import subprocess
import shutil


def build():
    root = Path(__file__).resolve().parent
    return [build_one(root, name) for name in ('table', 'far')]


def build_one(root, name):
    output = root / ('ghost_{}.dll'.format(name) if os.name == 'nt' else 'libghost_{}.so'.format(name))
    compiler = os.environ.get('CC', 'gcc')
    # No fast-math: preserve the validated double-precision interpolation.
    flags = ['-O3', '-std=c99', '-ffp-contract=off', '-shared']
    flags += ['-static-libgcc'] if os.name == 'nt' else ['-fPIC']
    resolved = shutil.which(compiler)
    if resolved is None:
        raise RuntimeError('No C compiler is available; the Python evaluator remains usable.')
    env = dict(os.environ, PATH=str(Path(resolved).resolve().parent)+os.pathsep+os.environ.get('PATH',''))
    startup = None
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
    subprocess.run([resolved, *flags, str(root/(name + '.c')), '-o', str(output)],
                   env=env, startupinfo=startup, check=True)
    return output


if __name__ == '__main__':
    print(build())
