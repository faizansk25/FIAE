"""
FIAE Build Script with Cython Compilation

This script compiles FIAE's core modules to C extensions for:
1. IP protection (source code obfuscation)
2. Performance improvement (10-30% faster execution)
3. Distribution flexibility (pre-compiled wheels)

Usage:
    python setup.py build_ext --inplace    # Build in-place for development
    python setup.py bdist_wheel             # Build wheel for distribution
"""

from setuptools import setup, Extension
import os
import sys

# Core modules to compile with Cython
# These contain the most sensitive algorithms
CORE_EXTENSIONS = [
    # Core funnel logic
    "src/fiae/funnel.py",
    # Feature operators (critical IP)
    "src/fiae/features/dag.py",
    "src/fiae/features/registry.py",
    # Search algorithms
    "src/fiae/search/triggers.py",
    "src/fiae/search/hints.py",
    # Experience memory
    "src/fiae/experience/store.py",
    "src/fiae/experience/retrieval.py",
]

# Build Cython extensions
def get_extensions():
    """Get Cython extension modules.

    Graceful degradation: if Cython or a C compiler is unavailable (pip
    install from sdist on Colab, notebooks, Windows without MSVC, ...),
    return an empty extension list so the wheel builds pure-Python. The
    package is fully functional without the compiled modules -- they are
    an optimization, never a requirement (doc 11 "never crash").
    """
    # Opt-out / opt-in switch.
    if os.environ.get("FIAE_NO_CYTHON"):
        return []
    try:
        import numpy as np
        from Cython.Build import cythonize  # noqa: F401 (availability probe)
    except ImportError:
        print("fiae setup: Cython/numpy unavailable; building pure-Python wheel", file=sys.stderr)
        return []
    # Compiler probe: a missing toolchain must degrade, not fail the build.
    try:
        from setuptools._distutils.ccompiler import new_compiler
        import setuptools._distutils as distutils
        comp = new_compiler()
        _ = distutils.errors.DistutilsPlatformError  # module sanity
        if not comp.has_function("printf", includes=["stdio.h"]):
            raise RuntimeError("no working C compiler")
    except Exception:
        print("fiae setup: no usable C compiler; building pure-Python wheel", file=sys.stderr)
        return []
    extensions = []
    for module_path in CORE_EXTENSIONS:
        if os.path.exists(module_path):
            # Convert path to module name
            # e.g., "src/fiae/funnel.py" -> "fiae.funnel"
            # (strip the leading "src." — otherwise compiled modules land in
            # a dead src.* namespace in wheels and are never imported)
            module_name = module_path.replace("/", ".").replace("\\", ".")[:-3]
            if module_name.startswith("src."):
                module_name = module_name[len("src."):]
            extensions.append(
                Extension(
                    module_name,
                    [module_path],
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-O2"],  # Optimization level
                )
            )
    return extensions


def _maybe_cythonize(extensions):
    """cythonize when extensions exist; otherwise return [] (pure-Python)."""
    if not extensions:
        return []
    from Cython.Build import cythonize
    return cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            # Annotations here are documentation; pure Python never enforces
            # them. Cython 3 defaults to annotation_typing=True, which turned
            # e.g. `candidate_values: list` into a hard C type check that
            # rejects numpy arrays the pure-Python function accepts (caught
            # by the verify-wheel dry-run). Treat annotations as objects.
            "annotation_typing": False,
            # Safety over speed: keep bounds/wraparound checks (an index bug
            # must raise, not segfault) and Python division semantics
            # (cdivision=True silently changed floor-div/mod behavior).
            "boundscheck": True,
            "wraparound": True,
            "cdivision": False,
        },
    )


setup(
    name="fiae",
    ext_modules=_maybe_cythonize(get_extensions()),
)
