"""Frozen-build entry point for the NEURON agent.

Kept separate from agent/agent.py on purpose: if PyInstaller's top-level script were named
`agent`, it would shadow the `agent` PACKAGE and `from agent import resource_guard` would try to
import from the half-initialized entry script (a circular import). A distinct launcher name avoids
that — the real logic still lives in agent/agent.py.
"""
from agent.agent import main

if __name__ == "__main__":
    main()
