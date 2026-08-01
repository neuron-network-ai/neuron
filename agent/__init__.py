"""NEURON agent package.

This file is load-bearing, not boilerplate. `agent/agent.py` is a module with the SAME
name as its package, and a fresh install runs it as a script (`python agent/agent.py`,
which is what INSTALL.md and install.py's systemd unit both do). Python then puts
`<repo>/agent` on sys.path ahead of everything the script adds itself, and a regular
module named `agent` anywhere on the path beats a namespace-package directory named
`agent` — so `from agent import local_chat` re-imported agent.py into itself:

    ImportError: cannot import name 'local_chat' from partially initialized module
    'agent' (most likely due to a circular import)

Making this a REGULAR package resolves `agent` to the directory at sys.path[0] instead.
The two live nodes never hit it because both had an `__init__.py` created by hand during
setup; a stranger cloning the repo hit it on the first command they ran.
"""
