"""Container-lane rig: assets, the host-side driver, and the in-container entrypoint.

Nothing in here is skill runtime. The package exists so the container-marked test
modules can import the shared Docker probe (:mod:`docker_probe`) rather than each
growing its own copy of the skip logic.
"""
