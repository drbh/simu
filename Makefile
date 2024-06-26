apply-patch:
	patch .venv/lib/python3.11/site-packages/gymnasium/envs/mujoco/mujoco_rendering.py < mujoco.patch
