"""Render the advanced MuJoCo scene without opening an interactive viewer."""

from pathlib import Path

import mujoco
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "medical_scene_advanced.xml"
OUTPUT = ROOT / "docs" / "scene-preview.png"


def main() -> None:
    # Reading the XML ourselves also works when the checkout path contains CJK characters.
    model = mujoco.MjModel.from_xml_string(SCENE.read_text(encoding="utf-8"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.azimuth = 135
    camera.elevation = -24
    camera.distance = 4.2
    camera.lookat[:] = [0.35, 0.1, 0.55]

    renderer = mujoco.Renderer(model, height=900, width=1440)
    try:
        renderer.update_scene(data, camera=camera)
        frame = renderer.render()
    finally:
        renderer.close()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(OUTPUT)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
