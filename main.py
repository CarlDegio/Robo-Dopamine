import os
from pathlib import Path

CUDA_DEVICE_ID = "0"
MODEL_PATH = "../ckpt/Robo-Dopamine-GRM-2.0-4B"
HTTP_PROXY = "http://127.0.0.1:7890"
HTTPS_PROXY = "http://127.0.0.1:7890"

os.environ["LOCAL_RANK"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE_ID
os.environ["http_proxy"] = HTTP_PROXY
os.environ["https_proxy"] = HTTPS_PROXY
os.environ["HTTP_PROXY"] = HTTP_PROXY
os.environ["HTTPS_PROXY"] = HTTPS_PROXY

from examples.inference import GRMInference

# examples.inference currently sets a default CUDA_VISIBLE_DEVICES during import.
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE_ID

model_path = Path(MODEL_PATH).expanduser()
if not model_path.is_absolute():
    candidates = [
        Path.cwd() / model_path,
        Path(__file__).resolve().parent / model_path,
    ]
    model_path = next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])

if not model_path.is_dir():
    raise FileNotFoundError(
        f"Local model path does not exist: {model_path}. "
        "Use an existing local checkpoint directory to avoid Hugging Face repo lookup."
    )

model = GRMInference(str(model_path))

TASK_INSTRUCTION = "Walk to the white table, pick up the gray cushion, place it in the white moving box, then return to the green sofa."
BASE_DEMO_PATH = "./examples/g1_demo"
OUTPUT_ROOT = "./results"

## Note: If no reference/goal image is provided, 
## please replace `GOAL_IMAGE_PATH` with the blank image "./examples/blank_goal.png".
GOAL_IMAGE_PATH = "./examples/demo_table/goal_image.png" # "./examples/blank_goal.png"

# select prediction model: Forward-Mode, Incremental-Mode or Backward-Mode
PREDICTION_MODE = "forward" # "incremental" or "backward"

# multi-view usage:
output_dir = model.run_pipeline(
    cam_high_path  = os.path.join(BASE_DEMO_PATH, "cam_high.mp4"),
    cam_left_path  = os.path.join(BASE_DEMO_PATH, "cam_left_wrist.mp4"),
    cam_right_path = os.path.join(BASE_DEMO_PATH, "cam_right_wrist.mp4"),
    out_root       = OUTPUT_ROOT,
    task           = TASK_INSTRUCTION,
    frame_interval = 10, # modify frame_interval as desired, but it shouldn't be set too small if using 'incremental'.
    batch_size     = 1, # please increase batch_size > 1, if you have enough GPU memory.
    goal_image     = GOAL_IMAGE_PATH,
    eval_mode      = PREDICTION_MODE,
    visualize      = True
)
print(f"Episode ({BASE_DEMO_PATH}) processed with multi-view {PREDICTION_MODE}-mode. Output at: {output_dir}")

# single-view usage:
# output_dir = model.run_pipeline(
#     cam_high_path  = os.path.join(BASE_DEMO_PATH, "cam_high.mp4"),
#     cam_left_path  = os.path.join(BASE_DEMO_PATH, "cam_high.mp4"), # repeat cam_high
#     cam_right_path = os.path.join(BASE_DEMO_PATH, "cam_high.mp4"), # repeat cam_high
#     out_root       = OUTPUT_ROOT,
#     task           = TASK_INSTRUCTION,
#     frame_interval = 10, # modify frame_interval as desired, but it shouldn't be set too small if using 'incremental'.
#     batch_size     = 1, # please increase batch_size > 1, if you have enough GPU memory.
#     goal_image     = GOAL_IMAGE_PATH,
#     eval_mode      = PREDICTION_MODE,
#     visualize      = True
# )
# print(f"Episode ({BASE_DEMO_PATH}) processed with single-view {PREDICTION_MODE}-mode. Output at: {output_dir}")
