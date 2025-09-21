import os.path

SEED = 42

def set_track(new_track="sub01_1"):
    global track 
    global VID_FOLDER
    global FMRI_FOLDER
    global OUTPUT_FOLDER
    global ROIs

    track = new_track
    data_clip = "sub03"

    OUTPUT_FOLDER = f"{PROJECT_FOLDER}output/"
    FMRI_FOLDER = f"{DATA_FOLDER}fmridata/{track}/"
    VID_FOLDER = f"{DATA_FOLDER}features/InterVideo/InternVideo-MM-B-16-768/{data_clip}/"
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # ROIs = ["FFA-1", "FFA-2", "LO", "OFA", "V1", "V2", "V3", "V4"]
    ROIs = ["FFA", "IPS", "LO", "MT", "OFA", "pSTS", "TPJ", "V1", "V2", "V3", "V4", "V3A", "V3B"]

# project settings
track = None
train_data_len = 1200
test_data_len = 120
num_subs = 3
subs = ["sub" + str(s + 1).zfill(2) for s in range(num_subs)]


# environment settings
PROJECT_FOLDER = "E:/LLM/fMRI-video-encoding-2023-lhy/"
DATA_FOLDER = "E:/LLM/Algonauts_2023/"
OUTPUT_FOLDER = "output"

FMRI_FOLDER = None
VID_FOLDER = None
ROIs = None
set_track("sub01_1")

COMPRESSION = True
autosklearn_config={"time_left_for_this_task": 600,   # train for 3 minutes
                    "per_run_time_limit": 100,      # the limit ti e for a single machine learning model
                    "ensemble_size": 10,       # the size of the ensemble built of the best models
                                         "ensemble_nbest": 10,         # only consider the 10 best models for building the ensemble
                                         "memory_limit": 102400}          # 10GB RAM limit

log_dir = f"logs/"


