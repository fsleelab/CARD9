# CARD9

Behavioral analysis code related to Lin, Kim et al., *Science*, 2026, on CARD9.

## Installation

1. Download this repository with `git clone --recurse-submodules https://github.com/fsleelab/CARD9.git`
2. Navigate into the downloaded repository and, using [Anaconda](https://www.anaconda.com/download), install the required packages with `conda env create -n card9 --file environment.yml`
3. Navigate into the `DeepLabCut`, `anipose`, and `camera-sync` submodules and follow their installation instructions, creating a separate conda environment for each.

### Additional requirements

1. Dataset and model files should be downloaded and placed in the corresponding locations:
    - `basic_behavior_analysis/models`: `aprl_model.pth`, `card9_gui_4.pickle`, `refined_initial_proj_data.pickle`
    - `fine_behaviors_analysis/models`: `more_test_params_model.sav`, `more_test_params_hdbscan_model.sav`
    - Repository root: the `SA_track-nate-2023-07-20` DeepLabCut model folder
2. Open `config.toml` in the repository root and set `model_folder` to the full path of your `SA_track-nate-2023-07-20` folder. This file is copied into every project folder created below, so edit it before creating projects.

## Input video requirements

1. To proceed with our DeepLabCut/Anipose 3D tracking pipeline, you will need:
    - Four matching video files, named like `<date>_<subject-ID>_cam<n>.avi`, for each recording trial.
    - Four matching calibration video files, named like `<date>_calibration_cam<n>.avi`, for each recording session (generally a single day). Essentially, any time the cameras are moved, you will need to record a new set of calibration videos. More information about these calibration videos can be found in the [Anipose documentation](https://anipose.readthedocs.io/en/latest/start3d.html).
2. All video recordings must begin with a visual cue, such as the pulsing of an LED. This cue allows us to synchronize the captured videos and extract tracking data across multiple views, to be later triangulated to points in 3D space.

## Video preprocessing

1. Navigate to the `camera-sync` directory and activate its conda environment.
2. Follow its instructions to sync all sets of videos.
3. Synced trial recordings (i.e., non-calibration videos) should then be moved to a single folder in preparation for DeepLabCut tracking.
4. Activate the DeepLabCut conda environment and run the following:

```python
import deeplabcut

config = '/path/to/SA_track-nate-2023-07-20/config.yaml'
vids = '/path/to/synced_vids'
deeplabcut.analyze_videos(config, vids, shuffle=77, videotype='avi')
deeplabcut.filterpredictions(config, vids, shuffle=77, videotype='avi', filtertype='median', windowlength=3)
```

### Project structure setup

1. You should now have a single folder containing synced videos and corresponding DeepLabCut-generated tracking files, which should look like:

```
<date>_subject1_cam<n>-frame_synced.avi
<date>_subject1_cam<n>-frame_syncedDLC_resnet50_inOFcamsApr5shuffle77_450000_filtered.csv
<date>_subject1_cam<n>-frame_syncedDLC_resnet50_inOFcamsApr5shuffle77_450000_filtered.h5
<date>_subject1_cam<n>-frame_syncedDLC_resnet50_inOFcamsApr5shuffle77_450000.h5
<date>_subject1_cam<n>-frame_syncedDLC_resnet50_inOFcamsApr5shuffle77_450000.pickle
...
<date>_subject3_cam<n>-frame_synced.avi
<date>_subject3_cam<n>-frame_syncedDLC_resnet50_inOFcamsApr5shuffle77_450000_filtered.csv
<date>_subject3_cam<n>-frame_syncedDLC_resnet50_inOFcamsApr5shuffle77_450000_filtered.h5
<date>_subject3_cam<n>-frame_syncedDLC_resnet50_inOFcamsApr5shuffle77_450000.h5
<date>_subject3_cam<n>-frame_syncedDLC_resnet50_inOFcamsApr5shuffle77_450000.pickle
```

2. Now you can automatically create project folders in the Anipose structure, as long as the file naming scheme described above was followed. From the repository root, with the `card9` environment activated, run:

```bash
python cli_project_creator.py --source '/path/to/synced_vids' --project '/path/to/projects'
```

Both paths must be full paths, and the `--project` directory must already exist. The script prints the structure it intends to create and the files it intends to copy, and waits for you to confirm each step. The output structure should look like:

```
-projects
    -date1
        config.toml
        -subject1
            -calibration
            -pose-2d
        -subject2
            -calibration
            -pose-2d
    -date2
        config.toml
        -subject1
            -calibration
            -pose-2d
        -subject2
            -calibration
            -pose-2d
```

3. DeepLabCut tracking files are copied into the created `pose-2d` folders. However, you will still need to manually copy the synced calibration videos into the `calibration` directory of one project folder for that recording date/session.
4. Finally, navigate into the date folder (the one containing `config.toml`), activate the anipose conda environment, and run `anipose calibrate`. The resulting `calibration.toml` and `detections.pickle` files that appear in the `calibration` folder can be copied to all other project folders that used the same camera setup.

### 3D tracking

1. With the anipose conda environment activated, you can now perform triangulation by running `python anipose_processing_pipeline.py --project '/path/to/projects/date'`. This will produce a 3D tracking file in a newly created `pose-3d` directory for each subject in the selected date directory.

## Basic behavior analysis (face grooming, body grooming, rearing)

Run these from the repository root with the `card9` conda environment activated.

1. If you want to train a multi-layer perceptron from scratch using the downloaded dataset, run:

```bash
python -m basic_behavior_analysis.train_mlp --dataset 'basic_behavior_analysis/models/refined_initial_proj_data.pickle'
```

2. To get class predictions for all subjects in a date folder, run:

```bash
python -m basic_behavior_analysis.cli_transform_predict --project '/path/to/projects/date' --mlp_path 'basic_behavior_analysis/models/aprl_model.pth'
```

Predictions are written as one `.csv` per subject into the directory given by `--output_dir` (default `output`, which must already exist). Add `--savefig` to also save a summary figure of the predicted labels. Replace `--mlp_path` with a model trained from scratch from step 1.

## Fine behavior analysis (face grooming, body grooming, rearing, twitching, hindpaw grooming, scratching)

Run these from the repository root with the `card9` conda environment activated.

1. If you have the annotated and unannotated project folders, you can fit and save a new UMAP reducer and HDBSCAN cluster model with:

```bash
python -m fine_behaviors_analysis.final_train_models --labeled_data_dir /path/to/all/data --save_model
```

The `--save_model` flag is required for the fitted models to be written to disk. Models, figures, and the annotated embeddings `.csv` are saved to `fine_behaviors_analysis/models` (change this with `--output_dir`). Output files are named after the parameter file passed via `--param_file` (default `fine_behaviors_analysis/models/more_test_params.txt`), so the embeddings file will be `more_test_params_to_cluster.csv`. Note that saving will fail if a model of the same name already exists.

2. You will then need to use those annotated embeddings to build a map from HDBSCAN labels to behavior names:

```bash
python -m fine_behaviors_analysis.get_behavior_hdbscanLabels --embed_dataset 'fine_behaviors_analysis/models/more_test_params_to_cluster.csv'
```

Change the `.csv` path to match the file exported by the previous step. After running this step, you will find the label mappings at `fine_behaviors_analysis/models/behavior_labels.json`.

3. To get class predictions for all subjects in a date folder without having run the previous steps, use our precomputed models and mapping:

```bash
python -m fine_behaviors_analysis.final_predict_models --root_dir /path/to/projects --projects date
```

`--projects` accepts multiple folder names as a single space-separated quoted string (e.g. `--projects 'date1 date2'`) or as the path to a `.json` file listing them. Predictions are written as `revised_cluster_labels.csv` alongside each subject's 3D tracking data. To use your own models and mapping, pass their locations with `--umap_path`, `--hdb_path`, and `--label_file`, respectively.
