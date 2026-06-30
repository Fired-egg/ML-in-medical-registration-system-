# Tutorial Code

## SuperRetina Inference for Registering Two Retinal Images

Before running [tutorial-inference.ipynb](./tutorial-inference.ipynb), make sure the configuration file has been prepared in the `config` directory. A demo configuration file is provided at [test.yaml](../config/test.yaml).

- In the notebooks, `os.chdir("..")` is used to set the working directory to the project root.
- `model_save_path` is the path to the saved SuperRetina model.
- The parameters `model_image_width`, `model_image_height`, `nms_size`, `nms_thresh`, and `knn_thresh` can be adjusted to obtain different registration results. `model_image_width` and `model_image_height` are used to resize the SuperRetina input, and `knn_thresh` is the threshold for the ratio test.

## SuperRetina Inference for Evaluating Registration Performance on FIRE

Similarly, [eval-registration-on-FIRE](./eval-registration-on-FIRE.ipynb) also requires the configuration file [test.yaml](../config/test.yaml).

- Download the FIRE dataset before running the evaluation, then specify the FIRE dataset path in the tutorial file. The dataset can be downloaded from [here](https://projects.ics.forth.gr/cvrl/fire/).
- The interface class `Predictor` is provided in [predictor.py](../predictor.py) to help evaluate SuperRetina. Its usage is shown in the notebook.
