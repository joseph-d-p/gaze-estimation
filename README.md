# gaze-estimation
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) ![PyPI](https://img.shields.io/pypi/v/opencv-python?label=opencv) ![GitHub](https://img.shields.io/github/license/joseph-d-p/gaze-estimation) 



Estimate person gaze based on head orientation and feature landmarks.

## Prerequisites

1. [OpenVINO 2020.3 or later](https://docs.openvinotoolkit.org/latest/index.html)

2. [Python 3.7](https://www.python.org/downloads/release/python-370/) or later

## Setup

1. Create virtual environment

```
$ python3 -m venv env
$ source env/bin/activate
```

2. Install packages

```
$ pip3 -r install requirements.txt
```

3. Download models

    ```bash
    # Set model downloader path
    export MODEL_DOWNLOADER_PATH=/opt/intel/openvino_2020.3.194/deployment_tools/tools/model_downloader
    ```

    3.1 **Face detection model**

    ```
    $> $MODEL_DOWNLOADER_PATH/downloader.py \
            --name face-detection-adas-0001 \
            --precisions FP32 \
            -o models
    ```
    
### Running

```
$ python main.py
```

### Output
TODO
