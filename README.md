
# TherNet: Thermal Segmentation Network Harnessing Physical Properties

## News
- **[23/8/2024]** The code of TherNetV1 has been all released, TI-Cityscapes is coming soon!!
- **[9/8/2025]** TherNet: Thermal Segmentation Network Harnessing Physical Properties has been accepted by TPAMI2025!!!


## Installation

This project is built upon the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) framework. We sincerely thank the OpenMMLab team for their remarkable work.

**Important Note:**
This project is based on a **legacy version** of `mmsegmentation`. To ensure proper functionality and reproducibility, we strongly recommend that you **use the code provided in this repository directly** instead of installing the latest version of `mmsegmentation` from its official source.

If you wish to integrate our methods into a newer version of `mmsegmentation`, you may need to manually migrate the core components of our project (e.g., models, data processing pipelines) to adapt to the API changes in the new framework.

### Installation Steps

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/mzzcdf/TherNetV1.git
    cd TherNetV1
    ```

2.  **Create and activate a Conda environment (recommended):**
    ```bash
    conda create -n TherNetV1 python=3.6 -y
    conda activate TherNetV1
    ```

3.  **Install dependencies:**
    ```bash
    # First, install PyTorch. Please adjust the command according to your CUDA version.
    # Install other requirements
    pip install -r requirements.txt
    ```

## Running

This section will guide you on how to train the **TherNetV1** model using this repository.

### 1. Implementation Overview

We provide support for TherNetV1 on the following three datasets:
- **TI-Cityscapes**
- **SODA**
- **SCUT-Seg**

The relevant core code can be found at:
- **Model File**: `main/mmseg/models/decode_heads/thernetv1_head.py` (containing `Thernetv1Head_TIC`, `Thernetv1Head_SODA`, and `Thernetv1Head_SCUT` classes for different datasets).
- **Dataset Definitions**:
  - `main/mmseg/dataset/TIC.py`
  - `main/mmseg/dataset/SODA.py`
  - `main/mmseg/dataset/SCUT-Seg.py`

### 2. Train Your Model (Example on TI-Cityscapes)

We provide a ready-to-use configuration file to help you get started easily.

**Step 1: Prepare the Dataset**
Please download the TI-Cityscapes dataset and organize your directory structure according to the format defined in `main/mmseg/dataset/TIC.py`.

**Step 2: Modify Paths in the Config File**
This is a critical step. Locate our example configuration file:
[`local_config/TherNetV1/TherNetV1.b5.480x480.TIC.160k.py`](https://github.com/mzzcdf/TherNetV1/blob/main/local_config/TherNetV1/TherNetV1.b5.480x480.TIC.160k.py)

Open this file and update the path variables (e.g., `data_root`) to point to your dataset location.

```python
# In the TherNetV1.b5.480x480.TIC.160k.py file, find and modify a section similar to this:
data_root = 'your/path/to/TI-Cityscapes' # <--- CHANGE THIS
```

**Step 3: Start Training**
We have set the configuration file above as the default in the tools/train.py script. Therefore, you can start training by simply running the following command in your terminal:

```bash
python tools/train.py
```

## Running

To run the optimizer, simply use

```shell
python 
```


## Evaluation
```shell
python 
```

## Dataset


## Visual Comparisons




## Data Available
#### 
#### 

## Citation
```

```

