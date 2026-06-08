---
license: mit
language:
- en
tags:
- physics
- PDE
- multi-physics
---
# Rayleigh-Bénard Convection Dataset

<div style="display: flex; gap: 10px;">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/661e07e02a8496916011c08a/n2cikSEomskDg5dED8u3Q.gif" width="45%">
  <img src="https://huggingface.co/datasets/ashiq24/Rayleigh_Benard_Convection/resolve/main/visualization/velocity_animation.gif" width="45%">
</div>


<!-- ![image/gif](https://cdn-uploads.huggingface.co/production/uploads/661e07e02a8496916011c08a/n2cikSEomskDg5dED8u3Q.gif) ![image/gif](https://cdn-lfs-us-1.hf.co/repos/48/2f/482f28b6cd923cd1162b531ea6775eaf0176abcc079954204abfc12e71406309/d59ee75d425fac8457969c0e114a4d695191be0e37f8b312f2be5ded92f126ca?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27velocity_animation.gif%3B+filename%3D%22velocity_animation.gif%22%3B&response-content-type=image%2Fgif&Expires=1749023394&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0OTAyMzM5NH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zLzQ4LzJmLzQ4MmYyOGI2Y2Q5MjNjZDExNjJiNTMxZWE2Nzc1ZWFmMDE3NmFiY2MwNzk5NTQyMDRhYmZjMTJlNzE0MDYzMDkvZDU5ZWU3NWQ0MjVmYWM4NDU3OTY5YzBlMTE0YTRkNjk1MTkxYmUwZTM3ZjhiMzEyZjJiZTVkZWQ5MmYxMjZjYT9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoifV19&Signature=cQF-IevrgtyA8OJyQQjjdlLhry03EpO2nOyIP5mttH18i8SqMSzlEQufm9HVOqZlPiLyb8-wKjvba2fQZJpZrOh1Hajui8iNG7PuijVxQXp6jN0hnF3tG2xaqDlcz2IIB5bfPkiQ12aqK7V54rBik8idkMRDzRTC7u4aCMzfvWy33ctyq-4Pq5aKWJbWmHmhLhqKjO9dJnGNz-rslAKA-PtgqynSuYOpFB9ncAhaL2BIFdOojL366YpP79aarutPcHG2Dv2iEWtXnwJbFu2tKv%7EZfJRm40jqIsecC0I0aq%7Ew8Hi2wle3Lu8flAgD%7EwWm3YWYbtHQtBTfEZZhGKtyEQ__&Key-Pair-Id=K24J24Z295AEI9) -->

This dataset contains data for Rayleigh-Bénard convection at different Rayleigh numbers. The data includes velocity fields and temperature distributions on a 128×128 grid.

## Dataset Description

The dataset consists of .npz files containing:
- `vx`: x-component of velocity field
- `vy`: y-component of velocity field
- `temp`: temperature field
- `time`: time points


### Sample Data Stats

<img src="https://cdn-uploads.huggingface.co/production/uploads/661e07e02a8496916011c08a/fyVw22n4JTL9c3gFEj4FQ.png" width="800">


## Installation and Download

```bash
pip install -r requirements.txt
```

Download the dataset

```python
from huggingface_hub import hf_hub_download
import os
os.makedirs("./rb_data", exist_ok=True)

# Download Ra=12,000 dataset
hf_hub_download(
    repo_id="ashiq24/Rayleigh_Benard_Convection",
    filename="data/data_12e3.npz",
    local_dir="./rb_data",
    repo_type="dataset"
)
```

## Dataset Loader Details

The dataset loader (`dataloader.py`) provides two main classes:

1. `RBFlowDataset`: A PyTorch Dataset class that handles:
   - Loading and preprocessing of .npz files
   - Single frame access
   - Sequence extraction
   - Statistical information

2. `load_rb_flow_data`: A utility function that creates a DataLoader with:
   - Batch processing
   - Data shuffling
   - Multi-worker support

### Accessing Data Statistics

```python
dataset = RBFlowDataset('data/data_12e3.npz')
stats = dataset.stats  # Dictionary containing field statistics
print(f"VX range: {stats['vx_min']} to {stats['vx_max']}")
print(f"Temperature mean: {stats['temp_mean']}")
```

### Basic Usage

```python
from dataloader import load_rb_flow_data

# Load the dataset
dataloader, dataset = load_rb_flow_data(
    data_path='data/data_12e3.npz',  # or data_20e3.npz
    batch_size=32,
    shuffle=True
)

# Iterate through batches
for batch in dataloader:
    vx = batch['vx']  # Shape: [batch_size, nx, ny]
    vy = batch['vy']
    temp = batch['temp']
    # Your processing here
```

### Loading Sequences

```python
from dataloader import RBFlowDataset

# Initialize dataset
dataset = RBFlowDataset('data/data_12e3.npz')

# Get a sequence of frames
sequence = dataset.get_sequence(start_idx=0, length=10)
vx_sequence = sequence['vx']  # Shape: [10, nx, ny]
```
## Visualization Tools

The repository includes tools for creating various visualizations of the flow fields.

### Creating Animations

```python
from visualize import RBFlowVisualizer

# Initialize visualizer
viz = RBFlowVisualizer('data/data_12e3.npz')

# Create velocity field animation
viz.create_velocity_animation(
    output_path='velocity_animation.gif',
    fps=30,
    skip=3  # Arrow density (smaller = denser)
)

# Create temperature field animation
viz.create_animation('temp', 'temperature_animation.gif', fps=30)
```

## Testing

The repository includes a comprehensive test suite (`test_functionality.py`) that verifies all functionality:

```bash
python test_functionality.py
```

The test suite checks:
1. Dataset loading and access
   - Basic loading functionality
   - Frame access
   - Sequence extraction
   - Data shapes and types

2. Data Processing
   - Normalization
   - Statistics computation
   - Batch processing
   - Sequence bounds checking

3. Visualization
   - Temperature field animations
   - Velocity field animations
   - File generation and saving

### Running Individual Tests

```python
import unittest
from test_functionality import TestRBFlowTools

# Run specific test
suite = unittest.TestLoader().loadTestsFromName('test_dataset_loading_12k', TestRBFlowTools)
unittest.TextTestRunner(verbosity=2).run(suite)
```




## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@article{rahman2024pretraining,
  title={Pretraining Codomain Attention Neural Operators for Solving Multiphysics PDEs},
  author={Rahman, Md Ashiqur and George, Robert Joseph and Elleithy, Mogab and Leibovici, Daniel and Li, Zongyi and Bonev, Boris and White, Colin and Berner, Julius and Yeh, Raymond A and Kossaifi, Jean and Azizzadenesheli, Kamyar and Anandkumar, Anima},
  journal={Advances in Neural Information Processing Systems},
  volume={37}
  year={2024}
}

```