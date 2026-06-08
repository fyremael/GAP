import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

class RBFlowDataset(Dataset):
    """
    Dataset class for Rayleigh-Bénard Flow data.
    The dataset contains velocity fields (vx, vy) and temperature field.
    """
    def __init__(self, data_path):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Path to the .npz file containing the data
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")
            
        # Load data
        data = np.load(self.data_path)
        self.vx = data['vx']  # x-component of velocity
        self.vy = data['vy']  # y-component of velocity
        self.temp = data['temp']  # temperature field
        if 'time' in data:
            self.time = data['time']
        else:
            self.time = None
            
        self.n_frames = len(self.vx)
        
        # Store statistics
        self.stats = self._compute_stats()
        
    def _compute_stats(self):
        """Compute statistics of the data."""
        stats = {
            'vx_mean': float(self.vx.mean()),
            'vy_mean': float(self.vy.mean()),
            'temp_mean': float(self.temp.mean()),
            'vx_min': float(self.vx.min()),
            'vx_max': float(self.vx.max()),
            'vy_min': float(self.vy.min()),
            'vy_max': float(self.vy.max()),
            'temp_min': float(self.temp.min()),
            'temp_max': float(self.temp.max())
        }
        return stats
    
    def __len__(self):
        """Return the number of frames in the dataset."""
        return self.n_frames
    
    def __getitem__(self, idx):
        """
        Get a single frame from the dataset.
        
        Args:
            idx (int): Frame index
            
        Returns:
            dict: Dictionary containing:
                - vx: x-component of velocity field
                - vy: y-component of velocity field
                - temp: temperature field
                - time: time value (if available)
        """
        data = {
            'vx': torch.from_numpy(self.vx[idx]).float(),
            'vy': torch.from_numpy(self.vy[idx]).float(),
            'temp': torch.from_numpy(self.temp[idx]).float(),
        }
        
        if self.time is not None:
            data['time'] = torch.tensor(self.time[idx]).float()
            
        return data
    
    def get_sequence(self, start_idx, length):
        """
        Get a sequence of consecutive frames.
        
        Args:
            start_idx (int): Starting frame index
            length (int): Number of frames to return
            
        Returns:
            dict: Dictionary containing sequences of fields
        """
        if start_idx + length > self.n_frames:
            raise ValueError(f"Requested sequence exceeds dataset length. Max start_idx: {self.n_frames - length}")
            
        end_idx = start_idx + length
        data = {
            'vx': torch.from_numpy(self.vx[start_idx:end_idx]).float(),
            'vy': torch.from_numpy(self.vy[start_idx:end_idx]).float(),
            'temp': torch.from_numpy(self.temp[start_idx:end_idx]).float(),
        }
        
        if self.time is not None:
            data['time'] = torch.from_numpy(self.time[start_idx:end_idx]).float()
            
        return data

def load_rb_flow_data(data_path, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a DataLoader for the RB Flow dataset.
    
    Args:
        data_path (str): Path to the .npz file
        batch_size (int): Batch size for the DataLoader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (DataLoader, Dataset)
    """
    dataset = RBFlowDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader, dataset 