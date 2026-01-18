import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class Helpers:

    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[1]
        self.property_file_path = self.root_dir / 'config' / 'properties.yaml'


    def load_properties(self) -> dict:
        """Load properties from a YAML configuration file."""
        try:
            with open(self.property_file_path, 'r') as file:
                properties = yaml.safe_load(file)

            if not isinstance(properties, dict):
                raise ValueError("properties.yaml is empty or invalid")
   
            return properties
        
        except FileNotFoundError as fnfe:
            raise FileNotFoundError(f"File not found at path: {self.property_file_path}") from fnfe
        
        except yaml.YAMLError as ye:
            raise ValueError(f"Error parsing YAML file: {str(ye)}") from ye
        

    def set_global_settings(self) -> None:
        """Set global setting for common libraries."""

        # Ingore warnings
        warnings.filterwarnings('ignore')

        # Pandas default settings
        pd.options.display.float_format = '{:.2f}'.format

        # Matplotlib style settings
        plt.rcParams.update({
            "figure.dpi": 200,
            "figure.edgecolor": "black",
            "figure.frameon": True
        })

        # Seaborn style settings

        custom_rc = {                  # Customize additional rc params
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "legend.loc": "best",
            "axes.titlesize": 14
        }

        sns.set_theme(
            context = "paper",         # Context: paper, notebook, talk, poster
            style = "darkgrid",        # Background style: whitegrid, darkgrid, white, ticks, dark
            palette = "deep",          # Color palette: deep, muted, pastel, bright, colorblind, etc.
            font = "sans-serif",       # Font family
            rc = custom_rc             # Custom rc parameters
        )


    def create_bar_chart(
            self, 
            df: pd.DataFrame, 
            x_axis: str, 
            hue: str = None, 
            rotate_labels: bool = False
        ) -> None:
        """Create a bar chart using seaborn."""

        plt.figure(figsize=(10, 6)) # Wider figure for better visibility
        ax = sns.countplot(data=df, x=x_axis, hue=hue)
        
        if rotate_labels:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.title(f'Count of {x_axis} by {hue}' if hue else f'Count of {x_axis}')
        plt.tight_layout() # Prevent clipping
        plt.show()

    def create_boxplot(
            self, 
            df: pd.DataFrame, 
            x_axis: str, 
            y_axis: str, 
            hue: str = None, 
            rotate_labels: bool = False
        ) -> None:
        """Create a boxplot using seaborn."""

        plt.figure(figsize=(10, 6)) # Wider figure for better visibility
        ax = sns.boxplot(data=df, x=x_axis, y=y_axis, hue=hue, notch=True, fliersize=3)
        
        if rotate_labels:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.title(f'Boxplot of {y_axis} by {x_axis}')
        plt.tight_layout() # Prevent clipping
        plt.show()

    
    def create_scatterplot(
            self, 
            df: pd.DataFrame, 
            x_axis: str, 
            y_axis: str, 
            hue: str = None
        ) -> None:
        """Create a scatter plot using seaborn."""

        plt.figure(figsize=(10, 6)) # Wider figure for better visibility
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=hue, alpha=0.7)
        
        plt.title(f'Scatterplot of {y_axis} vs {x_axis}')
        plt.tight_layout() # Prevent clipping
        plt.show()