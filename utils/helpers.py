from typing import Tuple
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from pathlib import Path
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV

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

        # NumPy default settings
        np.random.seed(100)

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


    def create_distribution_plot(
            self, 
            df: pd.DataFrame, 
            column: str, 
            bins: int = 30, 
            hue: str = None
        ) -> None:
        """Create a distribution plot using seaborn."""

        plt.figure(figsize=(10, 6)) # Wider figure for better visibility
        sns.histplot(data=df, x=column, hue=hue, bins=bins, kde=True, stat='density')

        # compute skew
        skew_val = df[column].skew()

        # add skew text
        plt.text(
            0.99, 0.88,                      # normalized location (right top)
            f"Skewness = {skew_val:.2f}",    # text
            transform=plt.gca().transAxes,   # coordinates relative to axes
            ha='right', va='top',
            fontsize=12, color="black",
            bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'alpha': 0.7}
        )

        line_mean = plt.axvline(df[column].mean(), color="darkred", linestyle="--")
        line_median = plt.axvline(df[column].median(), color="darkgreen", linestyle="--")

        plt.legend(
            handles=[line_mean, line_median],
            labels=["Mean", "Median"],
            loc="upper right"
        )

        plt.title(f'Distribution of {column}')
        plt.tight_layout() # Prevent clipping
        plt.show()


    def create_correlation_heatmap(
            self, 
            df: pd.DataFrame, 
            method: str = 'pearson', 
            annot: bool = True, 
            cmap: str = 'magma'
        ) -> None:
        """Create a correlation heatmap using seaborn."""

        plt.figure(figsize=(12, 10)) # Larger figure for better visibility
        corr = df.corr(method=method)
        sns.heatmap(corr, annot=annot, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, center=0, square=True, cbar_kws={"shrink": .8})
        
        plt.title(f'Correlation Heatmap ({method.capitalize()} method)')
        plt.tight_layout() # Prevent clipping
        plt.show()


    def power_transform(
            self,
            df: pd.DataFrame,
            columns: list,
            method: str = 'yeo-johnson'
        ) -> Tuple[pd.DataFrame, PowerTransformer]:
        """Apply power transformation to specified columns in the DataFrame."""
        
        pt = PowerTransformer(method=method)
        df_out = df.copy()
        # fit_transform expects 2D array
        df_out.loc[:, columns] = pt.fit_transform(df_out[columns].values)
        return df_out, pt
    

    def reverse_power_transform(
            self,
            df: pd.DataFrame,
            columns: list,
            pt: PowerTransformer,
        ) -> pd.DataFrame:
        """Reverse power transformation on specified columns in the DataFrame."""
        
        df_out = df.copy()
        df_out.loc[:, columns] = pt.inverse_transform(df_out[columns].values)
        return df_out
    

    def find_best_params(
            self,
            model: object,
            search_space: dict,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            metric: str,
            iterations: int = 50,
            random_state: int = 42,
        ) -> BayesSearchCV:
        """Find best hyperparameters using Bayesian Optimization."""

        opt = BayesSearchCV(
            estimator=model,
            search_spaces=search_space,
            n_iter=iterations,
            cv=StratifiedKFold(n_splits=3, random_state=random_state, shuffle=True),
            scoring=metric,
            n_points=2,
            pre_dispatch='2*n_jobs',
            random_state=random_state
        )

        opt.fit(X_train, y_train)
        print("Best Parameters:")
        pprint(dict(opt.best_params_))

        return opt.best_estimator_
