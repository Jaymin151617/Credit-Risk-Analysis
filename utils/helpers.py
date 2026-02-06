from typing import Tuple, Optional
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
from shap import Explanation
from lime.lime_tabular import LimeTabularExplainer

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


    def create_shap_waterfall(
            self,
            shap_object: Explanation,
            index: int
        ) -> None:
        """
        Create and display a horizontal waterfall plot for a single SHAP explanation.
        Shifts SHAP contributions so baseline is at 0 and converts to percent.
        """
        # --- Gather and prepare data ---
        features = list(shap_object[index].feature_names)
        vals = np.array(shap_object[index].values, dtype=float).flatten()
        base = float(shap_object[index].base_values)

        # Shift each contribution so base is effectively 0 and convert to percent
        shifted_shap = (vals + base / len(features)) * 100.0  # percentage-scale contributions
        total_pred = sum(shifted_shap)

        # Order features by absolute contribution (largest magnitude first)
        order = np.argsort(np.abs(shifted_shap))[::-1]
        feat_ordered = [features[i] for i in order]
        contribs = shifted_shap[order]

        # --- plotting setup ---
        n = len(contribs)
        fig_w = 10
        fig_h = max(3, n * 0.45)
        _, ax = plt.subplots(figsize=(fig_w, fig_h))

        y = np.arange(n)

        # sns.color_palette() reads the active theme/palette
        palette = sns.color_palette()
        pos_color, neg_color = palette[0], palette[1]

        # determine x limits and padding
        x_min = min(contribs.min(), 0.0)
        x_max = max(contribs.max(), 0.0)
        x_range = x_max - x_min if (x_max - x_min) != 0 else 1.0
        pad = x_range * 0.03
        ax.set_xlim(x_min - pad * 4, x_max + pad * 4)

        # assign per-bar color by sign (but all positives same, all negatives same)
        colors = [pos_color if v >= 0 else neg_color for v in contribs]

        # Draw bars starting at 0 so negatives extend left automatically
        ax.barh(y, contribs, left=0, height=0.6, align='center',
                color=colors, edgecolor='k', linewidth=0.3)

        # Baseline at 0
        ax.axvline(0.0, color='black')

        # Add value labels next to each bar (rounded to 2 decimals, signed)
        for i, val in enumerate(contribs):
            label = f"{val:+.2f}%"
            if val >= 0:
                text_x = val + pad
                ha = 'left'
            else:
                text_x = val - pad
                ha = 'right'
            ax.text(text_x, y[i], label, va='center', ha=ha, color=colors[i], fontsize=9, fontweight='bold')

        pred_text = f"Chance: {total_pred:.2f}%"
        ax.annotate(
            pred_text, xy=(1.1, 0.02), xycoords='axes fraction',
            ha='right', va='bottom', fontsize=12, fontweight='bold',
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "none"}
        )

        # Y labels and formatting
        ax.set_yticks(y)
        ax.set_yticklabels(feat_ordered, fontsize=10)
        ax.invert_yaxis()  # largest on top
        ax.set_xlabel("Contribution to prediction (%)", fontsize=11)
        ax.grid(axis='x', linestyle=':', linewidth=0.6, alpha=0.7)

        plt.title("SHAP contributions", fontsize=12)
        plt.tight_layout()
        plt.show()


    def get_recommended_values(
            self,
            data,
            model,
            lime_explainer: LimeTabularExplainer,
            feature_idx: int,
            prob_threshold: float = 0.75,
            min_limit: float = -2.0,
            max_limit: float = 2.0,
            window: float = 1.0,
            step: float = 0.1
        ) -> Optional[float]:
        """Get recommended values for a particular feature to improve the probability of loan approval."""

        exp = lime_explainer.explain_instance(data_row=data, predict_fn=model.predict_proba)
        
        slope = None
        for idx, weight in exp.local_exp[1]:
            if idx == feature_idx:
                slope = float(weight)
                break

        if slope is None:
            slope = 0.0     # feature not in the local explanation (weight implicitly zero)

        if abs(slope) < 1e-12:
            print("Slope zero")
            # If slope is effectively zero, the local linear model says changing this
            # feature won't change predicted probability
            return None
        
        intercept = float(exp.intercept[1])

        # intercept + slope * feature_value = prob_threshold  => feature_value = (prob_threshold - intercept) / slope
        center_value = (prob_threshold - intercept) / slope

        start = max(min_limit, center_value - window)
        end = min(max_limit, center_value + window)

        values = []
        v = start
        while v <= end + 1e-9:
            values.append(round(v, 2))
            v += step

        passing = []
        for val in values:
            row = data.copy()
            row[feature_idx] = val
            prob = float(model.predict_proba(row.reshape(1, -1))[0, 1])
            if prob >= prob_threshold:
                passing.append(val)

        if not passing:
            return None

        return min(passing)