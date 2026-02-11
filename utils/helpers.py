# ---------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------
from typing import Tuple, Optional  # Type hinting for clearer APIs
import warnings                     # Warning control for cleaner outputs
from pprint import pprint           # Structured pretty-printing (debugging)


# ---------------------------------------------------------------------
# Third-Party Scientific Computing Libraries
# ---------------------------------------------------------------------
import numpy as np                  # Numerical computing (arrays, math ops)
import pandas as pd                 # DataFrame-based data manipulation


# ---------------------------------------------------------------------
# Visualization Libraries
# ---------------------------------------------------------------------
import matplotlib.pyplot as plt     # Base plotting library
import seaborn as sns               # Statistical visualization (built on matplotlib)


# ---------------------------------------------------------------------
# Machine Learning - Preprocessing & Validation
# ---------------------------------------------------------------------
from sklearn.preprocessing import PowerTransformer
# Used to stabilize variance and make data more Gaussian-like
# Helpful for skewed distributions

from sklearn.model_selection import StratifiedKFold
# Cross-validation strategy that preserves class distribution
# Important for imbalanced classification problems

from sklearn.base import BaseEstimator
# Base class for all estimators in sklearn

# ---------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------
from skopt import BayesSearchCV
# Bayesian optimization for hyperparameter tuning
# More efficient than GridSearchCV for large search spaces


# ---------------------------------------------------------------------
# Model Explainability
# ---------------------------------------------------------------------
from shap import Explanation
# SHAP Explanation object used to interpret model predictions
# Often paired with TreeExplainer or KernelExplainer

from lime.lime_tabular import LimeTabularExplainer
# Local Interpretable Model-agnostic Explanations (LIME)
# Explains individual predictions using local surrogate models


# =====================================================================
# Visualization Helpers
# =====================================================================

def set_global_settings() -> None:
    """
    Configure global settings for common data science libraries.

    This function standardizes:
    - Warning visibility
    - Random seed for reproducibility
    - Pandas display formatting
    - Matplotlib figure styling
    - Seaborn visualization theme

    Notes
    -----
    - This function modifies global library state.
      It should be called once at application startup.
    - Re-seeding NumPy affects all subsequent random operations.
    - Suppressing warnings globally may hide important runtime signals.

    Returns
    -------
    None
        This function performs side effects only.
    """

    # Suppress all warnings to keep notebook/script output clean
    warnings.filterwarnings('ignore')

    # Set NumPy global random seed for reproducibility
    np.random.seed(100)

    # Configure Pandas to display floating-point numbers with 2 decimals
    pd.options.display.float_format = '{:.2f}'.format

    # Configure default Matplotlib rendering parameters
    plt.rcParams.update({
        "figure.dpi": 200,        # High-resolution figures
        "figure.edgecolor": "black",
        "figure.frameon": True
    })

    # Additional Seaborn rc configuration overrides
    # These modify matplotlib defaults under seaborn's theme
    custom_rc = {
        "axes.labelsize": 12,     # Axis label font size
        "legend.fontsize": 10,    # Legend text size
        "legend.loc": "best",     # Automatic legend placement
        "axes.titlesize": 14      # Plot title size
    }

    # Apply Seaborn global theme configuration
    # Context adjusts scaling for different use cases
    sns.set_theme(
        context="paper",          # paper, notebook, talk, poster
        style="darkgrid",         # whitegrid, darkgrid, white, ticks, dark
        palette="deep",           # deep, muted, pastel, bright, colorblind
        font="sans-serif",        # Global font family
        rc=custom_rc              # Custom matplotlib rc overrides
    )


def create_bar_chart(
        df: pd.DataFrame,
        x_axis: str,
        hue: str | None = None,
        rotate_labels: bool = False
) -> None:
    """
    Create a categorical count-based bar chart using Seaborn.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing categorical columns.
    x_axis : str
        Column name to display on the x-axis.
    hue : str | None, optional
        Optional secondary categorical variable for grouped bars.
        Default is None (no grouping).
    rotate_labels : bool, optional
        If True, rotate x-axis labels by 45 degrees for readability.
        Useful when category names are long. Default is False.

    Returns
    -------
    None
        Displays the generated plot.

    Notes
    -----
    - Uses seaborn.countplot(), which counts occurrences automatically.
    - Assumes `x_axis` (and `hue`, if provided) exist in the DataFrame.
    - This function creates a new figure each time it is called.
    """

    # Initialize figure with larger width for improved readability
    plt.figure(figsize=(10, 6))

    # Create count-based bar plot.
    # Seaborn automatically computes value counts of categorical variable
    ax = sns.countplot(data=df, x=x_axis, hue=hue)

    # Optionally rotate x-axis labels to avoid overlap for long categories
    if rotate_labels:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Dynamically generate plot title based on grouping variable
    title = f"Count of {x_axis} by {hue}" if hue else f"Count of {x_axis}"
    plt.title(title)

    # Adjust layout to prevent clipping of labels or title
    plt.tight_layout()

    # Render plot to output
    plt.show()


def create_boxplot(
        df: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        hue: str | None = None,
        rotate_labels: bool = False
) -> None:
    """
    Create a boxplot visualization using Seaborn.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing categorical and numerical columns.
    x_axis : str
        Categorical column to group data along the x-axis.
    y_axis : str
        Numerical column representing the distribution to visualize.
    hue : str | None, optional
        Additional categorical grouping variable for nested boxplots.
        Default is None (no subgrouping).
    rotate_labels : bool, optional
        If True, rotate x-axis labels by 45 degrees for readability.
        Useful for long category names. Default is False.

    Returns
    -------
    None
        Displays the generated plot.

    Notes
    -----
    - Uses seaborn.boxplot() with:
        notch=True     → Displays confidence interval around median.
        fliersize=3    → Controls outlier marker size.
    - Assumes provided column names exist in the DataFrame.
    - A new matplotlib figure is created on each call.
    """

    # Initialize figure with improved width for readability
    plt.figure(figsize=(10, 6))

    # Create boxplot
    ax = sns.boxplot(
        data=df,
        x=x_axis,
        y=y_axis,
        hue=hue,
        notch=True,      # notch=True visually indicates median confidence interval
        fliersize=3      # controls visibility of outliers
    )

    # Optionally rotate x-axis labels to prevent overlap
    if rotate_labels:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Set plot title dynamically based on selected axes
    plt.title(f"Boxplot of {y_axis} by {x_axis}")

    # Adjust layout to prevent clipping of axis labels and title
    plt.tight_layout()

    # Render plot to output
    plt.show()


def create_scatterplot(
        df: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        hue: str | None = None
) -> None:
    """
    Create a scatter plot using Seaborn.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing numerical columns.
    x_axis : str
        Column name for the x-axis (independent variable).
    y_axis : str
        Column name for the y-axis (dependent variable).
    hue : str | None, optional
        Optional categorical variable for color-based grouping.
        Default is None (single-color scatter).

    Returns
    -------
    None
        Displays the generated plot.

    Notes
    -----
    - Uses seaborn.scatterplot() for visualization.
    - Assumes provided column names exist in the DataFrame.
    - Alpha transparency is set to 0.7 to reduce overplotting.
    - Creates a new matplotlib figure on each call.
    """

    # Initialize figure with wider layout for better visibility
    plt.figure(figsize=(10, 6))

    # Create scatter plot
    sns.scatterplot(
        data=df,
        x=x_axis,
        y=y_axis,
        hue=hue,      # hue enables categorical color differentiation
        alpha=0.7     # alpha=0.7 improves readability when points overlap
    )

    # Dynamically set descriptive plot title
    plt.title(f"Scatterplot of {y_axis} vs {x_axis}")

    # Adjust layout to prevent clipping of labels or title
    plt.tight_layout()

    # Render plot to output
    plt.show()


def create_distribution_plot(
        df: pd.DataFrame, 
        column: str, 
        bins: int = 30, 
        hue: str = None
) -> None:
    """
    Create a histogram-based distribution plot with KDE overlay.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing numerical columns.
    column : str
        Numerical column to visualize.
    bins : int, optional
        Number of histogram bins. Default is 30.
    hue : str | None, optional
        Optional categorical variable for grouped distributions.
        Default is None.

    Returns
    -------
    None
        Displays the generated plot.

    Notes
    -----
    - Uses seaborn.histplot() with:
        kde=True        → overlays kernel density estimate curve.
        stat='density'  → normalizes histogram.
    - Displays skewness statistic inside the plot.
    - Adds vertical lines for mean and median.
    - Assumes the specified column exists and is numeric.
    """

    # Initialize figure with wider layout for improved readability
    plt.figure(figsize=(10, 6))

    # Create histogram with KDE overlay
    sns.histplot(
        data=df,
        x=column,
        hue=hue,
        bins=bins,
        kde=True,
        stat='density'      # stat='density' normalizes area under histogram to 1
    )

    # Compute skewness of the distribution
    skew_val = df[column].skew()

    # Display skewness annotation in top-right corner of the axes
    # Coordinates are normalized (0 to 1) relative to axes
    plt.text(
        0.99, 0.88,
        f"Skewness = {skew_val:.2f}",
        transform=plt.gca().transAxes,
        ha='right',
        va='top',
        fontsize=12,
        color="black",
        bbox={
            'boxstyle': 'round,pad=0.3',
            'facecolor': 'white',
            'alpha': 0.7
        }
    )

    # Add vertical reference lines for mean and median
    line_mean = plt.axvline(
        df[column].mean(),
        color="darkred",
        linestyle="--"
    )

    line_median = plt.axvline(
        df[column].median(),
        color="darkgreen",
        linestyle="--"
    )

    # Add legend for statistical reference lines
    plt.legend(
        handles=[line_mean, line_median],
        labels=["Mean", "Median"],
        loc="upper right"
    )

    # Set descriptive title
    plt.title(f"Distribution of {column}")

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Render plot
    plt.show()


def create_correlation_heatmap(
        df: pd.DataFrame,
        method: str = "pearson",
        annot: bool = True,
        cmap: str = "magma"
) -> None:
    """
    Create a correlation matrix heatmap using Seaborn.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing numerical columns.
    method : str, optional
        Correlation method to use:
            - 'pearson'  → Linear correlation (default)
            - 'spearman' → Rank-based correlation
            - 'kendall'  → Ordinal association measure
    annot : bool, optional
        If True, annotate heatmap cells with correlation values.
        Default is True.
    cmap : str, optional
        Colormap used for visualization. Default is 'magma'.

    Returns
    -------
    None
        Displays the generated heatmap.

    Notes
    -----
    - Only numerical columns are considered in correlation computation.
    - Correlation values range from -1 to 1.
    - The heatmap is centered at 0 for visual symmetry.
    - A new matplotlib figure is created on each call.
    """

    # Initialize larger figure for improved readability of matrix
    plt.figure(figsize=(12, 10))

    # Compute correlation matrix using selected method
    # Non-numeric columns are automatically excluded
    corr = df.corr(method=method)

    # Create heatmap visualization
    sns.heatmap(
        corr,
        annot=annot,                      # annot=True displays numeric values in each cell
        fmt=".2f",
        cmap=cmap,
        vmin=-1,                          # vmin/vmax fix color scale between -1 and 1
        vmax=1,
        center=0,                         # center=0 ensures neutral midpoint for diverging colormaps
        square=True,                      # square=True keeps cells proportionally square
        cbar_kws={"shrink": 0.8}
    )

    # Set descriptive title including correlation method used
    plt.title(f"Correlation Heatmap ({method.capitalize()} method)")

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Render heatmap
    plt.show()


def create_shap_waterfall(
        shap_object: Explanation,
        index: int
) -> None:
    """
    Create a custom horizontal waterfall-style SHAP contribution plot.

    This function:
    - Extracts SHAP values for a single observation
    - Shifts contributions so the baseline is centered at 0
    - Converts contributions to percentage scale
    - Orders features by absolute impact
    - Displays positive and negative contributions distinctly

    Parameters
    ----------
    shap_object : shap.Explanation
        SHAP Explanation object returned by SHAP explainer.
    index : int
        Index of the instance to visualize.

    Returns
    -------
    None
        Displays the generated plot.

    Notes
    -----
    - Contributions are shifted by distributing the base value evenly
      across features before converting to percentage scale.
    - This is a customized visualization and does NOT replicate
      the exact behavior of shap.plots.waterfall().
    - Intended for classification probability-style interpretation.
    """
    
    # Extract feature names, SHAP values, and base value
    features = list(shap_object[index].feature_names)

    # Ensure values are numeric and flattened
    vals = np.array(shap_object[index].values, dtype=float).flatten()

    # Base value (expected value of model output)
    base = float(shap_object[index].base_values)

    # Shift each contribution so base is effectively 0
    # Each contribution is adjusted by distributing base value equally across features, then scaled by 100
    shifted_shap = (vals + base / len(features)) * 100.0
    total_pred = sum(shifted_shap)

    # Order features by absolute magnitude (largest first)
    order = np.argsort(np.abs(shifted_shap))[::-1]
    feat_ordered = [features[i] for i in order]
    contribs = shifted_shap[order]

    # Plot configuration
    n = len(contribs)
    fig_w = 10
    fig_h = max(3, n * 0.45)    # Dynamic height scaling
    _, ax = plt.subplots(figsize=(fig_w, fig_h))

    y = np.arange(n)

    # Use active seaborn palette
    palette = sns.color_palette()
    pos_color, neg_color = palette[0], palette[1]

    # Determine x-axis limits with padding
    x_min = min(contribs.min(), 0.0)
    x_max = max(contribs.max(), 0.0)

    x_range = x_max - x_min if (x_max - x_min) != 0 else 1.0
    pad = x_range * 0.03

    ax.set_xlim(x_min - pad * 4, x_max + pad * 4)

    # Assign color based on sign of contribution
    colors = [pos_color if v >= 0 else neg_color for v in contribs]

    # Draw horizontal bars centered at 0
    # Negative values extend left automatically
    ax.barh(
        y,
        contribs,
        left=0,
        height=0.6,
        align="center",
        color=colors,
        edgecolor="k",
        linewidth=0.3
    )

    # Draw baseline reference at 0
    ax.axvline(0.0, color='black')

    # Annotate each bar with signed percentage value
    for i, val in enumerate(contribs):
        label = f"{val:+.2f}%"

        if val >= 0:
            text_x = val + pad
            ha = 'left'
        else:
            text_x = val - pad
            ha = 'right'
        
        ax.text(
            text_x, 
            y[i], 
            label, 
            va='center', 
            ha=ha, 
            color=colors[i], 
            fontsize=9, 
            fontweight='bold'
        )

    # Display predicted probability summary
    pred_text = f"Chance: {total_pred:.2f}%"
    ax.annotate(
        pred_text, xy=(1.1, 0.02), xycoords='axes fraction',
        ha='right', va='bottom', fontsize=12, fontweight='bold',
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "none"}
    )

    # Axis formatting
    ax.set_yticks(y)
    ax.set_yticklabels(feat_ordered, fontsize=10)
    ax.invert_yaxis()  # Largest contributor on top
    ax.set_xlabel("Contribution to prediction (%)", fontsize=11)
    ax.grid(axis='x', linestyle=':', linewidth=0.6, alpha=0.7)

    # Set descriptive title
    plt.title("SHAP contributions", fontsize=12)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Render the waterfall
    plt.show()


# =====================================================================
# Preprocessing Utilities
# =====================================================================


def power_transform(
        df: pd.DataFrame,
        columns: list[str],
        method: str = "yeo-johnson"
) -> Tuple[pd.DataFrame, PowerTransformer]:
    """
    Apply a power transformation to specified columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : list[str]
        List of numerical column names to transform.
    method : str, optional
        Transformation method:
            - 'yeo-johnson' (default) → Handles zero and negative values.
            - 'box-cox'               → Requires strictly positive values.

    Returns
    -------
    Tuple[pd.DataFrame, PowerTransformer]
        df_out : pd.DataFrame
            Copy of the original DataFrame with transformed columns.
        pt : PowerTransformer
            Fitted transformer object (can be reused on new data).

    Notes
    -----
    - A copy of the DataFrame is created to avoid mutating original data.
    - fit_transform expects a 2D array (hence `.values` usage).
    - For production pipelines, consider fitting on training data
      and using `pt.transform()` on validation/test sets.
    """

    # Initialize PowerTransformer with selected method
    # Yeo-Johnson works with non-positive values; Box-Cox does not
    pt = PowerTransformer(method=method)

    # Create a copy of the input DataFrame to prevent side effects
    df_out = df.copy()

    # Apply transformation
    # `.values` ensures a 2D NumPy array is passed to fit_transform()
    # The result replaces only the specified columns.
    df_out.loc[:, columns] = pt.fit_transform(df_out[columns].values)

    # Return transformed DataFrame and fitted transformer
    return df_out, pt


def reverse_power_transform(
        df: pd.DataFrame,
        columns: list[str],
        pt: PowerTransformer,
) -> pd.DataFrame:
    """
    Reverse a previously applied power transformation on specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing transformed numerical columns.
    columns : list[str]
        List of column names to inverse transform.
    pt : sklearn.preprocessing.PowerTransformer
        Fitted PowerTransformer instance used during transformation.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with specified columns restored
        to their original scale.

    Notes
    -----
    - The provided PowerTransformer must already be fitted.
    - A copy of the DataFrame is created to prevent mutation of input data.
    - Columns must match those originally used when fitting the transformer.
    """

    # Create a copy of the DataFrame to avoid side effects
    df_out = df.copy()

    # Apply inverse transformation
    # `.values` ensures a 2D NumPy array is passed as required
    df_out.loc[:, columns] = pt.inverse_transform(df_out[columns].values)

    # Return transformed DataFrame
    return df_out


# =====================================================================
# ML & Explanability Utilities
# =====================================================================


def find_best_params(
        model: object,
        search_space: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        metric: str,
        iterations: int = 50,
        random_state: int = 42,
) -> BaseEstimator:
    """
    Perform Bayesian hyperparameter optimization using BayesSearchCV.

    Parameters
    ----------
    model : object
        Scikit-learn compatible estimator.
    search_space : dict
        Dictionary defining hyperparameter search space for Bayesian optimization.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels.
    metric : str
        Scoring metric compatible with sklearn (e.g., 'accuracy', 'f1', 'roc_auc').
    iterations : int, optional
        Number of optimization iterations. Default is 50.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    object
        Best estimator found during optimization (already fitted).

    Notes
    -----
    - Uses StratifiedKFold cross-validation (3 splits).
    - Designed for classification tasks due to stratification.
    - Bayesian optimization is typically more efficient than GridSearchCV
      for large or continuous search spaces.
    """

    # Configure cross-validation strategy
    # StratifiedKFold preserves class distribution across folds
    cv_strategy = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=random_state
    )

    # Initialize Bayesian optimizer
    opt = BayesSearchCV(
        estimator=model,
        search_spaces=search_space,
        n_iter=iterations,
        cv=cv_strategy,
        scoring=metric,
        n_points=2,                    # n_points=2 allows parallel evaluation of two parameter sets
        pre_dispatch="2*n_jobs",       # pre_dispatch controls job allocation
        random_state=random_state
    )

    # Execute hyperparameter search.
    opt.fit(X_train, y_train)

    # Display best parameters found
    print("Best Parameters:")
    pprint(dict(opt.best_params_))

    # Return best fitted estimator
    return opt.best_estimator_


def get_recommended_values(
        data: np.ndarray,
        model,
        lime_explainer: LimeTabularExplainer,
        feature_idx: int,
        prob_threshold: float = 0.75,
        min_limit: float = -2.0,
        max_limit: float = 2.0,
        window: float = 1.0,
        step: float = 0.1
) -> Optional[float]:
    """
    Determine a recommended value for a specific feature to increase
    predicted probability beyond a desired threshold.

    This function:
    1. Uses LIME to extract a local linear approximation of the model.
    2. Solves analytically for the feature value needed to reach a
       probability threshold under the local linear model.
    3. Performs a bounded search around that analytical solution.
    4. Returns the smallest feature value that satisfies the threshold.

    Parameters
    ----------
    data : np.ndarray
        Single input instance (1D array).
    model : object
        Trained classifier with `predict_proba()` method.
    lime_explainer : LimeTabularExplainer
        Pre-configured LIME explainer instance.
    feature_idx : int
        Index of the feature to adjust.
    prob_threshold : float, optional
        Desired minimum probability for class 1. Default is 0.75.
    min_limit : float, optional
        Minimum allowed value for the feature search space.
    max_limit : float, optional
        Maximum allowed value for the feature search space.
    window : float, optional
        Search window centered around analytical solution.
    step : float, optional
        Increment size for brute-force search. Default is 0.1.

    Returns
    -------
    Optional[float]
        Smallest feature value achieving the probability threshold.
        Returns None if no such value exists or slope is zero.

    Notes
    -----
    - Assumes binary classification (class index 1).
    - LIME local explanation is linear and approximate.
    - Brute-force search validates analytical solution against true model.
    """

    # Generate local explanation using LIME
    exp = lime_explainer.explain_instance(
        data_row=data,
        predict_fn=model.predict_proba
    )

    # Extract local slope (feature weight) for the target class (class 1)
    slope = None
    for idx, weight in exp.local_exp[1]:
        if idx == feature_idx:
            slope = float(weight)
            break

    # If feature not present in explanation, weight is implicitly zero
    if slope is None:
        slope = 0.0

    # If slope is ~0, changing this feature has negligible local effect
    if np.isclose(slope, 0.0):
        print("Slope zero — feature has negligible local impact.")
        return None

    # Extract intercept of local linear model for class 1
    # Local linear model: prob ≈ intercept + slope * feature_value
    intercept = float(exp.intercept[1])

    # Solve analytically:
    # intercept + slope * x = prob_threshold
    # => x = (prob_threshold - intercept) / slope
    center_value = (prob_threshold - intercept) / slope

    # Define bounded search region around analytical solution
    start = max(min_limit, center_value - window)
    end = min(max_limit, center_value + window)

    # Generate candidate values
    values = np.arange(start, end + step, step).round(2)

    # Validate using actual model predictions (not LIME approximation)
    passing = []
    for val in values:
        row = data.copy()
        row[feature_idx] = val

        prob = float(model.predict_proba(row.reshape(1, -1))[0, 1])

        if prob >= prob_threshold:
            passing.append(val)

    # Return smallest valid adjustment if exists
    if not passing:
        return None

    return min(passing)
