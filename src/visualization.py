"""
Visualization Module for Aadhaar Societal Intelligence Project
Publication-quality plots for analysis and presentation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


class AadhaarVisualizer:
    """
    Comprehensive visualization toolkit for Aadhaar analysis
    Generates publication-quality plots
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize visualizer with configuration
        
        Args:
            config: Visualization configuration
        """
        self.config = config or {}
        self.style = self.config.get('style', 'seaborn-v0_8-darkgrid')
        self.palette = self.config.get('palette', 'Set2')
        self.figsize = self.config.get('figure_size', (12, 8))
        self.dpi = self.config.get('dpi', 300)
        
        # Set default style
        plt.style.use(self.style)
        sns.set_palette(self.palette)
        
        print("✓ Visualizer initialized")
    
    # ========== UNIVARIATE VISUALIZATIONS ==========
    
    def plot_distribution(self, df: pd.DataFrame, column: str, 
                         title: Optional[str] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of a single variable
        
        Args:
            df: DataFrame
            column: Column name
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram with KDE
        axes[0].hist(df[column], bins=50, alpha=0.7, edgecolor='black', density=True)
        df[column].plot(kind='kde', ax=axes[0], color='red', linewidth=2)
        axes[0].set_title(f'Distribution of {column}')
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('Density')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(df[column].dropna(), vert=True)
        axes[1].set_title(f'Box Plot of {column}')
        axes[1].set_ylabel(column)
        axes[1].grid(True, alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_time_series(self, df: pd.DataFrame, date_col: str, value_col: str,
                        group_by: Optional[str] = None,
                        title: Optional[str] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot time series with optional grouping
        
        Args:
            df: DataFrame
            date_col: Date column name
            value_col: Value column name
            group_by: Column to group by
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if group_by:
            for group in df[group_by].unique()[:10]:  # Limit to 10 groups
                group_data = df[df[group_by] == group]
                ax.plot(group_data[date_col], group_data[value_col], 
                       label=group, alpha=0.7, linewidth=2)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.plot(df[date_col], df[value_col], linewidth=2)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(value_col, fontsize=12)
        ax.set_title(title or f'{value_col} Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    # ========== BIVARIATE VISUALIZATIONS ==========
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, 
                                features: Optional[List[str]] = None,
                                title: Optional[str] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation heatmap
        
        Args:
            df: DataFrame
            features: List of features to include
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        if features:
            corr = df[features].corr()
        else:
            corr = df.select_dtypes(include=[np.number]).corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title(title or 'Feature Correlation Matrix', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_scatter_with_regression(self, df: pd.DataFrame, 
                                    x_col: str, y_col: str,
                                    hue: Optional[str] = None,
                                    title: Optional[str] = None,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Scatter plot with regression line
        
        Args:
            df: DataFrame
            x_col: X-axis column
            y_col: Y-axis column
            hue: Column for color coding
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.regplot(data=df, x=x_col, y=y_col, ax=ax, 
                   scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
        
        if hue and hue in df.columns:
            for category in df[hue].unique():
                subset = df[df[hue] == category]
                ax.scatter(subset[x_col], subset[y_col], label=category, alpha=0.6)
            ax.legend()
        
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(title or f'{y_col} vs {x_col}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    # ========== TRIVARIATE VISUALIZATIONS ==========
    
    def plot_3d_surface(self, df: pd.DataFrame, 
                       x_col: str, y_col: str, z_col: str,
                       title: Optional[str] = None,
                       save_path: Optional[str] = None):
        """
        3D surface plot using Plotly
        
        Args:
            df: DataFrame
            x_col: X-axis column
            y_col: Y-axis column
            z_col: Z-axis column (height)
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            Plotly figure
        """
        # Pivot data for surface plot
        pivot = df.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc='mean')
        
        fig = go.Figure(data=[go.Surface(z=pivot.values, x=pivot.columns, y=pivot.index)])
        
        fig.update_layout(
            title=title or f'{z_col} by {x_col} and {y_col}',
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            ),
            width=900,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_animated_heatmap(self, df: pd.DataFrame,
                             x_col: str, y_col: str, z_col: str,
                             time_col: str,
                             title: Optional[str] = None,
                             save_path: Optional[str] = None):
        """
        Animated heatmap over time using Plotly
        
        Args:
            df: DataFrame
            x_col: X-axis column
            y_col: Y-axis column
            z_col: Value column
            time_col: Time column for animation
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            Plotly figure
        """
        # Prepare data
        df_sorted = df.sort_values(time_col)
        
        fig = px.density_heatmap(
            df_sorted,
            x=x_col,
            y=y_col,
            z=z_col,
            animation_frame=time_col,
            title=title or f'{z_col} Heatmap Over Time',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            width=1000,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    # ========== GEOGRAPHIC VISUALIZATIONS ==========
    
    def plot_choropleth_map(self, df: pd.DataFrame,
                           location_col: str, value_col: str,
                           title: Optional[str] = None,
                           save_path: Optional[str] = None):
        """
        Choropleth map visualization
        
        Args:
            df: DataFrame
            location_col: Location column (state/district)
            value_col: Value to visualize
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            Plotly figure
        """
        # Aggregate by location
        map_data = df.groupby(location_col)[value_col].mean().reset_index()
        
        fig = px.choropleth(
            map_data,
            locations=location_col,
            locationmode='country names',  # Adjust based on data
            color=value_col,
            title=title or f'{value_col} by {location_col}',
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(
            width=1000,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    # ========== SOCIETAL INDICATOR VISUALIZATIONS ==========
    
    def plot_identity_stability_dashboard(self, df: pd.DataFrame,
                                         save_path: Optional[str] = None):
        """
        Comprehensive dashboard for Identity Stability Score
        
        Args:
            df: DataFrame with identity_stability_score
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Distribution
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.hist(df['identity_stability_score'], bins=50, edgecolor='black', alpha=0.7)
        ax1.set_title('Identity Stability Score Distribution', fontweight='bold')
        ax1.set_xlabel('Stability Score')
        ax1.set_ylabel('Frequency')
        
        # 2. Box plot by state
        ax2 = fig.add_subplot(gs[0, 2])
        df.boxplot(column='identity_stability_score', by='state', ax=ax2)
        ax2.set_title('By State')
        ax2.set_xlabel('')
        plt.sca(ax2)
        plt.xticks(rotation=45, ha='right')
        
        # 3. Time series
        ax3 = fig.add_subplot(gs[1, :])
        df_time = df.groupby('date')['identity_stability_score'].mean()
        ax3.plot(df_time.index, df_time.values, linewidth=2, color='green')
        ax3.set_title('Identity Stability Over Time', fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Average Stability Score')
        ax3.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 4. Top 10 stable districts
        ax4 = fig.add_subplot(gs[2, :])
        top_stable = df.groupby('district')['identity_stability_score'].mean().nlargest(10)
        ax4.barh(range(len(top_stable)), top_stable.values, color='skyblue')
        ax4.set_yticks(range(len(top_stable)))
        ax4.set_yticklabels(top_stable.index)
        ax4.set_title('Top 10 Most Stable Districts', fontweight='bold')
        ax4.set_xlabel('Average Stability Score')
        ax4.grid(axis='x', alpha=0.3)
        
        fig.suptitle('Identity Stability Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_migration_flow_sankey(self, df: pd.DataFrame,
                                   source_col: str, target_col: str, value_col: str,
                                   title: Optional[str] = None,
                                   save_path: Optional[str] = None):
        """
        Sankey diagram for migration flows
        
        Args:
            df: DataFrame
            source_col: Source location column
            target_col: Target location column
            value_col: Flow value column
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            Plotly figure
        """
        # Prepare data
        flow_data = df.groupby([source_col, target_col])[value_col].sum().reset_index()
        
        # Create unique labels
        all_locations = list(set(flow_data[source_col].unique()) | 
                           set(flow_data[target_col].unique()))
        location_map = {loc: i for i, loc in enumerate(all_locations)}
        
        # Map to indices
        source_indices = [location_map[x] for x in flow_data[source_col]]
        target_indices = [location_map[x] for x in flow_data[target_col]]
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_locations
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=flow_data[value_col]
            )
        )])
        
        fig.update_layout(
            title_text=title or "Migration Flow Diagram",
            font_size=12,
            width=1200,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


def quick_visualize(df: pd.DataFrame, feature: str, plot_type: str = 'distribution'):
    """
    Quick visualization wrapper
    
    Args:
        df: DataFrame
        feature: Feature to visualize
        plot_type: Type of plot ('distribution', 'timeseries', etc.)
    """
    viz = AadhaarVisualizer()
    
    if plot_type == 'distribution':
        return viz.plot_distribution(df, feature)
    elif plot_type == 'timeseries':
        return viz.plot_time_series(df, 'date', feature)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")


if __name__ == "__main__":
    print("Visualization module loaded successfully")
    print("Available visualization methods:")
    viz = AadhaarVisualizer()
    methods = [m for m in dir(viz) if m.startswith('plot_')]
    for method in methods:
        print(f"  - {method}")
