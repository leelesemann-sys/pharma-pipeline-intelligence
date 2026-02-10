"""
Reusable Plotly chart components for Pharma Pipeline Dashboard.
All charts use a consistent professional color palette and styling.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ─────────────────────────────────────────────
# Color Palette
# ─────────────────────────────────────────────

PRIMARY = "#2563eb"
PRIMARY_LIGHT = "#93c5fd"
SECONDARY = "#64748b"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"
DANGER = "#ef4444"
BG_LIGHT = "#f8fafc"

# Professional sequential palette for heatmaps
HEATMAP_COLORS = [
    "#f0f9ff",  # near white
    "#bae6fd",  # light blue
    "#38bdf8",  # medium blue
    "#2563eb",  # primary blue
    "#1e40af",  # dark blue
    "#7c3aed",  # purple
    "#dc2626",  # red (crowded)
]

# Categorical palette for MoA classes
MOA_COLORS = {
    "GLP-1 Receptor Agonist": "#2563eb",
    "SGLT2 Inhibitor": "#059669",
    "DPP-4 Inhibitor": "#d97706",
    "Dual GIP/GLP-1 Agonist": "#7c3aed",
    "Triple Agonist": "#dc2626",
    "Oral GLP-1": "#0891b2",
    "Insulin": "#64748b",
    "Biguanide": "#be185d",
    "Sulfonylurea": "#ca8a04",
    "Meglitinide": "#78716c",
    "Alpha-Glucosidase Inhibitor": "#4f46e5",
    "Amylin Analog": "#0d9488",
    "Dopamine Agonist": "#c026d3",
    "Bile Acid Sequestrant": "#ea580c",
    "Thiazolidinedione": "#16a34a",
}

# Status color mapping
STATUS_COLORS = {
    "completed": "#22c55e",
    "recruiting": "#2563eb",
    "active_not_recruiting": "#0891b2",
    "not_yet_recruiting": "#93c5fd",
    "terminated": "#ef4444",
    "withdrawn": "#f97316",
    "suspended": "#f59e0b",
    "unknown": "#94a3b8",
    "enrolling_by_invitation": "#8b5cf6",
}

# Common chart layout settings (without margin to avoid conflicts)
_LAYOUT_BASE = dict(
    font=dict(family="Inter, system-ui, sans-serif", color="#0f172a"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Inter, system-ui, sans-serif",
    ),
)

DEFAULT_MARGIN = dict(l=40, r=20, t=40, b=40)


def _layout(margin=None, **kwargs):
    """Build layout dict with base defaults + custom overrides."""
    layout = {**_LAYOUT_BASE, "margin": margin or DEFAULT_MARGIN}
    layout.update(kwargs)
    return layout


# Keep LAYOUT_DEFAULTS for backward compat (includes default margin)
LAYOUT_DEFAULTS = {**_LAYOUT_BASE, "margin": DEFAULT_MARGIN}


# ─────────────────────────────────────────────
# Chart Functions
# ─────────────────────────────────────────────

def kpi_metric(label, value, delta=None, delta_color="normal"):
    """Format a KPI value for display. Returns formatted string."""
    if isinstance(value, (int, float)):
        if value >= 1000:
            formatted = f"{value:,.0f}"
        else:
            formatted = str(value)
    else:
        formatted = str(value)
    return formatted


def pipeline_phase_bar(df):
    """Horizontal bar chart: drugs by highest development phase."""
    fig = px.bar(
        df,
        y="phase",
        x="drug_count",
        orientation="h",
        text="drug_count",
        color_discrete_sequence=[PRIMARY],
    )
    fig.update_traces(
        textposition="outside",
        textfont_size=13,
        marker_line_width=0,
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        xaxis_title="Number of Drugs",
        yaxis_title="",
        showlegend=False,
        height=300,
        yaxis=dict(autorange="reversed"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", gridwidth=1)
    fig.update_yaxes(showgrid=False)
    return fig


def donut_chart(df, names_col, values_col, title="", color_map=None, height=350):
    """Donut chart for status/phase breakdowns."""
    colors = None
    if color_map:
        colors = [color_map.get(name, "#94a3b8") for name in df[names_col]]

    fig = go.Figure(data=[go.Pie(
        labels=df[names_col],
        values=df[values_col],
        hole=0.55,
        marker=dict(colors=colors) if colors else {},
        textinfo="label+percent",
        textposition="outside",
        textfont_size=11,
        hovertemplate="<b>%{label}</b><br>Trials: %{value:,}<br>Share: %{percent}<extra></extra>",
    )])
    fig.update_layout(
        **_layout(margin=dict(l=20, r=20, t=50, b=20)),
        title=dict(text=title, font_size=14, x=0.5),
        showlegend=False,
        height=height,
    )
    return fig


def heatmap_chart(df, x_col, y_col, z_col, title="", height=500):
    """MoA x Indication heatmap."""
    pivot = df.pivot_table(index=y_col, columns=x_col, values=z_col, fill_value=0)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.0, "#f0f9ff"],
            [0.1, "#bae6fd"],
            [0.3, "#38bdf8"],
            [0.5, "#2563eb"],
            [0.7, "#1e40af"],
            [0.9, "#7c3aed"],
            [1.0, "#dc2626"],
        ],
        hovertemplate="<b>%{y}</b> x <b>%{x}</b><br>Count: %{z}<extra></extra>",
        showscale=True,
        colorbar=dict(title=z_col.replace("_", " ").title(), thickness=15),
    ))
    fig.update_layout(
        **_layout(margin=dict(l=180, r=40, t=50, b=120)),
        title=dict(text=title, font_size=14),
        height=height,
        xaxis=dict(tickangle=-45, tickfont_size=11),
        yaxis=dict(tickfont_size=11, autorange="reversed"),
    )
    return fig


def trend_line_chart(df, x_col, y_col, color_col, title="", height=400):
    """Line chart for trends over time (one line per category)."""
    color_map = MOA_COLORS

    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        color_discrete_map=color_map,
        markers=True,
    )
    fig.update_traces(line_width=2.5, marker_size=5)
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=title, font_size=14),
        height=height,
        xaxis_title="Year",
        yaxis_title="Trial Starts",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.5,
            font_size=10,
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", dtick=1)
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
    return fig


def stacked_area_chart(df, x_col, y_col, color_col, title="", height=400):
    """Stacked area chart for trial starts over time by phase."""
    fig = px.area(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=title, font_size=14),
        height=height,
        xaxis_title="Year",
        yaxis_title="Trial Starts",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font_size=10,
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", dtick=2)
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
    return fig


def horizontal_bar_chart(df, x_col, y_col, color_col=None, title="", height=500,
                          color_map=None, text_col=None):
    """Horizontal bar chart (for sponsors, termination rates, etc.)."""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        color_discrete_map=color_map,
        orientation="h",
        text=text_col or x_col,
    )
    fig.update_traces(
        textposition="outside",
        textfont_size=11,
        marker_line_width=0,
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=title, font_size=14),
        height=height,
        xaxis_title="",
        yaxis_title="",
        showlegend=bool(color_col),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
    fig.update_yaxes(showgrid=False)
    return fig


def trial_timeline_scatter(df, title="", height=400):
    """Scatter plot: trial start date vs phase, color by status."""
    if df.empty:
        return go.Figure().update_layout(**LAYOUT_DEFAULTS, height=height)

    # Fill NaN enrollment with 0 for size parameter
    plot_df = df.copy()
    plot_df["enrollment"] = plot_df["enrollment"].fillna(0).astype(float)

    fig = px.scatter(
        plot_df,
        x="start_date",
        y="phase",
        color="overall_status",
        color_discrete_map=STATUS_COLORS,
        size="enrollment",
        size_max=20,
        hover_data=["nct_id", "title", "lead_sponsor_name", "enrollment"],
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=title, font_size=14),
        height=height,
        xaxis_title="Start Date",
        yaxis_title="Phase",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.5,
            font_size=10,
            title_text="",
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
    return fig


def competitive_comparison_bar(df, drug_name, title="", height=350):
    """Bar chart comparing trial counts of drugs in same MoA class."""
    if df.empty:
        return go.Figure().update_layout(**LAYOUT_DEFAULTS, height=height)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["inn"],
        x=df["trial_count"],
        name="Total Trials",
        orientation="h",
        marker_color=PRIMARY_LIGHT,
        text=df["trial_count"],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        y=df["inn"],
        x=df["active_trials"],
        name="Active Trials",
        orientation="h",
        marker_color=PRIMARY,
        text=df["active_trials"],
        textposition="outside",
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=title, font_size=14),
        height=height,
        barmode="overlay",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
    return fig
