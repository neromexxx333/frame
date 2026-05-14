"""
Dashboard Streamlit untuk analisis portal 2D dan keandalan.

Jalankan dengan:
    streamlit run streamlit_app.py
"""

import io
import base64
import html
import json
import os
import re
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
try:
    import plotly.graph_objects as go
except Exception:
    go = None
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import proj3d
from scipy import stats

from main import PortalReliabilityAnalysis
from modules.display_conventions import (
    SPECIAL_BEAM_JOINT_RAW_SIGN_NODE_IDS,
    get_displayed_element_moment_values,
    get_joint_equilibrium_moment,
)
from modules.excel_reader import ExcelReader
from modules.plotting import PortalPlotter
from modules.reliability import AXIAL_DEMAND_TOLERANCE_KN, PerformanceFunction
from modules.stiffness_matrix import Portal2D


DEFAULT_INPUT_FILE = "input_template.xlsx"
ULM_LOGO_PATH = Path(__file__).with_name("Logo_ULM.png")
DETERMINISTIC_MODE_LABEL = "Deterministik (SNI 2847:2019)"
PROBABILISTIC_MODE_LABEL = "Probabilistik"
BASE_ESTIMATE_ELEMENTS = 9.0
BASE_ESTIMATE_RANDOM_VARIABLES = 42.0
CALIBRATION_REFERENCE_NUM_SIMULATIONS = 100000.0
CALIBRATION_REFERENCE_PROBABILISTIC_SECONDS = 400
BASE_DETERMINISTIC_SECONDS = 5
BASE_FULL_ANALYSIS_FIXED_OVERHEAD_SECONDS = BASE_DETERMINISTIC_SECONDS
BASE_MONTE_CARLO_SECONDS_PER_SAMPLE = max(
    (CALIBRATION_REFERENCE_PROBABILISTIC_SECONDS - BASE_FULL_ANALYSIS_FIXED_OVERHEAD_SECONDS)
    / CALIBRATION_REFERENCE_NUM_SIMULATIONS,
    1e-9
)
ZOOMABLE_PLOT_VIEWER_HEIGHT = 540
PLOT_DOWNLOAD_IMAGE_DPI = 320
PLOT_DOWNLOAD_JPEG_QUALITY = 95
SAFE_CLOUD_COLOR = '#0000ff'
PHYSICAL_NONLINEAR_CONTOUR_COLOR = '#c026d3'
FAILURE_CLOUD_MAX_STORED_POINTS = 12000
FAILURE_CLOUD_MAX_FAILED_POINTS = 4000
LIMIT_STATE_PHYSICAL_CLOUD_MAX_POINTS = 1200
LIMIT_STATE_PHYSICAL_CLOUD_MAX_FAILED_POINTS = 420
PHYSICAL_G_CONTOUR_GRID_SIZE = 120
PHYSICAL_LIMIT_STATE_FUNCTION_GRID_SIZE = 36
AXIAL_MOMENT_PM_CLOUD_MAX_POINTS = 1400
AXIAL_MOMENT_PM_CLOUD_MAX_FAILED_POINTS = 420
FAILURE_SURFACE_MAX_CLASS_POINTS = 2200
FAILURE_SURFACE_GRID_SIZE = 160
FAILURE_SURFACE_MAX_SCATTER_POINTS_PER_CLASS = 1800
FAILURE_SURFACE_3D_GRID_SIZE = 34
MOMENT_EQUILIBRIUM_TOLERANCE_KNM = 1e-6
DETERMINISTIC_RISK_WEIGHT_SEVERITY = 0.60
DETERMINISTIC_RISK_WEIGHT_SENSITIVITY = 0.40

RISK_LEVEL_COLORS = {
    'Rendah': '#0000ff',
    'Sedang': '#00b050',
    'Tinggi': '#ffc000',
    'Kritis': '#ff0000',
    'Tidak Ada Data': '#94a3b8'
}

RISK_LEVEL_ORDER = {
    'Tidak Ada Data': -1,
    'Rendah': 0,
    'Sedang': 1,
    'Tinggi': 2,
    'Kritis': 3
}

RISK_LEVEL_STYLES = {
    'Rendah': (
        'background-color: #dcfce7; '
        'font-weight: 700; '
        'color: #166534;'
    ),
    'Sedang': (
        'background-color: #fef3c7; '
        'font-weight: 700; '
        'color: #92400e;'
    ),
    'Tinggi': (
        'background-color: #ffedd5; '
        'font-weight: 700; '
        'color: #9a3412;'
    ),
    'Kritis': (
        'background-color: #fee2e2; '
        'font-weight: 700; '
        'color: #991b1b;'
    ),
    'Tidak Ada Data': (
        'background-color: #e5e7eb; '
        'font-weight: 700; '
        'color: #475569;'
    )
}

RISK_LEVEL_ENGLISH_LABELS = {
    'Rendah': 'Low',
    'Sedang': 'Moderate',
    'Tinggi': 'High',
    'Kritis': 'Critical',
    'Tidak Ada Data': 'No Data'
}

RISK_LEVEL_CANONICAL_LOOKUP = {
    'rendah': 'Rendah',
    'low': 'Rendah',
    'sedang': 'Sedang',
    'moderate': 'Sedang',
    'tinggi': 'Tinggi',
    'high': 'Tinggi',
    'kritis': 'Kritis',
    'critical': 'Kritis',
    'tidak ada data': 'Tidak Ada Data',
    'no data': 'Tidak Ada Data'
}

HEADER_GROUP_PALETTES = {
    'default': {
        'background': '#eef2f7',
        'border': '#cbd5e1'
    },
    'identity': {
        'background': '#e5e7eb',
        'border': '#9ca3af'
    },
    'summary': {
        'background': '#ffedd5',
        'border': '#fb923c'
    },
    'beam_system': {
        'background': '#ffedd5',
        'border': '#fb923c'
    },
    'column_system': {
        'background': '#dbeafe',
        'border': '#60a5fa'
    },
    'portal_system': {
        'background': '#fce7f3',
        'border': '#f472b6'
    },
    'overall': {
        'background': '#dbeafe',
        'border': '#60a5fa'
    },
    'load': {
        'background': '#fef3c7',
        'border': '#f59e0b'
    },
    'axial': {
        'background': '#dbeafe',
        'border': '#60a5fa'
    },
    'shear': {
        'background': '#dcfce7',
        'border': '#4ade80'
    },
    'moment': {
        'background': '#fef3c7',
        'border': '#f59e0b'
    },
    'axial_moment': {
        'background': '#ede9fe',
        'border': '#8b5cf6'
    },
    'risk': {
        'background': '#fee2e2',
        'border': '#ef4444'
    },
    'sensitivity': {
        'background': '#ede9fe',
        'border': '#8b5cf6'
    }
}

INTERNAL_FORCE_MAX_HIGHLIGHT_STYLES = {
    'B': (
        'background-color: #ffedd5; '
        'font-weight: 700; '
        'color: #9a3412;'
    ),
    'K': (
        'background-color: #dbeafe; '
        'font-weight: 700; '
        'color: #1d4ed8;'
    )
}

JOINT_MOMENT_EQUILIBRIUM_STATUS_STYLES = {
    'OK': (
        'background-color: #dcfce7; '
        'font-weight: 700; '
        'color: #166534;'
    ),
    'PERLU CEK': (
        'background-color: #fee2e2; '
        'font-weight: 700; '
        'color: #991b1b;'
    )
}


def format_metric(value, decimals: int = 4) -> str:
    """Format nilai numerik untuk metrik UI."""
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return "-"
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"
        return f"{float(value):.{decimals}f}"
    return str(value)


def format_beta_table_display(value, decimals: int = 4) -> str:
    """Format Beta(table) agar konsisten dengan label tabel, termasuk Inf/-Inf."""
    if value is None:
        return "-"
    if isinstance(value, str):
        text_value = str(value).strip()
        lowered = text_value.lower()
        if lowered in {"inf", "+inf", "infinity", "+infinity"}:
            return "Inf"
        if lowered in {"-inf", "-infinity"}:
            return "-Inf"
        return text_value
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return "-"
        if np.isposinf(value):
            return "Inf"
        if np.isneginf(value):
            return "-Inf"
        return f"{float(value):.{decimals}f}"
    return str(value)


def format_metric_comma(value, decimals: int = 2) -> str:
    """Format nilai numerik dengan pemisah ribuan koma."""
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return "-"
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"
        return f"{float(value):,.{decimals}f}"
    return str(value)


def format_error_message(exc: Exception) -> str:
    """Ringkas pesan error untuk ditampilkan di UI tanpa traceback."""
    message = str(exc).strip()
    if not message:
        return exc.__class__.__name__
    return message


def multiply_finite_values(left_value, right_value) -> Optional[float]:
    """Kalikan dua angka jika keduanya valid dan finite."""
    try:
        left_numeric = float(left_value)
        right_numeric = float(right_value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(left_numeric) or not np.isfinite(right_numeric):
        return None
    return left_numeric * right_numeric


def style_input_dataframe(df: pd.DataFrame,
                          table_min_width_px: Optional[int] = None):
    """Bangun styler tabel dengan header dan border yang lebih jelas."""
    subset_cols = list(df.columns)
    styler = df.style

    if subset_cols:
        styler = styler.set_properties(
            subset=subset_cols,
            **{
                'text-align': 'center',
                'border': '1px solid #d1d5db',
                'padding': '0.35rem 0.5rem'
            }
        )

    table_props = [
        ('border-collapse', 'collapse'),
        ('width', '100%'),
        ('font-size', '0.92rem')
    ]
    if table_min_width_px is not None:
        table_props.append(('min-width', f'{int(table_min_width_px)}px'))

    table_styles = [
        {
            'selector': 'table',
            'props': table_props
        },
        {
            'selector': 'th',
            'props': [
                ('text-align', 'center'),
                ('background-color', '#eef2f7'),
                ('color', '#000000'),
                ('border', '1px solid #cbd5e1'),
                ('padding', '0.4rem 0.55rem'),
                ('font-weight', '700'),
                ('font-size', '0.98rem')
            ]
        }
    ]
    styler = styler.set_table_styles(table_styles, overwrite=False)

    formatters = {}
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            formatters[col] = lambda value: "-" if pd.isna(value) else f"{int(value)}"
        elif pd.api.types.is_float_dtype(df[col]):
            formatters[col] = lambda value: "-" if pd.isna(value) else f"{float(value):,.4f}"
    if formatters:
        styler = styler.format(formatters, na_rep='-')

    return styler


def render_input_table(df: pd.DataFrame, styler: Optional[object] = None) -> None:
    """Render tabel HTML agar styling dataframe selalu konsisten."""
    active_styler = styler if styler is not None else style_input_dataframe(df)
    table_html = active_styler.hide(axis='index').to_html()
    st.markdown(
        f'<div style="overflow-x:auto; width:100%;">{table_html}</div>',
        unsafe_allow_html=True
    )


def apply_grouped_header_styles(styler,
                                df: pd.DataFrame,
                                grouped_columns: Dict[str, List[str]]):
    """Beri warna header per kelompok kolom agar tabel lebih mudah dibaca."""
    header_styles = []
    for group_name, columns in grouped_columns.items():
        palette = HEADER_GROUP_PALETTES.get(group_name, HEADER_GROUP_PALETTES['default'])
        for column in columns:
            if column not in df.columns:
                continue
            column_index = df.columns.get_loc(column)
            header_styles.append({
                'selector': f'th.col_heading.level0.col{column_index}',
                'props': [
                    ('background-color', palette['background']),
                    ('color', '#000000'),
                    ('border', f"1px solid {palette['border']}"),
                    ('padding', '0.4rem 0.55rem'),
                    ('font-weight', '700'),
                    ('font-size', '0.98rem')
                ]
            })
    if header_styles:
        styler = styler.set_table_styles(header_styles, overwrite=False)
    return styler


def get_max_abs_winner_mask(series: pd.Series,
                            rtol: float = 1e-9,
                            atol: float = 1e-12) -> pd.Series:
    """Kembalikan mask untuk semua nilai yang sama dengan maksimum absolut."""
    numeric_series = pd.to_numeric(series, errors='coerce')
    winner_mask = pd.Series(False, index=series.index)
    valid_series = numeric_series.dropna()
    if valid_series.empty:
        return winner_mask

    max_abs = float(valid_series.abs().max())
    if max_abs <= 0.0:
        winner_values = np.isclose(
            valid_series.to_numpy(dtype=float),
            0.0,
            rtol=rtol,
            atol=atol
        )
    else:
        winner_values = np.isclose(
            valid_series.abs().to_numpy(dtype=float),
            max_abs,
            rtol=rtol,
            atol=atol
        )

    winner_mask.loc[valid_series.index] = winner_values
    return winner_mask


def style_max_abs_dataframe(df: pd.DataFrame,
                            highlight_columns: Optional[List[str]] = None,
                            identity_columns: Optional[List[str]] = None):
    """Highlight nilai maksimum absolut per kolom tanpa mengubah nilainya."""
    styler = style_input_dataframe(df)
    if df.empty:
        return styler

    if highlight_columns is None:
        highlight_columns = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        ]
    highlight_columns = [col for col in highlight_columns if col in df.columns]
    if not highlight_columns:
        return styler

    identity_columns = [col for col in (identity_columns or []) if col in df.columns]
    highlight_style = (
        'background-color: #f8d7da; '
        'font-weight: 700; '
        'color: #7a0019;'
    )

    def highlight_max_abs(dataframe: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame('', index=dataframe.index, columns=dataframe.columns)
        for col in highlight_columns:
            series = pd.to_numeric(dataframe[col], errors='coerce')
            winner_mask = get_max_abs_winner_mask(series)
            if not winner_mask.any():
                continue

            styles.loc[winner_mask, col] = highlight_style
            if identity_columns:
                styles.loc[winner_mask, identity_columns] = highlight_style
        return styles

    return styler.apply(highlight_max_abs, axis=None)


def load_preview_input_data(path_text: str, uploaded_file):
    """Baca input Excel untuk preview sebelum analisis dijalankan."""
    source = None
    source_label = None

    if uploaded_file is not None:
        source = io.BytesIO(uploaded_file.getvalue())
        source_label = uploaded_file.name
    elif path_text:
        file_path = os.path.abspath(path_text)
        if os.path.exists(file_path):
            source = file_path
            source_label = file_path

    if source is None:
        return None, source_label

    reader = ExcelReader(source)
    return reader.get_all_data(), source_label


def get_geometry_elements_for_mode(input_data: Dict, is_probabilistic: bool) -> np.ndarray:
    """Pilih data elemen geometri sesuai mode analisis."""
    geometry = input_data['geometry']
    if is_probabilistic:
        return geometry.get('elements_mean', geometry['elements']).astype(float)
    return geometry.get('elements_deterministic', geometry.get('elements_mean', geometry['elements'])).astype(float)


def build_preview_portal(input_data: Dict, is_probabilistic: bool) -> Tuple[np.ndarray, list]:
    """Bangun elemen portal ringan untuk preview geometri input."""
    nodes = input_data['geometry']['nodes'].astype(float)
    elements_data = get_geometry_elements_for_mode(input_data, is_probabilistic)
    boundary_conditions = input_data.get('boundary', {})
    elastic_modulus = float(
        input_data['geometry']['E_mean']
        if is_probabilistic else
        input_data['geometry'].get('E_deterministic', input_data['geometry']['E_mean'])
    )
    portal = Portal2D(nodes, elements_data, boundary_conditions, elastic_modulus)
    return nodes, portal.elements


def build_preview_distributed_loads(input_data: Dict, is_probabilistic: bool) -> Dict[int, float]:
    """Gabungkan beban mati dan hidup untuk preview gambar input."""
    loads_by_element: Dict[int, float] = {}

    for load_key in ('dead_load', 'live_load'):
        load_data = input_data.get(load_key, {})
        element_ids = [int(elem_id) for elem_id in load_data.get('elements', [])]
        if not element_ids:
            continue

        if is_probabilistic:
            raw_values = load_data.get('mean')
        else:
            raw_values = load_data.get('deterministic')
            if raw_values is None:
                raw_values = load_data.get('mean')

        if raw_values is None:
            continue

        values = np.asarray(raw_values, dtype=float).reshape(-1)
        if values.size == 0:
            continue
        if values.size == 1 and len(element_ids) > 1:
            values = np.full(len(element_ids), float(values[0]), dtype=float)
        if values.size != len(element_ids):
            continue

        for index, elem_id in enumerate(element_ids):
            loads_by_element[elem_id] = loads_by_element.get(elem_id, 0.0) + float(values[index])

    return {
        elem_id: value
        for elem_id, value in loads_by_element.items()
        if abs(value) > 1e-12
    }


def prepare_input_file(path_text: str, uploaded_file) -> Tuple[Optional[str], Optional[str]]:
    """Resolve sumber file Excel dari path lokal atau upload."""
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix or ".xlsx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name, tmp_file.name

    if not path_text:
        return None, None

    file_path = os.path.abspath(path_text)
    if not os.path.exists(file_path):
        return None, None

    return file_path, None


def count_probabilistic_random_variables(input_data: Dict) -> int:
    """Hitung jumlah variabel random yang aktif pada mode probabilistik."""
    bias_count = len(input_data.get('geometry', {}).get('fb_by_element', {}))
    concrete_count = len(input_data.get('concrete', {}).get('by_element', {}))
    steel_count = len(input_data.get('steel', {}).get('by_element', {})) * 3
    dead_load_count = len(input_data.get('dead_load', {}).get('by_element', {}))
    live_load_count = len(input_data.get('live_load', {}).get('by_element', {}))
    return int(bias_count + concrete_count + steel_count + dead_load_count + live_load_count)


def estimate_analysis_runtime_seconds(input_data: Optional[Dict],
                                      is_probabilistic: bool,
                                      num_simulations: int) -> Optional[float]:
    """
    Estimasi runtime lokal berbasis benchmark mesin ini dan kompleksitas input.
    Nilai ini bersifat perkiraan, bukan jaminan waktu eksekusi.
    """
    if input_data is None:
        return None

    elements_data = get_geometry_elements_for_mode(input_data, is_probabilistic)
    num_elements = max(float(len(elements_data)), 1.0)

    if not is_probabilistic:
        complexity_factor = max(num_elements / BASE_ESTIMATE_ELEMENTS, 0.5)
        return float(BASE_DETERMINISTIC_SECONDS * complexity_factor)

    num_random_variables = max(
        float(count_probabilistic_random_variables(input_data)),
        1.0
    )
    complexity_factor = (
        0.75 * (num_elements / BASE_ESTIMATE_ELEMENTS)
        + 0.25 * (num_random_variables / BASE_ESTIMATE_RANDOM_VARIABLES)
    )
    complexity_factor = max(complexity_factor, 0.35)
    monte_carlo_seconds = (
        BASE_MONTE_CARLO_SECONDS_PER_SAMPLE
        * float(num_simulations)
        * complexity_factor
    )
    return float(BASE_FULL_ANALYSIS_FIXED_OVERHEAD_SECONDS + monte_carlo_seconds)


def format_duration_text(seconds: Optional[float]) -> str:
    """Format durasi estimasi menjadi teks ringkas."""
    if seconds is None:
        return "-"

    total_seconds = max(float(seconds), 0.0)
    if total_seconds < 60.0:
        return f"{total_seconds:.0f} detik"

    minutes = int(total_seconds // 60)
    remaining_seconds = int(round(total_seconds - (minutes * 60)))
    if remaining_seconds == 60:
        minutes += 1
        remaining_seconds = 0

    if minutes < 60:
        return f"{minutes} menit {remaining_seconds} detik"

    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours} jam {remaining_minutes} menit"


def cleanup_temporary_file(file_path: Optional[str]) -> None:
    """Hapus file temp upload tanpa membuat dashboard gagal."""
    if not file_path or not os.path.exists(file_path):
        return

    try:
        os.remove(file_path)
    except PermissionError:
        # Pada Windows file Excel kadang masih dipegang engine pembaca.
        # Cleanup gagal tidak boleh membuat analisis Streamlit ikut gagal.
        pass


def build_nodes_df(input_data: Dict) -> pd.DataFrame:
    df = pd.DataFrame(
        input_data['geometry']['nodes'],
        columns=['Node_ID', 'X (mm)', 'Y (mm)']
    )
    df['Node_ID'] = df['Node_ID'].astype(int)
    return df


def build_elements_df(input_data: Dict, is_probabilistic: bool) -> pd.DataFrame:
    geometry = input_data['geometry']
    props_by_element = geometry.get('properties_by_element', {})
    e_column_name = 'E_mean (MPa)' if is_probabilistic else 'E_deterministic (MPa)'
    include_bias_columns = any(
        ('fb_mean' in props) or ('fb_stdev' in props)
        for props in props_by_element.values()
    )

    if props_by_element:
        rows = []
        for elem_id in sorted(props_by_element):
            props = props_by_element[elem_id]
            row = {
                'Element_ID': int(elem_id),
                'Kode': props.get('code', ''),
                'Node_Start': int(props.get('node_start', 0)),
                'Node_End': int(props.get('node_end', 0)),
                'b (mm)': props.get('b'),
                'h (mm)': props.get('h'),
                'Area (mm2)': props.get('area'),
                'Inertia (mm4)': props.get('inertia'),
                e_column_name: (
                    props.get('E_mean')
                    if is_probabilistic else
                    props.get('E_deterministic')
                )
            }
            if include_bias_columns:
                row['fb_mean'] = props.get('fb_mean')
                row['fb_stdev'] = props.get('fb_stdev')
                if is_probabilistic:
                    row['E_acuan_mean (MPa)'] = multiply_finite_values(
                        props.get('E_mean'),
                        props.get('fb_mean')
                    )
            rows.append(row)
        return pd.DataFrame(rows)

    if is_probabilistic:
        base_elements = geometry.get('elements_mean', geometry['elements'])
    else:
        base_elements = geometry.get('elements_deterministic', geometry.get('elements_mean', geometry['elements']))

    df = pd.DataFrame(
        np.asarray(base_elements[:, :5]),
        columns=['Element_ID', 'Node_Start', 'Node_End', 'Area (mm2)', 'Inertia (mm4)']
    )
    if np.asarray(base_elements).shape[1] >= 6:
        df[e_column_name] = np.asarray(base_elements)[:, 5]
    if is_probabilistic and 'E_mean (MPa)' in df.columns:
        bias_lookup = geometry.get('fb_by_element', {}) or {}
        if bias_lookup:
            df['fb_mean'] = df['Element_ID'].map(
                lambda elem_id: get_by_element_value(bias_lookup, int(elem_id), {}).get('mean')
                if isinstance(get_by_element_value(bias_lookup, int(elem_id), {}), dict) else None
            )
            df['fb_stdev'] = df['Element_ID'].map(
                lambda elem_id: get_by_element_value(bias_lookup, int(elem_id), {}).get('stddev')
                if isinstance(get_by_element_value(bias_lookup, int(elem_id), {}), dict) else None
            )
            df['E_acuan_mean (MPa)'] = [
                multiply_finite_values(e_value, fb_value)
                for e_value, fb_value in zip(df['E_mean (MPa)'], df['fb_mean'])
            ]
    for col in ('Element_ID', 'Node_Start', 'Node_End'):
        df[col] = df[col].astype(int)
    return df


def build_boundary_df(boundary_data: Dict) -> pd.DataFrame:
    rows = []
    for node_id, restraint in sorted(boundary_data.items()):
        rows.append({
            'Node_ID': int(node_id),
            'Restrain_X': restraint.get('X', 0),
            'Restrain_Y': restraint.get('Y', 0),
            'Restrain_R': restraint.get('R', 0)
        })
    return pd.DataFrame(rows)


def build_nodal_load_df(nodal_loads: Dict) -> pd.DataFrame:
    rows = []
    for node_id, load in sorted(nodal_loads.items()):
        rows.append({
            'Node_ID': int(node_id),
            'Fx': load.get('Fx', 0.0),
            'Fy': load.get('Fy', 0.0),
            'Mz': load.get('Mz', 0.0)
        })
    return pd.DataFrame(rows)


def build_random_variable_df(random_variables: Dict) -> pd.DataFrame:
    rows = []
    for name, info in sorted(random_variables.items()):
        rows.append({
            'Variable': name,
            'Distribution': info.get('distribution', '-'),
            'Mean': info.get('mean'),
            'StdDev': info.get('stddev')
        })
    return pd.DataFrame(rows)


def build_latest_sample_df(latest_simulation: Dict) -> pd.DataFrame:
    sample = latest_simulation.get('random_sample') or {}
    return pd.DataFrame([
        {'Variable': key, 'Sample Value': value}
        for key, value in sorted(sample.items())
    ])


def build_effective_modulus_snapshot_df(input_data: Dict,
                                        latest_simulation: Dict,
                                        is_probabilistic: bool) -> pd.DataFrame:
    """Bangun snapshot E efektif per elemen untuk simulasi aktif."""
    if not is_probabilistic:
        return pd.DataFrame()

    geometry = input_data.get('geometry', {})
    props_by_element = geometry.get('properties_by_element', {}) or {}
    if not props_by_element:
        return pd.DataFrame()

    random_sample = latest_simulation.get('random_sample') or {}
    rows = []
    for elem_id in sorted(props_by_element):
        props = props_by_element[elem_id] or {}
        e_mean = props.get('E_mean')
        fb_mean = props.get('fb_mean')
        if e_mean is None or fb_mean is None:
            continue

        active_fb = random_sample.get(f'fb_E{int(elem_id)}')
        if active_fb is None:
            active_fb = fb_mean

        rows.append({
            'Element_ID': int(elem_id),
            'Kode': props.get('code', ''),
            'E_mean (MPa)': e_mean,
            'fb_mean': fb_mean,
            'fb_dipakai_DSM (-)': active_fb,
            'E_acuan_mean (MPa)': multiply_finite_values(e_mean, fb_mean),
            'E_dipakai_DSM (MPa)': multiply_finite_values(e_mean, active_fb)
        })

    return pd.DataFrame(rows)


def build_concrete_input_df(input_data: Dict) -> pd.DataFrame:
    rows = []
    for elem_id, props in sorted(input_data.get('concrete', {}).get('by_element', {}).items()):
        rows.append({
            'Element_ID': int(elem_id),
            'fc_Mean (MPa)': props.get('mean'),
            'fc_StdDev (MPa)': props.get('stddev'),
            'Distribution': props.get('distribution'),
            'fc_Deterministic (MPa)': props.get('deterministic')
        })
    return pd.DataFrame(rows)


def build_steel_input_df(input_data: Dict) -> pd.DataFrame:
    rows = []
    for elem_id, props in sorted(input_data.get('steel', {}).get('by_element', {}).items()):
        rows.append({
            'Element_ID': int(elem_id),
            'fy_tarik_Mean (MPa)': props.get('tarik_mean'),
            'fy_tarik_StdDev (MPa)': props.get('tarik_stddev'),
            'fy_tarik_Distribution': props.get('tarik_distribution'),
            'fy_tekan_Mean (MPa)': props.get('tekan_mean'),
            'fy_tekan_StdDev (MPa)': props.get('tekan_stddev'),
            'fy_tekan_Distribution': props.get('tekan_distribution'),
            'fy_geser_Mean (MPa)': props.get('geser_mean'),
            'fy_geser_StdDev (MPa)': props.get('geser_stddev'),
            'fy_geser_Distribution': props.get('geser_distribution'),
            'fy_tarik_Deterministic (MPa)': props.get('tarik_deterministic'),
            'fy_tekan_Deterministic (MPa)': props.get('tekan_deterministic'),
            'fy_geser_Deterministic (MPa)': props.get('geser_deterministic')
        })
    return pd.DataFrame(rows)


def build_reinforcement_input_df(input_data: Dict) -> pd.DataFrame:
    rows = []
    reinforcement_by_element = input_data.get('reinforcement', {}).get('by_element', {})
    steel_by_element = input_data.get('steel', {}).get('by_element', {})

    for elem_id, props in sorted(reinforcement_by_element.items()):
        steel_props = (
            steel_by_element.get(int(elem_id))
            or steel_by_element.get(str(int(elem_id)))
            or {}
        )
        row = {
            'Element_ID': int(elem_id),
            'ds_tarik (mm)': props.get('ds_tarik'),
            'ds_tekan (mm)': props.get('ds_tekan'),
            'd_tarik (mm)': props.get('d_tarik'),
            'd_tekan (mm)': props.get('d_tekan'),
            'n_tarik': props.get('n_tarik'),
            'du_tarik': props.get('du_tarik'),
            'As_tarik (mm2)': props.get('As_tarik'),
            'n_tekan': props.get('n_tekan'),
            'du_tekan': props.get('du_tekan'),
            'As_tekan (mm2)': props.get('As_tekan'),
            'n_geser': props.get('n_geser'),
            'du_geser': props.get('du_geser'),
            'As_geser (mm2)': props.get('As_geser'),
            'Spasi_geser (mm)': props.get('Spasi_geser'),
            'fy_geser_Mean (MPa)': steel_props.get('geser_mean'),
            'fy_geser_StdDev (MPa)': steel_props.get('geser_stddev'),
            'fy_geser_Distribution': steel_props.get('geser_distribution'),
            'fy_geser_Deterministic (MPa)': steel_props.get('geser_deterministic')
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_distributed_load_input_df(load_data: Dict) -> pd.DataFrame:
    rows = []
    for elem_id, props in sorted(load_data.get('by_element', {}).items()):
        rows.append({
            'Element_ID': int(elem_id),
            'Mean (kN/m)': props.get('mean'),
            'StdDev (kN/m)': props.get('stddev'),
            'Distribution': props.get('distribution'),
            'Deterministic (kN/m)': props.get('deterministic')
        })
    return pd.DataFrame(rows)


def build_displacement_df(nodes: np.ndarray, displacements: np.ndarray) -> pd.DataFrame:
    rows = []
    for node in nodes:
        node_id = int(node[0])
        dof = (node_id - 1) * 3
        rows.append({
            'Node_ID (-)': node_id,
            'Ux (mm)': displacements[dof],
            'Uy (mm)': displacements[dof + 1],
            'Rz (rad)': displacements[dof + 2]
        })
    return pd.DataFrame(rows)


def build_reaction_df(nodes: np.ndarray, boundary: Dict, reactions: np.ndarray) -> pd.DataFrame:
    rows = []
    for node in nodes:
        node_id = int(node[0])
        restraints = boundary.get(node_id, {})
        if not any(restraints.get(axis, 0) == 1 for axis in ('X', 'Y', 'R')):
            continue

        dof = (node_id - 1) * 3
        rows.append({
            'Node_ID (-)': node_id,
            'Rx (kN)': reactions[dof] if restraints.get('X', 0) == 1 else np.nan,
            'Ry (kN)': reactions[dof + 1] if restraints.get('Y', 0) == 1 else np.nan,
            'Mz (kN.m)': reactions[dof + 2] if restraints.get('R', 0) == 1 else np.nan
        })
    return pd.DataFrame(rows)


def get_element_code_from_input(input_data: Optional[Dict], elem_id: int) -> str:
    """Ambil kode elemen B/K dari input, fallback ke inferensi orientasi."""
    if not input_data:
        return "-"

    geometry_lookup = input_data.get('geometry', {}).get('properties_by_element', {})
    geometry_props = (
        geometry_lookup.get(int(elem_id))
        or geometry_lookup.get(str(int(elem_id)))
        or {}
    )
    raw_code = str(geometry_props.get('code', '') or '').strip().upper()
    if raw_code in {'B', 'K'}:
        return raw_code

    node_start = int(geometry_props.get('node_start', 0) or 0)
    node_end = int(geometry_props.get('node_end', 0) or 0)
    node_lookup = {
        int(row[0]): np.asarray(row[1:3], dtype=float)
        for row in input_data.get('geometry', {}).get('nodes', [])
    }
    start = node_lookup.get(node_start)
    end = node_lookup.get(node_end)
    if start is None or end is None:
        return "-"

    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    return 'B' if abs(float(delta[0])) >= abs(float(delta[1])) else 'K'


def get_element_type_label(code: str) -> str:
    """Label jenis elemen dari kode B/K."""
    mapping = {
        'B': 'Balok',
        'K': 'Kolom'
    }
    return mapping.get(str(code).strip().upper(), '-')


def build_internal_force_df(latest_result: Dict, input_data: Optional[Dict] = None) -> pd.DataFrame:
    rows = []
    connectivity = build_element_connectivity_lookup(input_data)
    for force in latest_result.get('element_forces', []):
        elem_id = int(force['elem_id'])
        code = get_element_code_from_input(input_data, elem_id)
        node_start, node_end = connectivity.get(elem_id, (None, None))
        moment_start, moment_end_joint, moment_end_internal = (
            get_displayed_element_moment_values(
                code=code,
                raw_start_joint=float(force['moment_start']),
                raw_end_joint=float(force['moment_end']),
                raw_end_internal=float(force.get('moment_end_internal')),
                node_start=node_start,
                node_end=node_end
            )
        )
        rows.append({
            'Element_ID (-)': elem_id,
            'Kode': code,
            'Jenis_Elemen': get_element_type_label(code),
            'Axial_Start (kN)': force['axial_start'],
            'Axial_End_Joint (kN)': force['axial_end'],
            'Axial_End_Internal (kN)': force.get('axial_end_internal'),
            'Max_Axial (kN)': force.get('max_axial'),
            'X_Max_Axial (m)': force.get('x_max_axial'),
            'Shear_Start (kN)': force['shear_start'],
            'Shear_End_Joint (kN)': force['shear_end'],
            'Shear_End_Internal (kN)': force.get('shear_end_internal'),
            'Max_Shear (kN)': force.get('max_shear'),
            'X_Max_Shear (m)': force.get('x_max_shear'),
            'Moment_Start (kN.m)': moment_start,
            'Moment_End_Joint (kN.m)': moment_end_joint,
            'Moment_End_Internal (kN.m)': moment_end_internal,
            'Max_Moment (kN.m)': force.get('max_moment'),
            'X_Max_Moment (m)': force.get('x_max_moment')
        })
    return pd.DataFrame(rows)


def build_internal_force_component_df(internal_force_df: pd.DataFrame,
                                      component: str) -> pd.DataFrame:
    """Ambil tabel gaya dalam per komponen dengan kolom identitas tetap."""
    component_columns = {
        'moment': [
            'Moment_Start (kN.m)',
            'Moment_End_Joint (kN.m)',
            'Moment_End_Internal (kN.m)',
            'Max_Moment (kN.m)',
            'X_Max_Moment (m)'
        ],
        'shear': [
            'Shear_Start (kN)',
            'Shear_End_Joint (kN)',
            'Shear_End_Internal (kN)',
            'Max_Shear (kN)',
            'X_Max_Shear (m)'
        ],
        'axial': [
            'Axial_Start (kN)',
            'Axial_End_Joint (kN)',
            'Axial_End_Internal (kN)',
            'Max_Axial (kN)',
            'X_Max_Axial (m)'
        ]
    }
    identity_columns = ['Element_ID (-)', 'Kode', 'Jenis_Elemen']
    selected_columns = identity_columns + component_columns.get(component, [])
    selected_columns = [
        column for column in selected_columns
        if column in internal_force_df.columns
    ]
    return internal_force_df.loc[:, selected_columns].copy()


def build_internal_force_sign_guide_figure():
    """Bangun panduan visual tanda gaya dalam untuk dibaca di dashboard."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)

    beam_ax, column_ax = axes
    for axis in axes:
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.axis('off')

    beam_ax.set_title("Balok B", fontsize=12, fontweight='bold')
    beam_y = 0.58
    beam_ax.plot([0.15, 0.85], [beam_y, beam_y], color='#1f2937', linewidth=3)
    beam_ax.scatter([0.15, 0.85], [beam_y, beam_y], color='#111827', s=28, zorder=3)
    beam_ax.text(0.15, beam_y + 0.05, "Start", ha='center', fontsize=9)
    beam_ax.text(0.85, beam_y + 0.05, "End", ha='center', fontsize=9)
    beam_ax.annotate(
        "",
        xy=(0.80, 0.88),
        xytext=(0.20, 0.88),
        arrowprops=dict(arrowstyle='-|>', color='#374151', linewidth=1.6)
    )
    beam_ax.text(0.50, 0.91, "Sumbu lokal x: Start -> End", ha='center', fontsize=9, color='#374151')

    beam_ax.annotate("", xy=(0.39, 0.72), xytext=(0.23, 0.72),
                     arrowprops=dict(arrowstyle='-|>', color='#b91c1c', linewidth=1.6))
    beam_ax.annotate("", xy=(0.61, 0.72), xytext=(0.77, 0.72),
                     arrowprops=dict(arrowstyle='-|>', color='#b91c1c', linewidth=1.6))
    beam_ax.text(0.50, 0.75, "Axial + = tekan", ha='center', fontsize=9, color='#991b1b')

    beam_ax.annotate("", xy=(0.23, 0.18), xytext=(0.39, 0.18),
                     arrowprops=dict(arrowstyle='-|>', color='#2563eb', linewidth=1.6))
    beam_ax.annotate("", xy=(0.77, 0.18), xytext=(0.61, 0.18),
                     arrowprops=dict(arrowstyle='-|>', color='#2563eb', linewidth=1.6))
    beam_ax.text(0.50, 0.12, "Axial - = tarik", ha='center', fontsize=9, color='#1d4ed8')

    x_beam = np.linspace(0.23, 0.77, 120)
    positive_curve = beam_y - 0.11 * np.sin(np.pi * (x_beam - x_beam.min()) / (x_beam.max() - x_beam.min()))
    negative_curve = beam_y + 0.11 * np.sin(np.pi * (x_beam - x_beam.min()) / (x_beam.max() - x_beam.min()))
    beam_ax.plot(x_beam, positive_curve, color='#2563eb', linewidth=2.2)
    beam_ax.plot(x_beam, negative_curve, color='#dc2626', linewidth=2.2)
    beam_ax.text(0.50, 0.34, "Momen + di tabel = sagging / lapangan", ha='center', fontsize=9, color='#1d4ed8')
    beam_ax.text(0.50, 0.80, "Momen - di tabel = hogging / tumpuan", ha='center', fontsize=9, color='#991b1b')
    beam_ax.text(
        0.50,
        0.03,
        "Untuk cek joint, konversi tanda balok ditangani otomatis; node 5 memakai tanda joint solver.",
        ha='center',
        fontsize=8.5,
        color='#374151',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#f8fafc', edgecolor='#cbd5e1')
    )

    column_ax.set_title("Kolom K", fontsize=12, fontweight='bold')
    column_x = 0.50
    column_ax.plot([column_x, column_x], [0.16, 0.84], color='#1f2937', linewidth=3)
    column_ax.scatter([column_x, column_x], [0.16, 0.84], color='#111827', s=28, zorder=3)
    column_ax.text(column_x + 0.07, 0.16, "Start", va='center', fontsize=9)
    column_ax.text(column_x + 0.07, 0.84, "End", va='center', fontsize=9)
    column_ax.annotate(
        "",
        xy=(0.18, 0.84),
        xytext=(0.18, 0.20),
        arrowprops=dict(arrowstyle='-|>', color='#374151', linewidth=1.6)
    )
    column_ax.text(0.21, 0.52, "Sumbu lokal x", rotation=90, va='center', fontsize=9, color='#374151')

    column_ax.annotate("", xy=(0.50, 0.42), xytext=(0.50, 0.27),
                       arrowprops=dict(arrowstyle='-|>', color='#b91c1c', linewidth=1.6))
    column_ax.annotate("", xy=(0.50, 0.58), xytext=(0.50, 0.73),
                       arrowprops=dict(arrowstyle='-|>', color='#b91c1c', linewidth=1.6))
    column_ax.text(0.68, 0.50, "Axial + = tekan", fontsize=9, color='#991b1b', va='center')

    column_ax.text(
        0.64,
        0.74,
        "Momen kolom di tabel\nlangsung mengikuti tanda lokal solver.",
        fontsize=9,
        color='#1f2937',
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#f8fafc', edgecolor='#cbd5e1')
    )
    column_ax.text(
        0.64,
        0.43,
        "Untuk cek joint,\nmomen kolom bisa dijumlah langsung.",
        fontsize=9,
        color='#1f2937',
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#f8fafc', edgecolor='#cbd5e1')
    )
    column_ax.text(
        0.64,
        0.18,
        "Shear + / - tetap mengikuti\nsumbu lokal solver per elemen.",
        fontsize=9,
        color='#1f2937',
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#f8fafc', edgecolor='#cbd5e1')
    )

    fig.suptitle("Panduan Visual Tanda Positif dan Negatif pada Tabel Gaya Dalam", fontsize=13, y=0.99)
    fig.tight_layout()
    return fig


def build_internal_force_sign_guide_df() -> pd.DataFrame:
    """Ringkasan aturan tanda gaya dalam yang tampil di tabel output."""
    return pd.DataFrame([
        {
            'Komponen': 'Axial',
            'Jenis': 'Balok dan Kolom',
            'Positif di Tabel': 'Tekan',
            'Negatif di Tabel': 'Tarik',
            'Catatan': 'Berlaku untuk Start, End_Joint, dan End_Internal.'
        },
        {
            'Komponen': 'Moment',
            'Jenis': 'Balok (B)',
            'Positif di Tabel': 'Sagging / lapangan',
            'Negatif di Tabel': 'Hogging / tumpuan',
            'Catatan': (
                'Nilai momen balok pada tabel umumnya dibalik dari solver; '
                'khusus node 5 momen joint balok ditampilkan langsung '
                'mengikuti tanda aksi joint solver.'
            )
        },
        {
            'Komponen': 'Moment',
            'Jenis': 'Kolom (K)',
            'Positif di Tabel': 'Sesuai aksi joint lokal solver',
            'Negatif di Tabel': 'Kebalikan aksi joint lokal solver',
            'Catatan': 'Nilai momen kolom tidak dibalik; bisa langsung dipakai pada cek joint.'
        },
        {
            'Komponen': 'Shear',
            'Jenis': 'Balok dan Kolom',
            'Positif di Tabel': 'Sesuai sumbu lokal solver',
            'Negatif di Tabel': 'Kebalikan sumbu lokal solver',
            'Catatan': 'Untuk cek node gunakan Start atau End_Joint, bukan End_Internal.'
        }
    ])


def build_element_connectivity_lookup(input_data: Optional[Dict]) -> Dict[int, Tuple[int, int]]:
    """Ambil pasangan node awal-akhir setiap elemen dari input geometri."""
    if not input_data:
        return {}

    geometry = input_data.get('geometry', {})
    connectivity: Dict[int, Tuple[int, int]] = {}

    for elem_id, props in (geometry.get('properties_by_element', {}) or {}).items():
        elem_id_int = int(elem_id)
        node_start = int(props.get('node_start', 0) or 0)
        node_end = int(props.get('node_end', 0) or 0)
        if node_start > 0 and node_end > 0:
            connectivity[elem_id_int] = (node_start, node_end)

    base_elements = geometry.get('elements')
    if base_elements is not None:
        for row in np.asarray(base_elements):
            if len(row) < 3:
                continue
            connectivity.setdefault(
                int(row[0]),
                (int(row[1]), int(row[2]))
            )

    return dict(sorted(connectivity.items()))


def build_joint_moment_equilibrium_df(latest_result: Dict,
                                      input_data: Optional[Dict] = None) -> pd.DataFrame:
    """Bangun tabel cek Sigma M per node dari tabel gaya dalam elemen."""
    if not latest_result or not input_data:
        return pd.DataFrame()

    internal_force_df = build_internal_force_df(latest_result, input_data=input_data)
    if internal_force_df.empty:
        return pd.DataFrame()

    force_lookup = {
        int(row['Element_ID (-)']): row
        for _, row in internal_force_df.iterrows()
    }
    connectivity = build_element_connectivity_lookup(input_data)
    if not connectivity:
        return pd.DataFrame()

    geometry_nodes = input_data.get('geometry', {}).get('nodes', [])
    node_ids = sorted({
        int(node[0]) for node in np.asarray(geometry_nodes)
    })
    nodal_load_lookup = input_data.get('nodal_loads', {}) or {}
    boundary_lookup = input_data.get('boundary', {}) or {}
    reactions = np.asarray(latest_result.get('reactions', []), dtype=float)
    nodal_load_vector = np.asarray(latest_result.get('nodal_loads', []), dtype=float)

    rows = []
    for node_id in node_ids:
        connected_count = 0
        sum_element_moment = 0.0
        contribution_labels = []

        for elem_id, (node_start, node_end) in connectivity.items():
            if elem_id not in force_lookup:
                continue

            row = force_lookup[elem_id]
            code = str(row.get('Kode', '') or '').strip().upper()

            table_moment = None
            end_label = None
            joint_node_id = None
            if node_id == node_start:
                table_moment = pd.to_numeric(row.get('Moment_Start (kN.m)'), errors='coerce')
                end_label = 'Start'
                joint_node_id = node_start
            elif node_id == node_end:
                table_moment = pd.to_numeric(row.get('Moment_End_Joint (kN.m)'), errors='coerce')
                end_label = 'End'
                joint_node_id = node_end

            if end_label is None or pd.isna(table_moment):
                continue

            contribution = get_joint_equilibrium_moment(
                float(table_moment),
                code=code,
                joint_node_id=joint_node_id
            )
            connected_count += 1
            sum_element_moment += contribution
            contribution_labels.append(
                f"E{elem_id}({end_label},{code}): tabel={float(table_moment):+.4f} -> cek={contribution:+.4f}"
            )

        dof = (node_id - 1) * 3
        if nodal_load_vector.size > dof + 2:
            nodal_moment = float(nodal_load_vector[dof + 2])
        else:
            nodal_moment = float(
                (nodal_load_lookup.get(node_id) or nodal_load_lookup.get(str(node_id)) or {}).get('Mz', 0.0) or 0.0
            )

        boundary_props = (
            boundary_lookup.get(node_id)
            or boundary_lookup.get(str(node_id))
            or {}
        )
        if reactions.size > dof + 2 and int(boundary_props.get('R', 0) or 0) == 1:
            reaction_moment = float(reactions[dof + 2])
        else:
            reaction_moment = 0.0

        residual = sum_element_moment + nodal_moment - reaction_moment
        status = (
            'OK'
            if abs(residual) <= MOMENT_EQUILIBRIUM_TOLERANCE_KNM else
            'PERLU CEK'
        )

        rows.append({
            'Node_ID (-)': int(node_id),
            'Jumlah_Elemen_Terkoneksi (-)': connected_count,
            'Kontribusi_Elemen_dari_Tabel (-)': (
                ' | '.join(contribution_labels) if contribution_labels else '-'
            ),
            'Sigma_M_Elemen_Cek (kN.m)': float(sum_element_moment),
            'Mz_Beban_Nodal (kN.m)': float(nodal_moment),
            'Mz_Reaksi (kN.m)': float(reaction_moment),
            'Residual_Sigma_M (kN.m)': float(residual),
            'Status_Equilibrium': status
        })

    return pd.DataFrame(rows)


def style_internal_force_df(df: pd.DataFrame):
    """Highlight nilai maksimum absolut terpisah untuk grup Balok dan Kolom."""
    styler = style_input_dataframe(df)
    styler = apply_grouped_header_styles(
        styler,
        df,
        {
            'identity': ['Element_ID (-)', 'Kode', 'Jenis_Elemen'],
            'axial': [
                'Axial_Start (kN)',
                'Axial_End_Joint (kN)',
                'Axial_End_Internal (kN)',
                'Max_Axial (kN)',
                'X_Max_Axial (m)'
            ],
            'shear': [
                'Shear_Start (kN)',
                'Shear_End_Joint (kN)',
                'Shear_End_Internal (kN)',
                'Max_Shear (kN)',
                'X_Max_Shear (m)'
            ],
            'moment': [
                'Moment_Start (kN.m)',
                'Moment_End_Joint (kN.m)',
                'Moment_End_Internal (kN.m)',
                'Max_Moment (kN.m)',
                'X_Max_Moment (m)'
            ]
        }
    )
    if df.empty or 'Kode' not in df.columns:
        return styler

    identity_columns = [
        'Element_ID (-)',
        'Kode',
        'Jenis_Elemen'
    ]
    identity_columns = [col for col in identity_columns if col in df.columns]

    highlight_columns = [
        'Axial_Start (kN)',
        'Axial_End_Joint (kN)',
        'Axial_End_Internal (kN)',
        'Max_Axial (kN)',
        'Shear_Start (kN)',
        'Shear_End_Joint (kN)',
        'Shear_End_Internal (kN)',
        'Max_Shear (kN)',
        'Moment_Start (kN.m)',
        'Moment_End_Joint (kN.m)',
        'Moment_End_Internal (kN.m)',
        'Max_Moment (kN.m)'
    ]
    highlight_columns = [col for col in highlight_columns if col in df.columns]

    def highlight_max_abs(dataframe: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame('', index=dataframe.index, columns=dataframe.columns)
        for code in ('B', 'K'):
            group_mask = dataframe['Kode'].astype(str).str.upper() == code
            if not group_mask.any():
                continue
            highlight_style = INTERNAL_FORCE_MAX_HIGHLIGHT_STYLES.get(
                code,
                (
                    'background-color: #f8d7da; '
                    'font-weight: 700; '
                    'color: #7a0019;'
                )
            )

            for col in highlight_columns:
                series = pd.to_numeric(dataframe.loc[group_mask, col], errors='coerce')
                winner_group_mask = get_max_abs_winner_mask(series)
                if not winner_group_mask.any():
                    continue
                winner_mask = pd.Series(False, index=dataframe.index)
                winner_mask.loc[winner_group_mask.index] = winner_group_mask
                styles.loc[winner_mask, col] = highlight_style
                if identity_columns:
                    styles.loc[winner_mask, identity_columns] = highlight_style
        return styles

    styler = styler.apply(highlight_max_abs, axis=None)
    return styler


def style_joint_moment_equilibrium_df(df: pd.DataFrame):
    """Warna header dan status untuk tabel cek Sigma M per node."""
    styler = style_input_dataframe(df)
    styler = apply_grouped_header_styles(
        styler,
        df,
        {
            'identity': [
                'Node_ID (-)',
                'Jumlah_Elemen_Terkoneksi (-)',
                'Kontribusi_Elemen_dari_Tabel (-)'
            ],
            'moment': [
                'Sigma_M_Elemen_Cek (kN.m)',
                'Residual_Sigma_M (kN.m)'
            ],
            'load': ['Mz_Beban_Nodal (kN.m)'],
            'overall': ['Mz_Reaksi (kN.m)', 'Status_Equilibrium']
        }
    )
    if df.empty:
        return styler

    def highlight_status(dataframe: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame('', index=dataframe.index, columns=dataframe.columns)
        residual_column = 'Residual_Sigma_M (kN.m)'
        status_column = 'Status_Equilibrium'
        residual_values = pd.to_numeric(dataframe[residual_column], errors='coerce')
        ok_mask = residual_values.abs() <= MOMENT_EQUILIBRIUM_TOLERANCE_KNM

        styles.loc[ok_mask, residual_column] = JOINT_MOMENT_EQUILIBRIUM_STATUS_STYLES['OK']
        styles.loc[~ok_mask, residual_column] = JOINT_MOMENT_EQUILIBRIUM_STATUS_STYLES['PERLU CEK']

        status_styles = dataframe[status_column].astype(str).map(
            lambda value: JOINT_MOMENT_EQUILIBRIUM_STATUS_STYLES.get(value, '')
        )
        styles.loc[:, status_column] = status_styles
        return styles

    return styler.apply(highlight_status, axis=None)


def style_performance_df(df: pd.DataFrame):
    """Warna header tabel nilai g per kelompok limit state."""
    styler = style_input_dataframe(df)
    return apply_grouped_header_styles(
        styler,
        df,
        {
            'identity': ['Element_ID (-)', 'Kode'],
            'summary': [
                'Kesimpulan_Akhir',
                'Basis_Kesimpulan',
                'Limit_State_Kontrol',
                'g_Kontrol',
                'Pf_Kontrol (-)',
                'Beta_Kontrol (-)',
                'Status_Akhir'
            ],
            'overall': [
                'Pf_Elemen (-)',
                'Beta_Elemen (-)',
                'Failures_Elemen (-)',
                'Status_Elemen'
            ],
            'moment': [
                'g_Moment (kN.m)',
                'Status_Moment',
                'Pf_Moment (-)',
                'Beta_Moment (-)',
                'Phi_Moment (-)',
                'Epsilon_t_Moment (-)',
                'Klasifikasi_Moment'
            ],
            'shear': [
                'g_Shear (kN)',
                'Status_Shear',
                'Pf_Shear (-)',
                'Beta_Shear (-)',
                'Phi_Shear (-)'
            ],
            'axial': [
                'g_Axial (kN)',
                'Status_Axial',
                'Pf_Axial (-)',
                'Beta_Axial (-)',
                'Phi_Axial (-)',
                'Epsilon_t_Axial (-)',
                'Klasifikasi_Axial'
            ],
            'axial_moment': [
                'g_Axial+Moment_Column (-)',
                'Status_Axial+Moment',
                'Pf_Axial+Moment (-)',
                'Beta_Axial+Moment (-)',
                'Phi_Axial+Moment (-)',
                'Epsilon_t_Axial+Moment (-)',
                'Klasifikasi_Axial+Moment'
            ]
        }
    )


def build_portal_system_reliability_df(system_results: Optional[List[Dict]]) -> pd.DataFrame:
    """Bangun tabel reliabilitas sistem portal gabungan."""
    rows = []
    for result in system_results or []:
        rows.append({
            'Kasus_Sistem': result.get('case_name'),
            'Deskripsi_Sistem': result.get('description'),
            'Sistem_Balok': result.get('beam_system'),
            'Jumlah_Balok (-)': result.get('num_beams'),
            'Failures_Balok (-)': result.get('beam_failures'),
            'Pf_Balok (-)': result.get('beam_pf'),
            'Beta_Balok (-)': result.get('beam_beta'),
            'Status_Balok': result.get('beam_status'),
            'Sistem_Kolom': result.get('column_system'),
            'Jumlah_Kolom (-)': result.get('num_columns'),
            'Failures_Kolom (-)': result.get('column_failures'),
            'Pf_Kolom (-)': result.get('column_pf'),
            'Beta_Kolom (-)': result.get('column_beta'),
            'Status_Kolom': result.get('column_status'),
            'Kriteria_Gagal_Portal': result.get('portal_failure_rule'),
            'Failures_Portal (-)': result.get('portal_failures'),
            'Pf_Portal (-)': result.get('portal_pf'),
            'Beta_Portal (-)': result.get('portal_beta'),
            'Status_Portal': result.get('portal_status')
        })
    df = pd.DataFrame(rows)
    numeric_columns = [
        'Jumlah_Balok (-)',
        'Failures_Balok (-)',
        'Pf_Balok (-)',
        'Beta_Balok (-)',
        'Jumlah_Kolom (-)',
        'Failures_Kolom (-)',
        'Pf_Kolom (-)',
        'Beta_Kolom (-)',
        'Failures_Portal (-)',
        'Pf_Portal (-)',
        'Beta_Portal (-)'
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


def style_portal_system_reliability_df(df: pd.DataFrame):
    """Warna header tabel reliabilitas sistem portal gabungan."""
    styler = style_input_dataframe(df)
    return apply_grouped_header_styles(
        styler,
        df,
        {
            'identity': ['Kasus_Sistem', 'Deskripsi_Sistem'],
            'beam_system': [
                'Sistem_Balok',
                'Jumlah_Balok (-)',
                'Failures_Balok (-)',
                'Pf_Balok (-)',
                'Beta_Balok (-)',
                'Status_Balok'
            ],
            'column_system': [
                'Sistem_Kolom',
                'Jumlah_Kolom (-)',
                'Failures_Kolom (-)',
                'Pf_Kolom (-)',
                'Beta_Kolom (-)',
                'Status_Kolom'
            ],
            'portal_system': [
                'Kriteria_Gagal_Portal',
                'Failures_Portal (-)',
                'Pf_Portal (-)',
                'Beta_Portal (-)',
                'Status_Portal'
            ]
        }
    )


def build_performance_df(latest_result: Dict,
                         input_data: Optional[Dict] = None,
                         element_reliability: Optional[Dict] = None) -> pd.DataFrame:
    target_beta_uls = 3.0
    moment_values = latest_result.get('performance', {})
    shear_values = latest_result.get('performance_shear', {})
    axial_values = latest_result.get('performance_axial', {})
    axial_moment_values = latest_result.get('performance_axial_moment', {})
    moment_meta_values = latest_result.get('performance_metadata', {})
    shear_meta_values = latest_result.get('performance_shear_metadata', {})
    axial_meta_values = latest_result.get('performance_axial_metadata', {})
    axial_moment_meta_values = latest_result.get('performance_axial_moment_metadata', {})
    geometry_lookup = {}
    element_reliability = element_reliability or {}
    if input_data:
        geometry_lookup = input_data.get('geometry', {}).get('properties_by_element', {})

    def get_by_element(source: Dict, elem_id: int, default=None):
        if elem_id in source:
            return source[elem_id]
        elem_id_str = str(int(elem_id))
        if elem_id_str in source:
            return source[elem_id_str]
        return default

    def get_reliability(limit_state: str, elem_id: int) -> Dict:
        return get_by_element(element_reliability.get(limit_state, {}), elem_id, {}) or {}

    def normalize_pf(value):
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(numeric):
            return None
        return numeric

    def normalize_beta(value):
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(numeric):
            return None
        return numeric

    def normalize_g(value):
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(numeric):
            return None
        return numeric

    def get_governing_reliability_summary(elem_id: int,
                                          moment_rel: Dict,
                                          shear_rel: Dict,
                                          axial_rel: Dict,
                                          axial_moment_rel: Dict) -> Dict:
        candidates = []
        for order, (label, key, reliability) in enumerate((
            ('Momen', 'moment', moment_rel),
            ('Geser', 'shear', shear_rel),
            ('Aksial', 'axial', axial_rel),
            ('Aksial+Momen', 'axial_moment', axial_moment_rel)
        )):
            pf_value = normalize_pf(reliability.get('Pf'))
            beta_value = normalize_beta(reliability.get('Beta'))
            if pf_value is None and beta_value is None:
                continue
            candidates.append({
                'label': label,
                'key': key,
                'Pf': pf_value,
                'Beta': beta_value,
                'order': order
            })

        if not candidates:
            return {
                'limit_state': '-',
                'Pf': None,
                'Beta': None,
                'status': '-',
                'kesimpulan': '-'
            }

        governing = min(
            candidates,
            key=lambda item: (
                -(item['Pf'] if item['Pf'] is not None else float('-inf')),
                item['Beta'] if item['Beta'] is not None else float('inf'),
                item['order']
            )
        )
        beta_value = governing['Beta']
        status = (
            '-'
            if beta_value is None else
            ('SAFE' if beta_value >= target_beta_uls else 'UNSAFE')
        )
        return {
            'basis': 'Reliabilitas',
            'limit_state': governing['label'],
            'g': None,
            'Pf': governing['Pf'],
            'Beta': governing['Beta'],
            'status': status,
            'kesimpulan': (
                '-'
                if status == '-' else
                f"{status} - dikontrol {governing['label']}"
            )
        }

    def get_governing_deterministic_summary(g_moment,
                                            g_shear,
                                            g_axial,
                                            g_axial_moment) -> Dict:
        candidates = []
        for order, (label, g_value) in enumerate((
            ('Momen', normalize_g(g_moment)),
            ('Geser', normalize_g(g_shear)),
            ('Aksial', normalize_g(g_axial)),
            ('Aksial+Momen', normalize_g(g_axial_moment))
        )):
            if g_value is None:
                continue
            candidates.append({
                'label': label,
                'g': g_value,
                'order': order
            })

        if not candidates:
            return {
                'basis': 'Deterministik (g)',
                'limit_state': '-',
                'g': None,
                'Pf': None,
                'Beta': None,
                'status': '-',
                'kesimpulan': '-'
            }

        governing = min(
            candidates,
            key=lambda item: (item['g'], item['order'])
        )
        status = 'SAFE' if governing['g'] >= 0.0 else 'UNSAFE'
        return {
            'basis': 'Deterministik (g)',
            'limit_state': governing['label'],
            'g': governing['g'],
            'Pf': None,
            'Beta': None,
            'status': status,
            'kesimpulan': f"{status} - dikontrol {governing['label']}"
        }

    element_id_keys = (
        set(moment_values.keys())
        | set(shear_values.keys())
        | set(axial_values.keys())
        | set(axial_moment_values.keys())
        | set(moment_meta_values.keys())
        | set(shear_meta_values.keys())
        | set(axial_meta_values.keys())
        | set(axial_moment_meta_values.keys())
    )
    element_ids = sorted({int(elem_id) for elem_id in element_id_keys})

    rows = []
    for elem_id in element_ids:
        geometry_props = (
            geometry_lookup.get(int(elem_id))
            or geometry_lookup.get(str(int(elem_id)))
            or {}
        )
        g_moment = get_by_element(moment_values, elem_id)
        g_shear = get_by_element(shear_values, elem_id)
        g_axial = get_by_element(axial_values, elem_id)
        g_axial_moment = get_by_element(axial_moment_values, elem_id)
        moment_meta = get_by_element(moment_meta_values, elem_id, {})
        shear_meta = get_by_element(shear_meta_values, elem_id, {})
        axial_meta = get_by_element(axial_meta_values, elem_id, {})
        axial_moment_meta = get_by_element(axial_moment_meta_values, elem_id, {})
        overall_reliability = get_reliability('overall', elem_id)
        moment_reliability = get_reliability('moment', elem_id)
        shear_reliability = get_reliability('shear', elem_id)
        axial_reliability = get_reliability('axial', elem_id)
        axial_moment_reliability = get_reliability('axial_moment', elem_id)
        governing_reliability = get_governing_reliability_summary(
            elem_id,
            moment_reliability,
            shear_reliability,
            axial_reliability,
            axial_moment_reliability
        )
        governing_deterministic = get_governing_deterministic_summary(
            g_moment,
            g_shear,
            g_axial,
            g_axial_moment
        )
        final_summary = (
            governing_reliability
            if governing_reliability.get('limit_state') != '-'
            else governing_deterministic
        )
        status_moment = "-" if g_moment is None else ("SAFE" if g_moment >= 0 else "UNSAFE")
        status_shear = "-" if g_shear is None else ("SAFE" if g_shear >= 0 else "UNSAFE")
        status_axial = "-" if g_axial is None else ("SAFE" if g_axial >= 0 else "UNSAFE")
        status_axial_moment = (
            "-"
            if g_axial_moment is None else
            ("SAFE" if g_axial_moment >= 0 else "UNSAFE")
        )
        element_safe_flags = [
            flag for flag in (g_moment, g_shear, g_axial, g_axial_moment)
            if flag is not None
        ]
        status_element = (
            "-"
            if not element_safe_flags else
            ("SAFE" if all(flag >= 0 for flag in element_safe_flags) else "UNSAFE")
        )
        rows.append({
            'Element_ID (-)': int(elem_id),
            'Kode': geometry_props.get('code', ''),
            'Kesimpulan_Akhir': final_summary.get('kesimpulan'),
            'Basis_Kesimpulan': final_summary.get('basis'),
            'Limit_State_Kontrol': final_summary.get('limit_state'),
            'g_Kontrol': final_summary.get('g'),
            'Pf_Kontrol (-)': final_summary.get('Pf'),
            'Beta_Kontrol (-)': final_summary.get('Beta'),
            'Status_Akhir': final_summary.get('status'),
            'Pf_Elemen (-)': overall_reliability.get('Pf'),
            'Beta_Elemen (-)': overall_reliability.get('Beta'),
            'Failures_Elemen (-)': overall_reliability.get('failures'),
            'g_Moment (kN.m)': g_moment,
            'Status_Moment': status_moment,
            'Pf_Moment (-)': moment_reliability.get('Pf'),
            'Beta_Moment (-)': moment_reliability.get('Beta'),
            'Phi_Moment (-)': moment_meta.get('phi'),
            'Epsilon_t_Moment (-)': moment_meta.get('epsilon_t'),
            'Klasifikasi_Moment': moment_meta.get('classification'),
            'g_Shear (kN)': g_shear,
            'Status_Shear': status_shear,
            'Pf_Shear (-)': shear_reliability.get('Pf'),
            'Beta_Shear (-)': shear_reliability.get('Beta'),
            'Phi_Shear (-)': shear_meta.get('phi'),
            'g_Axial (kN)': g_axial,
            'Status_Axial': status_axial,
            'Pf_Axial (-)': axial_reliability.get('Pf'),
            'Beta_Axial (-)': axial_reliability.get('Beta'),
            'Phi_Axial (-)': axial_meta.get('phi'),
            'Epsilon_t_Axial (-)': axial_meta.get('epsilon_t'),
            'Klasifikasi_Axial': axial_meta.get('classification'),
            'g_Axial+Moment_Column (-)': g_axial_moment,
            'Status_Axial+Moment': status_axial_moment,
            'Pf_Axial+Moment (-)': axial_moment_reliability.get('Pf'),
            'Beta_Axial+Moment (-)': axial_moment_reliability.get('Beta'),
            'Phi_Axial+Moment (-)': axial_moment_meta.get('phi'),
            'Epsilon_t_Axial+Moment (-)': axial_moment_meta.get('epsilon_t'),
            'Klasifikasi_Axial+Moment': axial_moment_meta.get('classification'),
            'Status_Elemen': status_element
        })

    return pd.DataFrame(rows)


def compute_limit_state_safety_factor(capacity: Any,
                                      demand: Any) -> Optional[float]:
    """Hitung safety factor SF = R / S bila kapasitas dan demand valid."""
    try:
        capacity_value = float(capacity)
        demand_value = float(demand)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(capacity_value) or not np.isfinite(demand_value):
        return None
    if abs(demand_value) <= 1e-12:
        return float('inf') if capacity_value > 0.0 else None
    return float(capacity_value / demand_value)


def build_limit_state_performance_tables(latest_result: Dict,
                                         input_data: Optional[Dict] = None,
                                         element_reliability: Optional[Dict] = None,
                                         is_probabilistic: bool = True) -> Dict[str, pd.DataFrame]:
    """Pisahkan tabel nilai g per elemen per limit state."""
    max_forces_values = latest_result.get('max_forces', {})
    moment_values = latest_result.get('performance', {})
    shear_values = latest_result.get('performance_shear', {})
    axial_values = latest_result.get('performance_axial', {})
    axial_moment_values = latest_result.get('performance_axial_moment', {})
    moment_meta_values = latest_result.get('performance_metadata', {})
    shear_meta_values = latest_result.get('performance_shear_metadata', {})
    axial_meta_values = latest_result.get('performance_axial_metadata', {})
    axial_moment_meta_values = latest_result.get('performance_axial_moment_metadata', {})
    geometry_lookup = {}
    element_reliability = element_reliability or {}
    if input_data:
        geometry_lookup = input_data.get('geometry', {}).get('properties_by_element', {})

    def get_by_element(source: Dict, elem_id: int, default=None):
        if elem_id in source:
            return source[elem_id]
        elem_id_str = str(int(elem_id))
        if elem_id_str in source:
            return source[elem_id_str]
        return default

    def normalize_numeric(value):
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(numeric):
            return None
        return numeric

    def get_status(g_value):
        g_numeric = normalize_numeric(g_value)
        if g_numeric is None:
            return "-"
        return "SAFE" if g_numeric >= 0.0 else "UNSAFE"

    def get_reliability(limit_state: str, elem_id: int) -> Dict:
        return get_by_element(element_reliability.get(limit_state, {}), elem_id, {}) or {}

    def collect_element_ids(*sources: Dict) -> list[int]:
        element_ids = set()
        for source in sources:
            for elem_id in (source or {}).keys():
                try:
                    element_ids.add(int(elem_id))
                except (TypeError, ValueError):
                    continue
        return sorted(element_ids)

    def get_element_code(elem_id: int) -> str:
        geometry_props = (
            geometry_lookup.get(int(elem_id))
            or geometry_lookup.get(str(int(elem_id)))
            or {}
        )
        return str(geometry_props.get('code', '') or '')

    def get_axial_demands(max_forces_entry: Dict) -> Dict[str, float]:
        force_data = (max_forces_entry or {}).get('forces', {}) or {}
        axial_values_local = [
            float(force_data.get('axial_start', 0.0)),
            float(force_data.get(
                'axial_end_internal',
                force_data.get('axial_end', 0.0)
            ))
        ]
        axial_values_local = [
            0.0 if abs(value) <= AXIAL_DEMAND_TOLERANCE_KN else value
            for value in axial_values_local
        ]
        compression_demand = max([0.0] + axial_values_local)
        tension_demand = max(0.0, -min(axial_values_local))
        return {
            'compression': float(compression_demand),
            'tension': float(tension_demand)
        }

    moment_rows = []
    for elem_id in collect_element_ids(
        max_forces_values,
        moment_values,
        moment_meta_values,
        element_reliability.get('moment', {})
    ):
        max_forces_entry = get_by_element(max_forces_values, elem_id, {}) or {}
        demand = normalize_numeric(max_forces_entry.get('max_moment'))
        demand = abs(demand) if demand is not None else None
        g_value = normalize_numeric(get_by_element(moment_values, elem_id))
        meta = get_by_element(moment_meta_values, elem_id, {}) or {}
        capacity = normalize_numeric(meta.get('phi_Mn'))
        if capacity is None and demand is not None and g_value is not None:
            capacity = g_value + demand
        reliability = get_reliability('moment', elem_id)
        row_data = {
            'Elemen (-)': int(elem_id),
            'Kode': get_element_code(elem_id),
            'Kapasitas R (kN.m)': capacity,
            'S dari Analisis Struktur (kN.m)': demand,
            'phi (-)': normalize_numeric(meta.get('phi'))
        }
        if is_probabilistic:
            row_data['Jumlah Gagal (-)'] = reliability.get('failures')
            row_data['g(x) (kN.m)'] = g_value
            row_data['Pf (-)'] = reliability.get('Pf')
            row_data['Beta (-)'] = reliability.get('Beta')
        else:
            row_data['g(x) (kN.m)'] = g_value
            row_data['SF = R/S (-)'] = compute_limit_state_safety_factor(capacity, demand)
        row_data['Status'] = get_status(g_value)
        moment_rows.append(row_data)

    shear_rows = []
    for elem_id in collect_element_ids(
        max_forces_values,
        shear_values,
        shear_meta_values,
        element_reliability.get('shear', {})
    ):
        max_forces_entry = get_by_element(max_forces_values, elem_id, {}) or {}
        demand = normalize_numeric(max_forces_entry.get('max_shear'))
        demand = abs(demand) if demand is not None else None
        g_value = normalize_numeric(get_by_element(shear_values, elem_id))
        meta = get_by_element(shear_meta_values, elem_id, {}) or {}
        capacity = normalize_numeric(meta.get('phi_Vn'))
        if capacity is None and demand is not None and g_value is not None:
            capacity = g_value + demand
        reliability = get_reliability('shear', elem_id)
        row_data = {
            'Elemen (-)': int(elem_id),
            'Kode': get_element_code(elem_id),
            'Kapasitas R (kN)': capacity,
            'S dari Analisis Struktur (kN)': demand,
            'phi (-)': normalize_numeric(meta.get('phi'))
        }
        if is_probabilistic:
            row_data['Jumlah Gagal (-)'] = reliability.get('failures')
            row_data['g(x) (kN)'] = g_value
            row_data['Pf (-)'] = reliability.get('Pf')
            row_data['Beta (-)'] = reliability.get('Beta')
        else:
            row_data['g(x) (kN)'] = g_value
            row_data['SF = R/S (-)'] = compute_limit_state_safety_factor(capacity, demand)
        row_data['Status'] = get_status(g_value)
        shear_rows.append(row_data)

    axial_rows = []
    for elem_id in collect_element_ids(
        max_forces_values,
        axial_values,
        axial_meta_values,
        element_reliability.get('axial', {})
    ):
        max_forces_entry = get_by_element(max_forces_values, elem_id, {}) or {}
        demands = get_axial_demands(max_forces_entry)
        g_value = normalize_numeric(get_by_element(axial_values, elem_id))
        meta = get_by_element(axial_meta_values, elem_id, {}) or {}
        controlling_state = str(meta.get('controlling_state', '') or '').strip().lower()
        phi_pn = normalize_numeric(meta.get('phi_Pn'))

        if controlling_state == 'absolute-axial':
            demand = normalize_numeric(meta.get('demand_axial_abs'))
            if demand is None:
                demand = max(
                    normalize_numeric(demands['compression']) or 0.0,
                    normalize_numeric(demands['tension']) or 0.0
                )
            capacity = (
                normalize_numeric(meta.get('phi_Pn_tekan'))
                if normalize_numeric(meta.get('phi_Pn_tekan')) is not None else
                phi_pn
            )
        elif controlling_state == 'tension':
            demand = normalize_numeric(demands['tension'])
            capacity = max(-phi_pn, 0.0) if phi_pn is not None else None
        else:
            demand = normalize_numeric(demands['compression'])
            capacity = phi_pn

        if capacity is None and demand is not None and g_value is not None:
            capacity = g_value + demand

        reliability = get_reliability('axial', elem_id)
        row_data = {
            'Elemen (-)': int(elem_id),
            'Kode': get_element_code(elem_id),
            'Kapasitas R (kN)': capacity,
            'S dari Analisis Struktur (kN)': demand,
            'phi (-)': normalize_numeric(meta.get('phi'))
        }
        if is_probabilistic:
            row_data['Jumlah Gagal (-)'] = reliability.get('failures')
            row_data['g(x) (kN)'] = g_value
            row_data['Pf (-)'] = reliability.get('Pf')
            row_data['Beta (-)'] = reliability.get('Beta')
        else:
            row_data['g(x) (kN)'] = g_value
            row_data['SF = R/S (-)'] = compute_limit_state_safety_factor(capacity, demand)
        row_data['Status'] = get_status(g_value)
        axial_rows.append(row_data)

    axial_moment_rows = []
    for elem_id in collect_element_ids(
        max_forces_values,
        axial_moment_values,
        axial_moment_meta_values,
        element_reliability.get('axial_moment', {})
    ):
        g_value = normalize_numeric(get_by_element(axial_moment_values, elem_id))
        meta = get_by_element(axial_moment_meta_values, elem_id, {}) or {}
        capacity = normalize_numeric(meta.get('lambda'))
        demand = 1.0 if capacity is not None or g_value is not None else None
        reliability = get_reliability('axial_moment', elem_id)
        row_data = {
            'Elemen (-)': int(elem_id),
            'Kode': get_element_code(elem_id),
            'Kapasitas R (-)': capacity,
            'S dari Analisis Struktur (-)': demand,
            'phi (-)': normalize_numeric(meta.get('phi'))
        }
        if is_probabilistic:
            row_data['Jumlah Gagal (-)'] = reliability.get('failures')
            row_data['g(x) (-)'] = g_value
            row_data['Pf (-)'] = reliability.get('Pf')
            row_data['Beta (-)'] = reliability.get('Beta')
        else:
            row_data['g(x) (-)'] = g_value
            row_data['SF = R/S (-)'] = compute_limit_state_safety_factor(capacity, demand)
        row_data['Status'] = get_status(g_value)
        axial_moment_rows.append(row_data)

    return {
        'lentur': pd.DataFrame(moment_rows),
        'geser': pd.DataFrame(shear_rows),
        'aksial': pd.DataFrame(axial_rows),
        'aksial_lentur': pd.DataFrame(axial_moment_rows)
    }


def style_limit_state_performance_df(df: pd.DataFrame,
                                     is_probabilistic: bool = True):
    """Styling sederhana untuk tabel limit state terpisah."""
    styler = style_input_dataframe(df)
    reliability_columns = (
        ['Jumlah Gagal (-)', 'Pf (-)', 'Beta (-)', 'Status']
        if is_probabilistic else
        ['Status']
    )
    return apply_grouped_header_styles(
        styler,
        df,
        {
            'identity': ['Elemen (-)', 'Kode'],
            'performance': [
                column
                for column in df.columns
                if column.startswith('Kapasitas R')
                or column.startswith('S dari Analisis Struktur')
                or column == 'phi (-)'
                or column == 'SF = R/S (-)'
                or column.startswith('g(x)')
            ],
            'reliability': reliability_columns
        }
    )


def build_limit_state_resume_df(limit_state_tables: Dict[str, pd.DataFrame],
                                is_probabilistic: bool = True) -> pd.DataFrame:
    """Ringkas hasil pengontrol per elemen dari tabel limit state terpisah."""
    state_specs = [
        ('lentur', 'Lentur', 'kN.m'),
        ('geser', 'Geser', 'kN'),
        ('aksial', 'Aksial', 'kN'),
        ('aksial_lentur', 'Aksial-Lentur', '-')
    ]

    def normalize_numeric(value):
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(numeric):
            return None
        return numeric

    candidates_by_element: Dict[int, list[Dict[str, Any]]] = {}
    for order, (table_key, label, unit) in enumerate(state_specs):
        table_df = limit_state_tables.get(table_key, pd.DataFrame())
        if table_df is None or table_df.empty:
            continue

        for _, row in table_df.iterrows():
            elem_id = row.get('Elemen (-)')
            try:
                elem_id = int(elem_id)
            except (TypeError, ValueError):
                continue

            candidate = {
                'Elemen (-)': elem_id,
                'Kode': row.get('Kode'),
                'Limit State Kontrol': label,
                'Satuan': unit,
                'Kapasitas R': row.get(next(
                    (col for col in table_df.columns if col.startswith('Kapasitas R')),
                    None
                )),
                'S dari Analisis Struktur': row.get(next(
                    (col for col in table_df.columns if col.startswith('S dari Analisis Struktur')),
                    None
                )),
                'phi (-)': row.get('phi (-)'),
                'SF = R/S (-)': row.get('SF = R/S (-)'),
                'g(x)': row.get(next(
                    (col for col in table_df.columns if col.startswith('g(x)')),
                    None
                )),
                'Jumlah Gagal (-)': row.get('Jumlah Gagal (-)'),
                'Pf (-)': row.get('Pf (-)'),
                'Beta (-)': row.get('Beta (-)'),
                'Status': row.get('Status'),
                'order': order
            }
            candidates_by_element.setdefault(elem_id, []).append(candidate)

    rows = []
    for elem_id in sorted(candidates_by_element):
        candidates = candidates_by_element[elem_id]
        reliability_candidates = [
            candidate for candidate in candidates
            if normalize_numeric(candidate.get('Pf (-)')) is not None
            or normalize_numeric(candidate.get('Beta (-)')) is not None
        ]

        if reliability_candidates:
            governing = min(
                reliability_candidates,
                key=lambda item: (
                    -(normalize_numeric(item.get('Pf (-)')) if normalize_numeric(item.get('Pf (-)')) is not None else float('-inf')),
                    normalize_numeric(item.get('Beta (-)')) if normalize_numeric(item.get('Beta (-)')) is not None else float('inf'),
                    item.get('order', 0)
                )
            )
        else:
            deterministic_candidates = [
                candidate for candidate in candidates
                if normalize_numeric(candidate.get('g(x)')) is not None
            ]
            if not deterministic_candidates:
                governing = min(candidates, key=lambda item: item.get('order', 0))
            else:
                governing = min(
                    deterministic_candidates,
                    key=lambda item: (
                        normalize_numeric(item.get('g(x)')),
                        item.get('order', 0)
                    )
                )

        row_data = {
            'Elemen (-)': governing.get('Elemen (-)'),
            'Kode': governing.get('Kode'),
            'Limit State Kontrol': governing.get('Limit State Kontrol'),
            'Satuan': governing.get('Satuan'),
            'Kapasitas R': governing.get('Kapasitas R'),
            'S dari Analisis Struktur': governing.get('S dari Analisis Struktur'),
            'phi (-)': governing.get('phi (-)')
        }
        if is_probabilistic:
            row_data['Jumlah Gagal (-)'] = governing.get('Jumlah Gagal (-)')
            row_data['g(x)'] = governing.get('g(x)')
            row_data['Pf (-)'] = governing.get('Pf (-)')
            row_data['Beta (-)'] = governing.get('Beta (-)')
        else:
            row_data['g(x)'] = governing.get('g(x)')
            row_data['SF = R/S (-)'] = governing.get('SF = R/S (-)')
        row_data['Status'] = governing.get('Status')
        rows.append(row_data)

    return pd.DataFrame(rows)


def style_limit_state_resume_df(df: pd.DataFrame,
                                is_probabilistic: bool = True):
    """Styling tabel resume limit state pengontrol."""
    styler = style_input_dataframe(df)
    performance_columns = ['Kapasitas R', 'S dari Analisis Struktur', 'phi (-)']
    if 'SF = R/S (-)' in df.columns:
        performance_columns.append('SF = R/S (-)')
    performance_columns.append('g(x)')
    reliability_columns = (
        ['Jumlah Gagal (-)', 'Pf (-)', 'Beta (-)', 'Status']
        if is_probabilistic else
        ['Status']
    )
    return apply_grouped_header_styles(
        styler,
        df,
        {
            'identity': ['Elemen (-)', 'Kode'],
            'control': ['Limit State Kontrol', 'Satuan'],
            'performance': performance_columns,
            'reliability': reliability_columns
        }
    )


def coerce_finite_float(value: Any) -> Optional[float]:
    """Konversi ke float finite, atau None bila invalid."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(numeric_value):
        return None
    return float(numeric_value)


def extract_element_id_from_variable_name(variable_name: str) -> Optional[int]:
    """Ambil nomor elemen dari nama variabel sensitivitas seperti fc_E7."""
    match = re.fullmatch(r'[A-Za-z_]+_E(\d+)', str(variable_name or '').strip())
    if not match:
        return None
    return int(match.group(1))


def normalize_nonnegative_mapping(raw_values: Dict[int, float]) -> Dict[int, float]:
    """Normalisasi mapping non-negatif ke rentang 0-1."""
    cleaned_values = {
        int(key): max(float(value), 0.0)
        for key, value in (raw_values or {}).items()
        if value is not None and np.isfinite(float(value))
    }
    if not cleaned_values:
        return {}

    max_value = max(cleaned_values.values())
    if max_value <= 0.0:
        return {int(key): 0.0 for key in cleaned_values}

    return {
        int(key): float(value / max_value)
        for key, value in cleaned_values.items()
    }


def get_risk_level_rank(level: str) -> int:
    """Urutan level risiko untuk sorting."""
    canonical_level = normalize_risk_level(level)
    return int(RISK_LEVEL_ORDER.get(canonical_level, -1))


def normalize_risk_level(level: Any) -> str:
    """Normalisasi label level risiko ke bentuk kanonik internal."""
    level_text = str(level or '').strip()
    if not level_text:
        return ''
    return RISK_LEVEL_CANONICAL_LOOKUP.get(level_text.lower(), level_text)


def get_risk_level_display_label(level: Any,
                                 language: str = 'id') -> str:
    """Ambil label level risiko untuk kebutuhan tampilan."""
    canonical_level = normalize_risk_level(level)
    if str(language or '').strip().lower() == 'en':
        return RISK_LEVEL_ENGLISH_LABELS.get(canonical_level, canonical_level or 'No Data')
    return canonical_level or str(level or '').strip() or '-'


def get_risk_level_style(level: Any) -> str:
    """Ambil style tabel berdasarkan level risiko tampilan maupun internal."""
    return RISK_LEVEL_STYLES.get(normalize_risk_level(level), '')


def describe_deterministic_priority_level(risk_score: Optional[float],
                                          sf_value: Optional[float]) -> str:
    """Tentukan level prioritas risiko deterministik per elemen."""
    score = max(coerce_finite_float(risk_score) or 0.0, 0.0)
    sf_numeric = coerce_finite_float(sf_value)

    if score >= 0.75:
        level = 'Kritis'
    elif score >= 0.55:
        level = 'Tinggi'
    elif score >= 0.30:
        level = 'Sedang'
    else:
        level = 'Rendah'

    if sf_numeric is None:
        return level
    if sf_numeric <= 0.75:
        return 'Kritis'
    if sf_numeric < 1.0:
        if level == 'Rendah':
            return 'Sedang'
        if level == 'Sedang':
            return 'Tinggi'
        return 'Kritis'
    return level


def describe_probabilistic_risk_level(pf_value: Optional[float],
                                      beta_value: Optional[float]) -> str:
    """Tentukan level risiko probabilistik dari kombinasi Pf dan Beta."""
    pf_numeric = coerce_finite_float(pf_value)
    beta_numeric = coerce_finite_float(beta_value)

    level = 'Tidak Ada Data'
    if pf_numeric is not None:
        if pf_numeric >= 1e-1:
            level = 'Kritis'
        elif pf_numeric >= 1e-2:
            level = 'Tinggi'
        elif pf_numeric >= 1e-3:
            level = 'Sedang'
        else:
            level = 'Rendah'

    if beta_numeric is not None:
        if beta_numeric < 1.5:
            level = 'Kritis'
        elif beta_numeric < 2.5:
            level = (
                'Tinggi'
                if get_risk_level_rank(level) < get_risk_level_rank('Tinggi') else
                level
            )
        elif beta_numeric < 3.0:
            level = (
                'Sedang'
                if get_risk_level_rank(level) < get_risk_level_rank('Sedang') else
                level
            )
        elif level == 'Tidak Ada Data':
            level = 'Rendah'

    return level


def build_element_risk_map_figure(nodes: np.ndarray,
                                  elements: List,
                                  boundary_conditions: Optional[Dict[int, Dict[str, Any]]],
                                  element_levels: Dict[int, str],
                                  title: str,
                                  subtitle: Optional[str] = None,
                                  x_label: str = 'X (mm)',
                                  y_label: str = 'Y (mm)',
                                  legend_title: str = 'Level',
                                  level_display_language: str = 'id') -> plt.Figure:
    """Bangun peta warna elemen berdasarkan level risiko/prioritas."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=180)
    boundary_conditions = boundary_conditions or {}
    nodes_array = np.asarray(nodes, dtype=float)
    structure_span = PortalPlotter.get_structure_span(nodes_array, elements)
    symbol_size = 0.025 * structure_span
    node_label_offset = 0.018 * structure_span
    x_values = np.asarray(nodes_array[:, 1], dtype=float)
    y_values = np.asarray(nodes_array[:, 2], dtype=float)
    support_bounds = (
        float(np.min(x_values)),
        float(np.max(x_values)),
        float(np.min(y_values)),
        float(np.max(y_values))
    )

    line_widths = {
        'Rendah': 3.8,
        'Sedang': 4.8,
        'Tinggi': 5.8,
        'Kritis': 6.8,
        'Tidak Ada Data': 3.4
    }

    present_levels = []
    for element in elements:
        elem_id = int(element.elem_id)
        level = normalize_risk_level(element_levels.get(elem_id, 'Tidak Ada Data'))
        color = RISK_LEVEL_COLORS.get(level, RISK_LEVEL_COLORS['Tidak Ada Data'])
        line_width = line_widths.get(level, line_widths['Tidak Ada Data'])
        start = np.asarray(element.coord_start, dtype=float)
        end = np.asarray(element.coord_end, dtype=float)
        present_levels.append(level)

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color='#0f172a',
            linewidth=line_width + 1.6,
            alpha=0.16,
            solid_capstyle='round',
            zorder=1
        )
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=line_width,
            solid_capstyle='round',
            zorder=2
        )

        mid_point = (start + end) / 2.0
        ax.text(
            mid_point[0],
            mid_point[1],
            f"E{elem_id}",
            fontsize=9,
            ha='center',
            va='center',
            zorder=4,
            bbox=dict(
                boxstyle='round,pad=0.18',
                facecolor='white',
                alpha=0.92,
                edgecolor=color,
                linewidth=1.0
            )
        )

    for node in nodes_array:
        node_id = int(node[0])
        coord = np.asarray(node[1:3], dtype=float)
        restraints = boundary_conditions.get(node_id, {})
        PortalPlotter._draw_support_symbol(
            ax,
            coord,
            restraints,
            symbol_size,
            support_bounds
        )
        ax.plot(
            coord[0],
            coord[1],
            marker='o',
            markersize=6,
            markerfacecolor='white',
            markeredgecolor='#1f2937',
            markeredgewidth=1.4,
            zorder=5
        )

        label_bbox = dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85)
        if any(int(restraints.get(axis, 0)) == 1 for axis in ('X', 'Y', 'R')):
            support_min_y = PortalPlotter._get_support_symbol_min_y(
                coord,
                restraints,
                symbol_size,
                support_bounds
            )
            ax.annotate(
                f"N{node_id}",
                xy=(float(coord[0]), float(support_min_y)),
                xycoords='data',
                xytext=(0, -6),
                textcoords='offset points',
                fontsize=8.5,
                ha='center',
                va='top',
                zorder=6,
                bbox=label_bbox
            )
        else:
            ax.text(
                coord[0],
                coord[1] - node_label_offset,
                f"N{node_id}",
                fontsize=8.5,
                ha='center',
                va='top',
                zorder=6,
                bbox=label_bbox
            )

    legend_levels = [
        level for level in ('Kritis', 'Tinggi', 'Sedang', 'Rendah', 'Tidak Ada Data')
        if level in set(present_levels)
    ]
    legend_handles = [
        Patch(
            facecolor=RISK_LEVEL_COLORS[level],
            edgecolor='#1e3a8a',
            linewidth=1.4,
            label=get_risk_level_display_label(level, language=level_display_language)
        )
        for level in legend_levels
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title=legend_title,
            loc='center left',
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0.0,
            framealpha=0.96
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.axis('equal')
    ax.margins(0.08)
    PortalPlotter.expand_axes_for_data_text(ax)
    fig.subplots_adjust(
        right=0.84 if legend_handles else 0.97,
        bottom=0.17 if subtitle else 0.11
    )
    if subtitle:
        fig.text(
            0.08,
            0.035,
            str(subtitle),
            fontsize=8.6,
            color='#374151',
            ha='left',
            va='bottom',
            wrap=True,
            bbox=dict(
                boxstyle='round,pad=0.24',
                facecolor='white',
                alpha=0.9,
                edgecolor='#cbd5e1'
            )
        )
    return fig


def build_deterministic_risk_priority_df(latest_result: Dict,
                                         input_data: Optional[Dict],
                                         deterministic_sensitivity_results: Optional[Dict]) -> pd.DataFrame:
    """Bangun tabel prioritas risiko deterministik per elemen."""
    if latest_result is None or not input_data:
        return pd.DataFrame()

    limit_state_tables = build_limit_state_performance_tables(
        latest_result,
        input_data=input_data,
        element_reliability={},
        is_probabilistic=False
    )
    resume_df = build_limit_state_resume_df(limit_state_tables, is_probabilistic=False)
    resume_lookup = {
        int(row['Elemen (-)']): row
        for _, row in resume_df.iterrows()
    } if not resume_df.empty else {}

    geometry_lookup = input_data.get('geometry', {}).get('properties_by_element', {}) or {}
    all_element_ids = sorted({
        int(elem_id) for elem_id in geometry_lookup.keys()
    } or {
        int(row[0]) for row in np.asarray(
            input_data.get('geometry', {}).get('elements', []),
            dtype=float
        )
    })

    sensitivity_lookup: Dict[int, Dict[str, Any]] = {}
    for variable_name, values in ((deterministic_sensitivity_results or {}).get('results') or {}).items():
        elem_id = extract_element_id_from_variable_name(variable_name)
        if elem_id is None:
            continue

        sensitivity_index = coerce_finite_float(values.get('sensitivity_index')) or 0.0
        if sensitivity_index < 0.0:
            sensitivity_index = 0.0

        entry = sensitivity_lookup.setdefault(int(elem_id), {
            'aggregate': 0.0,
            'count': 0,
            'top_variable': '-',
            'top_sensitivity': -1.0,
            'top_effect': '-',
            'top_delta_g': None,
            'variables': []
        })
        entry['aggregate'] += float(sensitivity_index)
        entry['count'] += 1
        entry['variables'].append({
            'name': str(variable_name),
            'sensitivity': float(sensitivity_index)
        })
        if (
            float(sensitivity_index) > float(entry['top_sensitivity'])
            or (
                np.isclose(float(sensitivity_index), float(entry['top_sensitivity']))
                and str(variable_name) < str(entry['top_variable'])
            )
        ):
            signed_effect = coerce_finite_float(values.get('signed_effect'))
            if signed_effect is None:
                signed_effect = coerce_finite_float(values.get('delta_g_plus'))
            entry['top_variable'] = str(variable_name)
            entry['top_sensitivity'] = float(sensitivity_index)
            entry['top_delta_g'] = signed_effect
            entry['top_effect'] = (
                str(values.get('worst_case')).strip()
                if str(values.get('worst_case', '')).strip()
                else describe_deterministic_g_effect(signed_effect)
            )

    severity_raw: Dict[int, float] = {}
    sensitivity_raw: Dict[int, float] = {}
    for elem_id in all_element_ids:
        resume_row = resume_lookup.get(int(elem_id), {})
        sf_value_raw = resume_row.get('SF = R/S (-)')
        try:
            sf_value = float(sf_value_raw)
        except (TypeError, ValueError):
            sf_value = None

        if sf_value is None:
            severity_raw[int(elem_id)] = 0.0
        elif np.isposinf(sf_value):
            severity_raw[int(elem_id)] = 0.0
        elif (not np.isfinite(sf_value)) or sf_value <= 0.0:
            severity_raw[int(elem_id)] = 10.0
        else:
            severity_raw[int(elem_id)] = float(min(1.0 / max(sf_value, 1e-6), 10.0))

        sensitivity_raw[int(elem_id)] = float(
            (sensitivity_lookup.get(int(elem_id), {}) or {}).get('aggregate', 0.0)
        )

    severity_index = normalize_nonnegative_mapping(severity_raw)
    sensitivity_index = normalize_nonnegative_mapping(sensitivity_raw)

    rows = []
    for elem_id in all_element_ids:
        resume_row = resume_lookup.get(int(elem_id), {})
        sf_value = resume_row.get('SF = R/S (-)')
        priority_score = (
            DETERMINISTIC_RISK_WEIGHT_SEVERITY * severity_index.get(int(elem_id), 0.0)
            + DETERMINISTIC_RISK_WEIGHT_SENSITIVITY * sensitivity_index.get(int(elem_id), 0.0)
        )
        level = describe_deterministic_priority_level(priority_score, sf_value)
        sensitivity_entry = sensitivity_lookup.get(int(elem_id), {}) or {}
        ordered_variable_names = [
            item.get('name', '-')
            for item in sorted(
                sensitivity_entry.get('variables', []),
                key=lambda item: (
                    -float(item.get('sensitivity', 0.0) or 0.0),
                    str(item.get('name', ''))
                )
            )
            if str(item.get('name', '')).strip()
        ]

        rows.append({
            'Elemen (-)': int(elem_id),
            'Kode': get_element_code_from_input(input_data, int(elem_id)),
            'Limit State Kontrol': resume_row.get('Limit State Kontrol', '-'),
            'Satuan': resume_row.get('Satuan', '-'),
            'g(x) Awal': resume_row.get('g(x)'),
            'SF Awal (-)': resume_row.get('SF = R/S (-)'),
            'Severity Index (-)': severity_index.get(int(elem_id), 0.0),
            'Agregat |Delta g|max': sensitivity_raw.get(int(elem_id), 0.0),
            'Sensitivity Index (-)': sensitivity_index.get(int(elem_id), 0.0),
            'Risk Priority Score (-)': float(priority_score),
            'Level Prioritas': level,
            'Variabel Dominan': sensitivity_entry.get('top_variable', '-'),
            'Delta g Variabel Dominan': sensitivity_entry.get('top_delta_g'),
            'Efek Variabel Dominan': sensitivity_entry.get('top_effect', '-'),
            'Jumlah Variabel Sensitivitas Elemen (-)': len(ordered_variable_names),
            'Daftar Variabel Sensitivitas Elemen (-)': (
                ', '.join(ordered_variable_names) if ordered_variable_names else '-'
            )
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df['_level_rank'] = df['Level Prioritas'].map(get_risk_level_rank).fillna(-1)
    df = df.sort_values(
        by=['_level_rank', 'Risk Priority Score (-)', 'Severity Index (-)', 'Elemen (-)'],
        ascending=[False, False, False, True]
    ).drop(columns=['_level_rank']).reset_index(drop=True)
    df = add_element_number_display_column(df)
    return df


def build_probabilistic_risk_map_df(latest_result: Dict,
                                    input_data: Optional[Dict],
                                    element_reliability: Optional[Dict]) -> pd.DataFrame:
    """Bangun tabel risk map probabilistik berbasis Pf/Beta elemen."""
    if latest_result is None or not input_data:
        return pd.DataFrame()

    element_reliability = element_reliability or {}
    overall_lookup = element_reliability.get('overall', {}) or {}

    limit_state_tables = build_limit_state_performance_tables(
        latest_result,
        input_data=input_data,
        element_reliability=element_reliability,
        is_probabilistic=True
    )
    resume_df = build_limit_state_resume_df(limit_state_tables, is_probabilistic=True)
    resume_lookup = {
        int(row['Elemen (-)']): row
        for _, row in resume_df.iterrows()
    } if not resume_df.empty else {}

    geometry_lookup = input_data.get('geometry', {}).get('properties_by_element', {}) or {}
    all_element_ids = sorted({
        int(elem_id) for elem_id in (
            set(geometry_lookup.keys()) | set(overall_lookup.keys()) | set(resume_lookup.keys())
        )
    })

    rows = []
    for elem_id in all_element_ids:
        overall_info = (
            get_by_element_value(overall_lookup, int(elem_id), {}) or {}
        )
        resume_row = resume_lookup.get(int(elem_id), {})
        pf_value = overall_info.get('Pf')
        beta_value = overall_info.get('Beta')
        level = describe_probabilistic_risk_level(pf_value, beta_value)

        rows.append({
            'Elemen (-)': int(elem_id),
            'Kode': get_element_code_from_input(input_data, int(elem_id)),
            'Limit State Kontrol': resume_row.get('Limit State Kontrol', '-'),
            'Pf Elemen (-)': pf_value,
            'Beta Elemen (-)': beta_value,
            'Jumlah Gagal Elemen (-)': overall_info.get('failures'),
            'Pf Kontrol (-)': resume_row.get('Pf (-)'),
            'Beta Kontrol (-)': resume_row.get('Beta (-)'),
            'Level Risiko': level
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df['_level_rank'] = df['Level Risiko'].map(get_risk_level_rank).fillna(-1)
    df['_pf_sort'] = pd.to_numeric(df['Pf Elemen (-)'], errors='coerce').fillna(-1.0)
    df['_beta_sort'] = pd.to_numeric(df['Beta Elemen (-)'], errors='coerce').fillna(np.inf)
    df = df.sort_values(
        by=['_level_rank', '_pf_sort', '_beta_sort', 'Elemen (-)'],
        ascending=[False, False, True, True]
    ).drop(columns=['_level_rank', '_pf_sort', '_beta_sort']).reset_index(drop=True)
    df = add_element_number_display_column(df)
    return df


def add_element_number_display_column(df: pd.DataFrame,
                                      element_column: str = 'Elemen (-)',
                                      display_column: str = 'Nomor Elemen') -> pd.DataFrame:
    """Tambahkan kolom tampilan nomor elemen dalam format E# tanpa mengubah kolom numerik asli."""
    if df is None or df.empty or element_column not in df.columns:
        return df

    result = df.copy()
    display_values = result[element_column].map(
        lambda value: (
            f"E{int(float(value))}"
            if pd.notna(value) and str(value).strip() not in {'', '-', 'nan'}
            else '-'
        )
    )
    if display_column in result.columns:
        result[display_column] = display_values
        return result

    result.insert(0, display_column, display_values)
    return result


def style_risk_level_dataframe(df: pd.DataFrame,
                               level_column: str,
                               grouped_columns: Dict[str, List[str]],
                               table_min_width_px: int = 1700):
    """Styling generik untuk tabel risk map berbasis level risiko."""
    styler = style_input_dataframe(df, table_min_width_px=table_min_width_px)
    styler = apply_grouped_header_styles(styler, df, grouped_columns)
    if df.empty or level_column not in df.columns:
        return styler

    def highlight_level(dataframe: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame('', index=dataframe.index, columns=dataframe.columns)
        mapped_styles = dataframe[level_column].astype(str).map(
            lambda value: get_risk_level_style(value)
        )
        styles.loc[:, level_column] = mapped_styles
        return styles

    return styler.apply(highlight_level, axis=None)


def style_deterministic_risk_priority_df(df: pd.DataFrame):
    """Styling tabel risk priority map deterministik."""
    return style_risk_level_dataframe(
        df,
        level_column='Level Prioritas',
        grouped_columns={
            'identity': ['Nomor Elemen', 'Kode'],
            'summary': ['Limit State Kontrol', 'Satuan', 'g(x) Awal', 'SF Awal (-)'],
            'sensitivity': [
                'Agregat |Delta g|max',
                'Sensitivity Index (-)',
                'Variabel Dominan',
                'Delta g Variabel Dominan',
                'Efek Variabel Dominan',
                'Jumlah Variabel Sensitivitas Elemen (-)',
                'Daftar Variabel Sensitivitas Elemen (-)'
            ],
            'risk': [
                'Severity Index (-)',
                'Risk Priority Score (-)',
                'Level Prioritas'
            ]
        },
        table_min_width_px=2400
    )


def style_probabilistic_risk_map_df(df: pd.DataFrame):
    """Styling tabel risk map probabilistik."""
    return style_risk_level_dataframe(
        df,
        level_column='Level Risiko',
        grouped_columns={
            'identity': ['Nomor Elemen', 'Kode'],
            'summary': ['Limit State Kontrol'],
            'overall': [
                'Pf Elemen (-)',
                'Beta Elemen (-)',
                'Jumlah Gagal Elemen (-)'
            ],
            'risk': [
                'Pf Kontrol (-)',
                'Beta Kontrol (-)',
                'Level Risiko'
            ]
        }
    )


def build_deterministic_risk_level_threshold_df() -> pd.DataFrame:
    """Panduan batas level untuk risk priority map deterministik."""
    return pd.DataFrame([
        {
            'Level': 'Kritis',
            'Batas Skor Dasar': 'Risk Priority Score >= 0.75',
            'Aturan Tambahan': 'Atau langsung Kritis bila SF <= 0.75'
        },
        {
            'Level': 'Tinggi',
            'Batas Skor Dasar': '0.55 <= Risk Priority Score < 0.75',
            'Aturan Tambahan': 'Bisa naik tingkat bila 0.75 < SF < 1.00'
        },
        {
            'Level': 'Sedang',
            'Batas Skor Dasar': '0.30 <= Risk Priority Score < 0.55',
            'Aturan Tambahan': 'Bisa naik tingkat bila 0.75 < SF < 1.00'
        },
        {
            'Level': 'Rendah',
            'Batas Skor Dasar': 'Risk Priority Score < 0.30',
            'Aturan Tambahan': 'Tetap rendah bila SF >= 1.00'
        }
    ])


def build_probabilistic_risk_level_threshold_df() -> pd.DataFrame:
    """Panduan batas level untuk risk map probabilistik."""
    return pd.DataFrame([
        {
            'Risk Level': 'Critical',
            'Pf Threshold': 'Pf >= 1e-1',
            'Beta Threshold': 'Beta < 1.5'
        },
        {
            'Risk Level': 'High',
            'Pf Threshold': '1e-2 <= Pf < 1e-1',
            'Beta Threshold': '1.5 <= Beta < 2.5'
        },
        {
            'Risk Level': 'Moderate',
            'Pf Threshold': '1e-3 <= Pf < 1e-2',
            'Beta Threshold': '2.5 <= Beta < 3.0'
        },
        {
            'Risk Level': 'Low',
            'Pf Threshold': 'Pf < 1e-3',
            'Beta Threshold': 'Beta >= 3.0'
        }
    ])


def style_risk_threshold_df(df: pd.DataFrame,
                            level_column: str = 'Level'):
    """Styling sederhana untuk tabel batas level risk map."""
    return style_risk_level_dataframe(
        df,
        level_column=level_column,
        grouped_columns={
            'risk': list(df.columns)
        }
    )


def get_risk_recommendation_catalog() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Daftar rekomendasi teknis umum berdasarkan jenis elemen dan level risiko."""
    return {
        'B': {
            'Kritis': {
                'indikasi': 'Kegagalan lentur/geser atau daktilitas rendah.',
                'sections': [
                    {
                        'title': 'Tindakan utama',
                        'items': [
                            'Tingkatkan kapasitas lentur.',
                            'Tambah luas tulangan tarik.',
                            'Optimasi tinggi efektif balok.',
                            'Perkuatan geser.',
                            'Rapatkan sengkang.',
                            'Gunakan sengkang tertutup (seismic hooks).'
                        ]
                    },
                    {
                        'title': 'Cek rasio tulangan',
                        'items': [
                            'Rasio tulangan terpasang harus memenuhi rasio tulangan minimum dan maksimum.'
                        ]
                    },
                    {
                        'title': 'Pelaksanaan, retrofit, dan perawatan',
                        'items': [
                            'Shoring / temporary support (WAJIB) untuk mencegah collapse selama retrofit.',
                            'Perbaikan retak (epoxy injection).',
                            'Perkuatan lentur: CFRP (flexural strengthening) atau steel plate bonding.',
                            'Perkuatan geser: wrap FRP U-jacket / full wrap.',
                            'Jacketing beton.',
                            'Tambah dimensi balok.',
                            'Tambah tulangan baru.',
                            'Coating anti-korosi.'
                        ]
                    },
                    {
                        'title': 'Perawatan',
                        'items': [
                            'Monitoring dan perawatan berkala.'
                        ]
                    }
                ]
            },
            'Tinggi': {
                'indikasi': 'Aman tetapi margin keamanan kecil.',
                'sections': [
                    {
                        'title': 'Tindakan utama',
                        'items': [
                            'Tambah luas tulangan tarik.',
                            'Perbaiki detailing sengkang di daerah kritis.',
                            'Evaluasi redistribusi momen.',
                            'Cek kontrol retak dan lendutan (serviceability).'
                        ]
                    },
                    {
                        'title': 'Perkuatan dan rehabilitasi',
                        'items': [
                            'Perkuatan lentur: CFRP (flexural strengthening) atau steel plate bonding.',
                            'Perkuatan geser: wrap FRP U-jacket / full wrap.',
                            'Perbaikan retak (epoxy injection).',
                            'Coating anti-korosi.'
                        ]
                    },
                    {
                        'title': 'Perawatan',
                        'items': [
                            'Monitoring dan perawatan berkala.'
                        ]
                    }
                ]
            },
            'Sedang': {
                'indikasi': 'Sesuai target keandalan.',
                'sections': [
                    {
                        'title': 'Tindakan utama',
                        'items': [
                            'Tidak perlu redesign besar.',
                            'Optimasi minor: cek spasi tulangan geser.',
                            'Optimasi minor: cek detailing tulangan angkur.',
                            'Validasi terhadap kombinasi beban SNI 1727:2020.'
                        ]
                    },
                    {
                        'title': 'Perbaikan minor dan perawatan',
                        'items': [
                            'Grouting.',
                            'Coating anti-korosi.',
                            'Monitoring dan perawatan berkala.'
                        ]
                    }
                ]
            },
            'Rendah': {
                'indikasi': 'Overdesign, kapasitas jauh lebih besar daripada beban.',
                'sections': [
                    {
                        'title': 'Tindakan utama',
                        'items': [
                            'Optimasi desain.',
                            'Kurangi luas tulangan tarik.',
                            'Kecilkan dimensi jika memungkinkan.'
                        ]
                    },
                    {
                        'title': 'Cek persyaratan desain',
                        'items': [
                            'Rasio tulangan terpasang harus memenuhi rasio tulangan minimum dan maksimum.',
                            'Pastikan daktilitas tidak menurun.'
                        ]
                    },
                    {
                        'title': 'Perbaikan minor dan perawatan',
                        'items': [
                            'Grouting.',
                            'Coating anti-korosi.',
                            'Monitoring dan perawatan berkala.'
                        ]
                    }
                ]
            }
        },
        'K': {
            'Kritis': {
                'indikasi': 'Risiko kegagalan aksial atau interaksi P-M.',
                'sections': [
                    {
                        'title': 'Tindakan utama',
                        'items': [
                            'Cek interaksi aksial-lentur.',
                            'Tingkatkan confinement.',
                            'Tambah rasio tulangan longitudinal (1% <= rho <= 8%).',
                            'Perbesar dimensi kolom.',
                            'Cek Strong Column Weak Beam (SCWB).'
                        ]
                    },
                    {
                        'title': 'Perawatan',
                        'items': [
                            'Monitoring dan perawatan berkala.'
                        ]
                    }
                ]
            },
            'Tinggi': {
                'indikasi': 'Mendekati batas kapasitas.',
                'sections': [
                    {
                        'title': 'Tindakan utama',
                        'items': [
                            'Tambah confinement lokal pada zona sendi plastis.',
                            'Tambah rasio tulangan longitudinal (1% <= rho <= 8%).',
                            'Evaluasi efek P-Delta (second order).'
                        ]
                    },
                    {
                        'title': 'Perawatan',
                        'items': [
                            'Monitoring dan perawatan berkala.'
                        ]
                    }
                ]
            },
            'Sedang': {
                'indikasi': 'Aman sesuai target.',
                'sections': [
                    {
                        'title': 'Tindakan utama',
                        'items': [
                            'Verifikasi detailing gempa.',
                            'Cek panjang penyaluran dan penggunaan kait sengkang 135 derajat.'
                        ]
                    },
                    {
                        'title': 'Perawatan',
                        'items': [
                            'Monitoring dan perawatan berkala.'
                        ]
                    }
                ]
            },
            'Rendah': {
                'indikasi': 'Overdesign, terlalu kuat dan kurang efisien.',
                'sections': [
                    {
                        'title': 'Tindakan utama',
                        'items': [
                            'Optimasi desain.',
                            'Kurangi luas tulangan tarik.',
                            'Kecilkan dimensi jika memungkinkan.'
                        ]
                    },
                    {
                        'title': 'Cek persyaratan desain',
                        'items': [
                            'Rasio tulangan terpasang harus memenuhi rasio tulangan minimum dan maksimum.',
                            'Memenuhi SCWB.',
                            'Menjaga kekakuan struktur.'
                        ]
                    },
                    {
                        'title': 'Perawatan',
                        'items': [
                            'Monitoring dan perawatan berkala.'
                        ]
                    }
                ]
            }
        }
    }


def get_risk_recommendation_card_palette(level: str) -> Dict[str, str]:
    """Palet warna kartu rekomendasi teknis per level risiko."""
    palette_mapping = {
        'Kritis': {
            'accent': '#ff0000',
            'background': '#fff1f2',
            'surface': '#ffe4e6',
            'text': '#7f1d1d'
        },
        'Tinggi': {
            'accent': '#ca8a04',
            'background': '#fefce8',
            'surface': '#fef3c7',
            'text': '#854d0e'
        },
        'Sedang': {
            'accent': '#15803d',
            'background': '#f0fdf4',
            'surface': '#dcfce7',
            'text': '#166534'
        },
        'Rendah': {
            'accent': "#110BB0",      # biru utama
            'background': '#EFF6FF',  # biru sangat muda (gantikan hijau)
            'surface': '#DBEAFE',     # biru muda untuk box isi
            'text': '#1E40AF'         # teks biru gelap
        }
    }
    return palette_mapping.get(
        str(level or '').strip().title(),
        {
            'accent': '#475569',
            'background': '#f8fafc',
            'surface': '#e2e8f0',
            'text': '#1e293b'
        }
    )


def build_risk_recommendation_level_counts(risk_df: pd.DataFrame,
                                           level_column: str) -> Dict[str, Dict[str, int]]:
    """Hitung jumlah elemen per level risiko untuk balok dan kolom."""
    levels = ('Kritis', 'Tinggi', 'Sedang', 'Rendah')
    counts = {
        'B': {level: 0 for level in levels},
        'K': {level: 0 for level in levels}
    }
    if risk_df is None or risk_df.empty or level_column not in risk_df.columns:
        return counts

    for _, row in risk_df.iterrows():
        code = str(row.get('Kode', '') or '').strip().upper()
        level = str(row.get(level_column, '') or '').strip().title()
        if code in counts and level in counts[code]:
            counts[code][level] += 1

    return counts


def build_risk_recommendation_element_lists(risk_df: pd.DataFrame,
                                            level_column: str) -> Dict[str, Dict[str, List[int]]]:
    """Kumpulkan nomor elemen per level risiko untuk balok dan kolom."""
    levels = ('Kritis', 'Tinggi', 'Sedang', 'Rendah')
    element_lists = {
        'B': {level: [] for level in levels},
        'K': {level: [] for level in levels}
    }
    if risk_df is None or risk_df.empty or level_column not in risk_df.columns:
        return element_lists

    for _, row in risk_df.iterrows():
        code = str(row.get('Kode', '') or '').strip().upper()
        level = str(row.get(level_column, '') or '').strip().title()
        elem_id = row.get('Elemen (-)')
        if code not in element_lists or level not in element_lists[code]:
            continue
        if pd.isna(elem_id) or str(elem_id).strip() in {'', '-', 'nan'}:
            continue
        try:
            element_lists[code][level].append(int(float(elem_id)))
        except (TypeError, ValueError):
            continue

    for code in element_lists:
        for level in element_lists[code]:
            element_lists[code][level] = sorted(set(element_lists[code][level]))
    return element_lists


def normalize_risk_recommendation_limit_state(limit_state: Any) -> str:
    """Normalisasi label limit state untuk dirangkum sebagai mode kegagalan."""
    normalized = str(limit_state or '').strip().lower().replace('_', '-').replace(' ', '-')
    mapping = {
        'lentur': 'Lentur',
        'geser': 'Geser',
        'aksial': 'Aksial',
        'aksial-lentur': 'Aksial-Lentur',
        'aksial+lentur': 'Aksial-Lentur'
    }
    return mapping.get(normalized, '')


def build_risk_recommendation_failure_mode_lists(
    risk_df: pd.DataFrame,
    level_column: str
) -> Dict[str, Dict[str, Dict[str, List[int]]]]:
    """Kumpulkan limit state kontrol per level agar indikasi bisa berbasis mode gagal."""
    levels = ('Kritis', 'Tinggi', 'Sedang', 'Rendah')
    modes = ('Lentur', 'Geser', 'Aksial', 'Aksial-Lentur')
    failure_mode_lists = {
        'B': {level: {mode: [] for mode in modes} for level in levels},
        'K': {level: {mode: [] for mode in modes} for level in levels}
    }
    if (
        risk_df is None
        or risk_df.empty
        or level_column not in risk_df.columns
        or 'Limit State Kontrol' not in risk_df.columns
    ):
        return failure_mode_lists

    for _, row in risk_df.iterrows():
        code = str(row.get('Kode', '') or '').strip().upper()
        level = str(row.get(level_column, '') or '').strip().title()
        elem_id = row.get('Elemen (-)')
        mode = normalize_risk_recommendation_limit_state(row.get('Limit State Kontrol'))
        if code not in failure_mode_lists or level not in failure_mode_lists[code] or not mode:
            continue
        if pd.isna(elem_id) or str(elem_id).strip() in {'', '-', 'nan'}:
            continue
        try:
            failure_mode_lists[code][level][mode].append(int(float(elem_id)))
        except (TypeError, ValueError):
            continue

    for code in failure_mode_lists:
        for level in failure_mode_lists[code]:
            for mode in failure_mode_lists[code][level]:
                failure_mode_lists[code][level][mode] = sorted(
                    set(failure_mode_lists[code][level][mode])
                )
    return failure_mode_lists


def build_risk_recommendation_indication_text(mode_lists: Optional[Dict[str, List[int]]],
                                              fallback_text: str) -> str:
    """Ringkas indikasi dari distribusi limit state kontrol pada level aktif."""
    display_labels = {
        'Lentur': 'lentur',
        'Geser': 'geser',
        'Aksial': 'aksial',
        'Aksial-Lentur': 'aksial+lentur'
    }
    active_modes = []
    for mode in ('Lentur', 'Geser', 'Aksial', 'Aksial-Lentur'):
        element_ids = sorted(set((mode_lists or {}).get(mode, []) or []))
        if element_ids:
            active_modes.append((mode, len(element_ids)))

    if not active_modes:
        return fallback_text

    active_modes = sorted(
        active_modes,
        key=lambda item: (-int(item[1]), ('Lentur', 'Geser', 'Aksial', 'Aksial-Lentur').index(item[0]))
    )
    phrases = [
        f"{display_labels.get(mode, str(mode).lower())} ({int(count)} elemen)"
        for mode, count in active_modes
    ]
    if len(phrases) == 1:
        return (
            "Kemungkinan mode kegagalan pengontrol pada level ini didominasi "
            f"{phrases[0]}, berdasarkan Limit State Kontrol elemen."
        )
    if len(phrases) == 2:
        tail_text = f"{phrases[0]} dan {phrases[1]}"
    else:
        tail_text = f"{', '.join(phrases[:-1])}, dan {phrases[-1]}"
    return (
        "Kemungkinan mode kegagalan pengontrol pada level ini didominasi "
        f"{tail_text}, berdasarkan Limit State Kontrol elemen."
    )


def build_risk_recommendation_cards_html(element_code: str,
                                         level_counts: Dict[str, int],
                                         element_lists: Optional[Dict[str, List[int]]] = None,
                                         failure_mode_lists: Optional[Dict[str, Dict[str, List[int]]]] = None,
                                         zoom_scale: float = 1.0,
                                         section_key: str = "risk-rec") -> str:
    """Bangun HTML kartu rekomendasi teknis yang responsif dan mudah dibaca."""
    recommendations = get_risk_recommendation_catalog().get(str(element_code).strip().upper(), {})
    element_label = get_element_type_label(str(element_code).strip().upper())
    levels = ('Kritis', 'Tinggi', 'Sedang', 'Rendah')
    dom_key = sanitize_dom_id(f"{section_key}-{element_code}-{int(zoom_scale * 100)}")

    cards_markup = []
    for level in levels:
        recommendation = recommendations.get(level)
        if not recommendation:
            continue

        palette = get_risk_recommendation_card_palette(level)
        active_elements = list((element_lists or {}).get(level, []) or [])
        element_list_text = (
            ", ".join(f"E{int(elem_id)}" for elem_id in active_elements)
            if active_elements else
            f"Belum muncul pada elemen {element_label.lower()} model aktif"
        )
        indication_text = (
            build_risk_recommendation_indication_text(
                (failure_mode_lists or {}).get(level, {}),
                str(recommendation.get("indikasi", "-"))
            )
            if active_elements else
            "-"
        )

        section_blocks = []
        for section in recommendation.get('sections', []):
            section_title = str(section.get('title', '-') or '-')
            items_markup = ''.join(
                f"<li>{html.escape(str(item))}</li>"
                for item in section.get('items', [])
            )
            section_blocks.append(
                (
                    f'<div class="{dom_key}-section-block">'
                    f'<div class="{dom_key}-section-title">'
                    f'{html.escape(section_title)}'
                    f'</div>'
                    f'<ul>{items_markup}</ul>'
                    f'</div>'
                )
            )

        cards_markup.append(
            (
                f'<article class="{dom_key}-card" '
                f'style="'
                f'--card-accent: {palette["accent"]}; '
                f'--card-background: {palette["background"]}; '
                f'--card-surface: {palette["surface"]}; '
                f'--card-text: {palette["text"]};'
                f'">'
                f'<div class="{dom_key}-card-header">'
                f'<div class="{dom_key}-level-badge">{html.escape(level.upper())}</div>'
                f'</div>'
                f'<div class="{dom_key}-indicator">'
                f'<strong>Indikasi:</strong> {html.escape(indication_text)}'
                f'</div>'
                f'<div class="{dom_key}-element-strip">'
                f'<span class="{dom_key}-element-strip-label">Nomor elemen:</span> '
                f'<span class="{dom_key}-element-strip-values">{html.escape(element_list_text)}</span>'
                f'</div>'
                f'{"".join(section_blocks)}'
                f'</article>'
            )
        )

    base_font_size = 0.96 * float(zoom_scale)
    indicator_font_size = 0.98 * float(zoom_scale)
    badge_font_size = 0.86 * float(zoom_scale)
    section_title_size = 0.92 * float(zoom_scale)

    style_markup = textwrap.dedent(
        f"""
        <style>
          .{dom_key}-wrapper {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1rem;
            margin-top: 0.25rem;
          }}
          .{dom_key}-card {{
            border: 1px solid var(--card-accent);
            background: linear-gradient(180deg, var(--card-background) 0%, #ffffff 100%);
            border-radius: 1rem;
            padding: 1rem 1rem 1.05rem 1rem;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
            color: #111827;
            font-size: {base_font_size:.3f}rem;
            line-height: 1.58;
          }}
          .{dom_key}-card-header {{
            display: flex;
            justify-content: flex-start;
            gap: 0.75rem;
            align-items: flex-start;
            flex-wrap: wrap;
            margin-bottom: 0.85rem;
          }}
          .{dom_key}-level-badge {{
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            background: var(--card-accent);
            color: #ffffff;
            padding: 0.28rem 0.75rem;
            font-size: {badge_font_size:.3f}rem;
            font-weight: 800;
            letter-spacing: 0.04em;
          }}
          .{dom_key}-indicator {{
            background: var(--card-surface);
            border-left: 5px solid var(--card-accent);
            border-radius: 0.8rem;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.9rem;
            color: var(--card-text);
            font-size: {indicator_font_size:.3f}rem;
          }}
          .{dom_key}-element-strip {{
            margin-bottom: 0.9rem;
            padding: 0.65rem 0.85rem;
            border-radius: 0.8rem;
            border: 1px dashed rgba(15, 23, 42, 0.14);
            background: rgba(255, 255, 255, 0.68);
            color: #0f172a;
            font-size: {indicator_font_size:.3f}rem;
            line-height: 1.5;
          }}
          .{dom_key}-element-strip-label {{
            font-weight: 800;
            color: var(--card-text);
          }}
          .{dom_key}-element-strip-values {{
            font-family: "Consolas", "Courier New", monospace;
            word-break: break-word;
          }}
          .{dom_key}-section-block + .{dom_key}-section-block {{
            margin-top: 0.8rem;
          }}
          .{dom_key}-section-title {{
            font-size: {section_title_size:.3f}rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.35rem;
          }}
          .{dom_key}-section-block ul {{
            margin: 0;
            padding-left: 1.15rem;
          }}
          .{dom_key}-section-block li {{
            margin: 0.18rem 0;
          }}
          @media (max-width: 760px) {{
            .{dom_key}-wrapper {{
              grid-template-columns: 1fr;
            }}
          }}
        </style>
        """
    ).strip()
    wrapper_markup = f'<div class="{dom_key}-wrapper">{"".join(cards_markup)}</div>'
    return "\n".join([style_markup, wrapper_markup])


def render_risk_map_technical_recommendations(risk_df: pd.DataFrame,
                                              level_column: str,
                                              heading_level: str = "####",
                                              section_key: str = "risk-recommendation") -> None:
    """Tampilkan daftar rekomendasi teknis berbasis level risiko pada tab Risk Map."""
    if risk_df is None or risk_df.empty or level_column not in risk_df.columns:
        return

    level_counts = build_risk_recommendation_level_counts(risk_df, level_column)
    element_lists = build_risk_recommendation_element_lists(risk_df, level_column)
    failure_mode_lists = build_risk_recommendation_failure_mode_lists(risk_df, level_column)
    zoom_percent = st.slider(
        "Zoom daftar rekomendasi teknis (%)",
        min_value=90,
        max_value=180,
        value=110,
        step=5,
        key=f"{section_key}-zoom"
    )
    zoom_scale = float(zoom_percent) / 100.0

    st.markdown(f"{heading_level} Daftar Rekomendasi Teknis berdasarkan Level Risiko")
    st.caption(
        "Daftar ini adalah panduan teknis praktis berdasarkan jenis elemen dan level risiko "
        "pada model aktif. Konten ini tidak mengubah hasil perhitungan struktur maupun reliability."
    )
    st.caption(
        "Gunakan slider `zoom` untuk memperbesar atau memperkecil ukuran teks rekomendasi "
        "agar nyaman dibaca pada desktop maupun HP."
    )
    st.caption(
        "Baris `Indikasi` diringkas dari distribusi `Limit State Kontrol` elemen pada tiap level, "
        "sehingga bisa menunjukkan kecenderungan gagal karena lentur, geser, aksial, atau aksial+lentur."
    )

    beam_count = int((risk_df['Kode'].astype(str).str.upper() == 'B').sum()) if 'Kode' in risk_df.columns else 0
    column_count = int((risk_df['Kode'].astype(str).str.upper() == 'K').sum()) if 'Kode' in risk_df.columns else 0
    metric_cols = st.columns(4)
    metric_cols[0].metric("Zoom Teks", f"{int(zoom_percent)}%")
    metric_cols[1].metric("Balok Terpetakan", str(beam_count))
    metric_cols[2].metric("Kolom Terpetakan", str(column_count))
    metric_cols[3].metric(
        "Level Acuan",
        ", ".join(
            level for level in ('Kritis', 'Tinggi', 'Sedang', 'Rendah')
            if level in set(risk_df[level_column].astype(str).str.title())
        ) or "-"
    )

    beam_tab, column_tab = st.tabs(["Balok", "Kolom"])
    with beam_tab:
        st.caption(
            "Rekomendasi balok difokuskan pada kontrol lentur, geser, detailing, "
            "retrofit, dan perawatan."
        )
        st.markdown(
            build_risk_recommendation_cards_html(
                element_code='B',
                level_counts=level_counts.get('B', {}),
                element_lists=element_lists.get('B', {}),
                failure_mode_lists=failure_mode_lists.get('B', {}),
                zoom_scale=zoom_scale,
                section_key=f"{section_key}-beam"
            ),
            unsafe_allow_html=True
        )

    with column_tab:
        st.caption(
            "Rekomendasi kolom difokuskan pada interaksi aksial-lentur, confinement, "
            "SCWB, stabilitas, dan perawatan."
        )
        st.markdown(
            build_risk_recommendation_cards_html(
                element_code='K',
                level_counts=level_counts.get('K', {}),
                element_lists=element_lists.get('K', {}),
                failure_mode_lists=failure_mode_lists.get('K', {}),
                zoom_scale=zoom_scale,
                section_key=f"{section_key}-column"
            ),
            unsafe_allow_html=True
        )


def render_risk_map_output_section(results_bundle: Dict,
                                   latest_result: Dict,
                                   input_data: Dict,
                                   nodes: np.ndarray,
                                   elements: List,
                                   is_probabilistic: bool,
                                   heading_level: str = "####") -> None:
    """Render risk map adaptif untuk mode deterministik atau probabilistik."""
    if latest_result is None or not input_data:
        st.info("Risk map akan tersedia setelah analisis dijalankan.")
        return

    if nodes is None or elements is None:
        nodes, elements = build_preview_portal(input_data, is_probabilistic)

    if is_probabilistic:
        risk_df = build_probabilistic_risk_map_df(
            latest_result,
            input_data=input_data,
            element_reliability=(results_bundle or {}).get('element_reliability', {})
        )
        if risk_df.empty:
            st.info("Data risk map probabilistik belum tersedia.")
            return

        top_row = risk_df.iloc[0]
        critical_count = int((risk_df['Level Risiko'] == 'Kritis').sum())
        element_level_map = {
            int(row['Elemen (-)']): str(row['Level Risiko'])
            for _, row in risk_df.iterrows()
        }

        st.markdown(f"{heading_level} Risk Map Probabilistik")
        st.caption(
            "Peta ini mewarnai setiap elemen berdasarkan `Pf/Beta` elemen secara keseluruhan. "
            "Warna menunjukkan level risiko, sedangkan tabel di bawah merangkum limit state "
            "kontrol yang paling dominan pada setiap elemen."
        )
        st.caption(
            "Klasifikasi yang dipakai: `Kritis` bila `Pf >= 1e-1` atau `Beta < 1.5`, "
            "`Tinggi` bila `Pf >= 1e-2` atau `Beta < 2.5`, `Sedang` bila "
            "`Pf >= 1e-3` atau `Beta < 3.0`, dan `Rendah` untuk kondisi di bawahnya."
        )
        with st.expander("Risk Level Thresholds", expanded=False):
            st.caption(
                "The final risk level follows the more critical condition between the "
                "`Pf` threshold and the `Beta` threshold."
            )
            probabilistic_risk_threshold_df = build_probabilistic_risk_level_threshold_df()
            render_input_table(
                probabilistic_risk_threshold_df,
                styler=style_risk_threshold_df(
                    probabilistic_risk_threshold_df,
                    level_column='Risk Level'
                )
            )

        metric_cols = st.columns(4)
        metric_cols[0].metric("Elemen Paling Kritis", f"E{int(top_row['Elemen (-)'])}")
        metric_cols[1].metric("Pf Maksimum", format_metric(top_row.get('Pf Elemen (-)'), 6))
        metric_cols[2].metric("Beta Minimum", format_metric(top_row.get('Beta Elemen (-)'), 4))
        metric_cols[3].metric("Elemen Kritis", str(critical_count))

        risk_fig = build_element_risk_map_figure(
            nodes,
            elements,
            boundary_conditions=input_data.get('boundary', {}),
            element_levels=element_level_map,
            title="Probabilistic Risk Map of Portal Frame Members",
            subtitle="Color coding is based on the member risk level derived from the overall Pf and Beta values.",
            x_label="Global X Coordinate (mm)",
            y_label="Global Y Coordinate (mm)",
            legend_title="Risk Level",
            level_display_language='en'
        )
        render_plot(
            risk_fig,
            interactive=True,
            viewer_key="probabilistic-risk-map",
            alt_text="Risk map probabilistik elemen portal",
            viewer_height=620,
            download_basename="risk-map-probabilistik"
        )

        st.markdown(f"{heading_level} Tabel Risk Map Probabilistik")
        probabilistic_risk_table_df = risk_df.drop(columns=['Elemen (-)'], errors='ignore')
        render_input_table(
            probabilistic_risk_table_df,
            styler=style_probabilistic_risk_map_df(probabilistic_risk_table_df)
        )
        render_risk_map_technical_recommendations(
            risk_df,
            level_column='Level Risiko',
            heading_level=heading_level,
            section_key="probabilistic-risk-recommendation"
        )
        return

    deterministic_results = (results_bundle or {}).get('deterministic_sensitivity_results', {}) or {}
    risk_df = build_deterministic_risk_priority_df(
        latest_result,
        input_data=input_data,
        deterministic_sensitivity_results=deterministic_results
    )
    if risk_df.empty:
        st.info("Data risk priority map deterministik belum tersedia.")
        return

    baseline_info = deterministic_results.get('baseline', {}) or {}
    baseline_label = str(baseline_info.get('limit_state_label', '-'))
    top_row = risk_df.iloc[0]
    critical_count = int((risk_df['Level Prioritas'] == 'Kritis').sum())
    top_sensitive_row = risk_df.sort_values(
        by=['Sensitivity Index (-)', 'Elemen (-)'],
        ascending=[False, True]
    ).iloc[0]
    element_level_map = {
        int(row['Elemen (-)']): str(row['Level Prioritas'])
        for _, row in risk_df.iterrows()
    }

    st.markdown(f"{heading_level} Risk Priority Map Deterministik")
    st.caption(
        "Peta ini menyusun prioritas risiko elemen dari dua sisi: "
        "`Severity Index` dari kebalikan `SF` elemen pengontrol, dan "
        "`Sensitivity Index` dari akumulasi `|Delta g|max` variabel-variabel yang "
        "terkait dengan elemen yang sama."
    )
    st.caption(
        f"Skor prioritas dihitung dengan rumus "
        f"`{DETERMINISTIC_RISK_WEIGHT_SEVERITY:.2f} x Severity + "
        f"{DETERMINISTIC_RISK_WEIGHT_SENSITIVITY:.2f} x Sensitivity`, "
        f"dengan sensitivitas mengacu pada limit state kontrol baseline `{baseline_label}`. "
        "Jenis limit state yang dibandingkan tetap sama pada semua perturbasi, "
        "namun elemen pengontrol dapat berubah."
    )
    with st.expander("Batas Level Risk Priority Map", expanded=False):
        st.caption(
            "`Risk Priority Score` dipakai sebagai batas dasar level, lalu `SF` "
            "dipakai sebagai pengaman tambahan agar elemen dengan margin sangat rendah "
            "tetap naik ke level prioritas yang lebih kritis."
        )
        st.caption(
            "`Severity Index` dan `Sensitivity Index` dinormalisasi pada model aktif, "
            "jadi level ini bersifat prioritas relatif dalam model yang sedang dianalisis."
        )
        render_input_table(
            build_deterministic_risk_level_threshold_df(),
            styler=style_risk_threshold_df(
                build_deterministic_risk_level_threshold_df()
            )
        )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Limit State Baseline", baseline_label)
    metric_cols[1].metric("Elemen Prioritas Tertinggi", f"E{int(top_row['Elemen (-)'])}")
    metric_cols[2].metric(
        "Skor Prioritas Maksimum",
        format_metric(top_row.get('Risk Priority Score (-)'), 4)
    )
    metric_cols[3].metric("Elemen Kritis", str(critical_count))

    st.caption(
        f"Elemen paling sensitif terhadap perubahan variabel adalah "
        f"`E{int(top_sensitive_row['Elemen (-)'])}` dengan "
        f"`Sensitivity Index = {format_metric(top_sensitive_row.get('Sensitivity Index (-)'), 4)}`."
    )

    risk_fig = build_element_risk_map_figure(
        nodes,
        elements,
        boundary_conditions=input_data.get('boundary', {}),
        element_levels=element_level_map,
        title="Risk Priority Map Deterministik Elemen Portal",
        subtitle=(
            "Basis warna: prioritas risiko elemen dari severity kontrol dan "
            "sensitivitas terhadap limit state baseline."
        )
    )
    render_plot(
        risk_fig,
        interactive=True,
        viewer_key="deterministic-risk-map",
        alt_text="Risk priority map deterministik elemen portal",
        viewer_height=620,
        download_basename="risk-map-deterministik"
    )

    st.markdown(f"{heading_level} Tabel Risk Priority Map Deterministik")
    st.caption(
        "Kolom `Jumlah Variabel Sensitivitas Elemen (-)` menunjukkan berapa banyak "
        "variabel perturbasi one-at-a-time yang dipetakan ke elemen tersebut dan ikut "
        "membentuk `Sensitivity Index`."
    )
    st.caption(
        "Kolom `Daftar Variabel Sensitivitas Elemen (-)` menuliskan nama variabel yang "
        "terkait dengan elemen itu, diurutkan dari `|Delta g|max` terbesar ke terkecil."
    )
    st.caption(
        "Kolom `g(x) Awal` adalah nilai fungsi kinerja elemen pada kondisi baseline "
        "sebelum perturbasi sensitivitas. Nilai positif berarti margin keamanan masih ada, "
        "sedangkan nilai negatif berarti margin keamanan sudah terlampaui."
    )
    st.caption(
        "Kolom `Delta g Variabel Dominan` menunjukkan perubahan `g(x)` bertanda dari "
        "variabel yang paling dominan pada elemen tersebut. Nilai positif berarti margin "
        "keamanan membesar, sedangkan nilai negatif berarti margin keamanan mengecil."
    )
    deterministic_risk_table_df = risk_df.drop(columns=['Elemen (-)'], errors='ignore')
    render_input_table(
        deterministic_risk_table_df,
        styler=style_deterministic_risk_priority_df(deterministic_risk_table_df)
    )
    render_risk_map_technical_recommendations(
        risk_df,
        level_column='Level Prioritas',
        heading_level=heading_level,
        section_key="deterministic-risk-recommendation"
    )


def get_probabilistic_histogram_variable_specs() -> List[Dict[str, str]]:
    """Spesifikasi variabel random yang ditampilkan pada tab histogram."""
    return [
        {
            'type': 'fc',
            'label': 'Mutu Beton fc',
            'plot_label': 'Concrete Compressive Strength fc',
            'distribution_label': 'Lognormal',
            'unit': 'MPa'
        },
        {
            'type': 'fy_tarik',
            'label': 'fy Tarik',
            'plot_label': 'Tensile Yield Strength fy',
            'distribution_label': 'Normal',
            'unit': 'MPa'
        },
        {
            'type': 'fy_tekan',
            'label': 'fy Tekan',
            'plot_label': 'Compressive Yield Strength fy',
            'distribution_label': 'Normal',
            'unit': 'MPa'
        },
        {
            'type': 'fy_geser',
            'label': 'fy Geser',
            'plot_label': 'Shear Yield Strength fy',
            'distribution_label': 'Normal',
            'unit': 'MPa'
        },
        {
            'type': 'qDL',
            'label': 'Beban Mati qDL',
            'plot_label': 'Dead Load qDL',
            'distribution_label': 'Normal',
            'unit': 'kN/m'
        },
        {
            'type': 'qLL',
            'label': 'Beban Hidup qLL',
            'plot_label': 'Live Load qLL',
            'distribution_label': 'Lognormal',
            'unit': 'kN/m'
        }
    ]


def build_histogram_variable_name(variable_type: str, elem_id: int) -> str:
    """Nama flat variabel random per elemen untuk tab histogram."""
    return f"{str(variable_type).strip()}_E{int(elem_id)}"


def build_probabilistic_histogram_summary_df(histogram_data: Dict[str, Dict[str, Any]],
                                             elem_id: int) -> pd.DataFrame:
    """Ringkas statistik histogram Monte Carlo untuk satu elemen."""
    rows = []
    for spec in get_probabilistic_histogram_variable_specs():
        variable_name = build_histogram_variable_name(spec['type'], int(elem_id))
        record = histogram_data.get(variable_name)
        if not record:
            continue

        rows.append({
            'Variabel Acak (-)': variable_name,
            'Distribusi Input': str(record.get('distribution', spec['distribution_label'])).title(),
            'Mean Input': record.get('mean'),
            'StdDev Input': record.get('stddev'),
            'Mean Sampel MC': record.get('sample_mean'),
            'StdDev Sampel MC': record.get('sample_std'),
            'Minimum Sampel': record.get('sample_min'),
            'Maksimum Sampel': record.get('sample_max'),
            'Jumlah Sampel (-)': record.get('sample_count'),
            'Satuan': record.get('unit', spec['unit'])
        })

    return pd.DataFrame(rows)


def build_probability_density_curve(distribution: str,
                                    mean_value: Optional[float],
                                    stddev_value: Optional[float],
                                    x_values: np.ndarray) -> Optional[np.ndarray]:
    """Hitung PDF teoritis untuk overlay histogram."""
    mean_numeric = coerce_finite_float(mean_value)
    stddev_numeric = coerce_finite_float(stddev_value)
    if mean_numeric is None or stddev_numeric is None or stddev_numeric <= 0.0:
        return None

    distribution_name = str(distribution or 'normal').strip().lower()
    x_array = np.asarray(x_values, dtype=float)
    if x_array.size == 0:
        return None

    if distribution_name == 'normal':
        return stats.norm.pdf(x_array, loc=mean_numeric, scale=stddev_numeric)

    if distribution_name == 'lognormal':
        if mean_numeric <= 0.0:
            return None
        variance_ratio = (stddev_numeric / mean_numeric) ** 2
        sigma = np.sqrt(np.log(1.0 + variance_ratio))
        mu = np.log(mean_numeric) - 0.5 * sigma ** 2
        clipped_x = np.clip(x_array, 1e-12, None)
        pdf_values = stats.lognorm.pdf(clipped_x, s=sigma, scale=np.exp(mu))
        pdf_values = np.where(x_array > 0.0, pdf_values, 0.0)
        return np.asarray(pdf_values, dtype=float)

    return None


def build_probabilistic_histogram_figure(histogram_data: Dict[str, Dict[str, Any]],
                                         elem_id: int) -> Optional[plt.Figure]:
    """Bangun figure histogram variabel random untuk satu elemen."""
    variable_specs = get_probabilistic_histogram_variable_specs()
    if not variable_specs:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.8), dpi=180)
    axes_list = list(np.asarray(axes).reshape(-1))
    color_map = {
        'fc': '#2563eb',
        'fy_tarik': '#dc2626',
        'fy_tekan': '#f59e0b',
        'fy_geser': '#7c3aed',
        'qDL': '#0f766e',
        'qLL': '#be185d'
    }
    plotted_any = False

    for axis, spec in zip(axes_list, variable_specs):
        variable_name = build_histogram_variable_name(spec['type'], int(elem_id))
        record = histogram_data.get(variable_name)
        plot_label = str(spec.get('plot_label', spec['label']))
        if not record:
            axis.axis('off')
            axis.text(
                0.5,
                0.5,
                f"Data for {plot_label} in E{int(elem_id)}\nis unavailable.",
                ha='center',
                va='center',
                fontsize=10,
                color='#475569',
                bbox=dict(
                    boxstyle='round,pad=0.25',
                    facecolor='#f8fafc',
                    edgecolor='#cbd5e1'
                )
            )
            continue

        hist_edges = np.asarray(record.get('hist_bin_edges', []), dtype=float)
        hist_density = np.asarray(record.get('hist_density', []), dtype=float)
        if hist_edges.size < 2 or hist_density.size == 0:
            axis.axis('off')
            continue

        bin_widths = np.diff(hist_edges)
        bin_centers = hist_edges[:-1] + (0.5 * bin_widths)
        face_color = color_map.get(spec['type'], '#2563eb')
        axis.bar(
            bin_centers,
            hist_density,
            width=bin_widths * 0.92,
            color=face_color,
            alpha=0.38,
            edgecolor='#1f2937',
            linewidth=0.8,
            label='Monte Carlo Histogram'
        )

        pdf_x = np.linspace(hist_edges[0], hist_edges[-1], 320)
        pdf_y = build_probability_density_curve(
            str(record.get('distribution', spec['distribution_label'])),
            record.get('mean'),
            record.get('stddev'),
            pdf_x
        )
        if pdf_y is not None and np.all(np.isfinite(pdf_y)):
            axis.plot(
                pdf_x,
                pdf_y,
                color='#111827',
                linewidth=1.8,
                label='Theoretical PDF'
            )

        input_mean = coerce_finite_float(record.get('mean'))
        sample_mean = coerce_finite_float(record.get('sample_mean'))
        if input_mean is not None:
            axis.axvline(
                input_mean,
                color='#dc2626',
                linestyle='--',
                linewidth=1.1,
                label='Input Mean'
            )
        if sample_mean is not None:
            axis.axvline(
                sample_mean,
                color='#2563eb',
                linestyle=':',
                linewidth=1.2,
                label='Sample Mean'
            )

        axis.set_title(
            f"{plot_label} | E{int(elem_id)}\nDistribution: {spec['distribution_label']}",
            fontsize=10.5,
            pad=10
        )
        axis.set_xlabel(f"Value ({record.get('unit', spec['unit'])})")
        axis.set_ylabel('Probability Density')
        axis.grid(True, alpha=0.22, linestyle='--')
        axis.legend(fontsize=8, loc='best')
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return None

    fig.suptitle(
        f"Probabilistic Random Variable Histograms | Element E{int(elem_id)}",
        fontsize=13,
        y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def get_probabilistic_limit_state_histogram_specs() -> List[Dict[str, str]]:
    """Spesifikasi limit-state untuk histogram respons R, Q, dan g(x)."""
    return [
        {
            'key': 'moment',
            'label': 'Lentur',
            'plot_label': 'Flexure',
            'unit': 'kN.m',
            'color': '#f59e0b'
        },
        {
            'key': 'shear',
            'label': 'Geser',
            'plot_label': 'Shear',
            'unit': 'kN',
            'color': '#16a34a'
        },
        {
            'key': 'axial',
            'label': 'Aksial',
            'plot_label': 'Axial',
            'unit': 'kN',
            'color': '#2563eb'
        },
        {
            'key': 'axial_moment',
            'label': 'Aksial+Lentur',
            'plot_label': 'Axial-Flexure Interaction',
            'unit': '(-)',
            'color': '#7c3aed'
        }
    ]


def build_limit_state_histogram_record_name(limit_state: str, elem_id: int) -> str:
    """Nama record histogram respons limit-state per elemen."""
    return f"{str(limit_state).strip()}_E{int(elem_id)}"


def build_probabilistic_limit_state_histogram_summary_df(
    histogram_data: Dict[str, Dict[str, Any]],
    elem_id: int
) -> pd.DataFrame:
    """Ringkas statistik histogram R, Q, dan g(x) per limit-state untuk satu elemen."""
    rows = []
    for spec in get_probabilistic_limit_state_histogram_specs():
        record = histogram_data.get(
            build_limit_state_histogram_record_name(spec['key'], int(elem_id))
        )
        if not record:
            continue

        r_summary = record.get('R', {}) or {}
        q_summary = record.get('Q', {}) or {}
        g_summary = record.get('g', {}) or {}
        rows.append({
            'Limit State': record.get('limit_state_label', spec['label']),
            'Mean R': r_summary.get('sample_mean'),
            'Mean Q': q_summary.get('sample_mean'),
            'Mean g(x)': g_summary.get('sample_mean'),
            'StdDev g(x)': g_summary.get('sample_std'),
            'Minimum g(x)': g_summary.get('sample_min'),
            'Maksimum g(x)': g_summary.get('sample_max'),
            'Jumlah Sampel Valid (-)': record.get('sample_count'),
            'Jumlah Gagal dari g(x) (-)': record.get('failure_count'),
            'Pf dari g(x) (-)': record.get('Pf_from_g'),
            'Satuan': record.get('unit', spec['unit'])
        })

    return pd.DataFrame(rows)


def plot_histogram_summary_on_axis(axis,
                                   hist_summary: Dict[str, Any],
                                   color: str,
                                   label: str,
                                   alpha_fill: float = 0.16) -> bool:
    """Plot histogram teringkas ke axis matplotlib tanpa data mentah."""
    hist_edges = np.asarray((hist_summary or {}).get('hist_bin_edges', []), dtype=float)
    hist_values = np.asarray((hist_summary or {}).get('hist_values', []), dtype=float)
    if hist_edges.size < 2 or hist_values.size == 0:
        return False

    axis.stairs(
        hist_values,
        hist_edges,
        fill=True,
        alpha=alpha_fill,
        color=color,
        linewidth=1.0,
        label=label
    )
    axis.stairs(
        hist_values,
        hist_edges,
        fill=False,
        color=color,
        linewidth=1.6
    )
    return True


def plot_histogram_theoretical_pdf_on_axis(axis,
                                           hist_summary: Dict[str, Any],
                                           color: str,
                                           label: str,
                                           distribution: str = 'normal',
                                           linestyle: str = '-',
                                           linewidth: float = 1.8,
                                           alpha: float = 0.95) -> bool:
    """Overlay PDF teoritis/aproksimasi berdasarkan mean dan simpangan baku respons."""
    hist_edges = np.asarray((hist_summary or {}).get('hist_bin_edges', []), dtype=float)
    if hist_edges.size < 2:
        return False

    pdf_x = np.linspace(hist_edges[0], hist_edges[-1], 320)
    pdf_y = build_probability_density_curve(
        distribution,
        (hist_summary or {}).get('sample_mean'),
        (hist_summary or {}).get('sample_std'),
        pdf_x
    )
    if pdf_y is None or not np.all(np.isfinite(pdf_y)):
        return False

    axis.plot(
        pdf_x,
        pdf_y,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label=label
    )
    return True


def build_histogram_frequency_summary(hist_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Ubah histogram kerapatan menjadi histogram frekuensi per bin."""
    hist_edges = np.asarray((hist_summary or {}).get('hist_bin_edges', []), dtype=float)
    hist_density = np.asarray((hist_summary or {}).get('hist_values', []), dtype=float)
    sample_count = int((hist_summary or {}).get('sample_count', 0) or 0)
    if hist_edges.size < 2 or hist_density.size == 0 or sample_count <= 0:
        return {}

    bin_widths = np.diff(hist_edges)
    if bin_widths.size == 0 or not np.all(np.isfinite(bin_widths)):
        return {}

    raw_counts = np.clip(hist_density * bin_widths * float(sample_count), 0.0, None)
    count_values = np.rint(raw_counts).astype(int)
    count_difference = int(sample_count - np.sum(count_values))

    if count_difference != 0 and count_values.size > 0:
        residuals = raw_counts - count_values.astype(float)
        if count_difference > 0:
            candidate_indices = np.argsort(-residuals)
            for idx in candidate_indices[:count_difference]:
                count_values[int(idx)] += 1
        else:
            candidate_indices = np.argsort(residuals)
            remaining = abs(count_difference)
            for idx in candidate_indices:
                idx = int(idx)
                if remaining <= 0:
                    break
                if count_values[idx] <= 0:
                    continue
                count_values[idx] -= 1
                remaining -= 1

    return {
        'hist_bin_edges': hist_edges.astype(float).tolist(),
        'hist_values': count_values.astype(int).tolist(),
        'sample_count': sample_count
    }


def plot_histogram_frequency_on_axis(axis,
                                     hist_summary: Dict[str, Any],
                                     color: str,
                                     label: str,
                                     alpha_fill: float = 0.24) -> bool:
    """Plot histogram frekuensi absolut ke axis matplotlib."""
    frequency_summary = build_histogram_frequency_summary(hist_summary)
    hist_edges = np.asarray(frequency_summary.get('hist_bin_edges', []), dtype=float)
    hist_values = np.asarray(frequency_summary.get('hist_values', []), dtype=float)
    if hist_edges.size < 2 or hist_values.size == 0:
        return False

    axis.stairs(
        hist_values,
        hist_edges,
        fill=True,
        alpha=alpha_fill,
        color=color,
        linewidth=1.0,
        label=label
    )
    axis.stairs(
        hist_values,
        hist_edges,
        fill=False,
        color=color,
        linewidth=1.6
    )
    return True


def build_probabilistic_limit_state_histogram_figure(
    histogram_data: Dict[str, Dict[str, Any]],
    elem_id: int
) -> Optional[plt.Figure]:
    """Bangun figure histogram R/Q dan g(x) untuk seluruh limit-state elemen terpilih."""
    state_specs = get_probabilistic_limit_state_histogram_specs()
    fig, axes = plt.subplots(4, 2, figsize=(15.5, 16.0), dpi=180)
    plotted_any = False

    for row_index, spec in enumerate(state_specs):
        left_axis = axes[row_index, 0]
        right_axis = axes[row_index, 1]
        plot_label = str(spec.get('plot_label', spec['label']))
        record = histogram_data.get(
            build_limit_state_histogram_record_name(spec['key'], int(elem_id))
        )
        if not record:
            for axis, panel_title in (
                (left_axis, f"{plot_label} | R and Q Histograms"),
                (right_axis, f"{plot_label} | g(x) Histogram")
            ):
                axis.axis('off')
                axis.text(
                    0.5,
                    0.5,
                    (
                        f"Data for {plot_label} in E{int(elem_id)}\n"
                        "is unavailable or not applicable."
                    ),
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='#475569',
                    bbox=dict(
                        boxstyle='round,pad=0.25',
                        facecolor='#f8fafc',
                        edgecolor='#cbd5e1'
                    )
                )
                axis.set_title(panel_title, fontsize=10.5, pad=10)
            continue

        r_summary = record.get('R', {}) or {}
        q_summary = record.get('Q', {}) or {}
        g_summary = record.get('g', {}) or {}
        unit_label = record.get('unit', spec['unit'])

        rq_plotted = False
        rq_plotted |= plot_histogram_summary_on_axis(
            left_axis,
            r_summary,
            color='#dc2626',
            label='R Histogram'
        )
        rq_plotted |= plot_histogram_summary_on_axis(
            left_axis,
            q_summary,
            color='#2563eb',
            label='Q Histogram'
        )
        rq_plotted |= plot_histogram_theoretical_pdf_on_axis(
            left_axis,
            r_summary,
            color='#991b1b',
            label='Normal PDF of R',
            distribution='normal',
            linestyle='--',
            linewidth=1.6
        )
        rq_plotted |= plot_histogram_theoretical_pdf_on_axis(
            left_axis,
            q_summary,
            color='#1d4ed8',
            label='Normal PDF of Q',
            distribution='normal',
            linestyle='-.',
            linewidth=1.6
        )
        r_mean = coerce_finite_float(r_summary.get('sample_mean'))
        q_mean = coerce_finite_float(q_summary.get('sample_mean'))
        if r_mean is not None:
            left_axis.axvline(
                r_mean,
                color='#dc2626',
                linestyle='--',
                linewidth=1.1,
                alpha=0.9
            )
        if q_mean is not None:
            left_axis.axvline(
                q_mean,
                color='#2563eb',
                linestyle=':',
                linewidth=1.2,
                alpha=0.95
            )
        left_axis.set_title(f"{plot_label} | R and Q Histograms", fontsize=10.5, pad=10)
        left_axis.set_xlabel(f"Value ({unit_label})")
        left_axis.set_ylabel('Probability Density')
        left_axis.grid(True, alpha=0.22, linestyle='--')
        if rq_plotted:
            left_axis.legend(loc='best', fontsize=8)

        g_plotted = plot_histogram_summary_on_axis(
            right_axis,
            g_summary,
            color=spec['color'],
            label='g(x) Histogram',
            alpha_fill=0.24
        )
        g_plotted |= plot_histogram_theoretical_pdf_on_axis(
            right_axis,
            g_summary,
            color='#111827',
            label='Normal PDF of g(x)',
            distribution='normal',
            linestyle='-',
            linewidth=1.8
        )
        right_axis.axvline(
            0.0,
            color='#111827',
            linestyle='--',
            linewidth=1.1,
            alpha=0.9,
            label='Failure Boundary g = 0'
        )
        g_mean = coerce_finite_float(g_summary.get('sample_mean'))
        if g_mean is not None:
            right_axis.axvline(
                g_mean,
                color=spec['color'],
                linestyle=':',
                linewidth=1.2,
                alpha=0.95,
                label='Mean g(x)'
            )
        right_axis.set_title(f"{plot_label} | g(x) Histogram", fontsize=10.5, pad=10)
        right_axis.set_xlabel(f"g(x) ({unit_label})")
        right_axis.set_ylabel('Probability Density')
        right_axis.grid(True, alpha=0.22, linestyle='--')
        if g_plotted:
            right_axis.legend(
                loc='center right',
                bbox_to_anchor=(1.0, 0.5),
                fontsize=8
            )

        right_axis.text(
            0.98,
            0.96,
            (
                f"Valid N = {int(record.get('sample_count', 0))}\n"
                f"Failures = {int(record.get('failure_count', 0))}\n"
                f"Pf = {float(record.get('Pf_from_g', 0.0)):.4f}"
            ),
            transform=right_axis.transAxes,
            ha='right',
            va='top',
            fontsize=8.5,
            bbox=dict(
                boxstyle='round,pad=0.25',
                facecolor='white',
                alpha=0.88,
                edgecolor='#cbd5e1'
            )
        )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return None

    fig.suptitle(
        f"Limit State Response Histograms | Element E{int(elem_id)}",
        fontsize=13,
        y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    return fig


def build_probabilistic_limit_state_g_frequency_figure(
    histogram_data: Dict[str, Dict[str, Any]],
    elem_id: int
) -> Optional[plt.Figure]:
    """Bangun figure histogram g(x) terhadap frekuensi untuk seluruh limit-state."""
    state_specs = get_probabilistic_limit_state_histogram_specs()
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.8), dpi=180)
    axes_list = list(np.asarray(axes).reshape(-1))
    plotted_any = False

    for axis, spec in zip(axes_list, state_specs):
        plot_label = str(spec.get('plot_label', spec['label']))
        record = histogram_data.get(
            build_limit_state_histogram_record_name(spec['key'], int(elem_id))
        )
        if not record:
            axis.axis('off')
            axis.text(
                0.5,
                0.5,
                (
                    f"Data for {plot_label} in E{int(elem_id)}\n"
                    "is unavailable or not applicable."
                ),
                ha='center',
                va='center',
                fontsize=10,
                color='#475569',
                bbox=dict(
                    boxstyle='round,pad=0.25',
                    facecolor='#f8fafc',
                    edgecolor='#cbd5e1'
                )
            )
            axis.set_title(f"{plot_label} | g(x) Frequency Histogram", fontsize=10.5, pad=10)
            continue

        g_summary = record.get('g', {}) or {}
        unit_label = record.get('unit', spec['unit'])
        g_plotted = plot_histogram_frequency_on_axis(
            axis,
            g_summary,
            color=spec['color'],
            label='g(x) Frequency',
            alpha_fill=0.26
        )
        axis.axvline(
            0.0,
            color='#111827',
            linestyle='--',
            linewidth=1.1,
            alpha=0.9,
            label='Failure Boundary g = 0'
        )
        g_mean = coerce_finite_float(g_summary.get('sample_mean'))
        if g_mean is not None:
            axis.axvline(
                g_mean,
                color=spec['color'],
                linestyle=':',
                linewidth=1.2,
                alpha=0.95,
                label='Mean g(x)'
            )

        axis.set_title(f"{plot_label} | g(x) Frequency Histogram", fontsize=10.5, pad=10)
        axis.set_xlabel(f"g(x) ({unit_label})")
        axis.set_ylabel('Frequency')
        axis.grid(True, alpha=0.22, linestyle='--')
        if g_plotted:
            axis.legend(loc='best', fontsize=8)

        axis.text(
            0.98,
            0.96,
            (
                f"Valid N = {int(record.get('sample_count', 0))}\n"
                f"Failures = {int(record.get('failure_count', 0))}\n"
                f"Pf = {float(record.get('Pf_from_g', 0.0)):.4f}"
            ),
            transform=axis.transAxes,
            ha='right',
            va='top',
            fontsize=8.5,
            bbox=dict(
                boxstyle='round,pad=0.25',
                facecolor='white',
                alpha=0.88,
                edgecolor='#cbd5e1'
            )
        )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return None

    fig.suptitle(
        f"g(x) Frequency Histograms | Element E{int(elem_id)}",
        fontsize=13,
        y=0.99
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def render_probabilistic_histogram_output_section(results_bundle: Dict,
                                                  heading_level: str = "####") -> None:
    """Tampilkan histogram variabel random untuk mode probabilistik."""
    histogram_data = (results_bundle or {}).get('probabilistic_histogram_data', {}) or {}
    limit_state_histogram_data = (
        (results_bundle or {}).get('probabilistic_limit_state_histogram_data', {}) or {}
    )
    summary = (results_bundle or {}).get('summary', {}) or {}
    if not histogram_data:
        st.info(
            "Histogram variabel acak belum tersedia. Jalankan analisis probabilistik "
            "agar sampel Monte Carlo dapat diringkas ke tab ini."
        )
        return

    allowed_types = {
        spec['type']
        for spec in get_probabilistic_histogram_variable_specs()
    }
    available_element_ids = sorted({
        int(record.get('element_id'))
        for record in histogram_data.values()
        if record.get('variable_type') in allowed_types
        and record.get('element_id') is not None
    })
    if not available_element_ids:
        st.info("Tidak ada elemen dengan data histogram variabel acak yang dapat ditampilkan.")
        return

    selected_element_id = st.selectbox(
        "Pilih elemen untuk histogram variabel acak",
        options=available_element_ids,
        format_func=lambda elem_id: f"E{int(elem_id)}",
        key="probabilistic_histogram_element_selector"
    )

    variable_specs = get_probabilistic_histogram_variable_specs()
    available_variable_count = int(sum(
        1
        for spec in variable_specs
        if build_histogram_variable_name(spec['type'], int(selected_element_id)) in histogram_data
    ))

    st.markdown(f"{heading_level} Histogram Variabel Acak")
    st.caption(
        "Tab ini menampilkan histogram hasil sampling Monte Carlo aktual untuk variabel acak "
        "per elemen, lalu dibandingkan dengan `PDF` teoritis sesuai distribusi input."
    )
    st.caption(
        "Distribusi yang dipakai pada tampilan ini adalah: `fc = lognormal`, "
        "`fy tarik/tekan/geser = normal`, `beban mati = normal`, dan "
        "`beban hidup = lognormal`."
    )
    st.caption(
        "Elemen yang tersedia pada model aktif: "
        + ", ".join(f"`E{int(elem_id)}`" for elem_id in available_element_ids)
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Jumlah Simulasi", format_metric_comma(summary.get('num_simulations'), 0))
    metric_cols[1].metric("Elemen Terpilih", f"E{int(selected_element_id)}")
    metric_cols[2].metric("Variabel Tersedia", str(available_variable_count))
    metric_cols[3].metric("Analisis Gagal", format_metric_comma(summary.get('analysis_failures'), 0))

    histogram_fig = build_probabilistic_histogram_figure(
        histogram_data,
        elem_id=int(selected_element_id)
    )
    if histogram_fig is not None:
        render_plot(
            histogram_fig,
            interactive=True,
            viewer_key=f"probabilistic-histogram-random-e{int(selected_element_id)}",
            alt_text=f"Histogram variabel acak probabilistik elemen {int(selected_element_id)}",
            viewer_height=760,
            download_basename=f"histogram-variabel-acak-e{int(selected_element_id)}"
        )
    else:
        st.info("Histogram untuk elemen yang dipilih belum dapat dibentuk.")

    histogram_summary_df = build_probabilistic_histogram_summary_df(
        histogram_data,
        elem_id=int(selected_element_id)
    )
    st.markdown(f"{heading_level} Tabel Ringkasan Histogram")
    st.caption(
        "Tabel ini membandingkan parameter input (`mean/stddev`) dengan statistik sampel Monte Carlo "
        "yang benar-benar terbentuk pada elemen terpilih."
    )
    if histogram_summary_df.empty:
        st.info("Ringkasan histogram untuk elemen yang dipilih belum tersedia.")
    else:
        render_input_table(
            histogram_summary_df,
            styler=style_input_dataframe(
                histogram_summary_df,
                table_min_width_px=1700
            )
        )

    st.markdown(f"{heading_level} Histogram Respons Limit State")
    st.caption(
        "Bagian ini menampilkan histogram `g(x)` untuk lentur, geser, aksial, dan "
        "aksial-lentur. Pada setiap limit state, panel kiri memperlihatkan histogram "
        "`R` dan `Q` dalam satu grafik, sedangkan panel kanan memperlihatkan histogram `g(x)`."
    )
    st.caption(
        "Setiap panel juga dilengkapi kurva `PDF` teoritis aproksimasi normal yang "
        "dibentuk dari `mean` dan `simpangan baku` respons Monte Carlo pada elemen terpilih."
    )
    st.caption(
        "Nilai `R`, `Q`, dan `g(x)` diambil langsung dari hasil setiap simulasi Monte Carlo "
        "yang sudah dihitung sebelumnya, sehingga tampilan ini tidak mengubah hasil perhitungan."
    )
    if int(summary.get('analysis_failures', 0) or 0) > 0:
        st.caption(
            "Simulasi yang gagal dieksekusi tidak ikut dihitung pada histogram respons limit state, "
            "karena tidak menghasilkan pasangan `R/Q/g(x)` yang valid."
        )

    if not limit_state_histogram_data:
        st.info("Histogram respons limit state belum tersedia untuk hasil analisis ini.")
        return

    limit_state_histogram_fig = build_probabilistic_limit_state_histogram_figure(
        limit_state_histogram_data,
        elem_id=int(selected_element_id)
    )
    if limit_state_histogram_fig is not None:
        render_plot(
            limit_state_histogram_fig,
            interactive=True,
            viewer_key=f"probabilistic-histogram-limit-state-e{int(selected_element_id)}",
            alt_text=f"Histogram limit state probabilistik elemen {int(selected_element_id)}",
            viewer_height=980,
            download_basename=f"histogram-limit-state-e{int(selected_element_id)}"
        )
    else:
        st.info("Histogram respons limit state untuk elemen yang dipilih belum dapat dibentuk.")

    st.caption(
        "Grafik berikut menampilkan hubungan `g(x)` dengan `frekuensi` absolut "
        "(jumlah sampel per bin), sehingga lebih mudah melihat sebaran jumlah kejadian."
    )
    limit_state_frequency_fig = build_probabilistic_limit_state_g_frequency_figure(
        limit_state_histogram_data,
        elem_id=int(selected_element_id)
    )
    if limit_state_frequency_fig is not None:
        render_plot(
            limit_state_frequency_fig,
            interactive=True,
            viewer_key=f"probabilistic-histogram-limit-state-frequency-e{int(selected_element_id)}",
            alt_text=f"Histogram g(x) terhadap frekuensi elemen {int(selected_element_id)}",
            viewer_height=780,
            download_basename=f"histogram-frekuensi-gx-e{int(selected_element_id)}"
        )
    else:
        st.info("Histogram frekuensi g(x) untuk elemen yang dipilih belum dapat dibentuk.")

    limit_state_histogram_summary_df = build_probabilistic_limit_state_histogram_summary_df(
        limit_state_histogram_data,
        elem_id=int(selected_element_id)
    )
    st.markdown(f"{heading_level} Tabel Ringkasan Histogram Respons")
    st.caption(
        "Tabel ini merangkum statistik akhir distribusi `R`, `Q`, dan `g(x)` "
        "untuk setiap limit state pada elemen terpilih."
    )
    if limit_state_histogram_summary_df.empty:
        st.info("Ringkasan histogram respons limit state untuk elemen yang dipilih belum tersedia.")
    else:
        render_input_table(
            limit_state_histogram_summary_df,
            styler=style_input_dataframe(
                limit_state_histogram_summary_df,
                table_min_width_px=1650
            )
        )


def build_probabilistic_limit_state_physical_cloud_data(
    analysis: Optional[PortalReliabilityAnalysis],
    max_points_per_state: int = LIMIT_STATE_PHYSICAL_CLOUD_MAX_POINTS,
    max_failed_points: int = LIMIT_STATE_PHYSICAL_CLOUD_MAX_FAILED_POINTS
) -> Dict[str, Any]:
    """Ringkas respons R/Q/g Monte Carlo untuk failure cloud di ruang fisik."""
    if analysis is None or not analysis.is_probabilistic or not analysis.mc_results:
        return {}

    analysis_history = list(analysis.mc_results.get('max_forces_history', []) or [])
    if not analysis_history:
        return {}
    sample_history = list(analysis.mc_results.get('random_samples_history', []) or [])
    random_variables = getattr(analysis, 'random_variables', {}) or {}
    sample_u_space_radii = build_random_sample_u_space_radius_array(
        sample_history,
        random_variables
    )

    state_specs = {
        str(spec['key']): spec
        for spec in get_probabilistic_limit_state_histogram_specs()
    }
    applicable_by_state = analysis._get_applicable_element_ids_by_limit_state()
    elements: Dict[str, Dict[str, Any]] = {}

    for limit_state, elem_ids in applicable_by_state.items():
        spec = state_specs.get(str(limit_state))
        if spec is None:
            continue

        for elem_id in sorted(int(value) for value in elem_ids):
            response_records = []
            for sample_index, analysis_result in enumerate(analysis_history):
                response = analysis._extract_limit_state_histogram_response(
                    analysis_result,
                    int(elem_id),
                    str(limit_state)
                )
                if response:
                    response_records.append({
                        'sample_index': int(sample_index),
                        'u_space_radius': (
                            float(sample_u_space_radii[sample_index])
                            if sample_index < sample_u_space_radii.size
                            and np.isfinite(sample_u_space_radii[sample_index])
                            else np.nan
                        ),
                        'response': dict(response),
                        'random_sample': (
                            sample_history[sample_index]
                            if (
                                sample_index < len(sample_history)
                                and isinstance(sample_history[sample_index], dict)
                            ) else
                            {}
                        )
                    })

            if not response_records:
                continue

            def build_material_sample_array(variable_key: str) -> np.ndarray:
                sampled_values = []
                for response_record in response_records:
                    material_value = get_physical_limit_state_material_sample_value(
                        analysis.data,
                        int(elem_id),
                        response_record.get('random_sample') or {},
                        variable_key
                    )
                    sampled_values.append(
                        np.nan if material_value is None else float(material_value)
                    )
                return np.asarray(sampled_values, dtype=float)

            r_values = np.asarray(
                [float(item['response']['R']) for item in response_records],
                dtype=float
            )
            q_values = np.asarray(
                [float(item['response']['Q']) for item in response_records],
                dtype=float
            )
            g_values = np.asarray(
                [float(item['response']['g']) for item in response_records],
                dtype=float
            )
            radius_array = np.asarray(
                [
                    np.nan if coerce_finite_float(item.get('u_space_radius')) is None else
                    float(coerce_finite_float(item.get('u_space_radius')))
                    for item in response_records
                ],
                dtype=float
            )
            estimated_projected_mpp = estimate_limit_state_projected_mpp(
                limit_state=str(limit_state),
                r_values=r_values,
                q_values=q_values,
                g_values=g_values,
                response_records=response_records,
                random_variables=random_variables
            )
            fc_values = build_material_sample_array('fc')
            fy_tarik_values = build_material_sample_array('fy_tarik')
            fy_tekan_values = build_material_sample_array('fy_tekan')
            fy_geser_values = build_material_sample_array('fy_geser')
            failure_mask = np.asarray(g_values < 0.0, dtype=bool)
            sample_indices = np.arange(g_values.size, dtype=int)
            failure_indices = sample_indices[failure_mask]

            if int(max_points_per_state or 0) <= 0:
                selected_failed = failure_indices.astype(int)
                selected_indices = sample_indices.astype(int)
            else:
                selected_failed = downsample_index_array(
                    failure_indices,
                    min(int(max_failed_points), int(max_points_per_state))
                )
                remaining_capacity = max(int(max_points_per_state) - int(selected_failed.size), 0)
                selected_safe = downsample_index_array(
                    sample_indices[~failure_mask],
                    remaining_capacity
                )
                selected_indices = np.sort(
                    np.concatenate([selected_failed, selected_safe]).astype(int)
                )
            projected_mpp_sample_index = coerce_finite_float(
                (estimated_projected_mpp or {}).get('sample_index')
            )
            if projected_mpp_sample_index is not None:
                projected_mpp_sample_index_int = int(projected_mpp_sample_index)
                if 0 <= projected_mpp_sample_index_int < g_values.size:
                    selected_indices = np.unique(
                        np.concatenate([
                            selected_indices.astype(int),
                            np.asarray([projected_mpp_sample_index_int], dtype=int)
                        ])
                    ).astype(int)
            if selected_indices.size == 0:
                continue

            element_code = get_element_code_from_input(analysis.data, int(elem_id))
            element_entry = elements.setdefault(
                str(int(elem_id)),
                {
                    'element_id': int(elem_id),
                    'element_code': str(element_code),
                    'element_type': get_element_type_label(element_code),
                    'states': {}
                }
            )
            element_entry['states'][str(limit_state)] = {
                'limit_state': str(limit_state),
                'limit_state_label': str(spec['label']),
                'plot_label': str(spec.get('plot_label', spec['label'])),
                'unit': str(spec.get('unit', '-')),
                'color': str(spec.get('color', '#2563eb')),
                'sample_count': int(g_values.size),
                'failure_count': int(np.sum(failure_mask)),
                'safe_count': int(np.sum(~failure_mask)),
                'Pf_from_g': float(np.mean(failure_mask)),
                'estimated_projected_mpp': dict(estimated_projected_mpp or {}),
                'used_downsampling': bool(selected_indices.size < g_values.size),
                'failed_points_truncated': bool(
                    selected_failed.size < failure_indices.size
                ),
                'R': r_values[selected_indices].astype(float).tolist(),
                'Q': q_values[selected_indices].astype(float).tolist(),
                'g': g_values[selected_indices].astype(float).tolist(),
                'u_space_radius': radius_array[selected_indices].astype(float).tolist(),
                'failure_mask': failure_mask[selected_indices].astype(bool).tolist(),
                'material_samples': {
                    'fc': fc_values[selected_indices].astype(float).tolist(),
                    'fy_tarik': fy_tarik_values[selected_indices].astype(float).tolist(),
                    'fy_tekan': fy_tekan_values[selected_indices].astype(float).tolist(),
                    'fy_geser': fy_geser_values[selected_indices].astype(float).tolist()
                }
            }

    element_ids = sorted(
        int(elem_id)
        for elem_id, record in elements.items()
        if (record or {}).get('states')
    )
    if not element_ids:
        return {}

    return {
        'num_simulations': int(
            analysis.mc_results.get('num_simulations', len(analysis_history))
            or len(analysis_history)
        ),
        'analysis_failures': int(analysis.mc_results.get('analysis_failures', 0) or 0),
        'element_ids': element_ids,
        'elements': elements
    }


def transform_random_sample_value_to_standard_normal_space(
    value: Any,
    variable_info: Optional[Dict[str, Any]]
) -> Optional[float]:
    """Transform satu nilai sampel acak ke ruang normal baku `U`."""
    numeric_value = coerce_finite_float(value)
    if numeric_value is None:
        return None

    variable_info = variable_info or {}
    distribution_name = str(
        variable_info.get('distribution', 'normal') or 'normal'
    ).strip().lower()
    mean_value = coerce_finite_float(variable_info.get('mean'))
    stddev_value = coerce_finite_float(variable_info.get('stddev'))

    if distribution_name == 'normal':
        if mean_value is None:
            mean_value = 0.0
        if stddev_value is None or stddev_value <= 0.0:
            return 0.0
        return float((float(numeric_value) - float(mean_value)) / float(stddev_value))

    if distribution_name == 'lognormal':
        if mean_value is None or mean_value <= 0.0:
            return None
        if stddev_value is None or stddev_value <= 0.0:
            return 0.0

        variance_ratio = (float(stddev_value) / float(mean_value)) ** 2
        sigma_ln = np.sqrt(np.log(1.0 + variance_ratio))
        if not np.isfinite(sigma_ln) or sigma_ln <= 0.0:
            return 0.0

        mu_ln = np.log(float(mean_value)) - 0.5 * sigma_ln ** 2
        clipped_value = max(float(numeric_value), 1e-12)
        return float((np.log(clipped_value) - mu_ln) / sigma_ln)

    if mean_value is None:
        mean_value = 0.0
    if stddev_value is None or stddev_value <= 0.0:
        return 0.0
    return float((float(numeric_value) - float(mean_value)) / float(stddev_value))


def compute_random_sample_u_space_radius(
    random_sample: Optional[Dict[str, Any]],
    random_variables: Optional[Dict[str, Dict[str, Any]]]
) -> Optional[float]:
    """Hitung radius sampel Monte Carlo pada ruang normal baku `U`."""
    if not isinstance(random_sample, dict) or not random_variables:
        return None

    u_components = []
    for variable_name, variable_info in (random_variables or {}).items():
        u_value = transform_random_sample_value_to_standard_normal_space(
            random_sample.get(variable_name),
            variable_info if isinstance(variable_info, dict) else {}
        )
        if u_value is None:
            continue
        u_components.append(float(u_value))

    if not u_components:
        return None

    u_array = np.asarray(u_components, dtype=float)
    u_array = u_array[np.isfinite(u_array)]
    if u_array.size == 0:
        return None
    return float(np.linalg.norm(u_array))


def build_random_sample_u_space_radius_array(
    sample_history: Optional[List[Dict[str, Any]]],
    random_variables: Optional[Dict[str, Dict[str, Any]]]
) -> np.ndarray:
    """Hitung radius `U-space` untuk seluruh sampel sekali saja agar bisa dipakai ulang."""
    if not sample_history or not random_variables:
        return np.asarray([], dtype=float)

    radius_values = []
    for random_sample in list(sample_history or []):
        sample_radius = compute_random_sample_u_space_radius(
            random_sample if isinstance(random_sample, dict) else {},
            random_variables
        )
        radius_values.append(
            np.nan if sample_radius is None else float(sample_radius)
        )
    return np.asarray(radius_values, dtype=float)


def estimate_limit_state_projected_mpp(
    limit_state: str,
    r_values: np.ndarray,
    q_values: np.ndarray,
    g_values: np.ndarray,
    response_records: Optional[List[Dict[str, Any]]],
    random_variables: Optional[Dict[str, Dict[str, Any]]]
) -> Dict[str, Any]:
    """Estimasi `projected MPP` cloud fisik dari boundary band SMC terdekat `g=0`."""
    r_array = np.asarray(r_values, dtype=float).reshape(-1)
    q_array = np.asarray(q_values, dtype=float).reshape(-1)
    g_array = np.asarray(g_values, dtype=float).reshape(-1)
    response_records = list(response_records or [])

    common_size = min(int(r_array.size), int(q_array.size), int(g_array.size), len(response_records))
    if common_size <= 0 or not random_variables:
        return {}

    radius_values = []
    for index in range(common_size):
        sample_radius = coerce_finite_float(
            (response_records[index] or {}).get('u_space_radius')
        )
        if sample_radius is None:
            sample_radius = compute_random_sample_u_space_radius(
                (response_records[index] or {}).get('random_sample') or {},
                random_variables
            )
        radius_values.append(
            np.nan if sample_radius is None else float(sample_radius)
        )
    radius_array = np.asarray(radius_values, dtype=float)

    valid_mask = (
        np.isfinite(r_array[:common_size])
        & np.isfinite(q_array[:common_size])
        & np.isfinite(g_array[:common_size])
        & np.isfinite(radius_array)
    )
    if not np.any(valid_mask):
        return {}

    valid_indices = np.where(valid_mask)[0]
    abs_g_values = np.abs(g_array[:common_size][valid_mask])
    valid_radii = radius_array[valid_mask]
    pool_size = int(
        np.clip(
            max(np.sqrt(valid_indices.size), valid_indices.size * 0.08),
            10,
            48
        )
    )
    pool_size = min(pool_size, int(valid_indices.size))
    if pool_size <= 0:
        return {}

    boundary_order = np.argsort(abs_g_values, kind='stable')
    boundary_local_indices = boundary_order[:pool_size]
    best_boundary_position = boundary_local_indices[
        int(np.nanargmin(valid_radii[boundary_local_indices]))
    ]
    best_global_index = int(valid_indices[best_boundary_position])

    selected_r = float(r_array[best_global_index])
    selected_q = float(q_array[best_global_index])
    selected_g = float(g_array[best_global_index])
    selected_beta = float(radius_array[best_global_index])

    if str(limit_state) == 'axial_moment':
        sample_x = selected_r
        sample_y = selected_g
        contour_x = 1.0
        contour_y = 0.0
    else:
        sample_x = selected_q
        sample_y = selected_r
        contour_coordinate = 0.5 * (selected_q + selected_r)
        contour_x = float(contour_coordinate)
        contour_y = float(contour_coordinate)

    sample_index = (
        (response_records[best_global_index] or {}).get('sample_index')
        if best_global_index < len(response_records) else
        None
    )
    projection_distance = float(
        np.linalg.norm(
            np.asarray([sample_x - contour_x, sample_y - contour_y], dtype=float)
        )
    )

    return {
        'beta': selected_beta,
        'sample_index': None if sample_index is None else int(sample_index),
        'candidate_g': selected_g,
        'sample_x': float(sample_x),
        'sample_y': float(sample_y),
        'contour_x': float(contour_x),
        'contour_y': float(contour_y),
        'projection_distance': projection_distance,
        'selection_pool_size': int(pool_size),
        'selection_method': 'boundary-band min-u-radius'
    }


def resolve_limit_state_projected_mpp_from_record(
    record: Dict[str, Any],
    limit_state: str,
    target_beta: Optional[float] = None
) -> Dict[str, Any]:
    """Pilih `projected MPP` fisik yang ditambatkan ke `Beta(table)` bila tersedia."""
    r_array = np.asarray((record or {}).get('R', []), dtype=float).reshape(-1)
    q_array = np.asarray((record or {}).get('Q', []), dtype=float).reshape(-1)
    g_array = np.asarray((record or {}).get('g', []), dtype=float).reshape(-1)
    radius_array = np.asarray((record or {}).get('u_space_radius', []), dtype=float).reshape(-1)

    common_size = min(
        int(r_array.size),
        int(q_array.size),
        int(g_array.size),
        int(radius_array.size)
    )
    if common_size <= 0:
        return {}

    r_array = r_array[:common_size]
    q_array = q_array[:common_size]
    g_array = g_array[:common_size]
    radius_array = radius_array[:common_size]

    valid_mask = (
        np.isfinite(r_array)
        & np.isfinite(q_array)
        & np.isfinite(g_array)
        & np.isfinite(radius_array)
    )
    if not np.any(valid_mask):
        return {}

    valid_indices = np.where(valid_mask)[0]
    abs_g_values = np.abs(g_array[valid_mask])
    valid_radii = radius_array[valid_mask]
    pool_size = int(
        np.clip(
            max(np.sqrt(valid_indices.size), valid_indices.size * 0.08),
            10,
            48
        )
    )
    pool_size = min(pool_size, int(valid_indices.size))
    if pool_size <= 0:
        return {}

    boundary_order = np.argsort(abs_g_values, kind='stable')
    boundary_local_indices = boundary_order[:pool_size]
    target_beta_numeric = coerce_finite_float(target_beta)

    if target_beta_numeric is not None:
        beta_distance = np.abs(valid_radii[boundary_local_indices] - float(target_beta_numeric))
        local_abs_g = abs_g_values[boundary_local_indices]
        best_within_pool = int(
            np.lexsort((local_abs_g, beta_distance))[0]
        )
        best_boundary_position = boundary_local_indices[best_within_pool]
        selection_method = 'boundary-band closest-to-beta-table'
    else:
        best_boundary_position = boundary_local_indices[
            int(np.nanargmin(valid_radii[boundary_local_indices]))
        ]
        selection_method = 'boundary-band min-u-radius'

    best_global_index = int(valid_indices[best_boundary_position])
    selected_r = float(r_array[best_global_index])
    selected_q = float(q_array[best_global_index])
    selected_g = float(g_array[best_global_index])
    selected_beta = float(radius_array[best_global_index])

    if str(limit_state) == 'axial_moment':
        sample_x = selected_r
        sample_y = selected_g
        contour_x = 1.0
        contour_y = 0.0
    else:
        sample_x = selected_q
        sample_y = selected_r
        contour_coordinate = 0.5 * (selected_q + selected_r)
        contour_x = float(contour_coordinate)
        contour_y = float(contour_coordinate)

    return {
        'beta': selected_beta,
        'display_beta': (
            float(target_beta_numeric)
            if target_beta_numeric is not None else
            float(selected_beta)
        ),
        'display_beta_raw': (
            target_beta
            if target_beta is not None else
            selected_beta
        ),
        'selected_index': int(best_global_index),
        'candidate_g': selected_g,
        'sample_x': float(sample_x),
        'sample_y': float(sample_y),
        'contour_x': float(contour_x),
        'contour_y': float(contour_y),
        'projection_distance': float(
            np.linalg.norm(
                np.asarray([sample_x - contour_x, sample_y - contour_y], dtype=float)
            )
        ),
        'selection_pool_size': int(pool_size),
        'selection_method': str(selection_method)
    }


def project_point_onto_polyline(point_x: float,
                                point_y: float,
                                polyline: np.ndarray) -> Optional[np.ndarray]:
    """Proyeksikan satu titik 2D ke titik terdekat pada polyline."""
    try:
        point = np.asarray([float(point_x), float(point_y)], dtype=float)
    except (TypeError, ValueError):
        return None

    polyline_array = np.asarray(polyline, dtype=float)
    if polyline_array.ndim != 2 or polyline_array.shape[0] < 2:
        return None

    vertices = polyline_array[:, :2]
    best_projection = None
    best_distance = np.inf
    for start_point, end_point in zip(vertices[:-1], vertices[1:]):
        segment = end_point - start_point
        segment_length_squared = float(np.dot(segment, segment))
        if segment_length_squared <= 1e-18:
            projected_point = np.asarray(start_point, dtype=float)
        else:
            projection_ratio = float(
                np.clip(
                    np.dot(point - start_point, segment) / segment_length_squared,
                    0.0,
                    1.0
                )
            )
            projected_point = start_point + projection_ratio * segment
        current_distance = float(np.linalg.norm(point - projected_point))
        if current_distance < best_distance:
            best_distance = current_distance
            best_projection = np.asarray(projected_point, dtype=float)

    if best_projection is None or best_projection.size != 2:
        return None
    return best_projection.astype(float)


def resolve_material_space_beta_table_overlay(record: Dict[str, Any],
                                              limit_state: str,
                                              x_axis_key: str,
                                              y_axis_key: str,
                                              target_beta: Optional[float],
                                              zero_contour_segment: Optional[np.ndarray]) -> Dict[str, Any]:
    """Bangun overlay `MPP/Beta(table)` pada peta material-space."""
    x_values = get_limit_state_physical_cloud_axis_values(record, x_axis_key)
    y_values = get_limit_state_physical_cloud_axis_values(record, y_axis_key)
    return resolve_physical_map_beta_table_overlay(
        record=record,
        limit_state=limit_state,
        x_values=x_values,
        y_values=y_values,
        target_beta=target_beta,
        zero_contour_segment=zero_contour_segment
    )


def resolve_physical_map_beta_table_overlay(record: Dict[str, Any],
                                            limit_state: str,
                                            x_values: np.ndarray,
                                            y_values: np.ndarray,
                                            target_beta: Optional[float],
                                            zero_contour_segment: Optional[np.ndarray]) -> Dict[str, Any]:
    """Bangun overlay `MPP/Beta(table)` pada peta fisik 2D dengan sumbu bebas."""
    beta_anchor = resolve_limit_state_projected_mpp_from_record(
        record,
        limit_state=limit_state,
        target_beta=target_beta
    )
    selected_index = (
        int(beta_anchor.get('selected_index'))
        if coerce_finite_float(beta_anchor.get('selected_index')) is not None else
        None
    )
    if selected_index is None:
        return {}

    x_values = np.asarray(x_values, dtype=float).reshape(-1)
    y_values = np.asarray(y_values, dtype=float).reshape(-1)
    common_size = min(int(x_values.size), int(y_values.size))
    if selected_index < 0 or selected_index >= common_size:
        return {}

    sample_x = coerce_finite_float(x_values[selected_index])
    sample_y = coerce_finite_float(y_values[selected_index])
    if sample_x is None or sample_y is None:
        return {}

    contour_point = project_point_onto_polyline(
        float(sample_x),
        float(sample_y),
        zero_contour_segment
    ) if zero_contour_segment is not None else None

    if contour_point is None or contour_point.size != 2:
        return {}

    return {
        'display_beta': beta_anchor.get('display_beta'),
        'sample_beta': beta_anchor.get('beta'),
        'selected_index': int(selected_index),
        'sample_x': float(sample_x),
        'sample_y': float(sample_y),
        'contour_x': float(contour_point[0]),
        'contour_y': float(contour_point[1]),
        'candidate_g': beta_anchor.get('candidate_g'),
        'selection_method': beta_anchor.get('selection_method')
    }


def resolve_axial_moment_map_beta_table_overlay(record: Dict[str, Any],
                                                x_axis_key: str,
                                                y_axis_key: str,
                                                target_beta: Optional[float],
                                                zero_contour_segment: Optional[np.ndarray]) -> Dict[str, Any]:
    """Bangun overlay `MPP/Beta(table)` untuk peta custom aksial-lentur."""
    x_values = get_axial_moment_custom_axis_values(record, x_axis_key)
    y_values = get_axial_moment_custom_axis_values(record, y_axis_key)
    return resolve_physical_map_beta_table_overlay(
        record=record,
        limit_state='axial_moment',
        x_values=x_values,
        y_values=y_values,
        target_beta=target_beta,
        zero_contour_segment=zero_contour_segment
    )


def resolve_axial_moment_plot_demand_axial(force_data: Optional[Dict[str, Any]],
                                           controlling_state: Any) -> float:
    """Tentukan aksial demand bertanda untuk plot `P-M`."""
    axial_demands = get_axial_demands_from_force_data(force_data or {})
    controlling_state_text = str(controlling_state or '').strip().lower()

    if (
        controlling_state_text == 'tension'
        and axial_demands['tension'] > AXIAL_DEMAND_TOLERANCE_KN
    ):
        return -float(axial_demands['tension'])
    if controlling_state_text == 'pure-bending':
        return 0.0
    if axial_demands['compression'] > AXIAL_DEMAND_TOLERANCE_KN:
        return float(axial_demands['compression'])
    if axial_demands['tension'] > AXIAL_DEMAND_TOLERANCE_KN:
        return -float(axial_demands['tension'])
    return 0.0


def build_probabilistic_axial_moment_pm_cloud_data(
    analysis: Optional[PortalReliabilityAnalysis],
    max_points: int = AXIAL_MOMENT_PM_CLOUD_MAX_POINTS,
    max_failed_points: int = AXIAL_MOMENT_PM_CLOUD_MAX_FAILED_POINTS
) -> Dict[str, Any]:
    """Bangun dataset Monte Carlo aksial-lentur pada ruang fisik `P-M`."""
    if analysis is None or not analysis.is_probabilistic or not analysis.mc_results:
        return {}

    analysis_history = list(analysis.mc_results.get('max_forces_history', []) or [])
    if not analysis_history:
        return {}
    sample_history = list(analysis.mc_results.get('random_samples_history', []) or [])
    random_variables = getattr(analysis, 'random_variables', {}) or {}
    sample_u_space_radii = build_random_sample_u_space_radius_array(
        sample_history,
        random_variables
    )

    applicable_element_ids = sorted(
        int(elem_id)
        for elem_id in (
            analysis._get_applicable_element_ids_by_limit_state().get('axial_moment', [])
            or []
        )
    )
    if not applicable_element_ids:
        return {}

    elements: Dict[str, Dict[str, Any]] = {}
    for elem_id in applicable_element_ids:
        demand_moment_values = []
        demand_axial_values = []
        boundary_moment_values = []
        boundary_axial_values = []
        g_values = []
        lambda_values = []
        response_records = []

        for sample_index, analysis_result in enumerate(analysis_history):
            if not analysis_result:
                continue

            max_forces_entry = (
                get_by_element_value(analysis_result.get('max_forces', {}), int(elem_id), {})
                or {}
            )
            meta = (
                get_by_element_value(
                    analysis_result.get('performance_axial_moment_metadata', {}),
                    int(elem_id),
                    {}
                ) or {}
            )
            g_value = get_by_element_value(
                analysis_result.get('performance_axial_moment', {}),
                int(elem_id),
                None
            )
            demand_moment = coerce_finite_float(max_forces_entry.get('max_moment'))
            boundary_moment = coerce_finite_float(meta.get('phi_Mn'))
            boundary_axial = coerce_finite_float(meta.get('phi_Pn'))
            lambda_value = coerce_finite_float(meta.get('lambda'))
            g_numeric = coerce_finite_float(g_value)
            if (
                demand_moment is None
                or boundary_moment is None
                or boundary_axial is None
                or lambda_value is None
                or g_numeric is None
            ):
                continue

            force_data = max_forces_entry.get('forces', {}) or {}
            demand_axial = resolve_axial_moment_plot_demand_axial(
                force_data,
                meta.get('controlling_state')
            )

            demand_moment_values.append(abs(float(demand_moment)))
            demand_axial_values.append(float(demand_axial))
            boundary_moment_values.append(float(boundary_moment))
            boundary_axial_values.append(float(boundary_axial))
            g_values.append(float(g_numeric))
            lambda_values.append(float(lambda_value))
            response_records.append({
                'sample_index': int(sample_index),
                'u_space_radius': (
                    float(sample_u_space_radii[sample_index])
                    if sample_index < sample_u_space_radii.size
                    and np.isfinite(sample_u_space_radii[sample_index])
                    else np.nan
                ),
                'random_sample': (
                    sample_history[sample_index]
                    if (
                        sample_index < len(sample_history)
                        and isinstance(sample_history[sample_index], dict)
                    ) else
                    {}
                )
            })

        if not g_values:
            continue

        def build_material_sample_array(variable_key: str) -> np.ndarray:
            sampled_values = []
            for response_record in response_records:
                material_value = get_physical_limit_state_material_sample_value(
                    analysis.data,
                    int(elem_id),
                    response_record.get('random_sample') or {},
                    variable_key
                )
                sampled_values.append(
                    np.nan if material_value is None else float(material_value)
                )
            return np.asarray(sampled_values, dtype=float)

        demand_moment_array = np.asarray(demand_moment_values, dtype=float)
        demand_axial_array = np.asarray(demand_axial_values, dtype=float)
        boundary_moment_array = np.asarray(boundary_moment_values, dtype=float)
        boundary_axial_array = np.asarray(boundary_axial_values, dtype=float)
        g_array = np.asarray(g_values, dtype=float)
        lambda_array = np.asarray(lambda_values, dtype=float)
        radius_array = np.asarray(
            [
                np.nan if coerce_finite_float(item.get('u_space_radius')) is None else
                float(coerce_finite_float(item.get('u_space_radius')))
                for item in response_records
            ],
            dtype=float
        )
        fc_array = build_material_sample_array('fc')
        fy_tarik_array = build_material_sample_array('fy_tarik')
        fy_tekan_array = build_material_sample_array('fy_tekan')
        failure_mask = np.asarray(g_array < 0.0, dtype=bool)
        sample_indices = np.arange(g_array.size, dtype=int)
        failure_indices = sample_indices[failure_mask]

        if int(max_points or 0) <= 0:
            selected_failed = failure_indices.astype(int)
            selected_indices = sample_indices.astype(int)
        else:
            selected_failed = downsample_index_array(
                failure_indices,
                min(int(max_failed_points), int(max_points))
            )
            remaining_capacity = max(int(max_points) - int(selected_failed.size), 0)
            selected_safe = downsample_index_array(
                sample_indices[~failure_mask],
                remaining_capacity
            )
            selected_indices = np.sort(
                np.concatenate([selected_failed, selected_safe]).astype(int)
            )
        if selected_indices.size == 0:
            continue

        mean_curve_moment = []
        mean_curve_axial = []
        try:
            section_inputs = get_section_capacity_inputs_from_input(analysis.data, int(elem_id))
            mean_material = get_element_material_snapshot(
                analysis.data,
                latest_simulation=None,
                is_probabilistic=True,
                elem_id=int(elem_id)
            )
            mean_curve = PerformanceFunction._get_column_interaction_curve(
                mean_material['fc'],
                mean_material['fy_tarik'],
                section_inputs['section_geometry'],
                section_inputs['steel_area'],
                fy_tekan=mean_material['fy_tekan'],
                use_code_phi=False
            )
            mean_curve_moment = [
                float(point['phi_Mn'])
                for point in (mean_curve or [])
                if coerce_finite_float(point.get('phi_Mn')) is not None
                and coerce_finite_float(point.get('phi_Pn')) is not None
            ]
            mean_curve_axial = [
                float(point['phi_Pn'])
                for point in (mean_curve or [])
                if coerce_finite_float(point.get('phi_Mn')) is not None
                and coerce_finite_float(point.get('phi_Pn')) is not None
            ]
        except Exception:
            mean_curve_moment = []
            mean_curve_axial = []

        elements[str(int(elem_id))] = {
            'element_id': int(elem_id),
            'sample_count': int(g_array.size),
            'failure_count': int(np.sum(failure_mask)),
            'safe_count': int(np.sum(~failure_mask)),
            'Pf_from_g': float(np.mean(failure_mask)),
            'used_downsampling': bool(selected_indices.size < g_array.size),
            'failed_points_truncated': bool(
                selected_failed.size < failure_indices.size
            ),
            'demand_moment': demand_moment_array[selected_indices].astype(float).tolist(),
            'demand_axial': demand_axial_array[selected_indices].astype(float).tolist(),
            'boundary_moment': boundary_moment_array[selected_indices].astype(float).tolist(),
            'boundary_axial': boundary_axial_array[selected_indices].astype(float).tolist(),
            'g': g_array[selected_indices].astype(float).tolist(),
            'lambda': lambda_array[selected_indices].astype(float).tolist(),
            'u_space_radius': radius_array[selected_indices].astype(float).tolist(),
            'failure_mask': failure_mask[selected_indices].astype(bool).tolist(),
            'material_samples': {
                'fc': fc_array[selected_indices].astype(float).tolist(),
                'fy_tarik': fy_tarik_array[selected_indices].astype(float).tolist(),
                'fy_tekan': fy_tekan_array[selected_indices].astype(float).tolist()
            },
            'mean_curve_moment': list(mean_curve_moment),
            'mean_curve_axial': list(mean_curve_axial)
        }

    if not elements:
        return {}

    return {
        'num_simulations': int(
            analysis.mc_results.get('num_simulations', len(analysis_history))
            or len(analysis_history)
        ),
        'analysis_failures': int(analysis.mc_results.get('analysis_failures', 0) or 0),
        'element_ids': sorted(int(elem_id) for elem_id in elements.keys()),
        'elements': elements
    }


def get_probabilistic_limit_state_physical_cloud_record(
    physical_cloud_data: Dict[str, Any],
    elem_id: int,
    limit_state: str
) -> Dict[str, Any]:
    """Ambil record failure cloud ruang fisik per elemen-limit-state."""
    element_record = (
        (physical_cloud_data or {}).get('elements', {}).get(str(int(elem_id)))
        or {}
    )
    return (
        (element_record.get('states', {}) or {}).get(str(limit_state))
        or {}
    )


def get_physical_limit_state_material_sample_value(input_data: Dict,
                                                   elem_id: int,
                                                   random_sample: Optional[Dict[str, Any]],
                                                   variable_key: str) -> Optional[float]:
    """Ambil nilai sampel material Monte Carlo untuk satu elemen."""
    elem_id = int(elem_id)
    random_sample = random_sample or {}
    variable_name = str(variable_key).strip()

    concrete_props = get_by_element_value(
        input_data.get('concrete', {}).get('by_element', {}),
        elem_id,
        {}
    ) or {}
    steel_props = get_by_element_value(
        input_data.get('steel', {}).get('by_element', {}),
        elem_id,
        {}
    ) or {}

    if variable_name == 'fc':
        raw_value = random_sample.get(f'fc_E{int(elem_id)}')
        if raw_value is None:
            raw_value = concrete_props.get('mean')
    elif variable_name == 'fy_tarik':
        raw_value = random_sample.get(f'fy_tarik_E{int(elem_id)}')
        if raw_value is None:
            raw_value = steel_props.get('tarik_mean')
    elif variable_name == 'fy_tekan':
        raw_value = random_sample.get(f'fy_tekan_E{int(elem_id)}')
        if raw_value is None:
            raw_value = steel_props.get('tekan_mean')
    elif variable_name == 'fy_geser':
        raw_value = random_sample.get(f'fy_geser_E{int(elem_id)}')
        if raw_value is None:
            raw_value = steel_props.get('geser_mean')
        if raw_value is None:
            raw_value = steel_props.get('tarik_mean')
    else:
        return None

    numeric_value = coerce_finite_float(raw_value)
    if numeric_value is None or numeric_value <= 0.0:
        return None
    return float(numeric_value)


def get_limit_state_physical_cloud_axis_values(record: Dict[str, Any],
                                               axis_key: str) -> np.ndarray:
    """Ambil array titik Monte Carlo untuk satu sumbu peta limit-state fisik."""
    normalized_axis_key = str(axis_key).strip()
    if normalized_axis_key == 'Q':
        return np.asarray((record or {}).get('Q', []), dtype=float)
    if normalized_axis_key == 'R':
        return np.asarray((record or {}).get('R', []), dtype=float)

    material_samples = (record or {}).get('material_samples', {}) or {}
    return np.asarray(material_samples.get(normalized_axis_key, []), dtype=float)


def get_axial_moment_custom_axis_values(record: Dict[str, Any],
                                        axis_key: str) -> np.ndarray:
    """Ambil array titik Monte Carlo untuk sumbu custom aksial-lentur."""
    normalized_axis_key = str(axis_key).strip()
    if normalized_axis_key == 'Pd':
        return np.asarray((record or {}).get('demand_axial', []), dtype=float)
    if normalized_axis_key == 'Md':
        return np.asarray((record or {}).get('demand_moment', []), dtype=float)

    material_samples = (record or {}).get('material_samples', {}) or {}
    return np.asarray(material_samples.get(normalized_axis_key, []), dtype=float)


def build_limit_state_physical_cloud_scatter_data(record: Dict[str, Any],
                                                  x_axis_key: str,
                                                  y_axis_key: str) -> Optional[Dict[str, Any]]:
    """Siapkan data titik safe/fail Monte Carlo untuk overlay pada peta fisik."""
    x_values = get_limit_state_physical_cloud_axis_values(record, x_axis_key)
    y_values = get_limit_state_physical_cloud_axis_values(record, y_axis_key)
    failure_mask = np.asarray((record or {}).get('failure_mask', []), dtype=bool)

    common_size = min(
        int(x_values.size),
        int(y_values.size),
        int(failure_mask.size)
    )
    if common_size <= 0:
        return None

    x_values = x_values[:common_size]
    y_values = y_values[:common_size]
    failure_mask = failure_mask[:common_size]
    valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(valid_mask):
        return None

    x_valid = x_values[valid_mask]
    y_valid = y_values[valid_mask]
    failure_valid = failure_mask[valid_mask]
    safe_valid = ~failure_valid
    return {
        'x': x_valid,
        'y': y_valid,
        'failure_mask': failure_valid,
        'safe_mask': safe_valid,
        'failure_count': int(np.sum(failure_valid)),
        'safe_count': int(np.sum(safe_valid))
    }


def build_axial_moment_custom_scatter_data(record: Dict[str, Any],
                                           x_axis_key: str,
                                           y_axis_key: str) -> Optional[Dict[str, Any]]:
    """Siapkan titik safe/fail Monte Carlo untuk custom map aksial-lentur."""
    x_values = get_axial_moment_custom_axis_values(record, x_axis_key)
    y_values = get_axial_moment_custom_axis_values(record, y_axis_key)
    failure_mask = np.asarray((record or {}).get('failure_mask', []), dtype=bool)

    common_size = min(
        int(x_values.size),
        int(y_values.size),
        int(failure_mask.size)
    )
    if common_size <= 0:
        return None

    x_values = x_values[:common_size]
    y_values = y_values[:common_size]
    failure_mask = failure_mask[:common_size]
    valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(valid_mask):
        return None

    x_valid = x_values[valid_mask]
    y_valid = y_values[valid_mask]
    failure_valid = failure_mask[valid_mask]
    safe_valid = ~failure_valid
    return {
        'x': x_valid,
        'y': y_valid,
        'failure_mask': failure_valid,
        'safe_mask': safe_valid,
        'failure_count': int(np.sum(failure_valid)),
        'safe_count': int(np.sum(safe_valid))
    }


def build_annotation_avoid_points(scatter_data: Optional[Dict[str, Any]] = None,
                                  x_values: Optional[np.ndarray] = None,
                                  y_values: Optional[np.ndarray] = None,
                                  extra_points: Optional[List[Tuple[float, float]]] = None,
                                  max_points: int = 240) -> List[Tuple[float, float]]:
    """Ringkas titik yang perlu dihindari oleh label anotasi."""
    if scatter_data:
        x_array = np.asarray((scatter_data or {}).get('x', []), dtype=float).reshape(-1)
        y_array = np.asarray((scatter_data or {}).get('y', []), dtype=float).reshape(-1)
    else:
        x_array = np.asarray(x_values if x_values is not None else [], dtype=float).reshape(-1)
        y_array = np.asarray(y_values if y_values is not None else [], dtype=float).reshape(-1)

    common_size = min(int(x_array.size), int(y_array.size))
    avoid_points: List[Tuple[float, float]] = []
    if common_size > 0:
        x_array = x_array[:common_size]
        y_array = y_array[:common_size]
        valid_mask = np.isfinite(x_array) & np.isfinite(y_array)
        x_array = x_array[valid_mask]
        y_array = y_array[valid_mask]
        if x_array.size > int(max_points):
            selected_indices = downsample_index_array(
                np.arange(x_array.size, dtype=int),
                int(max_points)
            )
            x_array = x_array[selected_indices]
            y_array = y_array[selected_indices]
        avoid_points.extend(
            (float(point_x), float(point_y))
            for point_x, point_y in zip(x_array, y_array)
        )

    for extra_point in list(extra_points or []):
        if not isinstance(extra_point, (tuple, list)) or len(extra_point) != 2:
            continue
        point_x = coerce_finite_float(extra_point[0])
        point_y = coerce_finite_float(extra_point[1])
        if point_x is None or point_y is None:
            continue
        avoid_points.append((float(point_x), float(point_y)))
    return avoid_points


def plot_limit_state_physical_cloud_scatter(axis,
                                            scatter_data: Optional[Dict[str, Any]],
                                            style_variant: str = 'default') -> None:
    """Overlay titik SMC safe/fail pada peta limit-state fisik."""
    if not scatter_data:
        return

    x_values = np.asarray(scatter_data.get('x', []), dtype=float)
    y_values = np.asarray(scatter_data.get('y', []), dtype=float)
    safe_mask = np.asarray(scatter_data.get('safe_mask', []), dtype=bool)
    failure_mask = np.asarray(scatter_data.get('failure_mask', []), dtype=bool)

    if x_values.size == 0 or y_values.size == 0:
        return

    variant_name = str(style_variant or 'default').strip().lower()
    use_physical_cloud_style = variant_name == 'physical_cloud'

    if np.any(safe_mask):
        axis.scatter(
            x_values[safe_mask],
            y_values[safe_mask],
            s=18 if use_physical_cloud_style else 24,
            color=SAFE_CLOUD_COLOR,
            alpha=0.34 if use_physical_cloud_style else 0.68,
            edgecolors='none' if use_physical_cloud_style else '#ffffff',
            linewidths=0.0 if use_physical_cloud_style else 0.28,
            zorder=4,
            label=(
                f"Safe ({int(np.sum(safe_mask)):,})"
                if use_physical_cloud_style else
                f"SMC safe ({int(np.sum(safe_mask)):,})"
            )
        )
    if np.any(failure_mask):
        axis.scatter(
            x_values[failure_mask],
            y_values[failure_mask],
            s=28 if use_physical_cloud_style else 30,
            color='#dc2626' if use_physical_cloud_style else '#ff0000',
            alpha=0.82 if use_physical_cloud_style else 0.88,
            edgecolors='#ffffff',
            linewidths=0.25 if use_physical_cloud_style else 0.30,
            zorder=5,
            label=(
                f"Failed ({int(np.sum(failure_mask)):,})"
                if use_physical_cloud_style else
                f"SMC fail ({int(np.sum(failure_mask)):,})"
            )
        )


def pin_contour_axes_to_grid(axis,
                             grid_x: np.ndarray,
                             grid_y: np.ndarray) -> None:
    """Paksa batas sumbu mengikuti area grid contour agar warna menempel ke bingkai."""
    x_values = np.asarray(grid_x, dtype=float)
    y_values = np.asarray(grid_y, dtype=float)
    finite_x = x_values[np.isfinite(x_values)]
    finite_y = y_values[np.isfinite(y_values)]
    if finite_x.size == 0 or finite_y.size == 0:
        return

    axis.set_xlim(float(np.min(finite_x)), float(np.max(finite_x)))
    axis.set_ylim(float(np.min(finite_y)), float(np.max(finite_y)))
    axis.margins(x=0.0, y=0.0)


def build_probabilistic_limit_state_physical_cloud_figure(
    physical_cloud_data: Dict[str, Any],
    element_reliability: Optional[Dict[str, Dict[int, Dict[str, Any]]]],
    elem_id: int
) -> Optional[plt.Figure]:
    """Bangun panel failure cloud di ruang fisik untuk empat limit state."""
    state_specs = get_probabilistic_limit_state_histogram_specs()
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.9), dpi=180)
    axes_list = list(np.asarray(axes).reshape(-1))
    plotted_any = False
    element_reliability = element_reliability or {}

    def build_nice_g_step(max_abs_value: float) -> float:
        """Pilih langkah kontur `g` yang cukup bersih untuk dibaca."""
        try:
            numeric_value = abs(float(max_abs_value))
        except (TypeError, ValueError):
            numeric_value = 0.0
        if not np.isfinite(numeric_value) or numeric_value <= 1e-12:
            return 1.0

        raw_step = numeric_value / 3.0
        exponent = np.floor(np.log10(raw_step))
        scale = 10.0 ** exponent
        normalized = raw_step / scale
        if normalized <= 1.0:
            factor = 1.0
        elif normalized <= 2.0:
            factor = 2.0
        elif normalized <= 5.0:
            factor = 5.0
        else:
            factor = 10.0
        return float(factor * scale)

    def add_qr_g_contours(axis,
                          x_limits: Tuple[float, float],
                          y_limits: Tuple[float, float],
                          g_samples: np.ndarray):
        """Tambahkan latar kontur `g = R - Q` pada bidang `Q-R`."""
        x_min, x_max = x_limits
        y_min, y_max = y_limits
        finite_g = np.asarray(g_samples, dtype=float)
        finite_g = finite_g[np.isfinite(finite_g)]

        max_abs_from_axes = max(
            abs(float(y_min - x_max)),
            abs(float(y_max - x_min)),
            1e-9
        )
        max_abs_from_samples = float(np.max(np.abs(finite_g))) if finite_g.size else 0.0
        max_abs_g = max(max_abs_from_axes, max_abs_from_samples, 1e-6)

        grid_x = np.linspace(x_min, x_max, PHYSICAL_G_CONTOUR_GRID_SIZE)
        grid_y = np.linspace(y_min, y_max, PHYSICAL_G_CONTOUR_GRID_SIZE)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        mesh_g = mesh_y - mesh_x

        fill_levels = np.linspace(-max_abs_g, max_abs_g, 19)
        contour_fill = axis.contourf(
            mesh_x,
            mesh_y,
            mesh_g,
            levels=fill_levels,
            cmap='RdYlBu',
            alpha=0.10,
            antialiased=True,
            zorder=0
        )

        contour_step = build_nice_g_step(max_abs_g)
        line_levels = contour_step * np.arange(-3, 4, dtype=float)
        nonzero_levels = np.asarray(
            [
                value for value in line_levels
                if (
                    not np.isclose(value, 0.0, atol=max(contour_step * 1e-6, 1e-12))
                    and abs(float(value)) <= max_abs_g * 1.02
                )
            ],
            dtype=float
        )
        if nonzero_levels.size > 0:
            contours = axis.contour(
                mesh_x,
                mesh_y,
                mesh_g,
                levels=np.sort(nonzero_levels),
                colors='#64748b',
                linewidths=0.65,
                alpha=0.62,
                linestyles='dashed',
                zorder=1
            )
            axis.clabel(
                contours,
                inline=True,
                fontsize=6.7,
                fmt=lambda value: f"g={format_metric(value, 1)}"
            )

        axis.text(
            0.02,
            0.98,
            "g = R - Q",
            transform=axis.transAxes,
            ha='left',
            va='top',
            fontsize=8.0,
            color='#334155',
            bbox=dict(
                boxstyle='round,pad=0.18',
                facecolor='white',
                alpha=0.82,
                edgecolor='#cbd5e1'
            )
        )
        return contour_fill

    def add_lambda_g_contours(axis,
                              x_limits: Tuple[float, float],
                              y_limits: Tuple[float, float],
                              g_samples: np.ndarray):
        """Tambahkan latar kontur `g` pada bidang `lambda-g`."""
        x_min, x_max = x_limits
        y_min, y_max = y_limits
        finite_g = np.asarray(g_samples, dtype=float)
        finite_g = finite_g[np.isfinite(finite_g)]

        max_abs_from_axes = max(abs(float(y_min)), abs(float(y_max)), 1e-9)
        max_abs_from_samples = float(np.max(np.abs(finite_g))) if finite_g.size else 0.0
        max_abs_g = max(max_abs_from_axes, max_abs_from_samples, 1e-6)

        grid_x = np.linspace(x_min, x_max, PHYSICAL_G_CONTOUR_GRID_SIZE)
        grid_y = np.linspace(y_min, y_max, PHYSICAL_G_CONTOUR_GRID_SIZE)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        mesh_g = mesh_y

        fill_levels = np.linspace(-max_abs_g, max_abs_g, 19)
        contour_fill = axis.contourf(
            mesh_x,
            mesh_y,
            mesh_g,
            levels=fill_levels,
            cmap='RdYlBu',
            alpha=0.10,
            antialiased=True,
            zorder=0
        )

        contour_step = build_nice_g_step(max_abs_g)
        line_levels = contour_step * np.arange(-3, 4, dtype=float)
        nonzero_levels = np.asarray(
            [
                value for value in line_levels
                if (
                    not np.isclose(value, 0.0, atol=max(contour_step * 1e-6, 1e-12))
                    and abs(float(value)) <= max_abs_g * 1.02
                )
            ],
            dtype=float
        )
        if nonzero_levels.size > 0:
            contours = axis.contour(
                mesh_x,
                mesh_y,
                mesh_g,
                levels=np.sort(nonzero_levels),
                colors='#64748b',
                linewidths=0.65,
                alpha=0.62,
                linestyles='dashed',
                zorder=1
            )
            axis.clabel(
                contours,
                inline=True,
                fontsize=6.7,
                fmt=lambda value: f"g={format_metric(value, 1)}"
            )

        axis.text(
            0.02,
            0.98,
            "g shown on Y-axis",
            transform=axis.transAxes,
            ha='left',
            va='top',
            fontsize=8.0,
            color='#334155',
            bbox=dict(
                boxstyle='round,pad=0.18',
                facecolor='white',
                alpha=0.82,
                edgecolor='#cbd5e1'
            )
        )
        return contour_fill

    def add_g_colorbar(axis, contour_fill, unit_label: str) -> None:
        """Tambahkan colorbar `g(x)` untuk subplot aktif."""
        if contour_fill is None:
            return
        colorbar = fig.colorbar(
            contour_fill,
            ax=axis,
            fraction=0.046,
            pad=0.04
        )
        colorbar.ax.tick_params(labelsize=7)
        colorbar.set_label(f"g(x) ({unit_label})", fontsize=8)

    for axis, spec in zip(axes_list, state_specs):
        record = get_probabilistic_limit_state_physical_cloud_record(
            physical_cloud_data,
            int(elem_id),
            str(spec['key'])
        )
        plot_label = str(spec.get('plot_label', spec['label']))
        if not record:
            axis.axis('off')
            axis.text(
                0.5,
                0.5,
                (
                    f"Data for {plot_label} in E{int(elem_id)}\n"
                    "is unavailable or not applicable."
                ),
                ha='center',
                va='center',
                fontsize=10,
                color='#475569',
                bbox=dict(
                    boxstyle='round,pad=0.25',
                    facecolor='#f8fafc',
                    edgecolor='#cbd5e1'
                )
            )
            axis.set_title(f"{plot_label} | Physical Failure Cloud", fontsize=10.5, pad=10)
            continue

        r_values = np.asarray(record.get('R', []), dtype=float)
        q_values = np.asarray(record.get('Q', []), dtype=float)
        g_values = np.asarray(record.get('g', []), dtype=float)
        failure_mask = np.asarray(record.get('failure_mask', []), dtype=bool)
        common_size = min(
            int(r_values.size),
            int(q_values.size),
            int(g_values.size),
            int(failure_mask.size)
        )
        if common_size <= 0:
            axis.axis('off')
            axis.set_title(f"{plot_label} | Physical Failure Cloud", fontsize=10.5, pad=10)
            continue

        r_values = r_values[:common_size]
        q_values = q_values[:common_size]
        g_values = g_values[:common_size]
        failure_mask = failure_mask[:common_size]
        safe_mask = ~failure_mask
        axis.set_facecolor('#f8fafc')

        reliability_record = get_by_element_value(
            element_reliability.get(str(spec['key']), {}),
            int(elem_id),
            {}
        ) or {}
        pf_value = coerce_finite_float(reliability_record.get('Pf'))
        beta_raw_value = reliability_record.get('Beta')

        projected_mpp = resolve_limit_state_projected_mpp_from_record(
            record,
            limit_state=str(spec['key']),
            target_beta=beta_raw_value
        )
        projected_mpp_beta = coerce_finite_float(projected_mpp.get('display_beta'))
        projected_mpp_sample_beta = coerce_finite_float(projected_mpp.get('beta'))
        projected_mpp_beta_label = format_beta_table_display(
            projected_mpp.get('display_beta_raw', projected_mpp.get('display_beta')),
            4
        )
        projected_mpp_x = coerce_finite_float(projected_mpp.get('contour_x'))
        projected_mpp_y = coerce_finite_float(projected_mpp.get('contour_y'))
        projected_sample_x = coerce_finite_float(projected_mpp.get('sample_x'))
        projected_sample_y = coerce_finite_float(projected_mpp.get('sample_y'))
        projected_sample_g = coerce_finite_float(projected_mpp.get('candidate_g'))

        if str(spec['key']) == 'axial_moment':
            x_values = r_values
            y_values = g_values
            valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
            x_values = x_values[valid_mask]
            y_values = y_values[valid_mask]
            failure_values = failure_mask[valid_mask]
            safe_values = ~failure_values
            valid_sample_count = int(x_values.size)
            safe_count_valid = int(np.sum(safe_values))
            failure_count_valid = int(np.sum(failure_values))

            if np.any(safe_values):
                axis.scatter(
                    x_values[safe_values],
                    y_values[safe_values],
                    s=18,
                    color=SAFE_CLOUD_COLOR,
                    alpha=0.34,
                    edgecolors='none',
                    label=f"Safe ({safe_count_valid:,})"
                )
            if np.any(failure_values):
                axis.scatter(
                    x_values[failure_values],
                    y_values[failure_values],
                    s=28,
                    color='#dc2626',
                    alpha=0.80,
                    edgecolors='#ffffff',
                    linewidths=0.25,
                    label=f"Failed ({failure_count_valid:,})"
                )

            x_limits = get_failure_cloud_axis_limits(
                extend_numeric_array_with_optional_values(
                    x_values,
                    projected_sample_x,
                    projected_mpp_x
                )
            )
            y_limits = get_failure_cloud_axis_limits(
                extend_numeric_array_with_optional_values(
                    y_values,
                    projected_sample_y,
                    projected_mpp_y
                )
            )
            line_x = np.linspace(x_limits[0], x_limits[1], 160)
            axis.plot(
                line_x,
                line_x - 1.0,
                color='#111827',
                linestyle='-.',
                linewidth=1.2,
                alpha=0.9,
                label='Response line, g(x) = lambda - 1'
            )
            axis.axvline(
                1.0,
                color=PHYSICAL_NONLINEAR_CONTOUR_COLOR,
                linestyle='--',
                linewidth=1.35,
                alpha=0.9,
                label='Limit-state contour, lambda = 1'
            )
            axis.axhline(
                0.0,
                color=PHYSICAL_NONLINEAR_CONTOUR_COLOR,
                linestyle=':',
                linewidth=1.45,
                alpha=0.95,
                label='Limit-state contour, g = 0'
            )
            x_mean = float(np.mean(x_values)) if x_values.size else None
            y_mean = float(np.mean(y_values)) if y_values.size else None
            x_label = 'Capacity Ratio, lambda (-)'
            y_label = 'g(x) = lambda - 1 (-)'
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)
            contour_fill = add_lambda_g_contours(
                axis,
                x_limits=x_limits,
                y_limits=y_limits,
                g_samples=y_values
            )
            add_g_colorbar(
                axis,
                contour_fill,
                unit_label='(-)'
            )
        else:
            x_values = q_values
            y_values = r_values
            valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
            x_values = x_values[valid_mask]
            y_values = y_values[valid_mask]
            failure_values = failure_mask[valid_mask]
            safe_values = ~failure_values
            valid_sample_count = int(x_values.size)
            safe_count_valid = int(np.sum(safe_values))
            failure_count_valid = int(np.sum(failure_values))

            if np.any(safe_values):
                axis.scatter(
                    x_values[safe_values],
                    y_values[safe_values],
                    s=18,
                    color=SAFE_CLOUD_COLOR,
                    alpha=0.34,
                    edgecolors='none',
                    label=f"Safe ({safe_count_valid:,})"
                )
            if np.any(failure_values):
                axis.scatter(
                    x_values[failure_values],
                    y_values[failure_values],
                    s=28,
                    color='#dc2626',
                    alpha=0.82,
                    edgecolors='#ffffff',
                    linewidths=0.25,
                    label=f"Failed ({failure_count_valid:,})"
                )

            axis_values = np.concatenate([x_values, y_values]) if x_values.size else np.asarray([], dtype=float)
            axis_values = extend_numeric_array_with_optional_values(
                axis_values,
                projected_sample_x,
                projected_sample_y,
                projected_mpp_x,
                projected_mpp_y
            )
            axis_limits = get_failure_cloud_axis_limits(axis_values)
            boundary_values = np.linspace(axis_limits[0], axis_limits[1], 160)
            axis.plot(
                boundary_values,
                boundary_values,
                color=PHYSICAL_NONLINEAR_CONTOUR_COLOR,
                linestyle='--',
                linewidth=1.35,
                alpha=0.92,
                label='Limit-state contour, g = 0 (R = Q)'
            )
            x_mean = float(np.mean(x_values)) if x_values.size else None
            y_mean = float(np.mean(y_values)) if y_values.size else None
            x_label = f"Demand Q ({record.get('unit', spec['unit'])})"
            y_label = f"Capacity R ({record.get('unit', spec['unit'])})"
            axis.set_xlim(*axis_limits)
            axis.set_ylim(*axis_limits)
            contour_fill = add_qr_g_contours(
                axis,
                x_limits=axis_limits,
                y_limits=axis_limits,
                g_samples=g_values[valid_mask]
            )
            add_g_colorbar(
                axis,
                contour_fill,
                unit_label=str(record.get('unit', spec['unit']))
            )

        if x_mean is not None and y_mean is not None:
            axis.scatter(
                [x_mean],
                [y_mean],
                marker='x',
                s=70,
                color='#0f172a',
                linewidths=1.5,
                label='Sample Mean'
            )

        if (
            x_mean is not None
            and y_mean is not None
            and projected_mpp_beta is not None
            and projected_mpp_x is not None
            and projected_mpp_y is not None
        ):
            axis.plot(
                [float(x_mean), float(projected_mpp_x)],
                [float(y_mean), float(projected_mpp_y)],
                linestyle='--',
                linewidth=1.35,
                color='#7e22ce',
                alpha=0.94,
                zorder=5,
                label=f"Projected beta line, Beta(table)={projected_mpp_beta_label}"
            )
            if (
                projected_sample_x is not None
                and projected_sample_y is not None
                and not (
                    np.isclose(projected_sample_x, projected_mpp_x, atol=1e-9, rtol=1e-9)
                    and np.isclose(projected_sample_y, projected_mpp_y, atol=1e-9, rtol=1e-9)
                )
            ):
                axis.plot(
                    [float(projected_sample_x), float(projected_mpp_x)],
                    [float(projected_sample_y), float(projected_mpp_y)],
                    linestyle=':',
                    linewidth=1.0,
                    color='#d946ef',
                    alpha=0.88,
                    zorder=5
                )
            axis.scatter(
                [float(projected_mpp_x)],
                [float(projected_mpp_y)],
                marker='*',
                s=175,
                color='#f59e0b',
                edgecolors='#111827',
                linewidths=0.80,
                zorder=7,
                label=f"Projected MPP, Beta(table)={projected_mpp_beta_label}"
            )
            annotation_lines = [
                "MPP",
                f"Beta(table)={projected_mpp_beta_label}"
            ]
            if str(spec['key']) == 'axial_moment':
                annotation_lines.extend([
                    f"lambda={format_metric(projected_mpp_x, 2)}",
                    (
                        f"g_samp={format_metric(projected_sample_g, 3)}"
                        if projected_sample_g is not None else
                        "g_samp=-"
                    )
                ])
            else:
                annotation_lines.extend([
                    f"Q={format_metric(projected_mpp_x, 2)}",
                    f"R={format_metric(projected_mpp_y, 2)}"
                ])
            annotation_text = "\n".join(annotation_lines)
            annotation_avoid_points = build_annotation_avoid_points(
                x_values=x_values,
                y_values=y_values,
                extra_points=[
                    (x_mean, y_mean),
                    (projected_sample_x, projected_sample_y),
                    (projected_mpp_x, projected_mpp_y)
                ],
                max_points=260
            )
            add_smart_mpp_annotation(
                axis,
                annotation_text,
                target_xy=(float(projected_mpp_x), float(projected_mpp_y)),
                avoid_points=annotation_avoid_points,
                bbox_edgecolor='#cbd5e1',
                text_color='#111827',
                fontsize=8.1,
                zorder=7,
                with_arrow=True
            )
        else:
            axis.text(
                0.02,
                0.03,
                (
                    "MPP estimate unavailable\n"
                    "Cloud valid, but no projected MPP\n"
                    "matching Beta(table) could be derived."
                ),
                transform=axis.transAxes,
                ha='left',
                va='bottom',
                fontsize=7.2,
                color='#7c2d12',
                bbox=dict(
                    boxstyle='round,pad=0.18',
                    facecolor='#fff7ed',
                    edgecolor='#fdba74',
                    alpha=0.94
                ),
                zorder=6
            )

        beta_text = format_beta_table_display(beta_raw_value, 4)
        projected_mpp_beta_text = (
            projected_mpp_beta_label
            if projected_mpp_beta is not None else
            'Unavailable'
        )
        pf_text = format_metric(pf_value, 6) if pf_value is not None else '-'

        axis.set_title(f"{plot_label} | Physical Failure Cloud", fontsize=10.5, pad=10)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.22, linestyle='--')
        axis.legend(loc='best', fontsize=7.8)
        axis.text(
            0.98,
            0.96,
            (
                f"Valid N = {valid_sample_count}\n"
                f"Safe = {safe_count_valid}\n"
                f"Failures = {failure_count_valid}\n"
                f"Pf(valid) = {float(record.get('Pf_from_g', 0.0)):.4f}\n"
                f"MPP Beta(table) = {projected_mpp_beta_text}\n"
                f"Beta(table) = {beta_text}\n"
                f"Pf(table) = {pf_text}"
            ),
            transform=axis.transAxes,
            ha='right',
            va='top',
            fontsize=8.2,
            bbox=dict(
                boxstyle='round,pad=0.25',
                facecolor='white',
                alpha=0.90,
                edgecolor='#cbd5e1'
            )
        )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return None

    fig.suptitle(
        f"Failure Cloud di Ruang Fisik | Element E{int(elem_id)}",
        fontsize=13,
        y=0.99
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return fig


def build_probabilistic_limit_state_physical_cloud_summary_df(
    physical_cloud_data: Dict[str, Any],
    element_reliability: Optional[Dict[str, Dict[int, Dict[str, Any]]]],
    elem_id: int
) -> pd.DataFrame:
    """Ringkas statistik failure cloud fisik dan kaitannya dengan Beta tabel."""
    rows = []
    element_reliability = element_reliability or {}

    for spec in get_probabilistic_limit_state_histogram_specs():
        record = get_probabilistic_limit_state_physical_cloud_record(
            physical_cloud_data,
            int(elem_id),
            str(spec['key'])
        )
        if not record:
            continue

        r_values = np.asarray(record.get('R', []), dtype=float)
        q_values = np.asarray(record.get('Q', []), dtype=float)
        g_values = np.asarray(record.get('g', []), dtype=float)
        reliability_record = get_by_element_value(
            element_reliability.get(str(spec['key']), {}),
            int(elem_id),
            {}
        ) or {}

        rows.append({
            'Limit State': record.get('limit_state_label', spec['label']),
            'Bidang Fisik': (
                'lambda vs g(x)'
                if str(spec['key']) == 'axial_moment' else
                'Q vs R'
            ),
            'Satuan': record.get('unit', spec['unit']),
            'Jumlah Sampel Valid (-)': record.get('sample_count'),
            'Jumlah Gagal Valid (-)': record.get('failure_count'),
            'Pf dari Cloud Valid (-)': record.get('Pf_from_g'),
            'Pf Tabel Reliability (-)': reliability_record.get('Pf'),
            'Beta Tabel Reliability (-)': reliability_record.get('Beta'),
            'Mean R': float(np.mean(r_values)) if r_values.size else None,
            'Mean Q': float(np.mean(q_values)) if q_values.size else None,
            'Mean g(x)': float(np.mean(g_values)) if g_values.size else None
        })

    return pd.DataFrame(rows)


def render_probabilistic_limit_state_physical_failure_cloud_section(
    physical_cloud_data: Dict[str, Any],
    axial_moment_pm_cloud_data: Dict[str, Any],
    results_bundle: Dict[str, Any],
    input_data: Optional[Dict],
    heading_level: str = "####"
) -> None:
    """Tampilkan failure cloud Monte Carlo di ruang fisik per limit-state."""
    available_element_ids = list((physical_cloud_data or {}).get('element_ids', []) or [])
    if not available_element_ids:
        st.info(
            "Failure cloud ruang fisik belum tersedia. Jalankan analisis probabilistik "
            "agar respons `R/Q/g(x)` per limit state dapat diproyeksikan."
        )
        return

    st.markdown(f"{heading_level} Failure Cloud di Ruang Fisik")
    st.caption(
        "Bagian ini menggambarkan sebaran sampel Monte Carlo langsung di ruang fisik respons."
    )
    st.caption(
        "Untuk lentur, geser, dan aksial, panel memakai pasangan `Q-R` dengan batas gagal "
        "`R = Q`, sehingga `g(x) = R - Q` terbaca langsung dari posisi titik terhadap diagonal. "
        "Kontur `iso-g` ringan dan `colorbar g(x)` juga ditampilkan pada bidang ini."
    )
    st.caption(
        "Untuk aksial-lentur, panel memakai `lambda` vs `g(x)` karena pada tabel reliability "
        "cek ini dinormalisasi sebagai `g(x) = lambda - 1` dengan `Q = 1`, sehingga `g` "
        "sudah tampil langsung pada sumbu vertikal dan diperkuat dengan `colorbar g(x)`."
    )
    st.caption(
        "Setiap subplot kini juga menampilkan `limit-state contour g=0`, `projected MPP`, "
        "serta `projected beta line`. Nilai `beta` yang tertulis tetap dibaca sebagai jarak di "
        "ruang normal baku, sedangkan marker/garis di bidang fisik adalah proyeksi bantu agar "
        "lokasi titik kritis lebih mudah dibaca."
    )

    selected_element_id = st.selectbox(
        "Pilih elemen untuk failure cloud ruang fisik",
        options=available_element_ids,
        format_func=lambda elem_id: (
            f"E{int(elem_id)} | "
            f"{get_element_type_label(get_element_code_from_input(input_data, int(elem_id)))}"
        ),
        key="physical_failure_cloud_element_selector"
    )

    element_record = (
        (physical_cloud_data or {}).get('elements', {}).get(str(int(selected_element_id)))
        or {}
    )
    state_records = list((element_record.get('states', {}) or {}).values())
    downsampled_labels = [
        str(record.get('limit_state_label', '-'))
        for record in state_records
        if bool(record.get('used_downsampling'))
    ]
    truncated_failed_labels = [
        str(record.get('limit_state_label', '-'))
        for record in state_records
        if bool(record.get('failed_points_truncated'))
    ]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Elemen Terpilih", f"E{int(selected_element_id)}")
    metric_cols[1].metric("Limit State Tersedia", str(len(state_records)))
    metric_cols[2].metric(
        "Jumlah Simulasi",
        format_metric_comma((physical_cloud_data or {}).get('num_simulations'), 0)
    )
    metric_cols[3].metric(
        "Analisis Gagal",
        format_metric_comma((physical_cloud_data or {}).get('analysis_failures'), 0)
    )

    if downsampled_labels:
        st.caption(
            "State berikut memakai subset titik agar dashboard tetap ringan: "
            + ", ".join(f"`{label}`" for label in downsampled_labels)
        )
    if truncated_failed_labels:
        st.caption(
            "Jumlah titik gagal yang dipadatkan secara khusus terjadi pada: "
            + ", ".join(f"`{label}`" for label in truncated_failed_labels)
        )

    physical_cloud_fig = build_probabilistic_limit_state_physical_cloud_figure(
        physical_cloud_data,
        (results_bundle or {}).get('element_reliability', {}),
        int(selected_element_id)
    )
    if physical_cloud_fig is not None:
        render_plot(
            physical_cloud_fig,
            interactive=True,
            viewer_key=f"physical-failure-cloud-e{int(selected_element_id)}",
            alt_text=f"Failure cloud ruang fisik elemen {int(selected_element_id)}",
            viewer_height=900,
            download_basename=f"failure-cloud-ruang-fisik-e{int(selected_element_id)}"
        )
    else:
        st.info("Failure cloud ruang fisik untuk elemen terpilih belum dapat dibentuk.")

    render_physical_limit_state_function_map_section(
        physical_cloud_data=physical_cloud_data,
        axial_moment_pm_cloud_data=axial_moment_pm_cloud_data or {},
        results_bundle=results_bundle,
        input_data=input_data,
        elem_id=int(selected_element_id),
        heading_level=heading_level
    )

    render_probabilistic_axial_moment_pm_cloud_section(
        axial_moment_pm_cloud_data=axial_moment_pm_cloud_data or {},
        element_reliability=(results_bundle or {}).get('element_reliability', {}),
        input_data=input_data,
        latest_simulation=(results_bundle or {}).get('latest_simulation', {}),
        latest_result=((results_bundle or {}).get('latest_simulation', {}) or {}).get('analysis_result'),
        elem_id=int(selected_element_id),
        heading_level=heading_level
    )

    summary_df = build_probabilistic_limit_state_physical_cloud_summary_df(
        physical_cloud_data,
        (results_bundle or {}).get('element_reliability', {}),
        int(selected_element_id)
    )
    st.markdown(f"{heading_level} Ringkasan Failure Cloud Fisik")
    st.caption(
        "Tabel ini merangkum ukuran cloud fisik dan mempertemukannya dengan `Pf/Beta` "
        "yang muncul pada menu `Output Reliability`."
    )
    if summary_df.empty:
        st.info("Ringkasan failure cloud fisik untuk elemen terpilih belum tersedia.")
    else:
        render_input_table(
            summary_df,
            styler=style_input_dataframe(
                summary_df,
                table_min_width_px=1500
            )
        )


def fit_physical_signed_margin_quadratic_surface(
    x_values: np.ndarray,
    y_values: np.ndarray,
    g_values: np.ndarray,
    boundary_x: Optional[np.ndarray] = None,
    boundary_y: Optional[np.ndarray] = None,
    max_training_points: int = FAILURE_SURFACE_MAX_CLASS_POINTS,
    boundary_weight: float = 3.0
) -> Optional[Dict[str, Any]]:
    """Fit surrogate kuadratik `g_hat(x)` pada ruang fisik dua dimensi."""
    x_array = np.asarray(x_values, dtype=float).reshape(-1)
    y_array = np.asarray(y_values, dtype=float).reshape(-1)
    g_array = np.asarray(g_values, dtype=float).reshape(-1)
    common_size = min(int(x_array.size), int(y_array.size), int(g_array.size))
    if common_size < 6:
        return None

    x_array = x_array[:common_size]
    y_array = y_array[:common_size]
    g_array = g_array[:common_size]
    valid_mask = np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(g_array)
    if int(np.sum(valid_mask)) < 6:
        return None

    x_array = x_array[valid_mask]
    y_array = y_array[valid_mask]
    g_array = g_array[valid_mask]
    sample_indices = downsample_index_array(
        np.arange(x_array.size, dtype=int),
        int(max_training_points)
    )
    x_train = x_array[sample_indices]
    y_train = y_array[sample_indices]
    g_train = g_array[sample_indices]

    boundary_train_x = np.asarray([], dtype=float)
    boundary_train_y = np.asarray([], dtype=float)
    if boundary_x is not None and boundary_y is not None:
        boundary_x_array = np.asarray(boundary_x, dtype=float).reshape(-1)
        boundary_y_array = np.asarray(boundary_y, dtype=float).reshape(-1)
        boundary_size = min(int(boundary_x_array.size), int(boundary_y_array.size))
        if boundary_size > 0:
            boundary_x_array = boundary_x_array[:boundary_size]
            boundary_y_array = boundary_y_array[:boundary_size]
            boundary_valid = np.isfinite(boundary_x_array) & np.isfinite(boundary_y_array)
            boundary_x_array = boundary_x_array[boundary_valid]
            boundary_y_array = boundary_y_array[boundary_valid]
            if boundary_x_array.size > 0:
                boundary_indices = downsample_index_array(
                    np.arange(boundary_x_array.size, dtype=int),
                    int(max_training_points)
                )
                boundary_train_x = boundary_x_array[boundary_indices]
                boundary_train_y = boundary_y_array[boundary_indices]

    combined_x = np.concatenate([x_train, boundary_train_x])
    combined_y = np.concatenate([y_train, boundary_train_y])
    if combined_x.size < 6 or combined_y.size < 6:
        return None

    x_center = float(np.mean(combined_x))
    y_center = float(np.mean(combined_y))
    x_scale = max(float(np.std(combined_x)), float(np.ptp(combined_x)) / 2.0, 1e-6)
    y_scale = max(float(np.std(combined_y)), float(np.ptp(combined_y)) / 2.0, 1e-6)

    x_train_scaled = (x_train - x_center) / x_scale
    y_train_scaled = (y_train - y_center) / y_scale
    demand_features = build_failure_surface_polynomial_features(
        x_train_scaled,
        y_train_scaled
    )
    targets = g_train.astype(float)
    weights = np.ones_like(targets, dtype=float)

    if boundary_train_x.size > 0:
        boundary_x_scaled = (boundary_train_x - x_center) / x_scale
        boundary_y_scaled = (boundary_train_y - y_center) / y_scale
        boundary_features = build_failure_surface_polynomial_features(
            boundary_x_scaled,
            boundary_y_scaled
        )
        demand_features = np.vstack([demand_features, boundary_features])
        targets = np.concatenate([
            targets,
            np.zeros(boundary_features.shape[0], dtype=float)
        ])
        weights = np.concatenate([
            weights,
            np.full(boundary_features.shape[0], float(boundary_weight), dtype=float)
        ])

    sqrt_weights = np.sqrt(np.asarray(weights, dtype=float))
    weighted_features = demand_features * sqrt_weights[:, np.newaxis]
    weighted_targets = targets * sqrt_weights
    ridge_lambda = 1e-3

    try:
        lhs = (
            weighted_features.T @ weighted_features
            + ridge_lambda * np.eye(weighted_features.shape[1], dtype=float)
        )
        rhs = weighted_features.T @ weighted_targets
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        try:
            coefficients = np.linalg.lstsq(
                weighted_features,
                weighted_targets,
                rcond=None
            )[0]
        except Exception:
            return None

    fitted_values = np.asarray(
        build_failure_surface_polynomial_features(
            x_train_scaled,
            y_train_scaled
        ) @ coefficients,
        dtype=float
    )
    if not np.any(np.isfinite(fitted_values)):
        return None

    return {
        'coefficients': np.asarray(coefficients, dtype=float),
        'x_center': float(x_center),
        'y_center': float(y_center),
        'x_scale': float(x_scale),
        'y_scale': float(y_scale),
        'fitted_min': float(np.nanmin(fitted_values)),
        'fitted_max': float(np.nanmax(fitted_values)),
        'method': 'quadratic-ridge'
    }


def evaluate_physical_signed_margin_surface_grid(
    model: Optional[Dict[str, Any]],
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    grid_size: int = PHYSICAL_G_CONTOUR_GRID_SIZE
) -> Optional[Dict[str, Any]]:
    """Evaluasi surrogate `g_hat(x)` pada grid ruang fisik 2D."""
    if not model:
        return None

    coefficients = np.asarray(model.get('coefficients', []), dtype=float).reshape(-1)
    if coefficients.size != 6 or not np.all(np.isfinite(coefficients)):
        return None

    x_center = coerce_finite_float(model.get('x_center'))
    y_center = coerce_finite_float(model.get('y_center'))
    x_scale = coerce_finite_float(model.get('x_scale'))
    y_scale = coerce_finite_float(model.get('y_scale'))
    if None in {x_center, y_center, x_scale, y_scale}:
        return None

    grid_x, grid_y = np.meshgrid(
        np.linspace(float(x_limits[0]), float(x_limits[1]), int(grid_size)),
        np.linspace(float(y_limits[0]), float(y_limits[1]), int(grid_size))
    )
    grid_x_scaled = (grid_x - float(x_center)) / max(float(x_scale), 1e-6)
    grid_y_scaled = (grid_y - float(y_center)) / max(float(y_scale), 1e-6)
    grid_features = build_failure_surface_polynomial_features(
        np.asarray(grid_x_scaled, dtype=float).ravel(),
        np.asarray(grid_y_scaled, dtype=float).ravel()
    )
    grid_g = np.asarray(grid_features @ coefficients, dtype=float).reshape(grid_x.shape)
    finite_g = grid_g[np.isfinite(grid_g)]
    if finite_g.size == 0:
        return None

    return {
        'grid_x': np.asarray(grid_x, dtype=float),
        'grid_y': np.asarray(grid_y, dtype=float),
        'grid_g': np.asarray(grid_g, dtype=float),
        'g_min': float(np.min(finite_g)),
        'g_max': float(np.max(finite_g)),
        'method': str(model.get('method', 'quadratic-ridge') or 'quadratic-ridge')
    }


def compute_polyline_length(polyline: np.ndarray) -> float:
    """Hitung panjang total polyline 2D."""
    polyline_array = np.asarray(polyline, dtype=float)
    if polyline_array.ndim != 2 or polyline_array.shape[0] < 2:
        return 0.0
    deltas = np.diff(polyline_array[:, :2], axis=0)
    segment_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    return float(np.sum(segment_lengths))


def compute_point_to_polyline_distances(points_x: np.ndarray,
                                        points_y: np.ndarray,
                                        polyline: np.ndarray) -> np.ndarray:
    """Hitung jarak minimum titik-titik 2D ke sebuah polyline."""
    x_array = np.asarray(points_x, dtype=float).reshape(-1)
    y_array = np.asarray(points_y, dtype=float).reshape(-1)
    common_size = min(int(x_array.size), int(y_array.size))
    if common_size <= 0:
        return np.asarray([], dtype=float)

    points = np.column_stack([x_array[:common_size], y_array[:common_size]]).astype(float)
    polyline_array = np.asarray(polyline, dtype=float)
    if polyline_array.ndim != 2 or polyline_array.shape[0] < 2:
        return np.full(points.shape[0], np.inf, dtype=float)

    vertices = polyline_array[:, :2]
    distances = np.full(points.shape[0], np.inf, dtype=float)
    for start_point, end_point in zip(vertices[:-1], vertices[1:]):
        segment = end_point - start_point
        segment_length_squared = float(np.dot(segment, segment))
        if segment_length_squared <= 1e-18:
            projected_points = np.repeat(start_point[np.newaxis, :], points.shape[0], axis=0)
        else:
            projection_ratio = np.clip(
                np.sum((points - start_point[np.newaxis, :]) * segment[np.newaxis, :], axis=1)
                / segment_length_squared,
                0.0,
                1.0
            )
            projected_points = (
                start_point[np.newaxis, :]
                + projection_ratio[:, np.newaxis] * segment[np.newaxis, :]
            )
        current_distances = np.hypot(
            points[:, 0] - projected_points[:, 0],
            points[:, 1] - projected_points[:, 1]
        )
        distances = np.minimum(distances, current_distances)
    return distances.astype(float)


def select_primary_zero_contour_segment(
    zero_segments: List[np.ndarray],
    reference_x: Optional[np.ndarray] = None,
    reference_y: Optional[np.ndarray] = None,
    max_reference_points: int = 260
) -> Dict[str, Any]:
    """Pilih cabang contour `g_hat=0` yang paling representatif terhadap cloud boundary."""
    cleaned_segments = []
    for segment in zero_segments or []:
        segment_array = np.asarray(segment, dtype=float)
        if segment_array.ndim != 2 or segment_array.shape[0] < 2:
            continue
        segment_array = segment_array[:, :2]
        valid_mask = np.isfinite(segment_array).all(axis=1)
        segment_array = segment_array[valid_mask]
        if segment_array.shape[0] < 2:
            continue
        cleaned_segments.append(segment_array.astype(float))

    if not cleaned_segments:
        return {
            'segments': [],
            'selected_segment': None,
            'branch_count': 0,
            'selection_applied': False
        }

    if len(cleaned_segments) == 1:
        return {
            'segments': cleaned_segments,
            'selected_segment': cleaned_segments[0],
            'branch_count': 1,
            'selection_applied': False
        }

    reference_x_array = np.asarray(reference_x, dtype=float).reshape(-1)
    reference_y_array = np.asarray(reference_y, dtype=float).reshape(-1)
    common_size = min(int(reference_x_array.size), int(reference_y_array.size))
    if common_size > 0:
        reference_x_array = reference_x_array[:common_size]
        reference_y_array = reference_y_array[:common_size]
        valid_reference_mask = (
            np.isfinite(reference_x_array)
            & np.isfinite(reference_y_array)
        )
        reference_x_array = reference_x_array[valid_reference_mask]
        reference_y_array = reference_y_array[valid_reference_mask]
        if reference_x_array.size > int(max_reference_points):
            reference_indices = downsample_index_array(
                np.arange(reference_x_array.size, dtype=int),
                int(max_reference_points)
            )
            reference_x_array = reference_x_array[reference_indices]
            reference_y_array = reference_y_array[reference_indices]

    def score_segment(segment: np.ndarray) -> Tuple[float, float, float, float]:
        segment_length = compute_polyline_length(segment)
        if reference_x_array.size == 0 or reference_y_array.size == 0:
            return (
                float('inf'),
                float('inf'),
                float('inf'),
                -float(segment_length)
            )

        distances = compute_point_to_polyline_distances(
            reference_x_array,
            reference_y_array,
            segment
        )
        finite_distances = distances[np.isfinite(distances)]
        if finite_distances.size == 0:
            return (
                float('inf'),
                float('inf'),
                float('inf'),
                -float(segment_length)
            )

        return (
            float(np.quantile(finite_distances, 0.50)),
            float(np.quantile(finite_distances, 0.85)),
            float(np.mean(finite_distances)),
            -float(segment_length)
        )

    selected_segment = min(cleaned_segments, key=score_segment)
    return {
        'segments': cleaned_segments,
        'selected_segment': selected_segment,
        'branch_count': int(len(cleaned_segments)),
        'selection_applied': True
    }


def add_physical_signed_margin_contours(
    axis,
    surface_grid: Optional[Dict[str, Any]],
    max_abs_g: Optional[float] = None,
    zero_reference_x: Optional[np.ndarray] = None,
    zero_reference_y: Optional[np.ndarray] = None,
    show_fill: bool = True,
    show_nonzero_guides: bool = True
):
    """Tambahkan kontur `g_hat(x)` dan garis nol pada axis fisik."""
    if not surface_grid:
        return {
            'contour_fill': None,
            'zero_contour_drawn': False,
            'zero_contour_branch_count': 0,
            'zero_contour_selection_applied': False
        }

    grid_x = np.asarray(surface_grid.get('grid_x', []), dtype=float)
    grid_y = np.asarray(surface_grid.get('grid_y', []), dtype=float)
    grid_g = np.asarray(surface_grid.get('grid_g', []), dtype=float)
    if grid_x.size == 0 or grid_y.size == 0 or grid_g.size == 0:
        return {
            'contour_fill': None,
            'zero_contour_drawn': False,
            'zero_contour_branch_count': 0,
            'zero_contour_selection_applied': False
        }

    finite_g = grid_g[np.isfinite(grid_g)]
    if finite_g.size == 0:
        return {
            'contour_fill': None,
            'zero_contour_drawn': False,
            'zero_contour_branch_count': 0,
            'zero_contour_selection_applied': False
        }

    max_abs_value = (
        float(max_abs_g)
        if max_abs_g is not None and np.isfinite(float(max_abs_g))
        else float(np.max(np.abs(finite_g)))
    )
    max_abs_value = max(max_abs_value, 1e-6)

    contour_fill = None
    if show_fill:
        fill_levels = np.linspace(-max_abs_value, max_abs_value, 19)
        contour_fill = axis.contourf(
            grid_x,
            grid_y,
            grid_g,
            levels=fill_levels,
            cmap='RdYlBu',
            alpha=0.14,
            antialiased=True,
            zorder=0
        )
    contour_step = build_nice_contour_step(max_abs_value)
    nonzero_levels = np.asarray(
        [
            value for value in (
                contour_step * np.arange(-4, 5, dtype=float)
            )
            if (
                not np.isclose(value, 0.0, atol=max(contour_step * 1e-6, 1e-12))
                and abs(float(value)) <= max_abs_value * 1.02
            )
        ],
        dtype=float
    )
    if show_nonzero_guides and nonzero_levels.size > 0:
        axis.contour(
            grid_x,
            grid_y,
            grid_g,
            levels=np.sort(nonzero_levels),
            colors='#64748b',
            linewidths=0.60,
            alpha=0.42,
            linestyles='dashed',
            zorder=1
        )
    zero_contour_drawn = False
    zero_contour_branch_count = 0
    zero_contour_selection_applied = False
    if float(np.min(finite_g)) <= 0.0 <= float(np.max(finite_g)):
        zero_contour = axis.contour(
            grid_x,
            grid_y,
            grid_g,
            levels=[0.0],
            colors=[PHYSICAL_NONLINEAR_CONTOUR_COLOR],
            linewidths=2.2,
            zorder=2
        )
        zero_segments = list((zero_contour.allsegs or [[]])[0])
        zero_contour.remove()

        zero_segment_info = select_primary_zero_contour_segment(
            zero_segments,
            reference_x=zero_reference_x,
            reference_y=zero_reference_y
        )
        zero_contour_branch_count = int(zero_segment_info.get('branch_count', 0) or 0)
        zero_contour_selection_applied = bool(
            zero_segment_info.get('selection_applied', False)
        )
        selected_segment = zero_segment_info.get('selected_segment')
        if selected_segment is not None:
            selected_segment_array = np.asarray(selected_segment, dtype=float)
            if selected_segment_array.ndim == 2 and selected_segment_array.shape[0] >= 2:
                axis.plot(
                    selected_segment_array[:, 0],
                    selected_segment_array[:, 1],
                    color=PHYSICAL_NONLINEAR_CONTOUR_COLOR,
                    linewidth=2.2,
                    zorder=2
                )
                zero_contour_drawn = True

    return {
        'contour_fill': contour_fill,
        'zero_contour_drawn': bool(zero_contour_drawn),
        'zero_contour_branch_count': int(zero_contour_branch_count),
        'zero_contour_selection_applied': bool(zero_contour_selection_applied)
    }


def get_physical_limit_state_function_map_specs() -> List[Dict[str, str]]:
    """Spesifikasi subplot peta fungsi limit-state pada ruang variabel fisik."""
    return [
        {
            'key': 'moment',
            'label': 'Flexure',
            'x_var': 'fc',
            'y_var': 'fy_tarik',
            'x_label': "Concrete strength fc' (MPa)",
            'y_label': "Tensile steel strength fy (MPa)",
            'demand_label': 'Moment demand Md',
            'unit': 'kN.m'
        },
        {
            'key': 'shear',
            'label': 'Shear',
            'x_var': 'fc',
            'y_var': 'fy_geser',
            'x_label': "Concrete strength fc' (MPa)",
            'y_label': "Shear steel strength fy (MPa)",
            'demand_label': 'Shear demand Vd',
            'unit': 'kN'
        },
        {
            'key': 'axial',
            'label': 'Axial',
            'x_var': 'fc',
            'y_var': 'fy_tarik',
            'x_label': "Concrete strength fc' (MPa)",
            'y_label': "Steel strength fy (MPa)",
            'demand_label': 'Axial demand Pd',
            'unit': 'kN'
        }
    ]


def get_physical_limit_state_demand_material_map_specs() -> List[Dict[str, str]]:
    """Spesifikasi subplot peta fungsi limit-state pada ruang demand-material."""
    return [
        {
            'key': 'moment',
            'label': 'Flexure',
            'material_var': 'fc',
            'x_label': 'Moment demand Md (kN.m)',
            'y_label': "Concrete strength fc' (MPa)",
            'unit': 'kN.m'
        },
        {
            'key': 'shear',
            'label': 'Shear',
            'material_var': 'fc',
            'x_label': 'Shear demand Vd (kN)',
            'y_label': "Concrete strength fc' (MPa)",
            'unit': 'kN'
        },
        {
            'key': 'axial',
            'label': 'Axial',
            'material_var': 'fc',
            'x_label': 'Axial demand Pd (kN)',
            'y_label': "Concrete strength fc' (MPa)",
            'unit': 'kN'
        }
    ]


def get_physical_limit_state_material_stat(input_data: Dict,
                                           elem_id: int,
                                           variable_key: str) -> Optional[Dict[str, Any]]:
    """Ambil statistik mean/stddev variabel fisik per elemen dari input."""
    elem_id = int(elem_id)
    if variable_key == 'fc':
        concrete_props = get_by_element_value(
            input_data.get('concrete', {}).get('by_element', {}),
            elem_id,
            {}
        ) or {}
        mean_value = read_positive_number(concrete_props.get('mean'))
        stddev_value = read_positive_number(concrete_props.get('stddev'))
        if mean_value <= 0.0:
            return None
        return {
            'key': 'fc',
            'mean': float(mean_value),
            'stddev': float(stddev_value),
            'unit': 'MPa'
        }

    steel_props = get_by_element_value(
        input_data.get('steel', {}).get('by_element', {}),
        elem_id,
        {}
    ) or {}
    if variable_key == 'fy_tarik':
        mean_value = read_positive_number(steel_props.get('tarik_mean'))
        stddev_value = read_positive_number(steel_props.get('tarik_stddev'))
    elif variable_key == 'fy_tekan':
        mean_value = read_positive_number(steel_props.get('tekan_mean'))
        stddev_value = read_positive_number(steel_props.get('tekan_stddev'))
    elif variable_key == 'fy_geser':
        mean_value = read_positive_number(
            steel_props.get('geser_mean', steel_props.get('tarik_mean'))
        )
        stddev_value = read_positive_number(
            steel_props.get('geser_stddev', steel_props.get('tarik_stddev'))
        )
    else:
        return None

    if mean_value <= 0.0:
        return None
    return {
        'key': str(variable_key),
        'mean': float(mean_value),
        'stddev': float(stddev_value),
        'unit': 'MPa'
    }


def build_physical_limit_state_variable_limits(stat: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    """Bangun rentang sumbu untuk variabel fisik dari mean/stddev input."""
    if not stat:
        return None
    mean_value = coerce_finite_float(stat.get('mean'))
    stddev_value = coerce_finite_float(stat.get('stddev'))
    if mean_value is None or mean_value <= 0.0:
        return None

    effective_std = (
        float(stddev_value)
        if stddev_value is not None and stddev_value > 0.0 else
        max(0.08 * float(mean_value), 1e-6)
    )
    lower_limit = max(float(mean_value) - 3.0 * effective_std, 0.15 * float(mean_value), 1e-6)
    upper_limit = max(
        float(mean_value) + 3.0 * effective_std,
        lower_limit + max(0.25 * float(mean_value), effective_std, 1e-6)
    )
    return float(lower_limit), float(upper_limit)


def get_mean_limit_state_demand_from_record(record: Dict[str, Any]) -> Optional[float]:
    """Ambil demand rata-rata dari record failure cloud fisik per limit-state."""
    q_values = np.asarray((record or {}).get('Q', []), dtype=float)
    q_values = q_values[np.isfinite(q_values)]
    if q_values.size == 0:
        return None
    return float(np.mean(q_values))


def get_physical_limit_state_response_stat(record: Dict[str, Any],
                                           response_key: str) -> Optional[Dict[str, Any]]:
    """Ambil statistik respons valid `R`/`Q` dari record failure cloud fisik."""
    response_values = np.asarray((record or {}).get(str(response_key), []), dtype=float)
    response_values = response_values[np.isfinite(response_values)]
    if response_values.size == 0:
        return None
    return {
        'mean': float(np.mean(response_values)),
        'stddev': float(np.std(response_values)),
        'min': float(np.min(response_values)),
        'max': float(np.max(response_values))
    }


def get_physical_limit_state_demand_stat(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Ambil statistik demand valid dari record failure cloud fisik."""
    return get_physical_limit_state_response_stat(record, 'Q')


def get_physical_limit_state_capacity_stat(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Ambil statistik kapasitas valid dari record failure cloud fisik."""
    return get_physical_limit_state_response_stat(record, 'R')


def build_physical_limit_state_response_limits(response_stat: Optional[Dict[str, Any]]
                                               ) -> Optional[Tuple[float, float]]:
    """Bangun rentang sumbu respons `R`/`Q` dari cloud fisik valid."""
    if not response_stat:
        return None
    response_mean = coerce_finite_float(response_stat.get('mean'))
    response_std = coerce_finite_float(response_stat.get('stddev'))
    response_min = coerce_finite_float(response_stat.get('min'))
    response_max = coerce_finite_float(response_stat.get('max'))
    if None in {response_mean, response_min, response_max}:
        return None

    if response_std is None or response_std <= 0.0:
        response_std = max(0.08 * max(float(response_mean), 1.0), 1e-6)
    lower_limit = max(
        min(float(response_min), float(response_mean) - 3.0 * float(response_std)),
        0.0
    )
    upper_limit = max(
        max(float(response_max), float(response_mean) + 3.0 * float(response_std)),
        lower_limit + max(0.20 * max(float(response_mean), 1.0), float(response_std), 1e-6)
    )
    if np.isclose(lower_limit, upper_limit, atol=1e-12, rtol=1e-9):
        upper_limit = lower_limit + max(0.20 * max(float(response_mean), 1.0), 1e-6)
    return float(lower_limit), float(upper_limit)


def build_physical_limit_state_demand_limits(demand_stat: Optional[Dict[str, Any]]
                                             ) -> Optional[Tuple[float, float]]:
    """Bangun rentang sumbu demand dari cloud fisik valid."""
    return build_physical_limit_state_response_limits(demand_stat)


def get_physical_limit_state_custom_axis_specs(limit_state: str) -> List[Dict[str, str]]:
    """Daftar sumbu yang bisa dipilih untuk peta custom per limit-state."""
    limit_state_key = str(limit_state).strip().lower()
    if limit_state_key == 'moment':
        return [
            {
                'key': 'Q',
                'label': 'Q: Moment demand Md',
                'short_label': 'Q = Md',
                'axis_label': 'Q: Moment demand Md (kN.m)',
                'kind': 'demand',
                'unit': 'kN.m'
            },
            {
                'key': 'R',
                'label': 'R: Flexural capacity',
                'short_label': 'R',
                'axis_label': 'R: Flexural capacity (kN.m)',
                'kind': 'capacity',
                'unit': 'kN.m'
            },
            {
                'key': 'fc',
                'label': "Concrete strength fc'",
                'short_label': "fc'",
                'axis_label': "Concrete strength fc' (MPa)",
                'kind': 'material',
                'unit': 'MPa'
            },
            {
                'key': 'fy_tarik',
                'label': 'Tensile steel strength fy',
                'short_label': 'fy',
                'axis_label': 'Tensile steel strength fy (MPa)',
                'kind': 'material',
                'unit': 'MPa'
            },
            {
                'key': 'fy_tekan',
                'label': "Compressive steel strength fy'",
                'short_label': "fy'",
                'axis_label': "Compressive steel strength fy' (MPa)",
                'kind': 'material',
                'unit': 'MPa'
            }
        ]
    if limit_state_key == 'shear':
        return [
            {
                'key': 'Q',
                'label': 'Q: Shear demand Vd',
                'short_label': 'Q = Vd',
                'axis_label': 'Q: Shear demand Vd (kN)',
                'kind': 'demand',
                'unit': 'kN'
            },
            {
                'key': 'R',
                'label': 'R: Shear capacity',
                'short_label': 'R',
                'axis_label': 'R: Shear capacity (kN)',
                'kind': 'capacity',
                'unit': 'kN'
            },
            {
                'key': 'fc',
                'label': "Concrete strength fc'",
                'short_label': "fc'",
                'axis_label': "Concrete strength fc' (MPa)",
                'kind': 'material',
                'unit': 'MPa'
            },
            {
                'key': 'fy_geser',
                'label': 'Shear steel strength fy',
                'short_label': 'fy_shear',
                'axis_label': 'Shear steel strength fy (MPa)',
                'kind': 'material',
                'unit': 'MPa'
            }
        ]
    if limit_state_key == 'axial':
        return [
            {
                'key': 'Q',
                'label': 'Q: Axial demand Pd',
                'short_label': 'Q = Pd',
                'axis_label': 'Q: Axial demand Pd (kN)',
                'kind': 'demand',
                'unit': 'kN'
            },
            {
                'key': 'R',
                'label': 'R: Axial capacity',
                'short_label': 'R',
                'axis_label': 'R: Axial capacity (kN)',
                'kind': 'capacity',
                'unit': 'kN'
            },
            {
                'key': 'fc',
                'label': "Concrete strength fc'",
                'short_label': "fc'",
                'axis_label': "Concrete strength fc' (MPa)",
                'kind': 'material',
                'unit': 'MPa'
            },
            {
                'key': 'fy_tarik',
                'label': 'Tensile steel strength fy',
                'short_label': 'fy',
                'axis_label': 'Tensile steel strength fy (MPa)',
                'kind': 'material',
                'unit': 'MPa'
            },
            {
                'key': 'fy_tekan',
                'label': "Compressive steel strength fy'",
                'short_label': "fy'",
                'axis_label': "Compressive steel strength fy' (MPa)",
                'kind': 'material',
                'unit': 'MPa'
            }
        ]
    if limit_state_key == 'axial_moment':
        return [
            {
                'key': 'Pd',
                'label': 'Q: Axial demand Pd',
                'short_label': 'Pd',
                'axis_label': 'Q: Axial demand Pd (kN)',
                'kind': 'demand_axial',
                'unit': 'kN'
            },
            {
                'key': 'Md',
                'label': 'Q: Moment demand Md',
                'short_label': 'Md',
                'axis_label': 'Q: Moment demand Md (kN.m)',
                'kind': 'demand_moment',
                'unit': 'kN.m'
            },
            {
                'key': 'fc',
                'label': "Concrete strength fc'",
                'short_label': "fc'",
                'axis_label': "Concrete strength fc' (MPa)",
                'kind': 'material',
                'unit': 'MPa'
            },
            {
                'key': 'fy_tarik',
                'label': 'Tensile steel strength fy',
                'short_label': 'fy',
                'axis_label': 'Tensile steel strength fy (MPa)',
                'kind': 'material',
                'unit': 'MPa'
            },
            {
                'key': 'fy_tekan',
                'label': "Compressive steel strength fy'",
                'short_label': "fy'",
                'axis_label': "Compressive steel strength fy' (MPa)",
                'kind': 'material',
                'unit': 'MPa'
            }
        ]
    return []


def evaluate_physical_limit_state_capacity_value(limit_state: str,
                                                 section_inputs: Dict[str, Dict],
                                                 fc_value: float,
                                                 fy_tarik_value: float,
                                                 fy_tekan_value: float,
                                                 fy_geser_value: float,
                                                 axial_branch: str = 'compression'
                                                 ) -> Optional[float]:
    """Hitung kapasitas exact `R` untuk satu limit-state pada snapshot fisik tertentu."""
    try:
        if limit_state == 'moment':
            return float(
                PerformanceFunction._get_moment_capacity(
                    fc_value,
                    fy_tarik_value,
                    section_inputs['section_geometry'],
                    section_inputs['steel_area'],
                    fy_tekan=fy_tekan_value,
                    use_code_phi=False
                )
            )
        if limit_state == 'shear':
            response = PerformanceFunction._get_shear_capacity_check_result(
                0.0,
                fc_value,
                fy_geser_value,
                section_inputs['section_geometry'],
                section_inputs['steel_area'].get('As_shear', 0.0),
                shear_spacing=section_inputs['steel_area'].get('shear_spacing', 0.0),
                use_code_phi=False
            )
            return coerce_finite_float(response.get('phi_Vn'))
        if limit_state == 'axial':
            phi_pn_tekan, phi_pn_tarik = PerformanceFunction._get_axial_capacities(
                fc_value,
                fy_tarik_value,
                section_inputs['section_geometry'],
                section_inputs['steel_area'],
                fy_tekan=fy_tekan_value,
                use_code_phi=False
            )
            if str(axial_branch).strip().lower() == 'tension':
                return float(phi_pn_tarik)
            return float(phi_pn_tekan)
    except Exception:
        return None
    return None


def build_physical_limit_state_custom_axis_note(limit_state: str,
                                                axis_specs: List[Dict[str, str]],
                                                selected_axis_keys: Tuple[str, str],
                                                reference_values: Dict[str, float],
                                                axial_branch: str = 'compression') -> str:
    """Catatan singkat tentang variabel yang ditahan pada nilai mean referensi."""
    selected_keys = {str(value) for value in selected_axis_keys}
    note_parts = []

    if 'Q' not in selected_keys:
        q_value = coerce_finite_float(reference_values.get('Q'))
        q_spec = next((spec for spec in axis_specs if str(spec['key']) == 'Q'), None)
        if q_value is not None and q_spec is not None:
            note_parts.append(
                f"{q_spec['short_label']} = {q_value:.2f} {q_spec['unit']}"
            )

    for spec in axis_specs:
        spec_key = str(spec['key'])
        if spec_key in selected_keys or str(spec.get('kind')) != 'material':
            continue
        reference_value = coerce_finite_float(reference_values.get(spec_key))
        if reference_value is None:
            continue
        note_parts.append(
            f"{spec['short_label']} = {reference_value:.1f} {spec['unit']}"
        )

    if limit_state == 'axial':
        note_parts.append(
            f"Branch: {str(axial_branch).replace('-', ' ').title()}"
        )

    if not note_parts:
        return "All remaining variables use the reference mean values."

    midpoint = max(int(np.ceil(len(note_parts) / 2.0)), 1)
    first_line = "Fixed at mean: " + " | ".join(note_parts[:midpoint])
    second_line = " | ".join(note_parts[midpoint:])
    if second_line:
        return f"{first_line}\n{second_line}"
    return first_line


def get_latest_axial_limit_state_branch(latest_result: Optional[Dict[str, Any]],
                                        elem_id: int) -> str:
    """Cabang aksial terbaru: compression/tension/absolute-axial."""
    metadata = get_by_element_value(
        (latest_result or {}).get('performance_axial_metadata', {}),
        int(elem_id),
        {}
    ) or {}
    branch = str(metadata.get('controlling_state', 'compression') or 'compression').strip().lower()
    if branch not in {'compression', 'tension', 'absolute-axial'}:
        return 'compression'
    return branch


def evaluate_physical_limit_state_function_grid(input_data: Dict,
                                                elem_id: int,
                                                limit_state: str,
                                                demand_value: float,
                                                x_var: str,
                                                y_var: str,
                                                axial_branch: str = 'compression',
                                                grid_size: int = PHYSICAL_LIMIT_STATE_FUNCTION_GRID_SIZE
                                                ) -> Optional[Dict[str, Any]]:
    """Evaluasi peta `g(x)` exact pada ruang variabel fisik dua dimensi."""
    elem_id = int(elem_id)
    demand_numeric = coerce_finite_float(demand_value)
    if demand_numeric is None:
        return None

    x_stat = get_physical_limit_state_material_stat(input_data, elem_id, x_var)
    y_stat = get_physical_limit_state_material_stat(input_data, elem_id, y_var)
    x_limits = build_physical_limit_state_variable_limits(x_stat)
    y_limits = build_physical_limit_state_variable_limits(y_stat)
    if x_stat is None or y_stat is None or x_limits is None or y_limits is None:
        return None

    section_inputs = get_section_capacity_inputs_from_input(input_data, elem_id)
    material_snapshot = get_element_material_snapshot(
        input_data,
        latest_simulation=None,
        is_probabilistic=True,
        elem_id=elem_id
    )
    steel_props = get_by_element_value(
        input_data.get('steel', {}).get('by_element', {}),
        elem_id,
        {}
    ) or {}
    fy_geser_mean = read_positive_number(
        steel_props.get('geser_mean', steel_props.get('tarik_mean'))
    )
    if fy_geser_mean <= 0.0:
        fy_geser_mean = material_snapshot['fy_tarik']

    grid_x_values = np.linspace(float(x_limits[0]), float(x_limits[1]), int(grid_size))
    grid_y_values = np.linspace(float(y_limits[0]), float(y_limits[1]), int(grid_size))
    grid_x, grid_y = np.meshgrid(grid_x_values, grid_y_values)
    grid_g = np.full_like(grid_x, np.nan, dtype=float)

    for row_index in range(grid_y_values.size):
        for col_index in range(grid_x_values.size):
            fc_value = float(material_snapshot['fc'])
            fy_tarik_value = float(material_snapshot['fy_tarik'])
            fy_tekan_value = float(material_snapshot['fy_tekan'])
            fy_geser_value = float(fy_geser_mean)

            for variable_key, variable_value in (
                (str(x_var), float(grid_x[row_index, col_index])),
                (str(y_var), float(grid_y[row_index, col_index]))
            ):
                if variable_key == 'fc':
                    fc_value = float(variable_value)
                elif variable_key == 'fy_tarik':
                    fy_tarik_value = float(variable_value)
                elif variable_key == 'fy_tekan':
                    fy_tekan_value = float(variable_value)
                elif variable_key == 'fy_geser':
                    fy_geser_value = float(variable_value)

            try:
                if limit_state == 'moment':
                    g_value = PerformanceFunction.moment_capacity_demand(
                        float(demand_numeric),
                        fc_value,
                        fy_tarik_value,
                        section_inputs['section_geometry'],
                        section_inputs['steel_area'],
                        fy_tekan=fy_tekan_value,
                        use_code_phi=False
                    )
                elif limit_state == 'shear':
                    g_value = PerformanceFunction.shear_capacity_demand(
                        float(demand_numeric),
                        fc_value,
                        fy_geser_value,
                        section_inputs['section_geometry'],
                        section_inputs['steel_area'].get('As_shear', 0.0),
                        shear_spacing=section_inputs['steel_area'].get('shear_spacing', 0.0),
                        use_code_phi=False
                    )
                elif limit_state == 'axial':
                    compression_demand = 0.0
                    tension_demand = 0.0
                    if str(axial_branch).strip().lower() == 'tension':
                        tension_demand = float(demand_numeric)
                    else:
                        compression_demand = float(demand_numeric)
                    g_value = PerformanceFunction.axial_capacity_demand(
                        compression_demand,
                        tension_demand,
                        fc_value,
                        fy_tarik_value,
                        section_inputs['section_geometry'],
                        section_inputs['steel_area'],
                        fy_tekan=fy_tekan_value,
                        use_code_phi=False
                    )
                else:
                    continue
            except Exception:
                continue

            if np.isfinite(float(g_value)):
                grid_g[row_index, col_index] = float(g_value)

    finite_g = grid_g[np.isfinite(grid_g)]
    if finite_g.size == 0:
        return None

    return {
        'grid_x': np.asarray(grid_x, dtype=float),
        'grid_y': np.asarray(grid_y, dtype=float),
        'grid_g': np.asarray(grid_g, dtype=float),
        'x_stat': x_stat,
        'y_stat': y_stat,
        'g_min': float(np.min(finite_g)),
        'g_max': float(np.max(finite_g))
    }


def evaluate_physical_limit_state_demand_material_grid(input_data: Dict,
                                                       elem_id: int,
                                                       limit_state: str,
                                                       material_var: str,
                                                       demand_limits: Tuple[float, float],
                                                       material_limits: Tuple[float, float],
                                                       axial_branch: str = 'compression',
                                                       grid_size: int = PHYSICAL_LIMIT_STATE_FUNCTION_GRID_SIZE
                                                       ) -> Optional[Dict[str, Any]]:
    """Evaluasi peta `g(x)` pada ruang demand-material fisik."""
    elem_id = int(elem_id)
    section_inputs = get_section_capacity_inputs_from_input(input_data, elem_id)
    material_snapshot = get_element_material_snapshot(
        input_data,
        latest_simulation=None,
        is_probabilistic=True,
        elem_id=elem_id
    )
    steel_props = get_by_element_value(
        input_data.get('steel', {}).get('by_element', {}),
        elem_id,
        {}
    ) or {}
    fy_geser_mean = read_positive_number(
        steel_props.get('geser_mean', steel_props.get('tarik_mean'))
    )
    if fy_geser_mean <= 0.0:
        fy_geser_mean = material_snapshot['fy_tarik']

    x_values = np.linspace(float(demand_limits[0]), float(demand_limits[1]), int(grid_size))
    y_values = np.linspace(float(material_limits[0]), float(material_limits[1]), int(grid_size))
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    grid_g = np.full_like(grid_x, np.nan, dtype=float)

    for row_index in range(y_values.size):
        for col_index in range(x_values.size):
            demand_value = float(grid_x[row_index, col_index])
            material_value = float(grid_y[row_index, col_index])
            fc_value = float(material_snapshot['fc'])
            fy_tarik_value = float(material_snapshot['fy_tarik'])
            fy_tekan_value = float(material_snapshot['fy_tekan'])
            fy_geser_value = float(fy_geser_mean)

            if str(material_var) == 'fc':
                fc_value = material_value
            elif str(material_var) == 'fy_tarik':
                fy_tarik_value = material_value
            elif str(material_var) == 'fy_tekan':
                fy_tekan_value = material_value
            elif str(material_var) == 'fy_geser':
                fy_geser_value = material_value

            try:
                if limit_state == 'moment':
                    g_value = PerformanceFunction.moment_capacity_demand(
                        demand_value,
                        fc_value,
                        fy_tarik_value,
                        section_inputs['section_geometry'],
                        section_inputs['steel_area'],
                        fy_tekan=fy_tekan_value,
                        use_code_phi=False
                    )
                elif limit_state == 'shear':
                    g_value = PerformanceFunction.shear_capacity_demand(
                        demand_value,
                        fc_value,
                        fy_geser_value,
                        section_inputs['section_geometry'],
                        section_inputs['steel_area'].get('As_shear', 0.0),
                        shear_spacing=section_inputs['steel_area'].get('shear_spacing', 0.0),
                        use_code_phi=False
                    )
                elif limit_state == 'axial':
                    compression_demand = 0.0
                    tension_demand = 0.0
                    if str(axial_branch).strip().lower() == 'tension':
                        tension_demand = demand_value
                    else:
                        compression_demand = demand_value
                    g_value = PerformanceFunction.axial_capacity_demand(
                        compression_demand,
                        tension_demand,
                        fc_value,
                        fy_tarik_value,
                        section_inputs['section_geometry'],
                        section_inputs['steel_area'],
                        fy_tekan=fy_tekan_value,
                        use_code_phi=False
                    )
                else:
                    continue
            except Exception:
                continue

            if np.isfinite(float(g_value)):
                grid_g[row_index, col_index] = float(g_value)

    finite_g = grid_g[np.isfinite(grid_g)]
    if finite_g.size == 0:
        return None
    return {
        'grid_x': np.asarray(grid_x, dtype=float),
        'grid_y': np.asarray(grid_y, dtype=float),
        'grid_g': np.asarray(grid_g, dtype=float),
        'g_min': float(np.min(finite_g)),
        'g_max': float(np.max(finite_g))
    }


def build_limit_state_function_physical_space_figure(physical_cloud_data: Dict[str, Any],
                                                     input_data: Dict,
                                                     latest_result: Optional[Dict],
                                                     element_reliability: Optional[Dict[str, Dict[int, Dict[str, Any]]]],
                                                     elem_id: int) -> Optional[plt.Figure]:
    """Bangun peta fungsi limit-state nonlinier di ruang variabel fisik asli."""
    state_specs = get_physical_limit_state_function_map_specs()
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.9), dpi=180)
    axes_list = list(np.asarray(axes).reshape(-1))
    contour_fill_reference = None
    plotted_any = False
    element_reliability = element_reliability or {}

    for axis, spec in zip(axes_list, state_specs):
        axis.set_facecolor('#f8fafc')
        record = get_probabilistic_limit_state_physical_cloud_record(
            physical_cloud_data,
            int(elem_id),
            str(spec['key'])
        )
        mean_demand = get_mean_limit_state_demand_from_record(record)
        if mean_demand is None:
            axis.axis('off')
            axis.text(
                0.5,
                0.5,
                f"No valid {spec['label'].lower()} cloud data\nfor element E{int(elem_id)}.",
                ha='center',
                va='center',
                fontsize=9.5,
                color='#475569',
                bbox=dict(
                    boxstyle='round,pad=0.25',
                    facecolor='#ffffff',
                    edgecolor='#cbd5e1'
                )
            )
            continue

        axial_branch = (
            get_latest_axial_limit_state_branch(latest_result, int(elem_id))
            if str(spec['key']) == 'axial' else
            'compression'
        )
        surface_grid = evaluate_physical_limit_state_function_grid(
            input_data=input_data,
            elem_id=int(elem_id),
            limit_state=str(spec['key']),
            demand_value=float(mean_demand),
            x_var=str(spec['x_var']),
            y_var=str(spec['y_var']),
            axial_branch=axial_branch
        )
        if surface_grid is None:
            axis.axis('off')
            axis.text(
                0.5,
                0.5,
                f"The nonlinear {spec['label'].lower()} map\ncould not be built.",
                ha='center',
                va='center',
                fontsize=9.5,
                color='#475569',
                bbox=dict(
                    boxstyle='round,pad=0.25',
                    facecolor='#ffffff',
                    edgecolor='#cbd5e1'
                )
            )
            continue

        grid_x = np.asarray(surface_grid['grid_x'], dtype=float)
        grid_y = np.asarray(surface_grid['grid_y'], dtype=float)
        grid_g = np.asarray(surface_grid['grid_g'], dtype=float)
        finite_g = grid_g[np.isfinite(grid_g)]
        max_abs_g = max(float(np.max(np.abs(finite_g))), 1e-6)
        fill_levels = np.linspace(-max_abs_g, max_abs_g, 19)
        contour_fill = axis.contourf(
            grid_x,
            grid_y,
            grid_g,
            levels=fill_levels,
            cmap='RdYlBu',
            alpha=0.92,
            antialiased=True
        )
        pin_contour_axes_to_grid(axis, grid_x, grid_y)
        contour_step = build_nice_contour_step(max_abs_g)
        line_levels = contour_step * np.arange(-4, 5, dtype=float)
        nonzero_levels = np.asarray(
            [
                value for value in line_levels
                if (
                    not np.isclose(value, 0.0, atol=max(contour_step * 1e-6, 1e-12))
                    and abs(float(value)) <= max_abs_g * 1.02
                )
            ],
            dtype=float
        )
        if nonzero_levels.size > 0:
            iso_contours = axis.contour(
                grid_x,
                grid_y,
                grid_g,
                levels=np.sort(nonzero_levels),
                colors='#ffffff',
                linewidths=0.8,
                alpha=0.72
            )
            axis.clabel(
                iso_contours,
                fmt=lambda value: f"{float(value):.0f}",
                inline=True,
                fontsize=7
            )

        scatter_data = build_limit_state_physical_cloud_scatter_data(
            record,
            str(spec['x_var']),
            str(spec['y_var'])
        )
        has_zero_crossing = bool(
            float(surface_grid['g_min']) <= 0.0 <= float(surface_grid['g_max'])
        )
        selected_zero_segment = None
        if has_zero_crossing:
            zero_contour = axis.contour(
                grid_x,
                grid_y,
                grid_g,
                levels=[0.0],
                colors=[PHYSICAL_NONLINEAR_CONTOUR_COLOR],
                linewidths=2.4
            )
            axis.clabel(
                zero_contour,
                fmt={0.0: 'g = 0'},
                inline=True,
                fontsize=8
            )
            zero_segment_info = select_primary_zero_contour_segment(
                list((zero_contour.allsegs or [[]])[0]),
                reference_x=None if not scatter_data else np.asarray(scatter_data.get('x', []), dtype=float),
                reference_y=None if not scatter_data else np.asarray(scatter_data.get('y', []), dtype=float)
            )
            selected_zero_segment = zero_segment_info.get('selected_segment')

        plot_limit_state_physical_cloud_scatter(
            axis,
            scatter_data
        )
        x_stat = surface_grid['x_stat']
        y_stat = surface_grid['y_stat']
        axis.scatter(
            [float(x_stat['mean'])],
            [float(y_stat['mean'])],
            marker='x',
            s=70,
            color='#111827',
            linewidths=1.6,
            zorder=6,
            label='Mean material point'
        )
        reliability_record = get_by_element_value(
            element_reliability.get(str(spec['key']), {}),
            int(elem_id),
            {}
        ) or {}
        beta_raw_value = reliability_record.get('Beta')
        material_space_mpp = resolve_material_space_beta_table_overlay(
            record,
            limit_state=str(spec['key']),
            x_axis_key=str(spec['x_var']),
            y_axis_key=str(spec['y_var']),
            target_beta=beta_raw_value,
            zero_contour_segment=selected_zero_segment
        )
        material_space_display_beta = coerce_finite_float(
            material_space_mpp.get('display_beta')
        )
        material_space_display_beta_label = format_beta_table_display(
            material_space_mpp.get('display_beta_raw', material_space_mpp.get('display_beta')),
            4
        )
        material_space_mpp_x = coerce_finite_float(material_space_mpp.get('contour_x'))
        material_space_mpp_y = coerce_finite_float(material_space_mpp.get('contour_y'))
        material_space_sample_x = coerce_finite_float(material_space_mpp.get('sample_x'))
        material_space_sample_y = coerce_finite_float(material_space_mpp.get('sample_y'))
        if (
            material_space_display_beta is not None
            and material_space_mpp_x is not None
            and material_space_mpp_y is not None
        ):
            axis.plot(
                [float(x_stat['mean']), float(material_space_mpp_x)],
                [float(y_stat['mean']), float(material_space_mpp_y)],
                linestyle='--',
                linewidth=1.35,
                color='#7e22ce',
                alpha=0.94,
                zorder=6,
                label=f"Projected beta line, Beta(table)={material_space_display_beta_label}"
            )
            if (
                material_space_sample_x is not None
                and material_space_sample_y is not None
                and not (
                    np.isclose(material_space_sample_x, material_space_mpp_x, atol=1e-9, rtol=1e-9)
                    and np.isclose(material_space_sample_y, material_space_mpp_y, atol=1e-9, rtol=1e-9)
                )
            ):
                axis.plot(
                    [float(material_space_sample_x), float(material_space_mpp_x)],
                    [float(material_space_sample_y), float(material_space_mpp_y)],
                    linestyle=':',
                    linewidth=1.0,
                    color='#d946ef',
                    alpha=0.88,
                    zorder=6
                )
            axis.scatter(
                [float(material_space_mpp_x)],
                [float(material_space_mpp_y)],
                marker='*',
                s=175,
                color='#f59e0b',
                edgecolors='#111827',
                linewidths=0.80,
                zorder=7,
                label=f"Projected MPP, Beta(table)={material_space_display_beta_label}"
            )
            add_smart_mpp_annotation(
                axis,
                "\n".join([
                    "MPP",
                    f"Beta(table)={material_space_display_beta_label}",
                    f"{str(spec['x_var'])}={format_metric(material_space_mpp_x, 2)}",
                    f"{str(spec['y_var'])}={format_metric(material_space_mpp_y, 2)}"
                ]),
                target_xy=(float(material_space_mpp_x), float(material_space_mpp_y)),
                avoid_points=build_annotation_avoid_points(
                    scatter_data=scatter_data,
                    extra_points=[
                        (float(x_stat['mean']), float(y_stat['mean'])),
                        (material_space_sample_x, material_space_sample_y),
                        (material_space_mpp_x, material_space_mpp_y)
                    ],
                    max_points=260
                ),
                bbox_edgecolor='#cbd5e1',
                text_color='#111827',
                fontsize=8.0,
                zorder=7,
                with_arrow=True
            )
        elif beta_raw_value is not None:
            axis.text(
                0.02,
                0.03,
                (
                    "MPP/Beta(table) overlay unavailable\n"
                    "for the current material-space contour."
                ),
                transform=axis.transAxes,
                ha='left',
                va='bottom',
                fontsize=7.1,
                color='#7c2d12',
                bbox=dict(
                    boxstyle='round,pad=0.18',
                    facecolor='#fff7ed',
                    edgecolor='#fdba74',
                    alpha=0.92
                ),
                zorder=7
            )
        if has_zero_crossing:
            axis.plot(
                [],
                [],
                color=PHYSICAL_NONLINEAR_CONTOUR_COLOR,
                linewidth=2.4,
                label='Nonlinear limit-state contour'
            )
        else:
            axis.text(
                0.02,
                0.03,
                "No g = 0 contour in the current range",
                transform=axis.transAxes,
                ha='left',
                va='bottom',
                fontsize=7.3,
                color='#7c2d12',
                bbox=dict(
                    boxstyle='round,pad=0.18',
                    facecolor='#fff7ed',
                    edgecolor='#fdba74',
                    alpha=0.92
                )
            )
        axis.set_xlabel(str(spec['x_label']))
        axis.set_ylabel(str(spec['y_label']))
        branch_note = ''
        if str(spec['key']) == 'axial':
            branch_note = f" | Branch: {str(axial_branch).replace('-', ' ').title()}"
        axis.set_title(
            (
                f"{spec['label']} | {spec['demand_label']} = {float(mean_demand):.2f} {spec['unit']}"
                f"{branch_note}"
            ),
            fontsize=10.2,
            pad=9
        )
        axis.grid(True, alpha=0.16, linestyle='--')
        axis.legend(loc='best', fontsize=7.4, frameon=True)
        contour_fill_reference = contour_fill
        plotted_any = True

    if not plotted_any or contour_fill_reference is None:
        plt.close(fig)
        return None

    fig.suptitle(
        f"Nonlinear Limit-State Function Maps in Physical Variable Space | Element E{int(elem_id)}",
        fontsize=13,
        y=0.99
    )
    fig.tight_layout(rect=[0, 0, 0.94, 0.965])
    colorbar_axis = fig.add_axes([0.948, 0.16, 0.014, 0.68])
    fig.colorbar(
        contour_fill_reference,
        cax=colorbar_axis,
        label='Limit-state function g(x)'
    )
    return fig


def build_limit_state_function_demand_material_space_figure(physical_cloud_data: Dict[str, Any],
                                                            input_data: Dict,
                                                            latest_result: Optional[Dict],
                                                            element_reliability: Optional[Dict[str, Dict[int, Dict[str, Any]]]],
                                                            elem_id: int) -> Optional[plt.Figure]:
    """Bangun peta fungsi limit-state nonlinier pada ruang demand-material."""
    state_specs = get_physical_limit_state_demand_material_map_specs()
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.9), dpi=180)
    axes_list = list(np.asarray(axes).reshape(-1))
    contour_fill_reference = None
    plotted_any = False
    element_reliability = element_reliability or {}

    for axis, spec in zip(axes_list, state_specs):
        axis.set_facecolor('#f8fafc')
        record = get_probabilistic_limit_state_physical_cloud_record(
            physical_cloud_data,
            int(elem_id),
            str(spec['key'])
        )
        demand_stat = get_physical_limit_state_demand_stat(record)
        material_stat = get_physical_limit_state_material_stat(
            input_data,
            int(elem_id),
            str(spec['material_var'])
        )
        demand_limits = build_physical_limit_state_demand_limits(demand_stat)
        material_limits = build_physical_limit_state_variable_limits(material_stat)
        if demand_stat is None or material_stat is None or demand_limits is None or material_limits is None:
            axis.axis('off')
            axis.text(
                0.5,
                0.5,
                f"No valid {spec['label'].lower()} data\nfor element E{int(elem_id)}.",
                ha='center',
                va='center',
                fontsize=9.5,
                color='#475569',
                bbox=dict(
                    boxstyle='round,pad=0.25',
                    facecolor='#ffffff',
                    edgecolor='#cbd5e1'
                )
            )
            continue

        axial_branch = (
            get_latest_axial_limit_state_branch(latest_result, int(elem_id))
            if str(spec['key']) == 'axial' else
            'compression'
        )
        surface_grid = evaluate_physical_limit_state_demand_material_grid(
            input_data=input_data,
            elem_id=int(elem_id),
            limit_state=str(spec['key']),
            material_var=str(spec['material_var']),
            demand_limits=demand_limits,
            material_limits=material_limits,
            axial_branch=axial_branch
        )
        if surface_grid is None:
            axis.axis('off')
            axis.text(
                0.5,
                0.5,
                f"The nonlinear {spec['label'].lower()} demand-material map\ncould not be built.",
                ha='center',
                va='center',
                fontsize=9.5,
                color='#475569',
                bbox=dict(
                    boxstyle='round,pad=0.25',
                    facecolor='#ffffff',
                    edgecolor='#cbd5e1'
                )
            )
            continue

        grid_x = np.asarray(surface_grid['grid_x'], dtype=float)
        grid_y = np.asarray(surface_grid['grid_y'], dtype=float)
        grid_g = np.asarray(surface_grid['grid_g'], dtype=float)
        finite_g = grid_g[np.isfinite(grid_g)]
        max_abs_g = max(float(np.max(np.abs(finite_g))), 1e-6)
        fill_levels = np.linspace(-max_abs_g, max_abs_g, 19)
        contour_fill = axis.contourf(
            grid_x,
            grid_y,
            grid_g,
            levels=fill_levels,
            cmap='RdYlBu',
            alpha=0.92,
            antialiased=True
        )
        pin_contour_axes_to_grid(axis, grid_x, grid_y)
        contour_step = build_nice_contour_step(max_abs_g)
        line_levels = contour_step * np.arange(-4, 5, dtype=float)
        nonzero_levels = np.asarray(
            [
                value for value in line_levels
                if (
                    not np.isclose(value, 0.0, atol=max(contour_step * 1e-6, 1e-12))
                    and abs(float(value)) <= max_abs_g * 1.02
                )
            ],
            dtype=float
        )
        if nonzero_levels.size > 0:
            iso_contours = axis.contour(
                grid_x,
                grid_y,
                grid_g,
                levels=np.sort(nonzero_levels),
                colors='#ffffff',
                linewidths=0.8,
                alpha=0.72
            )
            axis.clabel(
                iso_contours,
                fmt=lambda value: f"{float(value):.0f}",
                inline=True,
                fontsize=7
            )

        scatter_data = build_limit_state_physical_cloud_scatter_data(
            record,
            'Q',
            str(spec['material_var'])
        )
        has_zero_crossing = bool(
            float(surface_grid['g_min']) <= 0.0 <= float(surface_grid['g_max'])
        )
        selected_zero_segment = None
        if has_zero_crossing:
            zero_contour = axis.contour(
                grid_x,
                grid_y,
                grid_g,
                levels=[0.0],
                colors=[PHYSICAL_NONLINEAR_CONTOUR_COLOR],
                linewidths=2.4
            )
            axis.clabel(
                zero_contour,
                fmt={0.0: 'g = 0'},
                inline=True,
                fontsize=8
            )
            zero_segment_info = select_primary_zero_contour_segment(
                list((zero_contour.allsegs or [[]])[0]),
                reference_x=None if not scatter_data else np.asarray(scatter_data.get('x', []), dtype=float),
                reference_y=None if not scatter_data else np.asarray(scatter_data.get('y', []), dtype=float)
            )
            selected_zero_segment = zero_segment_info.get('selected_segment')

        plot_limit_state_physical_cloud_scatter(
            axis,
            scatter_data
        )
        axis.scatter(
            [float(demand_stat['mean'])],
            [float(material_stat['mean'])],
            marker='x',
            s=70,
            color='#111827',
            linewidths=1.6,
            zorder=6,
            label='Mean operating point'
        )
        reliability_record = get_by_element_value(
            element_reliability.get(str(spec['key']), {}),
            int(elem_id),
            {}
        ) or {}
        beta_raw_value = reliability_record.get('Beta')
        demand_material_mpp = resolve_material_space_beta_table_overlay(
            record,
            limit_state=str(spec['key']),
            x_axis_key='Q',
            y_axis_key=str(spec['material_var']),
            target_beta=beta_raw_value,
            zero_contour_segment=selected_zero_segment
        )
        demand_material_display_beta = coerce_finite_float(
            demand_material_mpp.get('display_beta')
        )
        demand_material_display_beta_label = format_beta_table_display(
            demand_material_mpp.get('display_beta_raw', demand_material_mpp.get('display_beta')),
            4
        )
        demand_material_mpp_x = coerce_finite_float(demand_material_mpp.get('contour_x'))
        demand_material_mpp_y = coerce_finite_float(demand_material_mpp.get('contour_y'))
        demand_material_sample_x = coerce_finite_float(demand_material_mpp.get('sample_x'))
        demand_material_sample_y = coerce_finite_float(demand_material_mpp.get('sample_y'))
        if (
            demand_material_display_beta is not None
            and demand_material_mpp_x is not None
            and demand_material_mpp_y is not None
        ):
            axis.plot(
                [float(demand_stat['mean']), float(demand_material_mpp_x)],
                [float(material_stat['mean']), float(demand_material_mpp_y)],
                linestyle='--',
                linewidth=1.35,
                color='#7e22ce',
                alpha=0.94,
                zorder=6,
                label=f"Projected beta line, Beta(table)={demand_material_display_beta_label}"
            )
            if (
                demand_material_sample_x is not None
                and demand_material_sample_y is not None
                and not (
                    np.isclose(demand_material_sample_x, demand_material_mpp_x, atol=1e-9, rtol=1e-9)
                    and np.isclose(demand_material_sample_y, demand_material_mpp_y, atol=1e-9, rtol=1e-9)
                )
            ):
                axis.plot(
                    [float(demand_material_sample_x), float(demand_material_mpp_x)],
                    [float(demand_material_sample_y), float(demand_material_mpp_y)],
                    linestyle=':',
                    linewidth=1.0,
                    color='#d946ef',
                    alpha=0.88,
                    zorder=6
                )
            axis.scatter(
                [float(demand_material_mpp_x)],
                [float(demand_material_mpp_y)],
                marker='*',
                s=175,
                color='#f59e0b',
                edgecolors='#111827',
                linewidths=0.80,
                zorder=7,
                label=f"Projected MPP, Beta(table)={demand_material_display_beta_label}"
            )
            add_smart_mpp_annotation(
                axis,
                "\n".join([
                    "MPP",
                    f"Beta(table)={demand_material_display_beta_label}",
                    f"Q={format_metric(demand_material_mpp_x, 2)}",
                    f"{str(spec['material_var'])}={format_metric(demand_material_mpp_y, 2)}"
                ]),
                target_xy=(float(demand_material_mpp_x), float(demand_material_mpp_y)),
                avoid_points=build_annotation_avoid_points(
                    scatter_data=scatter_data,
                    extra_points=[
                        (float(demand_stat['mean']), float(material_stat['mean'])),
                        (demand_material_sample_x, demand_material_sample_y),
                        (demand_material_mpp_x, demand_material_mpp_y)
                    ],
                    max_points=260
                ),
                bbox_edgecolor='#cbd5e1',
                text_color='#111827',
                fontsize=8.0,
                zorder=7,
                with_arrow=True
            )
        elif beta_raw_value is not None:
            axis.text(
                0.02,
                0.03,
                (
                    "MPP/Beta(table) overlay unavailable\n"
                    "for the current demand-material contour."
                ),
                transform=axis.transAxes,
                ha='left',
                va='bottom',
                fontsize=7.1,
                color='#7c2d12',
                bbox=dict(
                    boxstyle='round,pad=0.18',
                    facecolor='#fff7ed',
                    edgecolor='#fdba74',
                    alpha=0.92
                ),
                zorder=7
            )
        if has_zero_crossing:
            axis.plot(
                [],
                [],
                color=PHYSICAL_NONLINEAR_CONTOUR_COLOR,
                linewidth=2.4,
                label='Nonlinear limit-state contour'
            )
        else:
            axis.text(
                0.02,
                0.03,
                "No g = 0 contour in the current range",
                transform=axis.transAxes,
                ha='left',
                va='bottom',
                fontsize=7.3,
                color='#7c2d12',
                bbox=dict(
                    boxstyle='round,pad=0.18',
                    facecolor='#fff7ed',
                    edgecolor='#fdba74',
                    alpha=0.92
                )
            )

        fy_note = ''
        if str(spec['key']) == 'moment':
            fy_note = " | fy fixed at mean"
        elif str(spec['key']) == 'shear':
            fy_note = " | fy_shear fixed at mean"
        elif str(spec['key']) == 'axial':
            branch_note = str(axial_branch).replace('-', ' ').title()
            fy_note = f" | fy fixed at mean | Branch: {branch_note}"

        axis.set_xlabel(str(spec['x_label']))
        axis.set_ylabel(str(spec['y_label']))
        axis.set_title(
            f"{spec['label']} | g(demand, material){fy_note}",
            fontsize=10.2,
            pad=9
        )
        axis.grid(True, alpha=0.16, linestyle='--')
        axis.legend(loc='best', fontsize=7.4, frameon=True)
        contour_fill_reference = contour_fill
        plotted_any = True

    if not plotted_any or contour_fill_reference is None:
        plt.close(fig)
        return None

    fig.suptitle(
        f"Nonlinear Limit-State Function Maps in Demand-Material Space | Element E{int(elem_id)}",
        fontsize=13,
        y=0.99
    )
    fig.tight_layout(rect=[0, 0, 0.94, 0.965])
    colorbar_axis = fig.add_axes([0.948, 0.16, 0.014, 0.68])
    fig.colorbar(
        contour_fill_reference,
        cax=colorbar_axis,
        label='Limit-state function g(x)'
    )
    return fig


def build_axial_moment_custom_axis_figure(axial_moment_pm_cloud_data: Dict[str, Any],
                                          input_data: Dict,
                                          beta_value: Optional[float],
                                          elem_id: int,
                                          x_axis_key: str,
                                          y_axis_key: str) -> Optional[plt.Figure]:
    """Bangun peta custom aksial-lentur pada ruang Pd/Md/material dengan `g = lambda - 1`."""
    axis_specs = get_physical_limit_state_custom_axis_specs('axial_moment')
    axis_lookup = {
        str(spec['key']): spec
        for spec in axis_specs
    }
    x_spec = axis_lookup.get(str(x_axis_key))
    y_spec = axis_lookup.get(str(y_axis_key))
    if x_spec is None or y_spec is None or str(x_spec['key']) == str(y_spec['key']):
        return None

    record = (
        (axial_moment_pm_cloud_data or {}).get('elements', {}).get(str(int(elem_id)))
        or {}
    )
    if not record:
        return None

    scatter_data = build_axial_moment_custom_scatter_data(
        record,
        str(x_spec['key']),
        str(y_spec['key'])
    )
    demand_axial_values = get_axial_moment_custom_axis_values(record, 'Pd')
    demand_moment_values = get_axial_moment_custom_axis_values(record, 'Md')
    demand_axial_values = demand_axial_values[np.isfinite(demand_axial_values)]
    demand_moment_values = demand_moment_values[np.isfinite(demand_moment_values)]

    material_snapshot = get_element_material_snapshot(
        input_data,
        latest_simulation=None,
        is_probabilistic=True,
        elem_id=int(elem_id)
    )
    section_inputs = get_section_capacity_inputs_from_input(input_data, int(elem_id))

    reference_values = {
        'Pd': float(np.mean(demand_axial_values)) if demand_axial_values.size else 0.0,
        'Md': float(np.mean(demand_moment_values)) if demand_moment_values.size else 0.0,
        'fc': float(material_snapshot['fc']),
        'fy_tarik': float(material_snapshot['fy_tarik']),
        'fy_tekan': float(material_snapshot['fy_tekan'])
    }

    def resolve_axis_limits(axis_spec: Dict[str, str]) -> Optional[Tuple[float, float]]:
        axis_kind = str(axis_spec.get('kind'))
        axis_key = str(axis_spec['key'])
        if axis_kind == 'material':
            material_stat = get_physical_limit_state_material_stat(
                input_data,
                int(elem_id),
                axis_key
            )
            return build_physical_limit_state_variable_limits(material_stat)

        demand_values = get_axial_moment_custom_axis_values(record, axis_key)
        demand_values = demand_values[np.isfinite(demand_values)]
        if demand_values.size > 0:
            return get_failure_cloud_axis_limits(demand_values)

        reference_value = coerce_finite_float(reference_values.get(axis_key))
        if reference_value is None:
            return None
        padding = max(abs(float(reference_value)) * 0.20, 1.0)
        return float(reference_value - padding), float(reference_value + padding)

    x_limits = resolve_axis_limits(x_spec)
    y_limits = resolve_axis_limits(y_spec)
    if x_limits is None or y_limits is None:
        return None

    grid_x_values = np.linspace(float(x_limits[0]), float(x_limits[1]), PHYSICAL_LIMIT_STATE_FUNCTION_GRID_SIZE)
    grid_y_values = np.linspace(float(y_limits[0]), float(y_limits[1]), PHYSICAL_LIMIT_STATE_FUNCTION_GRID_SIZE)
    grid_x, grid_y = np.meshgrid(grid_x_values, grid_y_values)
    grid_g = np.full_like(grid_x, np.nan, dtype=float)

    for row_index in range(grid_y_values.size):
        for col_index in range(grid_x_values.size):
            current_values = dict(reference_values)
            current_values[str(x_spec['key'])] = float(grid_x[row_index, col_index])
            current_values[str(y_spec['key'])] = float(grid_y[row_index, col_index])

            demand_axial = float(current_values['Pd'])
            demand_moment = abs(float(current_values['Md']))
            compression_demand = 0.0
            tension_demand = 0.0
            if demand_axial > AXIAL_DEMAND_TOLERANCE_KN:
                compression_demand = float(demand_axial)
            elif demand_axial < -AXIAL_DEMAND_TOLERANCE_KN:
                tension_demand = float(-demand_axial)

            try:
                response = PerformanceFunction._get_axial_moment_capacity_check_result(
                    compression_demand,
                    tension_demand,
                    demand_moment,
                    float(current_values['fc']),
                    float(current_values['fy_tarik']),
                    section_inputs['section_geometry'],
                    section_inputs['steel_area'],
                    fy_tekan=float(current_values['fy_tekan']),
                    use_code_phi=False
                )
            except Exception:
                continue

            g_value = coerce_finite_float((response or {}).get('g'))
            if g_value is not None:
                grid_g[row_index, col_index] = float(g_value)

    finite_g = grid_g[np.isfinite(grid_g)]
    if finite_g.size == 0:
        return None

    fig, axis = plt.subplots(figsize=(9.0, 6.2), dpi=180)
    axis.set_facecolor('#f8fafc')

    max_abs_g = max(float(np.max(np.abs(finite_g))), 1e-6)
    fill_levels = np.linspace(-max_abs_g, max_abs_g, 19)
    contour_fill = axis.contourf(
        grid_x,
        grid_y,
        grid_g,
        levels=fill_levels,
        cmap='RdYlBu',
        alpha=0.92,
        antialiased=True
    )
    pin_contour_axes_to_grid(axis, grid_x, grid_y)

    contour_step = build_nice_contour_step(max_abs_g)
    line_levels = contour_step * np.arange(-4, 5, dtype=float)
    nonzero_levels = np.asarray(
        [
            value for value in line_levels
            if (
                not np.isclose(value, 0.0, atol=max(contour_step * 1e-6, 1e-12))
                and abs(float(value)) <= max_abs_g * 1.02
            )
        ],
        dtype=float
    )
    if nonzero_levels.size > 0:
        iso_contours = axis.contour(
            grid_x,
            grid_y,
            grid_g,
            levels=np.sort(nonzero_levels),
            colors='#ffffff',
            linewidths=0.8,
            alpha=0.72
        )
        axis.clabel(
            iso_contours,
            fmt=lambda value: f"{float(value):.0f}",
            inline=True,
            fontsize=7
        )

    has_zero_crossing = bool(float(np.min(finite_g)) <= 0.0 <= float(np.max(finite_g)))
    selected_zero_segment = None
    if has_zero_crossing:
        zero_contour = axis.contour(
            grid_x,
            grid_y,
            grid_g,
            levels=[0.0],
            colors=[PHYSICAL_NONLINEAR_CONTOUR_COLOR],
            linewidths=2.4
        )
        axis.clabel(
            zero_contour,
            fmt={0.0: 'g = 0'},
            inline=True,
            fontsize=8
        )
        zero_segment_info = select_primary_zero_contour_segment(
            list((zero_contour.allsegs or [[]])[0]),
            reference_x=np.asarray((scatter_data or {}).get('x', []), dtype=float),
            reference_y=np.asarray((scatter_data or {}).get('y', []), dtype=float)
        )
        selected_zero_segment = zero_segment_info.get('selected_segment')

    plot_limit_state_physical_cloud_scatter(
        axis,
        scatter_data
    )
    reference_x = coerce_finite_float(reference_values.get(str(x_spec['key'])))
    reference_y = coerce_finite_float(reference_values.get(str(y_spec['key'])))
    if reference_x is not None and reference_y is not None:
        axis.scatter(
            [float(reference_x)],
            [float(reference_y)],
            marker='x',
            s=70,
            color='#111827',
            linewidths=1.6,
            zorder=6,
            label='Mean reference point'
        )
    axial_moment_mpp = resolve_axial_moment_map_beta_table_overlay(
        record,
        x_axis_key=str(x_spec['key']),
        y_axis_key=str(y_spec['key']),
        target_beta=beta_value,
        zero_contour_segment=selected_zero_segment
    )
    axial_moment_display_beta = coerce_finite_float(axial_moment_mpp.get('display_beta'))
    axial_moment_display_beta_label = format_beta_table_display(
        axial_moment_mpp.get('display_beta_raw', axial_moment_mpp.get('display_beta')),
        4
    )
    axial_moment_mpp_x = coerce_finite_float(axial_moment_mpp.get('contour_x'))
    axial_moment_mpp_y = coerce_finite_float(axial_moment_mpp.get('contour_y'))
    axial_moment_sample_x = coerce_finite_float(axial_moment_mpp.get('sample_x'))
    axial_moment_sample_y = coerce_finite_float(axial_moment_mpp.get('sample_y'))
    if (
        reference_x is not None
        and reference_y is not None
        and axial_moment_display_beta is not None
        and axial_moment_mpp_x is not None
        and axial_moment_mpp_y is not None
    ):
        axis.plot(
            [float(reference_x), float(axial_moment_mpp_x)],
            [float(reference_y), float(axial_moment_mpp_y)],
            linestyle='--',
            linewidth=1.35,
            color='#7e22ce',
            alpha=0.94,
            zorder=6,
            label=f"Projected beta line, Beta(table)={axial_moment_display_beta_label}"
        )
        if (
            axial_moment_sample_x is not None
            and axial_moment_sample_y is not None
            and not (
                np.isclose(axial_moment_sample_x, axial_moment_mpp_x, atol=1e-9, rtol=1e-9)
                and np.isclose(axial_moment_sample_y, axial_moment_mpp_y, atol=1e-9, rtol=1e-9)
            )
        ):
            axis.plot(
                [float(axial_moment_sample_x), float(axial_moment_mpp_x)],
                [float(axial_moment_sample_y), float(axial_moment_mpp_y)],
                linestyle=':',
                linewidth=1.0,
                color='#d946ef',
                alpha=0.88,
                zorder=6
            )
        axis.scatter(
            [float(axial_moment_mpp_x)],
            [float(axial_moment_mpp_y)],
            marker='*',
            s=175,
            color='#f59e0b',
            edgecolors='#111827',
            linewidths=0.80,
            zorder=7,
            label=f"Projected MPP, Beta(table)={axial_moment_display_beta_label}"
        )
        add_smart_mpp_annotation(
            axis,
            "\n".join([
                "MPP",
                f"Beta(table)={axial_moment_display_beta_label}",
                f"{x_spec['short_label']}={format_metric(axial_moment_mpp_x, 2)}",
                f"{y_spec['short_label']}={format_metric(axial_moment_mpp_y, 2)}"
            ]),
            target_xy=(float(axial_moment_mpp_x), float(axial_moment_mpp_y)),
            avoid_points=build_annotation_avoid_points(
                scatter_data=scatter_data,
                extra_points=[
                    (reference_x, reference_y),
                    (axial_moment_sample_x, axial_moment_sample_y),
                    (axial_moment_mpp_x, axial_moment_mpp_y)
                ],
                max_points=260
            ),
            bbox_edgecolor='#cbd5e1',
            text_color='#111827',
            fontsize=8.0,
            zorder=7,
            with_arrow=True
        )
    elif beta_value is not None:
        axis.text(
            0.02,
            0.03,
            (
                "MPP/Beta(table) overlay unavailable\n"
                "for the current axial-flexure contour."
            ),
            transform=axis.transAxes,
            ha='left',
            va='bottom',
            fontsize=7.1,
            color='#7c2d12',
            bbox=dict(
                boxstyle='round,pad=0.18',
                facecolor='#fff7ed',
                edgecolor='#fdba74',
                alpha=0.92
            ),
            zorder=7
        )

    if has_zero_crossing:
        axis.plot(
            [],
            [],
            color=PHYSICAL_NONLINEAR_CONTOUR_COLOR,
            linewidth=2.4,
            label='Limit-state contour g = 0'
        )
    else:
        axis.text(
            0.02,
            0.03,
            "No g = 0 contour in the current range",
            transform=axis.transAxes,
            ha='left',
            va='bottom',
            fontsize=7.3,
            color='#7c2d12',
            bbox=dict(
                boxstyle='round,pad=0.18',
                facecolor='#fff7ed',
                edgecolor='#fdba74',
                alpha=0.92
            )
        )

    note_parts = []
    for axis_key in ('Pd', 'Md', 'fc', 'fy_tarik', 'fy_tekan'):
        if axis_key in {str(x_spec['key']), str(y_spec['key'])}:
            continue
        reference_value = coerce_finite_float(reference_values.get(axis_key))
        axis_spec = axis_lookup.get(axis_key)
        if reference_value is None or axis_spec is None:
            continue
        decimals = 2 if axis_key in {'Pd', 'Md'} else 1
        note_parts.append(
            f"{axis_spec['short_label']} = {reference_value:.{decimals}f} {axis_spec['unit']}"
        )
    note_text = "Fixed at mean: " + " | ".join(note_parts) if note_parts else "All remaining variables use the reference mean values."
    axis.text(
        0.02,
        0.98,
        note_text,
        transform=axis.transAxes,
        ha='left',
        va='top',
        fontsize=7.2,
        color='#334155',
        bbox=dict(
            boxstyle='round,pad=0.20',
            facecolor='#ffffff',
            edgecolor='#cbd5e1',
            alpha=0.92
        )
    )
    axis.text(
        0.98,
        0.98,
        "Interaction check: g = lambda - 1",
        transform=axis.transAxes,
        ha='right',
        va='top',
        fontsize=7.3,
        color='#334155',
        bbox=dict(
            boxstyle='round,pad=0.18',
            facecolor='#ffffff',
            edgecolor='#cbd5e1',
            alpha=0.92
        )
    )

    axis.set_xlabel(str(x_spec['axis_label']))
    axis.set_ylabel(str(y_spec['axis_label']))
    axis.set_title(
        f"Axial-Flexure Interaction | Custom Physical g(x) Map: {x_spec['short_label']} vs {y_spec['short_label']}",
        fontsize=10.8,
        pad=10
    )
    axis.grid(True, alpha=0.16, linestyle='--')
    axis.legend(loc='best', fontsize=7.5, frameon=True)

    fig.suptitle(
        f"Custom Nonlinear Limit-State Map | Element E{int(elem_id)}",
        fontsize=13,
        y=0.985
    )
    fig.tight_layout(rect=[0, 0, 0.93, 0.96])
    colorbar_axis = fig.add_axes([0.94, 0.16, 0.015, 0.68])
    fig.colorbar(
        contour_fill,
        cax=colorbar_axis,
        label='Limit-state function g(x)'
    )
    return fig


def build_limit_state_function_custom_axis_figure(physical_cloud_data: Dict[str, Any],
                                                  axial_moment_pm_cloud_data: Dict[str, Any],
                                                  input_data: Dict,
                                                  latest_result: Optional[Dict],
                                                  element_reliability: Optional[Dict[str, Dict[int, Dict[str, Any]]]],
                                                  elem_id: int,
                                                  limit_state: str,
                                                  x_axis_key: str,
                                                  y_axis_key: str) -> Optional[plt.Figure]:
    """Bangun peta fungsi limit-state custom dengan kedua sumbu dapat dipilih."""
    limit_state_key = str(limit_state).strip().lower()
    element_reliability = element_reliability or {}
    if limit_state_key == 'axial_moment':
        axial_moment_reliability = get_by_element_value(
            element_reliability.get('axial_moment', {}),
            int(elem_id),
            {}
        ) or {}
        return build_axial_moment_custom_axis_figure(
            axial_moment_pm_cloud_data=axial_moment_pm_cloud_data,
            input_data=input_data,
            beta_value=axial_moment_reliability.get('Beta'),
            elem_id=int(elem_id),
            x_axis_key=str(x_axis_key),
            y_axis_key=str(y_axis_key)
        )

    axis_specs = get_physical_limit_state_custom_axis_specs(limit_state_key)
    axis_lookup = {
        str(spec['key']): spec
        for spec in axis_specs
    }
    x_spec = axis_lookup.get(str(x_axis_key))
    y_spec = axis_lookup.get(str(y_axis_key))
    if x_spec is None or y_spec is None or str(x_spec['key']) == str(y_spec['key']):
        return None

    record = get_probabilistic_limit_state_physical_cloud_record(
        physical_cloud_data,
        int(elem_id),
        limit_state_key
    )
    demand_stat = get_physical_limit_state_demand_stat(record)
    capacity_stat = get_physical_limit_state_capacity_stat(record)
    if demand_stat is None:
        return None
    scatter_data = build_limit_state_physical_cloud_scatter_data(
        record,
        str(x_spec['key']),
        str(y_spec['key'])
    )
    scatter_x_values = np.asarray((scatter_data or {}).get('x', []), dtype=float)
    scatter_y_values = np.asarray((scatter_data or {}).get('y', []), dtype=float)

    section_inputs = get_section_capacity_inputs_from_input(input_data, int(elem_id))
    material_snapshot = get_element_material_snapshot(
        input_data,
        latest_simulation=None,
        is_probabilistic=True,
        elem_id=int(elem_id)
    )
    steel_props = get_by_element_value(
        input_data.get('steel', {}).get('by_element', {}),
        int(elem_id),
        {}
    ) or {}
    fy_geser_mean = read_positive_number(
        steel_props.get('geser_mean', steel_props.get('tarik_mean'))
    )
    if fy_geser_mean <= 0.0:
        fy_geser_mean = material_snapshot['fy_tarik']

    axial_branch = (
        get_latest_axial_limit_state_branch(latest_result, int(elem_id))
        if limit_state_key == 'axial' else
        'compression'
    )
    selected_axis_keys = {str(x_spec['key']), str(y_spec['key'])}
    reference_values = {
        'Q': float(demand_stat['mean']),
        'fc': float(material_snapshot['fc']),
        'fy_tarik': float(material_snapshot['fy_tarik']),
        'fy_tekan': float(material_snapshot['fy_tekan']),
        'fy_geser': float(fy_geser_mean)
    }
    exact_reference_capacity = evaluate_physical_limit_state_capacity_value(
        limit_state_key,
        section_inputs,
        float(reference_values['fc']),
        float(reference_values['fy_tarik']),
        float(reference_values['fy_tekan']),
        float(reference_values['fy_geser']),
        axial_branch=axial_branch
    )
    mean_capacity = coerce_finite_float(
        (capacity_stat or {}).get('mean')
    )
    if mean_capacity is None:
        mean_capacity = exact_reference_capacity
    if mean_capacity is None:
        return None
    reference_values['R'] = float(mean_capacity)

    def resolve_axis_limits(axis_spec: Dict[str, str]) -> Optional[Tuple[float, float]]:
        axis_kind = str(axis_spec.get('kind'))
        axis_key = str(axis_spec['key'])
        if axis_kind == 'material':
            material_stat = get_physical_limit_state_material_stat(
                input_data,
                int(elem_id),
                axis_key
            )
            return build_physical_limit_state_variable_limits(material_stat)
        if axis_kind == 'demand':
            return build_physical_limit_state_response_limits(demand_stat)
        if axis_kind == 'capacity':
            return build_physical_limit_state_response_limits(capacity_stat)
        return None

    x_limits = resolve_axis_limits(x_spec)
    y_limits = resolve_axis_limits(y_spec)
    if x_limits is None or y_limits is None:
        return None

    use_physical_qr_alignment = selected_axis_keys == {'Q', 'R'}
    if (
        use_physical_qr_alignment
        and scatter_x_values.size > 0
        and scatter_y_values.size > 0
    ):
        common_limits = get_failure_cloud_axis_limits(
            np.concatenate([scatter_x_values, scatter_y_values])
        )
        x_limits = common_limits
        y_limits = common_limits

    grid_x_values = np.linspace(float(x_limits[0]), float(x_limits[1]), PHYSICAL_LIMIT_STATE_FUNCTION_GRID_SIZE)
    grid_y_values = np.linspace(float(y_limits[0]), float(y_limits[1]), PHYSICAL_LIMIT_STATE_FUNCTION_GRID_SIZE)
    grid_x, grid_y = np.meshgrid(grid_x_values, grid_y_values)
    grid_g = np.full_like(grid_x, np.nan, dtype=float)

    for row_index in range(grid_y_values.size):
        for col_index in range(grid_x_values.size):
            current_values = dict(reference_values)
            current_values[str(x_spec['key'])] = float(grid_x[row_index, col_index])
            current_values[str(y_spec['key'])] = float(grid_y[row_index, col_index])

            demand_value = coerce_finite_float(current_values.get('Q'))
            if demand_value is None:
                continue

            if 'R' in selected_axis_keys:
                capacity_value = coerce_finite_float(current_values.get('R'))
            else:
                capacity_value = evaluate_physical_limit_state_capacity_value(
                    limit_state_key,
                    section_inputs,
                    float(current_values['fc']),
                    float(current_values['fy_tarik']),
                    float(current_values['fy_tekan']),
                    float(current_values['fy_geser']),
                    axial_branch=axial_branch
                )
            if capacity_value is None:
                continue

            if 'Q' not in selected_axis_keys:
                demand_value = float(reference_values['Q'])

            g_value = float(capacity_value) - float(demand_value)
            if np.isfinite(g_value):
                grid_g[row_index, col_index] = g_value

    finite_g = grid_g[np.isfinite(grid_g)]
    if finite_g.size == 0:
        return None

    fig, axis = plt.subplots(figsize=(9.0, 6.2), dpi=180)
    axis.set_facecolor('#f8fafc')

    max_abs_g = max(float(np.max(np.abs(finite_g))), 1e-6)
    fill_levels = np.linspace(-max_abs_g, max_abs_g, 19)
    contour_fill = axis.contourf(
        grid_x,
        grid_y,
        grid_g,
        levels=fill_levels,
        cmap='RdYlBu',
        alpha=0.92,
        antialiased=True
    )
    pin_contour_axes_to_grid(axis, grid_x, grid_y)
    if use_physical_qr_alignment and hasattr(axis, 'set_box_aspect'):
        axis.set_box_aspect(1.0)

    contour_step = build_nice_contour_step(max_abs_g)
    line_levels = contour_step * np.arange(-4, 5, dtype=float)
    nonzero_levels = np.asarray(
        [
            value for value in line_levels
            if (
                not np.isclose(value, 0.0, atol=max(contour_step * 1e-6, 1e-12))
                and abs(float(value)) <= max_abs_g * 1.02
            )
        ],
        dtype=float
    )
    if nonzero_levels.size > 0:
        iso_contours = axis.contour(
            grid_x,
            grid_y,
            grid_g,
            levels=np.sort(nonzero_levels),
            colors='#ffffff',
            linewidths=0.8,
            alpha=0.72
        )
        axis.clabel(
            iso_contours,
            fmt=lambda value: f"{float(value):.0f}",
            inline=True,
            fontsize=7
        )

    g_min = float(np.min(finite_g))
    g_max = float(np.max(finite_g))
    has_zero_crossing = bool(g_min <= 0.0 <= g_max)
    selected_zero_segment = None
    if has_zero_crossing:
        zero_contour = axis.contour(
            grid_x,
            grid_y,
            grid_g,
            levels=[0.0],
            colors=[PHYSICAL_NONLINEAR_CONTOUR_COLOR],
            linewidths=2.4
        )
        axis.clabel(
            zero_contour,
            fmt={0.0: 'g = 0'},
            inline=True,
            fontsize=8
        )
        zero_segment_info = select_primary_zero_contour_segment(
            list((zero_contour.allsegs or [[]])[0]),
            reference_x=np.asarray((scatter_data or {}).get('x', []), dtype=float),
            reference_y=np.asarray((scatter_data or {}).get('y', []), dtype=float)
        )
        selected_zero_segment = zero_segment_info.get('selected_segment')

    plot_limit_state_physical_cloud_scatter(
        axis,
        scatter_data,
        style_variant='physical_cloud' if use_physical_qr_alignment else 'default'
    )
    if (
        use_physical_qr_alignment
        and scatter_x_values.size > 0
        and scatter_y_values.size > 0
    ):
        reference_x = float(np.mean(scatter_x_values))
        reference_y = float(np.mean(scatter_y_values))
        reference_label = 'Sample Mean'
    else:
        reference_x = coerce_finite_float(reference_values.get(str(x_spec['key'])))
        reference_y = coerce_finite_float(reference_values.get(str(y_spec['key'])))
        reference_label = 'Mean reference point'
    if reference_x is not None and reference_y is not None:
        axis.scatter(
            [float(reference_x)],
            [float(reference_y)],
            marker='x',
            s=70,
            color='#111827',
            linewidths=1.6,
            zorder=6,
            label=reference_label
        )
    reliability_record = get_by_element_value(
        element_reliability.get(limit_state_key, {}),
        int(elem_id),
        {}
    ) or {}
    beta_raw_value = reliability_record.get('Beta')
    custom_map_mpp = resolve_material_space_beta_table_overlay(
        record,
        limit_state=limit_state_key,
        x_axis_key=str(x_spec['key']),
        y_axis_key=str(y_spec['key']),
        target_beta=beta_raw_value,
        zero_contour_segment=selected_zero_segment
    )
    custom_map_display_beta = coerce_finite_float(custom_map_mpp.get('display_beta'))
    custom_map_display_beta_label = format_beta_table_display(
        custom_map_mpp.get('display_beta_raw', custom_map_mpp.get('display_beta')),
        4
    )
    custom_map_mpp_x = coerce_finite_float(custom_map_mpp.get('contour_x'))
    custom_map_mpp_y = coerce_finite_float(custom_map_mpp.get('contour_y'))
    custom_map_sample_x = coerce_finite_float(custom_map_mpp.get('sample_x'))
    custom_map_sample_y = coerce_finite_float(custom_map_mpp.get('sample_y'))
    if (
        reference_x is not None
        and reference_y is not None
        and custom_map_display_beta is not None
        and custom_map_mpp_x is not None
        and custom_map_mpp_y is not None
    ):
        axis.plot(
            [float(reference_x), float(custom_map_mpp_x)],
            [float(reference_y), float(custom_map_mpp_y)],
            linestyle='--',
            linewidth=1.35,
            color='#7e22ce',
            alpha=0.94,
            zorder=6,
            label=f"Projected beta line, Beta(table)={custom_map_display_beta_label}"
        )
        if (
            custom_map_sample_x is not None
            and custom_map_sample_y is not None
            and not (
                np.isclose(custom_map_sample_x, custom_map_mpp_x, atol=1e-9, rtol=1e-9)
                and np.isclose(custom_map_sample_y, custom_map_mpp_y, atol=1e-9, rtol=1e-9)
            )
        ):
            axis.plot(
                [float(custom_map_sample_x), float(custom_map_mpp_x)],
                [float(custom_map_sample_y), float(custom_map_mpp_y)],
                linestyle=':',
                linewidth=1.0,
                color='#d946ef',
                alpha=0.88,
                zorder=6
            )
        axis.scatter(
            [float(custom_map_mpp_x)],
            [float(custom_map_mpp_y)],
            marker='*',
            s=175,
            color='#f59e0b',
            edgecolors='#111827',
            linewidths=0.80,
            zorder=7,
            label=f"Projected MPP, Beta(table)={custom_map_display_beta_label}"
        )
        add_smart_mpp_annotation(
            axis,
            "\n".join([
                "MPP",
                f"Beta(table)={custom_map_display_beta_label}",
                f"{x_spec['short_label']}={format_metric(custom_map_mpp_x, 2)}",
                f"{y_spec['short_label']}={format_metric(custom_map_mpp_y, 2)}"
            ]),
            target_xy=(float(custom_map_mpp_x), float(custom_map_mpp_y)),
            avoid_points=build_annotation_avoid_points(
                scatter_data=scatter_data,
                extra_points=[
                    (reference_x, reference_y),
                    (custom_map_sample_x, custom_map_sample_y),
                    (custom_map_mpp_x, custom_map_mpp_y)
                ],
                max_points=260
            ),
            bbox_edgecolor='#cbd5e1',
            text_color='#111827',
            fontsize=8.0,
            zorder=7,
            with_arrow=True
        )
    elif beta_raw_value is not None:
        axis.text(
            0.02,
            0.03,
            (
                "MPP/Beta(table) overlay unavailable\n"
                "for the current custom-axis contour."
            ),
            transform=axis.transAxes,
            ha='left',
            va='bottom',
            fontsize=7.1,
            color='#7c2d12',
            bbox=dict(
                boxstyle='round,pad=0.18',
                facecolor='#fff7ed',
                edgecolor='#fdba74',
                alpha=0.92
            ),
            zorder=7
        )

    if has_zero_crossing:
        axis.plot(
            [],
            [],
            color=PHYSICAL_NONLINEAR_CONTOUR_COLOR,
            linewidth=2.4,
            label='Limit-state contour g = 0'
        )
    else:
        axis.text(
            0.02,
            0.03,
            "No g = 0 contour in the current range",
            transform=axis.transAxes,
            ha='left',
            va='bottom',
            fontsize=7.3,
            color='#7c2d12',
            bbox=dict(
                boxstyle='round,pad=0.18',
                facecolor='#fff7ed',
                edgecolor='#fdba74',
                alpha=0.92
            )
        )

    if 'R' in selected_axis_keys:
        axis.text(
            0.98,
            0.98,
            "Direct R axis: g = R - Q",
            transform=axis.transAxes,
            ha='right',
            va='top',
            fontsize=7.3,
            color='#334155',
            bbox=dict(
                boxstyle='round,pad=0.18',
                facecolor='#ffffff',
                edgecolor='#cbd5e1',
                alpha=0.92
            )
        )

    reference_note = build_physical_limit_state_custom_axis_note(
        limit_state_key,
        axis_specs,
        (str(x_spec['key']), str(y_spec['key'])),
        reference_values,
        axial_branch=axial_branch
    )
    axis.text(
        0.02,
        0.98,
        reference_note,
        transform=axis.transAxes,
        ha='left',
        va='top',
        fontsize=7.2,
        color='#334155',
        bbox=dict(
            boxstyle='round,pad=0.20',
            facecolor='#ffffff',
            edgecolor='#cbd5e1',
            alpha=0.92
        )
    )

    state_display_spec = next(
        (
            spec for spec in get_probabilistic_limit_state_histogram_specs()
            if str(spec['key']) == limit_state_key
        ),
        {
            'plot_label': str(limit_state_key).replace('_', ' ').title()
        }
    )
    axis.set_xlabel(str(x_spec['axis_label']))
    axis.set_ylabel(str(y_spec['axis_label']))
    axis.set_title(
        (
            f"{state_display_spec.get('plot_label', str(limit_state_key).title())} | "
            f"Custom Physical g(x) Map: {x_spec['short_label']} vs {y_spec['short_label']}"
        ),
        fontsize=10.8,
        pad=10
    )
    axis.grid(True, alpha=0.22 if use_physical_qr_alignment else 0.16, linestyle='--')
    axis.legend(
        loc='best',
        fontsize=7.8 if use_physical_qr_alignment else 7.5,
        frameon=True
    )

    fig.suptitle(
        f"Custom Nonlinear Limit-State Map | Element E{int(elem_id)}",
        fontsize=13,
        y=0.985
    )
    fig.tight_layout(rect=[0, 0, 0.93, 0.96])
    colorbar_axis = fig.add_axes([0.94, 0.16, 0.015, 0.68])
    fig.colorbar(
        contour_fill,
        cax=colorbar_axis,
        label='Limit-state function g(x)'
    )
    return fig


def render_physical_limit_state_function_map_section(physical_cloud_data: Dict[str, Any],
                                                     axial_moment_pm_cloud_data: Dict[str, Any],
                                                     results_bundle: Dict[str, Any],
                                                     input_data: Dict,
                                                     elem_id: int,
                                                     heading_level: str = "####") -> None:
    """Render peta fungsi limit-state nonlinier pada ruang variabel fisik."""
    latest_result = (
        ((results_bundle or {}).get('latest_simulation', {}) or {}).get('analysis_result')
    )
    st.markdown(f"{heading_level} Nonlinear Limit-State Maps in Physical Space")
    st.caption(
        "Untuk lentur, geser, dan aksial, bidang `Q-R` memang cenderung linear karena "
        "`g = R - Q`. Agar fungsi limit nonlinier benar-benar terlihat, peta berikut "
        "digambar pada ruang variabel fisik asli penampang."
    )
    st.caption(
        "Anda sekarang juga bisa memilih pasangan sumbu custom, termasuk `R` dan `Q`, "
        "termasuk pasangan `Pd-Md/material` untuk `axial+flexure`, atau tetap memakai "
        "tampilan preset seperti `Demand-Material Space` dan `Material-Variable Space`."
    )
    st.caption(
        "Overlay titik SMC pada peta: bullet merah = sampel fail (`g(x) < 0`), "
        "bullet biru = sampel safe (`g(x) >= 0`)."
    )
    map_view_mode = st.radio(
        "Map view",
        options=[
            "Custom Axis Map",
            "Demand-Material Space",
            "Material-Variable Space"
        ],
        index=0,
        horizontal=True,
        key=f"physical_limit_state_map_view_mode_e{int(elem_id)}"
    )
    if map_view_mode == "Custom Axis Map":
        st.caption(
            "Pada mode ini, kedua sumbu dapat dipilih bebas. Jika `R` dipilih di salah "
            "satu sumbu, peta akan mengikuti hubungan langsung `g = R - Q`; sedangkan "
            "variabel yang tidak dipilih ditahan pada nilai mean referensinya."
        )
        available_limit_state_keys = {'moment', 'shear', 'axial'}
        has_axial_moment_data = bool(
            (axial_moment_pm_cloud_data or {}).get('elements', {}).get(str(int(elem_id)))
        )
        if has_axial_moment_data:
            available_limit_state_keys.add('axial_moment')

        limit_state_specs = [
            spec for spec in get_probabilistic_limit_state_histogram_specs()
            if str(spec['key']) in available_limit_state_keys
        ]
        limit_state_lookup = {
            str(spec['key']): str(spec.get('plot_label', spec['label']))
            for spec in limit_state_specs
        }
        control_columns = st.columns(3)
        with control_columns[0]:
            selected_limit_state = st.selectbox(
                "Limit state",
                options=list(limit_state_lookup.keys()),
                index=0,
                format_func=lambda value: limit_state_lookup.get(str(value), str(value)),
                key=f"physical_limit_state_custom_state_e{int(elem_id)}"
            )
        axis_specs = get_physical_limit_state_custom_axis_specs(selected_limit_state)
        axis_lookup = {
            str(spec['key']): spec
            for spec in axis_specs
        }
        axis_keys = list(axis_lookup.keys())
        default_x_index = axis_keys.index('Q') if 'Q' in axis_keys else 0
        with control_columns[1]:
            selected_x_axis = st.selectbox(
                "X-axis",
                options=axis_keys,
                index=default_x_index,
                format_func=lambda value: str(axis_lookup[str(value)]['label']),
                key=f"physical_limit_state_custom_x_axis_e{int(elem_id)}_{selected_limit_state}"
            )
        y_axis_options = [
            axis_key for axis_key in axis_keys
            if axis_key != str(selected_x_axis)
        ]
        default_y_axis = (
            'fc'
            if 'fc' in y_axis_options else
            y_axis_options[0]
        )
        with control_columns[2]:
            selected_y_axis = st.selectbox(
                "Y-axis",
                options=y_axis_options,
                index=y_axis_options.index(default_y_axis),
                format_func=lambda value: str(axis_lookup[str(value)]['label']),
                key=(
                    f"physical_limit_state_custom_y_axis_e{int(elem_id)}_"
                    f"{selected_limit_state}_{selected_x_axis}"
                )
            )
        if 'R' in {str(selected_x_axis), str(selected_y_axis)}:
            st.caption(
                "Catatan: saat `R` dipilih sebagai sumbu, contour mengikuti relasi langsung "
                "`g = R - Q`, sehingga bentuknya dapat menjadi lebih linear dibanding "
                "peta yang hanya memakai variabel material dan demand."
            )
        figure = build_limit_state_function_custom_axis_figure(
            physical_cloud_data=physical_cloud_data,
            axial_moment_pm_cloud_data=axial_moment_pm_cloud_data,
            input_data=input_data,
            latest_result=latest_result,
            element_reliability=(results_bundle or {}).get('element_reliability', {}),
            elem_id=int(elem_id),
            limit_state=str(selected_limit_state),
            x_axis_key=str(selected_x_axis),
            y_axis_key=str(selected_y_axis)
        )
        viewer_key = (
            f"physical-limit-state-custom-map-e{int(elem_id)}-"
            f"{selected_limit_state}-{selected_x_axis}-{selected_y_axis}"
        )
        download_basename = (
            f"custom-nonlinear-limit-state-map-e{int(elem_id)}-"
            f"{selected_limit_state}-{selected_x_axis}-{selected_y_axis}"
        )
        alt_text = (
            f"Custom nonlinear limit-state map for element {int(elem_id)} "
            f"in {selected_limit_state} with axes {selected_x_axis} and {selected_y_axis}"
        )
    elif map_view_mode == "Demand-Material Space":
        st.caption(
            "Pada mode ini, sumbu horizontal adalah demand fisik, sedangkan sumbu vertikal "
            "adalah satu variabel material fisik. Ini lebih dekat ke contoh `g(Md, fc)`, "
            "`g(Vd, fc)`, dan `g(Pd, fc)`."
        )
        figure = build_limit_state_function_demand_material_space_figure(
            physical_cloud_data=physical_cloud_data,
            input_data=input_data,
            latest_result=latest_result,
            element_reliability=(results_bundle or {}).get('element_reliability', {}),
            elem_id=int(elem_id)
        )
        viewer_key = f"physical-limit-state-demand-material-maps-e{int(elem_id)}"
        download_basename = f"nonlinear-limit-state-demand-material-space-e{int(elem_id)}"
        alt_text = (
            f"Nonlinear limit-state maps in demand-material space for element {int(elem_id)}"
        )
    else:
        st.caption(
            "Pada mode ini, demand dipatok pada nilai rata-rata cloud Monte Carlo valid, "
            "lalu `g(x)` dievaluasi langsung terhadap pasangan variabel material fisik. "
            "Marker `MPP` dan `projected beta line` berbasis `Beta(table)` juga ditampilkan "
            "pada contour `g=0` agar konsisten dengan panel Failure Cloud di Ruang Fisik."
        )
        figure = build_limit_state_function_physical_space_figure(
            physical_cloud_data=physical_cloud_data,
            input_data=input_data,
            latest_result=latest_result,
            element_reliability=(results_bundle or {}).get('element_reliability', {}),
            elem_id=int(elem_id)
        )
        viewer_key = f"physical-limit-state-material-variable-maps-e{int(elem_id)}"
        download_basename = f"nonlinear-limit-state-material-variable-space-e{int(elem_id)}"
        alt_text = (
            f"Nonlinear limit-state maps in material-variable space for element {int(elem_id)}"
        )
    if figure is None:
        st.info(
            "Peta fungsi limit-state nonlinier untuk lentur, geser, aksial, atau axial-flexure "
            "belum dapat dibentuk pada elemen terpilih."
        )
        return

    render_plot(
        figure,
        interactive=True,
        viewer_key=viewer_key,
        alt_text=alt_text,
        viewer_height=680,
        download_basename=download_basename
    )


def build_probabilistic_axial_moment_pm_cloud_figure(
    axial_moment_pm_cloud_data: Dict[str, Any],
    elem_id: int,
    beta_value: Optional[float] = None,
    show_envelope: bool = True,
    show_nonlinear_contour: bool = True,
    current_demand_moment: Optional[float] = None,
    current_demand_axial: Optional[float] = None,
    current_boundary_moment: Optional[float] = None,
    current_boundary_axial: Optional[float] = None,
    active_curve_moment: Optional[List[float]] = None,
    active_curve_axial: Optional[List[float]] = None,
    active_boundary_label: str = 'Active boundary point'
) -> Optional[plt.Figure]:
    """Bangun figure contour nonlinier aksial-lentur yang disederhanakan pada ruang `P-M`."""
    record = (
        (axial_moment_pm_cloud_data or {}).get('elements', {}).get(str(int(elem_id)))
        or {}
    )
    if not record:
        return None

    demand_moment = np.asarray(record.get('demand_moment', []), dtype=float)
    demand_axial = np.asarray(record.get('demand_axial', []), dtype=float)
    boundary_moment = np.asarray(record.get('boundary_moment', []), dtype=float)
    boundary_axial = np.asarray(record.get('boundary_axial', []), dtype=float)
    g_values = np.asarray(record.get('g', []), dtype=float)
    common_size = min(
        int(demand_moment.size),
        int(demand_axial.size),
        int(boundary_moment.size),
        int(boundary_axial.size),
        int(g_values.size)
    )
    if common_size <= 0:
        return None

    demand_moment = demand_moment[:common_size]
    demand_axial = demand_axial[:common_size]
    boundary_moment = boundary_moment[:common_size]
    boundary_axial = boundary_axial[:common_size]
    g_values = g_values[:common_size]

    valid_mask = (
        np.isfinite(demand_moment)
        & np.isfinite(demand_axial)
        & np.isfinite(boundary_moment)
        & np.isfinite(boundary_axial)
        & np.isfinite(g_values)
    )
    if not np.any(valid_mask):
        return None

    demand_moment = demand_moment[valid_mask]
    demand_axial = demand_axial[valid_mask]
    boundary_moment = boundary_moment[valid_mask]
    boundary_axial = boundary_axial[valid_mask]
    g_values = g_values[valid_mask]
    failure_mask = np.asarray(g_values < 0.0, dtype=bool)
    safe_mask = ~failure_mask

    active_curve_moment_array = np.asarray(active_curve_moment or [], dtype=float)
    active_curve_axial_array = np.asarray(active_curve_axial or [], dtype=float)
    active_curve_size = min(
        int(active_curve_moment_array.size),
        int(active_curve_axial_array.size)
    )
    active_curve_moment_array = active_curve_moment_array[:active_curve_size]
    active_curve_axial_array = active_curve_axial_array[:active_curve_size]
    active_curve_valid = (
        np.isfinite(active_curve_moment_array)
        & np.isfinite(active_curve_axial_array)
    )
    active_curve_moment_array = active_curve_moment_array[active_curve_valid]
    active_curve_axial_array = active_curve_axial_array[active_curve_valid]

    current_demand_moment = coerce_finite_float(current_demand_moment)
    current_demand_axial = coerce_finite_float(current_demand_axial)
    current_boundary_moment = coerce_finite_float(current_boundary_moment)
    current_boundary_axial = coerce_finite_float(current_boundary_axial)

    overlay_x_values = np.asarray(
        [
            value for value in (
                current_demand_moment,
                current_boundary_moment
            )
            if value is not None
        ],
        dtype=float
    )
    overlay_y_values = np.asarray(
        [
            value for value in (
                current_demand_axial,
                current_boundary_axial
            )
            if value is not None
        ],
        dtype=float
    )

    def build_boundary_envelope_band(x_values: np.ndarray,
                                     y_values: np.ndarray,
                                     lower_quantile: float = 0.10,
                                     upper_quantile: float = 0.90) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bangun band statistik lokal `P10-P90` dari cloud boundary aktif Monte Carlo."""
        finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
        if int(np.sum(finite_mask)) < 4:
            empty = np.asarray([], dtype=float)
            return empty, empty, empty

        x_valid = np.asarray(x_values[finite_mask], dtype=float)
        y_valid = np.asarray(y_values[finite_mask], dtype=float)
        sort_order = np.lexsort((y_valid, x_valid))
        x_sorted = x_valid[sort_order]
        y_sorted = y_valid[sort_order]
        sample_count = int(x_sorted.size)
        min_points_per_group = 8
        max_groups = max(sample_count // min_points_per_group, 1)
        group_count = int(np.clip(max_groups, 4, 24))
        grouped_indices = [
            group_indices
            for group_indices in np.array_split(np.arange(sample_count, dtype=int), group_count)
            if int(group_indices.size) > 0
        ]
        if len(grouped_indices) < 2:
            empty = np.asarray([], dtype=float)
            return empty, empty, empty

        band_x = []
        band_lower = []
        band_upper = []
        for group_indices in grouped_indices:
            x_group = x_sorted[group_indices]
            y_group = y_sorted[group_indices]
            if x_group.size == 0 or y_group.size == 0:
                continue
            band_x.append(float(np.median(x_group)))
            band_lower.append(float(np.quantile(y_group, lower_quantile)))
            band_upper.append(float(np.quantile(y_group, upper_quantile)))

        if len(band_x) < 2:
            empty = np.asarray([], dtype=float)
            return empty, empty, empty

        band_x_array = np.asarray(band_x, dtype=float)
        band_lower_array = np.asarray(band_lower, dtype=float)
        band_upper_array = np.asarray(band_upper, dtype=float)
        valid_band_mask = (
            np.isfinite(band_x_array)
            & np.isfinite(band_lower_array)
            & np.isfinite(band_upper_array)
        )
        if int(np.sum(valid_band_mask)) < 2:
            empty = np.asarray([], dtype=float)
            return empty, empty, empty

        band_x_array = band_x_array[valid_band_mask]
        band_lower_array = band_lower_array[valid_band_mask]
        band_upper_array = band_upper_array[valid_band_mask]
        band_lower_original = band_lower_array.copy()
        band_upper_original = band_upper_array.copy()
        band_lower_array = np.minimum(band_lower_original, band_upper_original)
        band_upper_array = np.maximum(band_lower_original, band_upper_original)
        return band_x_array, band_lower_array, band_upper_array

    envelope_moment, envelope_axial_lower, envelope_axial_upper = build_boundary_envelope_band(
        boundary_moment,
        boundary_axial
    )

    full_x_values = np.concatenate(
        [
            demand_moment,
            boundary_moment,
            active_curve_moment_array if active_curve_moment_array.size else np.asarray([], dtype=float),
            overlay_x_values
        ]
    )
    full_y_values = np.concatenate(
        [
            demand_axial,
            boundary_axial,
            active_curve_axial_array if active_curve_axial_array.size else np.asarray([], dtype=float),
            overlay_y_values
        ]
    )
    full_x_limits = get_failure_cloud_axis_limits(full_x_values)
    full_y_limits = get_failure_cloud_axis_limits(full_y_values)

    zoom_x_values = np.concatenate([demand_moment, boundary_moment, overlay_x_values])
    zoom_y_values = np.concatenate([demand_axial, boundary_axial, overlay_y_values])
    zoom_x_limits = get_failure_cloud_axis_limits(zoom_x_values)
    zoom_y_limits = get_failure_cloud_axis_limits(zoom_y_values)

    surface_full = None
    surface_zoom = None
    surface_max_abs_g = None
    surface_status_text = None
    surface_has_zero_crossing = False
    if show_nonlinear_contour:
        surface_model = fit_physical_signed_margin_quadratic_surface(
            demand_moment,
            demand_axial,
            g_values,
            boundary_x=boundary_moment,
            boundary_y=boundary_axial
        )
        if surface_model is not None:
            surface_full = evaluate_physical_signed_margin_surface_grid(
                surface_model,
                x_limits=full_x_limits,
                y_limits=full_y_limits
            )
            surface_zoom = evaluate_physical_signed_margin_surface_grid(
                surface_model,
                x_limits=zoom_x_limits,
                y_limits=zoom_y_limits
            )
        if surface_full is not None or surface_zoom is not None:
            surface_finite_values = []
            for surface_record in (surface_full, surface_zoom):
                if not surface_record:
                    continue
                candidate_values = np.asarray(surface_record.get('grid_g', []), dtype=float)
                candidate_values = candidate_values[np.isfinite(candidate_values)]
                if candidate_values.size > 0:
                    surface_finite_values.append(float(np.max(np.abs(candidate_values))))
            if surface_finite_values:
                surface_max_abs_g = max(max(surface_finite_values), 1e-6)
        surface_has_zero_crossing = any(
            bool(
                surface_record
                and float(surface_record.get('g_min', 1.0)) <= 0.0
                and float(surface_record.get('g_max', -1.0)) >= 0.0
            )
            for surface_record in (surface_full, surface_zoom)
        )
        if surface_max_abs_g is None:
            surface_status_text = (
                "The nonlinear contour `g_hat(M,P)=0` is not stable enough yet "
                "to be built from the current sample cloud."
            )
        elif not surface_has_zero_crossing:
            surface_status_text = (
                "The surrogate `g_hat(M,P)` has been fitted, but within the current plot range "
                "it does not cross `g=0`, so the zero contour is not shown."
            )

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.9), dpi=180)
    axes_list = list(np.asarray(axes).reshape(-1))
    panel_surface_grids = [surface_full, surface_zoom]
    multiple_zero_branch_detected = False

    for axis, surface_grid in zip(axes_list, panel_surface_grids):
        axis.set_facecolor('#f8fafc')
        if show_nonlinear_contour and surface_grid is not None:
            contour_result = add_physical_signed_margin_contours(
                axis,
                surface_grid,
                max_abs_g=surface_max_abs_g,
                zero_reference_x=boundary_moment,
                zero_reference_y=boundary_axial,
                show_fill=True,
                show_nonzero_guides=False
            )
            if contour_result.get('zero_contour_drawn'):
                if int(contour_result.get('zero_contour_branch_count', 0) or 0) > 1:
                    multiple_zero_branch_detected = True
                axis.plot(
                    [],
                    [],
                    color=PHYSICAL_NONLINEAR_CONTOUR_COLOR,
                    linewidth=2.2,
                    label=(
                        'Principal nonlinear contour g_hat(M, P)=0'
                        if contour_result.get('zero_contour_selection_applied') else
                        'Nonlinear contour g_hat(M, P)=0'
                    )
                )

        if np.any(safe_mask):
            axis.scatter(
                demand_moment[safe_mask],
                demand_axial[safe_mask],
                s=18,
                color=SAFE_CLOUD_COLOR,
                alpha=0.30,
                edgecolors='none',
                zorder=2.1,
                label=f"Safe cloud ({int(np.sum(safe_mask)):,})"
            )
        if np.any(failure_mask):
            axis.scatter(
                demand_moment[failure_mask],
                demand_axial[failure_mask],
                s=28,
                color='#dc2626',
                alpha=0.80,
                edgecolors='#ffffff',
                linewidths=0.25,
                zorder=2.2,
                label=f"Fail cloud ({int(np.sum(failure_mask)):,})"
            )
        if show_envelope and envelope_moment.size > 1:
            axis.fill_between(
                envelope_moment,
                envelope_axial_lower,
                envelope_axial_upper,
                facecolor='#9ca3af',
                edgecolor='none',
                linewidth=0.0,
                alpha=0.22,
                zorder=2.0,
                label='Envelope boundary'
            )
        if active_curve_moment_array.size > 1:
            axis.plot(
                active_curve_moment_array,
                active_curve_axial_array,
                color='#2563eb',
                linestyle='-',
                linewidth=1.9,
                alpha=0.88,
                label='Active interaction curve (current sample)'
            )
        if (
            current_demand_moment is not None
            and current_demand_axial is not None
            and current_boundary_moment is not None
            and current_boundary_axial is not None
        ):
            axis.plot(
                [current_demand_moment, current_boundary_moment],
                [current_demand_axial, current_boundary_axial],
                color='#2a9d8f',
                linestyle='-.',
                linewidth=1.35,
                alpha=0.92,
                label='Direction line to active boundary'
            )
            axis.scatter(
                [current_boundary_moment],
                [current_boundary_axial],
                color='#ffffff',
                marker='*',
                s=260,
                linewidths=0.0,
                zorder=6.8,
                label='_nolegend_'
            )
            axis.scatter(
                [current_boundary_moment],
                [current_boundary_axial],
                color='#7c3aed',
                marker='*',
                s=185,
                edgecolors='#7c3aed',
                linewidths=0.6,
                zorder=7.0,
                label=str(active_boundary_label)
            )
            axis.scatter(
                [current_boundary_moment],
                [current_boundary_axial],
                color='#581c87',
                marker='o',
                s=24,
                edgecolors='#ffffff',
                linewidths=0.70,
                zorder=7.2,
                label='_nolegend_'
            )
        if current_demand_moment is not None and current_demand_axial is not None:
            axis.scatter(
                [current_demand_moment],
                [current_demand_axial],
                marker='X',
                s=116,
                color='#111827',
                edgecolors='#ffffff',
                linewidths=0.75,
                zorder=6.4,
                label='Current point'
            )

        axis.set_xlabel('Moment M (kN.m)')
        axis.set_ylabel('Axial P (kN)')
        axis.grid(True, alpha=0.22, linestyle='--')

    axes_list[0].set_xlim(*full_x_limits)
    axes_list[0].set_ylim(*full_y_limits)
    axes_list[0].set_title('Physical P-M Space | Full View', fontsize=11, pad=10)

    axes_list[1].set_xlim(*zoom_x_limits)
    axes_list[1].set_ylim(*zoom_y_limits)
    axes_list[1].set_title(
        'Physical P-M Space | Zoomed Current Point and g=0 Contour',
        fontsize=11,
        pad=10
    )
    axes_list[1].text(
        0.98,
        0.96,
        (
            f"Valid N = {int(record.get('sample_count', 0))}\n"
            f"Failures = {int(record.get('failure_count', 0))}\n"
            f"Pf(valid) = {float(record.get('Pf_from_g', 0.0)):.4f}\n"
            f"Current Point Md = {format_metric(current_demand_moment, 2)} kN.m\n"
            f"Current Point Pd = {format_metric(current_demand_axial, 2)} kN"
        ),
        transform=axes_list[1].transAxes,
        ha='right',
        va='top',
        fontsize=8.2,
        bbox=dict(
            boxstyle='round,pad=0.25',
            facecolor='white',
            alpha=0.90,
            edgecolor='#cbd5e1'
        )
    )
    if surface_status_text is not None:
        axes_list[0].text(
            0.02,
            0.02,
            surface_status_text,
            transform=axes_list[0].transAxes,
            ha='left',
            va='bottom',
            fontsize=7.8,
            color='#7c2d12',
            bbox=dict(
                boxstyle='round,pad=0.22',
                facecolor='#fff7ed',
                edgecolor='#fdba74',
                alpha=0.92
            )
        )
    elif multiple_zero_branch_detected:
        axes_list[0].text(
            0.02,
            0.02,
            (
                "Several mathematical branches of `g_hat(M,P)=0` were detected.\n"
                "Only the branch closest to the sampled boundary cloud is shown."
            ),
            transform=axes_list[0].transAxes,
            ha='left',
            va='bottom',
            fontsize=7.7,
            color='#334155',
            bbox=dict(
                boxstyle='round,pad=0.22',
                facecolor='#ffffff',
                edgecolor='#cbd5e1',
                alpha=0.93
            )
        )

    legend_handles, legend_labels = axes_list[1].get_legend_handles_labels()
    beta_text = format_metric(beta_value, 4) if beta_value is not None else '-'
    legend_handles.append(
        Line2D([], [], linestyle='None', marker='', color='none')
    )
    legend_labels.append(f"Beta (axial-flexure) = {beta_text}")
    axes_list[1].legend(
        legend_handles,
        legend_labels,
        loc='best',
        fontsize=7.3
    )

    fig.suptitle(
        f"Nonlinear Contour g(x)=0 | Column Element E{int(elem_id)} | Physical P-M Space",
        fontsize=13,
        y=0.99
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def render_probabilistic_axial_moment_pm_cloud_section(
    axial_moment_pm_cloud_data: Dict[str, Any],
    element_reliability: Optional[Dict[str, Dict[int, Dict[str, Any]]]],
    input_data: Optional[Dict],
    latest_simulation: Optional[Dict[str, Any]],
    latest_result: Optional[Dict[str, Any]],
    elem_id: int,
    heading_level: str = "####"
) -> None:
    """Render the axial-flexure figure in physical `P-M` space."""
    fallback_elem_id = int(elem_id)
    available_column_ids = sorted(
        int(candidate_id)
        for candidate_id in list((axial_moment_pm_cloud_data or {}).get('element_ids', []) or [])
        if (
            get_element_code_from_input(input_data, int(candidate_id)) == 'K'
            and (axial_moment_pm_cloud_data or {}).get('elements', {}).get(str(int(candidate_id)))
        )
    )

    st.markdown(f"{heading_level} Contour Nonlinier g(x)=0 in Physical Space")
    st.caption(
        "Panel ini menggambarkan `nonlinear contour g(x)=0` pada ruang fisik `P-M` "
        "untuk elemen kolom, dibangun dari sampel probabilistik yang tersimpan."
    )
    st.caption(
        "Plot ini mempertahankan titik cloud `safe/fail`, marker `current point`, "
        "latar contour berwarna `g(x)`, `direction line`, dan `active interaction curve`. "
        "Arsiran abu-abu `envelope boundary` adalah band statistik lokal `P10-P90` dan bisa "
        "ditampilkan atau disembunyikan."
    )
    st.caption(
        "Garis panduan `Current Md` dan `Current Pd` tidak ditampilkan. Posisi simulasi "
        "aktif ditunjukkan langsung oleh marker `current point`, sedangkan `direction line` "
        "tetap diarahkan ke boundary aktif yang sekarang diberi marker bintang ungu."
    )
    st.caption(
        "Jika data simulasi aktif tersedia, dashboard menampilkan `active interaction curve` "
        "sebagai garis biru solid agar marker boundary aktif tepat punya kurva acuan yang "
        "konsisten."
    )
    st.caption(
        "Jika surrogate kuadratik membentuk lebih dari satu cabang matematis `g_hat(M,P)=0`, "
        "dashboard hanya menampilkan cabang utama yang paling dekat ke cloud boundary sampel "
        "agar interpretasi visual tetap stabil."
    )

    if not available_column_ids:
        st.info(
            "Contour nonlinier ruang fisik hanya tersedia untuk elemen kolom yang memiliki "
            "dataset aksial-lentur probabilistik."
        )
        return

    default_elem_id = (
        fallback_elem_id
        if fallback_elem_id in available_column_ids else
        available_column_ids[0]
    )
    selected_pm_elem_id = st.selectbox(
        "Pilih elemen kolom untuk contour nonlinier ruang fisik",
        options=available_column_ids,
        index=available_column_ids.index(int(default_elem_id)),
        format_func=lambda candidate_id: f"E{int(candidate_id)} | Kolom",
        key="physical_nonlinear_contour_column_selector"
    )
    selected_pm_elem_id = int(selected_pm_elem_id)
    record = (
        (axial_moment_pm_cloud_data or {}).get('elements', {}).get(str(selected_pm_elem_id))
        or {}
    )
    if not record:
        st.info(
            "Dataset aksial-lentur probabilistik untuk kolom terpilih belum dapat dibaca."
        )
        return

    show_envelope = st.checkbox(
        "Tampilkan envelope boundary sampel (P10-P90)",
        value=True,
        key=f"axial_moment_pm_envelope_toggle_e{selected_pm_elem_id}"
    )
    show_nonlinear_contour = st.checkbox(
        "Tampilkan contour nonlinier g(x)=0 dari sampel SMC",
        value=True,
        key=f"axial_moment_pm_contour_toggle_e{selected_pm_elem_id}"
    )
    beta_value = (
        get_by_element_value(
            (element_reliability or {}).get('axial_moment', {}),
            selected_pm_elem_id,
            {}
        ) or {}
    ).get('Beta')
    current_overlay = get_current_axial_moment_physical_overlay_from_latest_result(
        input_data,
        latest_simulation,
        latest_result,
        int(selected_pm_elem_id)
    )

    pm_fig = build_probabilistic_axial_moment_pm_cloud_figure(
        axial_moment_pm_cloud_data,
        selected_pm_elem_id,
        beta_value=beta_value,
        show_envelope=bool(show_envelope),
        show_nonlinear_contour=bool(show_nonlinear_contour),
        current_demand_moment=(
            None if not current_overlay else current_overlay.get('demand_moment')
        ),
        current_demand_axial=(
            None if not current_overlay else current_overlay.get('demand_axial')
        ),
        current_boundary_moment=(
            None if not current_overlay else current_overlay.get('boundary_moment')
        ),
        current_boundary_axial=(
            None if not current_overlay else current_overlay.get('boundary_axial')
        ),
        active_curve_moment=(
            [] if not current_overlay else list(current_overlay.get('active_curve_moment', []) or [])
        ),
        active_curve_axial=(
            [] if not current_overlay else list(current_overlay.get('active_curve_axial', []) or [])
        ),
        active_boundary_label=(
            'Active boundary point (exact)'
            if (current_overlay or {}).get('boundary_source') == 'exact' else
            'Active boundary point (interp)'
        )
    )
    if pm_fig is not None:
        render_plot(
            pm_fig,
            interactive=True,
            viewer_key=f"axial-moment-pm-cloud-e{selected_pm_elem_id}",
            alt_text=f"Contour nonlinier ruang fisik P-M elemen {selected_pm_elem_id}",
            viewer_height=720,
            download_basename=f"contour-nonlinier-ruang-fisik-pm-e{selected_pm_elem_id}"
        )
    else:
        st.info("Contour nonlinier ruang fisik untuk kolom terpilih belum dapat dibentuk.")


def build_output_reliability_beta_sketch_df(
    element_reliability: Optional[Dict[str, Dict[int, Dict[str, Any]]]],
    elem_id: int
) -> pd.DataFrame:
    """Bangun tabel ringkas Pf/Beta per limit-state untuk satu elemen."""
    rows = []
    element_reliability = element_reliability or {}

    for spec in get_probabilistic_limit_state_histogram_specs():
        reliability_record = get_by_element_value(
            element_reliability.get(str(spec['key']), {}),
            int(elem_id),
            {}
        ) or {}
        if not reliability_record:
            continue

        pf_value = reliability_record.get('Pf')
        beta_value = reliability_record.get('Beta')
        rows.append({
            'Limit State': spec['label'],
            'Pf (-)': pf_value,
            'Beta (-)': beta_value,
            'Jumlah Gagal (-)': reliability_record.get('failures'),
            'Level Risiko': describe_probabilistic_risk_level(pf_value, beta_value)
        })

    return pd.DataFrame(rows)


def build_output_reliability_beta_sketch_figure(
    element_reliability: Optional[Dict[str, Dict[int, Dict[str, Any]]]],
    elem_id: int,
    target_beta: float = 3.0
) -> Optional[plt.Figure]:
    """Sketsa batang nilai Beta tabel reliability untuk satu elemen."""
    summary_df = build_output_reliability_beta_sketch_df(
        element_reliability,
        int(elem_id)
    )
    if summary_df.empty:
        return None

    labels = summary_df['Limit State'].astype(str).tolist()
    pf_values = [
        coerce_finite_float(value)
        for value in summary_df['Pf (-)'].tolist()
    ]
    beta_raw_values = summary_df['Beta (-)'].tolist()
    beta_numeric_values: List[Optional[float]] = []
    finite_beta_values: List[float] = []
    has_positive_infinity = False
    has_negative_infinity = False

    for raw_value in beta_raw_values:
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            numeric_value = None
        if numeric_value is None or np.isnan(numeric_value):
            beta_numeric_values.append(None)
            continue
        if np.isposinf(numeric_value):
            has_positive_infinity = True
        elif np.isneginf(numeric_value):
            has_negative_infinity = True
        elif np.isfinite(numeric_value):
            finite_beta_values.append(float(numeric_value))
        beta_numeric_values.append(float(numeric_value))

    y_min = min([-0.4, target_beta - 3.4] + finite_beta_values) if finite_beta_values else -0.4
    y_max = max([target_beta + 1.0] + finite_beta_values) if finite_beta_values else (target_beta + 1.0)
    if has_negative_infinity:
        y_min = min(y_min, -2.0)
    if has_positive_infinity:
        y_max = max(y_max, target_beta + 1.8)
    if np.isclose(y_min, y_max, atol=1e-12, rtol=1e-9):
        y_max = y_min + 1.0

    y_padding = max((y_max - y_min) * 0.12, 0.45)
    plot_min = y_min - y_padding * 0.35
    plot_max = y_max + y_padding
    plotted_beta_values = []
    for numeric_value in beta_numeric_values:
        if numeric_value is None:
            plotted_beta_values.append(np.nan)
        elif np.isposinf(numeric_value):
            plotted_beta_values.append(plot_max - y_padding * 0.35)
        elif np.isneginf(numeric_value):
            plotted_beta_values.append(plot_min + y_padding * 0.35)
        else:
            plotted_beta_values.append(float(numeric_value))

    bar_colors = [
        RISK_LEVEL_COLORS.get(
            describe_probabilistic_risk_level(pf_value, raw_beta),
            RISK_LEVEL_COLORS['Tidak Ada Data']
        )
        for pf_value, raw_beta in zip(pf_values, beta_raw_values)
    ]

    fig, axis = plt.subplots(figsize=(10.4, 5.8), dpi=180)
    positions = np.arange(len(labels), dtype=float)
    bars = axis.bar(
        positions,
        plotted_beta_values,
        color=bar_colors,
        alpha=0.88,
        edgecolor='#0f172a',
        linewidth=0.65
    )
    axis.axhline(
        float(target_beta),
        color='#111827',
        linestyle='--',
        linewidth=1.25,
        label=f'Target Beta = {target_beta:.1f}'
    )
    axis.axhline(
        0.0,
        color='#94a3b8',
        linestyle=':',
        linewidth=1.0,
        alpha=0.95
    )

    for bar, beta_value, plotted_value, pf_value in zip(
        bars,
        beta_numeric_values,
        plotted_beta_values,
        pf_values
    ):
        if beta_value is None or not np.isfinite(float(plotted_value)):
            continue
        if np.isposinf(beta_value):
            beta_text = 'Infinity'
        elif np.isneginf(beta_value):
            beta_text = '-Infinity'
        else:
            beta_text = f"{float(beta_value):.3f}"
        pf_text = '-' if pf_value is None else f"Pf={pf_value:.2e}"
        text_offset = y_padding * 0.08
        if float(plotted_value) >= 0.0:
            y_text = float(plotted_value) + text_offset
            vertical_alignment = 'bottom'
        else:
            y_text = float(plotted_value) - text_offset
            vertical_alignment = 'top'
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_text,
            f"{beta_text}\n{pf_text}",
            ha='center',
            va=vertical_alignment,
            fontsize=8.2,
            color='#0f172a'
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel('Indeks Keandalan, Beta (-)')
    axis.set_title(f"Sketsa Beta Reliability | Element E{int(elem_id)}", fontsize=12, pad=12)
    axis.set_ylim(plot_min, plot_max)
    axis.grid(True, axis='y', alpha=0.22, linestyle='--')
    axis.legend(loc='upper right', fontsize=8.4)
    fig.tight_layout()
    return fig


def render_output_reliability_beta_sketch_section(
    results_bundle: Dict[str, Any],
    input_data: Optional[Dict],
    heading_level: str = "####"
) -> None:
    """Render sketsa visual nilai Beta yang tampil pada tabel reliability."""
    element_reliability = (results_bundle or {}).get('element_reliability', {}) or {}
    available_element_ids = sorted({
        int(elem_id)
        for state_name in ('moment', 'shear', 'axial', 'axial_moment')
        for elem_id in (element_reliability.get(state_name, {}) or {}).keys()
    })
    if not available_element_ids:
        st.info("Sketsa Beta belum tersedia karena data reliability per elemen belum ada.")
        return

    st.markdown(f"{heading_level} Sketsa Beta per Elemen")
    st.caption(
        "Grafik ini memvisualkan nilai `Beta` yang sama dengan tabel `Output Reliability`, "
        "sehingga perbandingan antar limit state pada satu elemen lebih cepat terbaca."
    )
    st.caption(
        "Garis putus-putus `beta = 3.0` dipakai sebagai acuan target ULS. "
        "Warna batang mengikuti level risiko yang diturunkan dari kombinasi `Pf/Beta`."
    )

    selected_element_id = st.selectbox(
        "Pilih elemen untuk sketsa Beta",
        options=available_element_ids,
        format_func=lambda elem_id: (
            f"E{int(elem_id)} | "
            f"{get_element_type_label(get_element_code_from_input(input_data, int(elem_id)))}"
        ),
        key="output_reliability_beta_selector"
    )

    beta_fig = build_output_reliability_beta_sketch_figure(
        element_reliability,
        int(selected_element_id)
    )
    if beta_fig is not None:
        render_plot(
            beta_fig,
            interactive=True,
            viewer_key=f"output-reliability-beta-e{int(selected_element_id)}",
            alt_text=f"Sketsa beta reliability elemen {int(selected_element_id)}",
            viewer_height=620,
            download_basename=f"sketsa-beta-reliability-e{int(selected_element_id)}"
        )
    else:
        st.info("Sketsa Beta untuk elemen terpilih belum dapat dibentuk.")

    beta_summary_df = build_output_reliability_beta_sketch_df(
        element_reliability,
        int(selected_element_id)
    )
    if beta_summary_df.empty:
        st.info("Ringkasan Beta untuk elemen terpilih belum tersedia.")
    else:
        render_input_table(
            beta_summary_df,
            styler=style_input_dataframe(
                beta_summary_df,
                table_min_width_px=1050
            )
        )


def get_failure_cloud_variable_prefix_specs() -> Dict[str, Dict[str, str]]:
    """Metadata label variabel acak untuk tab Failure Cloud."""
    return {
        'fb': {
            'label': 'Faktor Bias Modulus E',
            'unit': '(-)'
        },
        'E': {
            'label': 'Modulus Elastisitas E',
            'unit': 'MPa'
        },
        'fc': {
            'label': 'Mutu Beton fc',
            'unit': 'MPa'
        },
        'fy_tarik': {
            'label': 'fy Tarik',
            'unit': 'MPa'
        },
        'fy_tekan': {
            'label': 'fy Tekan',
            'unit': 'MPa'
        },
        'fy_geser': {
            'label': 'fy Geser',
            'unit': 'MPa'
        },
        'qDL': {
            'label': 'Beban Mati qDL',
            'unit': 'kN/m'
        },
        'qLL': {
            'label': 'Beban Hidup qLL',
            'unit': 'kN/m'
        }
    }


def extract_variable_prefix_from_name(variable_name: str) -> str:
    """Ambil prefix nama variabel acak flat seperti `fc` dari `fc_E7`."""
    match = re.fullmatch(r'([A-Za-z_]+)_E\d+', str(variable_name or '').strip())
    if match:
        return str(match.group(1))
    return str(variable_name or '').strip()


def downsample_index_array(indices: np.ndarray, max_points: int) -> np.ndarray:
    """Ambil subset indeks merata dengan urutan tetap."""
    index_array = np.asarray(indices, dtype=int).reshape(-1)
    if max_points <= 0 or index_array.size == 0:
        return np.asarray([], dtype=int)
    if index_array.size <= int(max_points):
        return index_array.astype(int)

    target_size = int(max_points)
    step = index_array.size / float(target_size)
    selected_positions = np.floor(np.arange(target_size, dtype=float) * step).astype(int)
    selected_positions = np.clip(selected_positions, 0, index_array.size - 1)
    return index_array[selected_positions].astype(int)


def build_probabilistic_failure_cloud_data(mc_results: Optional[Dict[str, Any]],
                                           random_variables: Optional[Dict[str, Dict[str, Any]]],
                                           input_data: Optional[Dict],
                                           max_points: int = FAILURE_CLOUD_MAX_STORED_POINTS,
                                           max_failed_points: int = FAILURE_CLOUD_MAX_FAILED_POINTS) -> Dict[str, Any]:
    """Bangun dataset ringkas untuk visualisasi failure cloud sistem."""
    if not mc_results or not random_variables:
        return {}

    sample_history = list(mc_results.get('random_samples_history', []) or [])
    if not sample_history:
        return {}

    num_samples = len(sample_history)
    valid_failure_indices = sorted({
        int(index)
        for index in (mc_results.get('failure_indices', []) or [])
        if 0 <= int(index) < num_samples
    })
    failure_set = set(valid_failure_indices)
    failure_index_array = np.asarray(valid_failure_indices, dtype=int)
    safe_index_array = np.asarray(
        [index for index in range(num_samples) if index not in failure_set],
        dtype=int
    )
    all_indices = np.arange(num_samples, dtype=int)

    if int(max_points or 0) <= 0:
        selected_failed = failure_index_array.astype(int)
        selected_indices = all_indices.astype(int)
    else:
        selected_failed = downsample_index_array(
            failure_index_array,
            min(int(max_failed_points), int(max_points))
        )
        remaining_capacity = max(int(max_points) - int(selected_failed.size), 0)
        selected_safe = downsample_index_array(safe_index_array, remaining_capacity)

        selected_indices = np.sort(
            np.concatenate([selected_failed, selected_safe]).astype(int)
        )
    if selected_indices.size == 0:
        return {}

    prefix_specs = get_failure_cloud_variable_prefix_specs()
    selected_failure_mask = np.asarray(
        [int(index) in failure_set for index in selected_indices],
        dtype=bool
    )

    variables: Dict[str, Dict[str, Any]] = {}
    for variable_name, variable_info in random_variables.items():
        prefix = extract_variable_prefix_from_name(variable_name)
        elem_id = extract_element_id_from_variable_name(variable_name)
        prefix_spec = prefix_specs.get(prefix, {})
        element_code = (
            get_element_code_from_input(input_data, int(elem_id))
            if elem_id is not None else
            '-'
        )
        element_type = (
            get_element_type_label(element_code)
            if elem_id is not None else
            '-'
        )
        sampled_values_list = []
        for index in selected_indices:
            raw_value = sample_history[int(index)].get(variable_name)
            numeric_value = coerce_finite_float(raw_value)
            sampled_values_list.append(
                np.nan if numeric_value is None else float(numeric_value)
            )
        sampled_values = np.asarray(sampled_values_list, dtype=float)

        variables[str(variable_name)] = {
            'variable_name': str(variable_name),
            'prefix': str(prefix),
            'element_id': None if elem_id is None else int(elem_id),
            'element_code': str(element_code),
            'element_type': str(element_type),
            'label': str(prefix_spec.get('label', prefix or variable_name)),
            'unit': str(prefix_spec.get('unit', '-')),
            'distribution': str(variable_info.get('distribution', '-')),
            'mean_input': coerce_finite_float(variable_info.get('mean')),
            'stddev_input': coerce_finite_float(variable_info.get('stddev')),
            'values': sampled_values.tolist()
        }

    return {
        'num_simulations': int(mc_results.get('num_simulations', num_samples) or num_samples),
        'failures': int(mc_results.get('failures', len(valid_failure_indices)) or 0),
        'analysis_failures': int(mc_results.get('analysis_failures', 0) or 0),
        'stored_sample_count': int(selected_indices.size),
        'stored_failure_count': int(np.sum(selected_failure_mask)),
        'stored_safe_count': int(selected_indices.size - np.sum(selected_failure_mask)),
        'used_downsampling': bool(selected_indices.size < num_samples),
        'failed_points_truncated': bool(selected_failed.size < failure_index_array.size),
        'failure_mask': selected_failure_mask.tolist(),
        'variables': variables
    }


def get_failure_cloud_variable_sort_order(failure_cloud_data: Dict[str, Any],
                                          sensitivity_results: Optional[Dict[str, Dict[str, Any]]] = None) -> List[str]:
    """Urutkan variabel berdasarkan ranking sensitivitas lalu alfabetis."""
    variable_names = list((failure_cloud_data or {}).get('variables', {}).keys())
    if not variable_names:
        return []

    sensitivity_lookup = {
        str(var_name): index
        for index, var_name in enumerate((sensitivity_results or {}).keys())
    }
    return sorted(
        variable_names,
        key=lambda var_name: (
            sensitivity_lookup.get(str(var_name), len(sensitivity_lookup) + 1000),
            str(var_name)
        )
    )


def get_failure_cloud_default_variable_names(failure_cloud_data: Dict[str, Any],
                                             sensitivity_results: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[Optional[str], Optional[str]]:
    """Pilih default sumbu `X/Y` dari variabel paling sensitif yang tersedia."""
    ordered_names = get_failure_cloud_variable_sort_order(
        failure_cloud_data,
        sensitivity_results=sensitivity_results
    )
    if not ordered_names:
        return None, None
    if len(ordered_names) == 1:
        return ordered_names[0], None
    return ordered_names[0], ordered_names[1]


def get_failure_cloud_default_three_variable_names(failure_cloud_data: Dict[str, Any],
                                                   sensitivity_results: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Pilih default sumbu `X/Y/Z` dari variabel paling sensitif yang tersedia."""
    ordered_names = get_failure_cloud_variable_sort_order(
        failure_cloud_data,
        sensitivity_results=sensitivity_results
    )
    if not ordered_names:
        return None, None, None
    if len(ordered_names) == 1:
        return ordered_names[0], None, None
    if len(ordered_names) == 2:
        return ordered_names[0], ordered_names[1], None
    return ordered_names[0], ordered_names[1], ordered_names[2]


def get_first_distinct_failure_cloud_variable_name(variable_names: List[str],
                                                   excluded_names: List[str]) -> Optional[str]:
    """Ambil variabel pertama yang tidak termasuk daftar eksklusi."""
    excluded = {str(name) for name in (excluded_names or [])}
    for variable_name in (variable_names or []):
        if str(variable_name) not in excluded:
            return str(variable_name)
    return None


def format_failure_cloud_variable_label(record: Dict[str, Any],
                                        short: bool = False) -> str:
    """Format label ramah pengguna untuk pilihan variabel failure cloud."""
    variable_name = str(record.get('variable_name', '-') or '-')
    label = str(record.get('label', variable_name) or variable_name)
    unit = str(record.get('unit', '-') or '-')
    elem_id = record.get('element_id')
    element_type = str(record.get('element_type', '-') or '-')

    if short:
        if elem_id is None:
            return f"{label} [{unit}]"
        return f"{label} | E{int(elem_id)} ({element_type}) [{unit}]"

    if elem_id is None:
        return f"{variable_name} | {label} [{unit}]"
    return (
        f"{variable_name} | E{int(elem_id)} ({element_type}) | "
        f"{label} [{unit}]"
    )


def prepare_failure_cloud_plot_data(failure_cloud_data: Dict[str, Any],
                                    x_variable_name: str,
                                    y_variable_name: str) -> Optional[Dict[str, Any]]:
    """Siapkan array `X/Y` dan mask failure untuk plotting failure cloud."""
    variables = (failure_cloud_data or {}).get('variables', {}) or {}
    x_record = variables.get(str(x_variable_name))
    y_record = variables.get(str(y_variable_name))
    if not x_record or not y_record:
        return None

    x_values = np.asarray(x_record.get('values', []), dtype=float)
    y_values = np.asarray(y_record.get('values', []), dtype=float)
    failure_mask = np.asarray(failure_cloud_data.get('failure_mask', []), dtype=bool)

    common_size = min(int(x_values.size), int(y_values.size), int(failure_mask.size))
    if common_size <= 0:
        return None

    x_values = x_values[:common_size]
    y_values = y_values[:common_size]
    failure_mask = failure_mask[:common_size]
    valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(valid_mask):
        return None

    x_valid = x_values[valid_mask]
    y_valid = y_values[valid_mask]
    failure_valid = failure_mask[valid_mask]
    safe_valid = ~failure_valid

    return {
        'x_record': x_record,
        'y_record': y_record,
        'x': x_valid,
        'y': y_valid,
        'failure_mask': failure_valid,
        'safe_mask': safe_valid,
        'plot_count': int(x_valid.size),
        'failure_count': int(np.sum(failure_valid)),
        'safe_count': int(np.sum(safe_valid))
    }


def get_failure_cloud_axis_limits(values: np.ndarray) -> Tuple[float, float]:
    """Tentukan batas sumbu dengan padding ringan agar cloud mudah dibaca."""
    numeric_values = np.asarray(values, dtype=float)
    if numeric_values.size == 0:
        return -1.0, 1.0

    data_min = float(np.min(numeric_values))
    data_max = float(np.max(numeric_values))
    if np.isclose(data_min, data_max, atol=1e-12, rtol=1e-9):
        span = max(abs(data_min) * 0.05, 1e-6)
        return data_min - span, data_max + span

    span = data_max - data_min
    padding = max(span * 0.06, 1e-6)
    return data_min - padding, data_max + padding


def extend_numeric_array_with_optional_values(values: np.ndarray,
                                              *extra_values: Any) -> np.ndarray:
    """Gabungkan array dengan nilai tambahan finite untuk kebutuhan limit sumbu."""
    base_array = np.asarray(values, dtype=float).reshape(-1)
    finite_extra_values = []
    for raw_value in extra_values:
        numeric_value = coerce_finite_float(raw_value)
        if numeric_value is not None:
            finite_extra_values.append(float(numeric_value))

    if not finite_extra_values:
        return base_array.astype(float)
    return np.concatenate([
        base_array.astype(float),
        np.asarray(finite_extra_values, dtype=float)
    ]).astype(float)


def build_failure_cloud_figure(failure_cloud_data: Dict[str, Any],
                               x_variable_name: str,
                               y_variable_name: str) -> Optional[plt.Figure]:
    """Bangun scatter `safe` vs `failed` pada ruang dua variabel acak."""
    plot_data = prepare_failure_cloud_plot_data(
        failure_cloud_data,
        x_variable_name,
        y_variable_name
    )
    if plot_data is None:
        return None

    x_values = np.asarray(plot_data['x'], dtype=float)
    y_values = np.asarray(plot_data['y'], dtype=float)
    safe_mask = np.asarray(plot_data['safe_mask'], dtype=bool)
    failure_mask = np.asarray(plot_data['failure_mask'], dtype=bool)
    x_record = plot_data['x_record']
    y_record = plot_data['y_record']

    fig, axis = plt.subplots(figsize=(10.8, 7.0), dpi=180)
    axis.set_facecolor('#f8fafc')

    if np.any(safe_mask):
        axis.scatter(
            x_values[safe_mask],
            y_values[safe_mask],
            s=16,
            color=SAFE_CLOUD_COLOR,
            alpha=0.30,
            edgecolors='none',
            label=f"Safe ({int(np.sum(safe_mask)):,})"
        )

    if np.any(failure_mask):
        axis.scatter(
            x_values[failure_mask],
            y_values[failure_mask],
            s=26,
            color='#dc2626',
            alpha=0.80,
            edgecolors='#ffffff',
            linewidths=0.25,
            label=f"Failed ({int(np.sum(failure_mask)):,})"
        )

    x_input_mean = coerce_finite_float(x_record.get('mean_input'))
    y_input_mean = coerce_finite_float(y_record.get('mean_input'))
    if x_input_mean is not None:
        axis.axvline(
            x_input_mean,
            color='#475569',
            linestyle='--',
            linewidth=1.0,
            alpha=0.95
        )
    if y_input_mean is not None:
        axis.axhline(
            y_input_mean,
            color='#475569',
            linestyle='--',
            linewidth=1.0,
            alpha=0.95
        )
    if x_input_mean is not None and y_input_mean is not None:
        axis.scatter(
            [x_input_mean],
            [y_input_mean],
            marker='x',
            s=74,
            color='#0f172a',
            linewidths=1.6,
            label='Input Mean'
        )

    axis.set_xlim(*get_failure_cloud_axis_limits(x_values))
    axis.set_ylim(*get_failure_cloud_axis_limits(y_values))
    axis.set_xlabel(format_failure_cloud_variable_label(x_record, short=True))
    axis.set_ylabel(format_failure_cloud_variable_label(y_record, short=True))
    axis.set_title("Failure Cloud of Monte Carlo Samples", fontsize=12, pad=12)
    axis.grid(True, alpha=0.24, linestyle='--')
    axis.legend(loc='best', fontsize=8.5)

    fig.suptitle(
        "Failure Cloud | System Failure Classification",
        fontsize=13,
        y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def build_failure_cloud_summary_df(failure_cloud_data: Dict[str, Any],
                                   x_variable_name: str,
                                   y_variable_name: str) -> pd.DataFrame:
    """Ringkas statistik subset titik yang diplot pada failure cloud."""
    plot_data = prepare_failure_cloud_plot_data(
        failure_cloud_data,
        x_variable_name,
        y_variable_name
    )
    if plot_data is None:
        return pd.DataFrame()

    x_values = np.asarray(plot_data['x'], dtype=float)
    y_values = np.asarray(plot_data['y'], dtype=float)
    failure_mask = np.asarray(plot_data['failure_mask'], dtype=bool)
    safe_mask = np.asarray(plot_data['safe_mask'], dtype=bool)
    total_points = max(int(plot_data.get('plot_count', 0)), 1)

    x_record = plot_data['x_record']
    y_record = plot_data['y_record']
    x_unit = str(x_record.get('unit', '-') or '-')
    y_unit = str(y_record.get('unit', '-') or '-')

    rows = []
    category_specs = [
        ('Semua Titik Cloud', np.ones(total_points, dtype=bool)),
        ('Safe', safe_mask),
        ('Failed', failure_mask)
    ]
    for category_label, mask in category_specs:
        if mask.size != total_points or not np.any(mask):
            continue

        x_subset = x_values[mask]
        y_subset = y_values[mask]
        rows.append({
            'Kategori': category_label,
            'Jumlah Titik (-)': int(x_subset.size),
            'Proporsi Cloud (%)': float((x_subset.size / total_points) * 100.0),
            f'X Mean ({x_unit})': float(np.mean(x_subset)),
            f'X StdDev ({x_unit})': float(np.std(x_subset)),
            f'Y Mean ({y_unit})': float(np.mean(y_subset)),
            f'Y StdDev ({y_unit})': float(np.std(y_subset))
        })

    return pd.DataFrame(rows)


def transform_failure_cloud_values_to_standard_normal_space(values: np.ndarray,
                                                            record: Dict[str, Any]) -> np.ndarray:
    """Transform nilai variabel acak ke ruang normal baku `U`."""
    numeric_values = np.asarray(values, dtype=float)
    distribution_name = str(record.get('distribution', 'normal') or 'normal').strip().lower()
    mean_value = coerce_finite_float(record.get('mean_input'))
    stddev_value = coerce_finite_float(record.get('stddev_input'))
    if numeric_values.size == 0:
        return np.asarray([], dtype=float)

    if distribution_name == 'normal':
        if mean_value is None:
            mean_value = 0.0
        if stddev_value is None or stddev_value <= 0.0:
            return np.zeros_like(numeric_values, dtype=float)
        return (numeric_values - float(mean_value)) / float(stddev_value)

    if distribution_name == 'lognormal':
        if mean_value is None or mean_value <= 0.0:
            return np.full_like(numeric_values, np.nan, dtype=float)
        if stddev_value is None or stddev_value <= 0.0:
            return np.zeros_like(numeric_values, dtype=float)

        variance_ratio = (float(stddev_value) / float(mean_value)) ** 2
        sigma_ln = np.sqrt(np.log(1.0 + variance_ratio))
        if not np.isfinite(sigma_ln) or sigma_ln <= 0.0:
            return np.zeros_like(numeric_values, dtype=float)

        mu_ln = np.log(float(mean_value)) - 0.5 * sigma_ln ** 2
        clipped_values = np.clip(numeric_values, 1e-12, None)
        return (np.log(clipped_values) - mu_ln) / sigma_ln

    if mean_value is None:
        mean_value = float(np.nanmean(numeric_values)) if numeric_values.size else 0.0
    if stddev_value is None or stddev_value <= 0.0:
        return np.zeros_like(numeric_values, dtype=float)
    return (numeric_values - float(mean_value)) / float(stddev_value)


def build_failure_surface_axis_label(record: Dict[str, Any]) -> str:
    """Label sumbu ringkas untuk ruang normal baku `U`."""
    variable_name = str(record.get('variable_name', '-') or '-')
    return f"u({variable_name}) (-)"


def prepare_failure_surface_plot_data(failure_cloud_data: Dict[str, Any],
                                      x_variable_name: str,
                                      y_variable_name: str) -> Optional[Dict[str, Any]]:
    """Siapkan pasangan data `u1-u2` dan kelas safe/failed untuk failure surface."""
    plot_data = prepare_failure_cloud_plot_data(
        failure_cloud_data,
        x_variable_name,
        y_variable_name
    )
    if plot_data is None:
        return None

    x_record = plot_data['x_record']
    y_record = plot_data['y_record']
    x_u = transform_failure_cloud_values_to_standard_normal_space(plot_data['x'], x_record)
    y_u = transform_failure_cloud_values_to_standard_normal_space(plot_data['y'], y_record)
    failure_mask = np.asarray(plot_data['failure_mask'], dtype=bool)

    valid_mask = np.isfinite(x_u) & np.isfinite(y_u)
    if not np.any(valid_mask):
        return None

    x_valid = np.asarray(x_u[valid_mask], dtype=float)
    y_valid = np.asarray(y_u[valid_mask], dtype=float)
    failure_valid = np.asarray(failure_mask[valid_mask], dtype=bool)
    safe_valid = ~failure_valid

    return {
        'x_record': x_record,
        'y_record': y_record,
        'u_x': x_valid,
        'u_y': y_valid,
        'failure_mask': failure_valid,
        'safe_mask': safe_valid,
        'plot_count': int(x_valid.size),
        'failure_count': int(np.sum(failure_valid)),
        'safe_count': int(np.sum(safe_valid))
    }


def prepare_failure_surface_3d_plot_data(failure_cloud_data: Dict[str, Any],
                                         x_variable_name: str,
                                         y_variable_name: str,
                                         z_variable_name: str) -> Optional[Dict[str, Any]]:
    """Siapkan pasangan data `u1-u2-u3` dan kelas safe/failed untuk failure surface 3D."""
    variables = (failure_cloud_data or {}).get('variables', {}) or {}
    x_record = variables.get(str(x_variable_name))
    y_record = variables.get(str(y_variable_name))
    z_record = variables.get(str(z_variable_name))
    if not x_record or not y_record or not z_record:
        return None

    x_values = np.asarray(x_record.get('values', []), dtype=float)
    y_values = np.asarray(y_record.get('values', []), dtype=float)
    z_values = np.asarray(z_record.get('values', []), dtype=float)
    failure_mask = np.asarray(failure_cloud_data.get('failure_mask', []), dtype=bool)

    common_size = min(
        int(x_values.size),
        int(y_values.size),
        int(z_values.size),
        int(failure_mask.size)
    )
    if common_size <= 0:
        return None

    x_values = x_values[:common_size]
    y_values = y_values[:common_size]
    z_values = z_values[:common_size]
    failure_mask = failure_mask[:common_size]

    x_u = transform_failure_cloud_values_to_standard_normal_space(x_values, x_record)
    y_u = transform_failure_cloud_values_to_standard_normal_space(y_values, y_record)
    z_u = transform_failure_cloud_values_to_standard_normal_space(z_values, z_record)
    valid_mask = np.isfinite(x_u) & np.isfinite(y_u) & np.isfinite(z_u)
    if not np.any(valid_mask):
        return None

    x_valid = np.asarray(x_u[valid_mask], dtype=float)
    y_valid = np.asarray(y_u[valid_mask], dtype=float)
    z_valid = np.asarray(z_u[valid_mask], dtype=float)
    failure_valid = np.asarray(failure_mask[valid_mask], dtype=bool)
    safe_valid = ~failure_valid

    return {
        'x_record': x_record,
        'y_record': y_record,
        'z_record': z_record,
        'u_x': x_valid,
        'u_y': y_valid,
        'u_z': z_valid,
        'failure_mask': failure_valid,
        'safe_mask': safe_valid,
        'plot_count': int(x_valid.size),
        'failure_count': int(np.sum(failure_valid)),
        'safe_count': int(np.sum(safe_valid))
    }


def build_beta_circle_values(x_limits: Tuple[float, float],
                             y_limits: Tuple[float, float]) -> List[float]:
    """Pilih radius lingkar beta yang masih relevan di area plot."""
    max_radius = max(
        abs(float(x_limits[0])),
        abs(float(x_limits[1])),
        abs(float(y_limits[0])),
        abs(float(y_limits[1]))
    )
    candidate_values = [1.0, 2.0, 3.0, 4.0]
    return [
        beta_value
        for beta_value in candidate_values
        if beta_value <= max_radius + 0.25
    ]


def build_nice_contour_step(max_abs_value: float,
                            target_half_steps: int = 4) -> float:
    """Pilih langkah kontur simetris yang cukup bersih untuk dibaca."""
    try:
        numeric_value = abs(float(max_abs_value))
    except (TypeError, ValueError):
        numeric_value = 0.0
    if not np.isfinite(numeric_value) or numeric_value <= 1e-12:
        return 1.0

    half_steps = max(int(target_half_steps), 1)
    raw_step = numeric_value / float(half_steps)
    exponent = np.floor(np.log10(raw_step))
    scale = 10.0 ** exponent
    normalized = raw_step / scale
    if normalized <= 1.0:
        factor = 1.0
    elif normalized <= 2.0:
        factor = 2.0
    elif normalized <= 5.0:
        factor = 5.0
    else:
        factor = 10.0
    return float(factor * scale)


def estimate_failure_surface_2d_mpp_from_grid(grid_x: np.ndarray,
                                              grid_y: np.ndarray,
                                              score_grid: np.ndarray) -> Optional[Dict[str, Any]]:
    """Estimasi MPP 2D dari titik potong `g_hat(u)=0` yang terdekat ke origin."""
    x_array = np.asarray(grid_x, dtype=float)
    y_array = np.asarray(grid_y, dtype=float)
    score_array = np.asarray(score_grid, dtype=float)
    if x_array.ndim != 2 or y_array.ndim != 2 or score_array.ndim != 2:
        return None
    if x_array.shape != y_array.shape or x_array.shape != score_array.shape:
        return None

    candidate_points: List[Tuple[float, float]] = []
    num_rows, num_cols = score_array.shape

    def append_zero_crossing(x1: float, y1: float, s1: float,
                             x2: float, y2: float, s2: float) -> None:
        if not all(np.isfinite(value) for value in (x1, y1, s1, x2, y2, s2)):
            return
        if s1 == 0.0 and s2 == 0.0:
            candidate_points.append((0.5 * (x1 + x2), 0.5 * (y1 + y2)))
            return
        if s1 == 0.0:
            candidate_points.append((x1, y1))
            return
        if s2 == 0.0:
            candidate_points.append((x2, y2))
            return
        if (s1 < 0.0 and s2 > 0.0) or (s1 > 0.0 and s2 < 0.0):
            denominator = (s2 - s1)
            if abs(float(denominator)) <= 1e-12:
                t_value = 0.5
            else:
                t_value = float(-s1 / denominator)
            t_value = float(np.clip(t_value, 0.0, 1.0))
            candidate_points.append((
                x1 + t_value * (x2 - x1),
                y1 + t_value * (y2 - y1)
            ))

    for row_index in range(num_rows):
        for col_index in range(num_cols - 1):
            append_zero_crossing(
                float(x_array[row_index, col_index]),
                float(y_array[row_index, col_index]),
                float(score_array[row_index, col_index]),
                float(x_array[row_index, col_index + 1]),
                float(y_array[row_index, col_index + 1]),
                float(score_array[row_index, col_index + 1])
            )

    for row_index in range(num_rows - 1):
        for col_index in range(num_cols):
            append_zero_crossing(
                float(x_array[row_index, col_index]),
                float(y_array[row_index, col_index]),
                float(score_array[row_index, col_index]),
                float(x_array[row_index + 1, col_index]),
                float(y_array[row_index + 1, col_index]),
                float(score_array[row_index + 1, col_index])
            )

    if not candidate_points:
        return None

    candidate_array = np.asarray(candidate_points, dtype=float)
    candidate_array = candidate_array[
        np.isfinite(candidate_array[:, 0]) & np.isfinite(candidate_array[:, 1])
    ]
    if candidate_array.size == 0:
        return None

    radii = np.linalg.norm(candidate_array, axis=1)
    if radii.size == 0 or not np.any(np.isfinite(radii)):
        return None

    best_index = int(np.nanargmin(radii))
    best_point = np.asarray(candidate_array[best_index], dtype=float)
    beta_value = float(radii[best_index])
    return {
        'point': best_point,
        'beta': beta_value
    }


def get_failure_surface_axis_limits(values: np.ndarray) -> Tuple[float, float]:
    """Batas sumbu simetris untuk tampilan ruang normal baku `U`."""
    numeric_values = np.asarray(values, dtype=float)
    if numeric_values.size == 0:
        return -4.0, 4.0

    finite_values = numeric_values[np.isfinite(numeric_values)]
    if finite_values.size == 0:
        return -4.0, 4.0

    max_abs = max(
        float(np.max(np.abs(finite_values))),
        3.0
    )
    limit = max_abs + 0.45
    return -limit, limit


def downsample_xy_series(x_values: np.ndarray,
                         y_values: np.ndarray,
                         max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Padatkan pasangan titik `x-y` secara merata."""
    x_array = np.asarray(x_values, dtype=float).reshape(-1)
    y_array = np.asarray(y_values, dtype=float).reshape(-1)
    common_size = min(int(x_array.size), int(y_array.size))
    if common_size <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    base_indices = np.arange(common_size, dtype=int)
    selected_indices = downsample_index_array(base_indices, int(max_points))
    return x_array[selected_indices], y_array[selected_indices]


def downsample_xyz_series(x_values: np.ndarray,
                          y_values: np.ndarray,
                          z_values: np.ndarray,
                          max_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Padatkan pasangan titik `x-y-z` secara merata."""
    x_array = np.asarray(x_values, dtype=float).reshape(-1)
    y_array = np.asarray(y_values, dtype=float).reshape(-1)
    z_array = np.asarray(z_values, dtype=float).reshape(-1)
    common_size = min(int(x_array.size), int(y_array.size), int(z_array.size))
    if common_size <= 0:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float)
        )

    base_indices = np.arange(common_size, dtype=int)
    selected_indices = downsample_index_array(base_indices, int(max_points))
    return (
        x_array[selected_indices],
        y_array[selected_indices],
        z_array[selected_indices]
    )


def build_failure_surface_3d_quadratic_features(x_values: np.ndarray,
                                                y_values: np.ndarray,
                                                z_values: np.ndarray) -> np.ndarray:
    """Bangun fitur kuadratik implisit 3D untuk surrogate batas gagal."""
    x_array = np.asarray(x_values, dtype=float).reshape(-1)
    y_array = np.asarray(y_values, dtype=float).reshape(-1)
    z_array = np.asarray(z_values, dtype=float).reshape(-1)
    return np.column_stack([
        np.ones_like(x_array, dtype=float),
        x_array,
        y_array,
        z_array,
        x_array * x_array,
        x_array * y_array,
        y_array * y_array,
        x_array * z_array,
        y_array * z_array,
        z_array * z_array
    ])


def build_failure_surface_polynomial_features(x_values: np.ndarray,
                                              y_values: np.ndarray) -> np.ndarray:
    """Bangun fitur kuadratik 2D untuk surrogate batas gagal."""
    x_array = np.asarray(x_values, dtype=float).reshape(-1)
    y_array = np.asarray(y_values, dtype=float).reshape(-1)
    return np.column_stack([
        np.ones_like(x_array, dtype=float),
        x_array,
        y_array,
        x_array * x_array,
        x_array * y_array,
        y_array * y_array
    ])


def estimate_failure_surface_grid_from_kde(failure_x: np.ndarray,
                                           failure_y: np.ndarray,
                                           safe_x: np.ndarray,
                                           safe_y: np.ndarray,
                                           grid_x: np.ndarray,
                                           grid_y: np.ndarray) -> Optional[np.ndarray]:
    """Estimasi score grid dengan pemisahan densitas `KDE`."""
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()])
    failure_points = np.vstack([failure_x, failure_y])
    safe_points = np.vstack([safe_x, safe_y])
    rng = np.random.default_rng(20260510)
    jitter_scale = 1e-6
    failure_points = failure_points + rng.normal(
        loc=0.0,
        scale=jitter_scale,
        size=failure_points.shape
    )
    safe_points = safe_points + rng.normal(
        loc=0.0,
        scale=jitter_scale,
        size=safe_points.shape
    )

    try:
        failure_kde = stats.gaussian_kde(failure_points)
        safe_kde = stats.gaussian_kde(safe_points)
        failure_density = np.asarray(failure_kde(grid_points), dtype=float)
        safe_density = np.asarray(safe_kde(grid_points), dtype=float)
    except Exception:
        return None

    epsilon = 1e-12
    failure_prior = float(failure_x.size) / float(failure_x.size + safe_x.size)
    safe_prior = float(safe_x.size) / float(failure_x.size + safe_x.size)
    score = (
        np.log(failure_density + epsilon)
        + np.log(failure_prior + epsilon)
        - np.log(safe_density + epsilon)
        - np.log(safe_prior + epsilon)
    )
    return np.asarray(score.reshape(grid_x.shape), dtype=float)


def estimate_failure_surface_grid_from_quadratic_surrogate(failure_x: np.ndarray,
                                                           failure_y: np.ndarray,
                                                           safe_x: np.ndarray,
                                                           safe_y: np.ndarray,
                                                           grid_x: np.ndarray,
                                                           grid_y: np.ndarray) -> Optional[np.ndarray]:
    """Fallback robust: surrogate kuadratik ridge dari klasifikasi safe/failed."""
    train_x = np.concatenate([safe_x, failure_x]).astype(float)
    train_y = np.concatenate([safe_y, failure_y]).astype(float)
    labels = np.concatenate([
        -np.ones_like(safe_x, dtype=float),
        np.ones_like(failure_x, dtype=float)
    ]).astype(float)
    if min(int(safe_x.size), int(failure_x.size)) < 2:
        return None

    features = build_failure_surface_polynomial_features(train_x, train_y)
    if features.shape[0] < 4:
        return None

    try:
        ridge_lambda = 1e-3
        lhs = features.T @ features + ridge_lambda * np.eye(features.shape[1], dtype=float)
        rhs = features.T @ labels
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        try:
            coefficients = np.linalg.lstsq(features, labels, rcond=None)[0]
        except Exception:
            return None

    train_scores = np.asarray(features @ coefficients, dtype=float)
    safe_scores = train_scores[:safe_x.size]
    failure_scores = train_scores[safe_x.size:]
    if safe_scores.size == 0 or failure_scores.size == 0:
        return None
    threshold = 0.5 * (
        float(np.mean(safe_scores))
        + float(np.mean(failure_scores))
    )

    grid_features = build_failure_surface_polynomial_features(
        np.asarray(grid_x, dtype=float).ravel(),
        np.asarray(grid_y, dtype=float).ravel()
    )
    score = (grid_features @ coefficients) - threshold
    if not np.any(np.isfinite(score)):
        return None
    return np.asarray(score.reshape(grid_x.shape), dtype=float)


def estimate_failure_surface_grid_from_linear_surrogate(failure_x: np.ndarray,
                                                        failure_y: np.ndarray,
                                                        safe_x: np.ndarray,
                                                        safe_y: np.ndarray,
                                                        grid_x: np.ndarray,
                                                        grid_y: np.ndarray) -> Optional[np.ndarray]:
    """Fallback paling sederhana: separator linear ridge pada ruang `U`."""
    train_x = np.concatenate([safe_x, failure_x]).astype(float)
    train_y = np.concatenate([safe_y, failure_y]).astype(float)
    labels = np.concatenate([
        -np.ones_like(safe_x, dtype=float),
        np.ones_like(failure_x, dtype=float)
    ]).astype(float)
    if min(int(safe_x.size), int(failure_x.size)) < 1:
        return None

    features = np.column_stack([
        np.ones_like(train_x, dtype=float),
        train_x,
        train_y
    ])
    if features.shape[0] < 2:
        return None

    try:
        ridge_lambda = 1e-3
        lhs = features.T @ features + ridge_lambda * np.eye(features.shape[1], dtype=float)
        rhs = features.T @ labels
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        try:
            coefficients = np.linalg.lstsq(features, labels, rcond=None)[0]
        except Exception:
            return None

    train_scores = np.asarray(features @ coefficients, dtype=float)
    safe_scores = train_scores[:safe_x.size]
    failure_scores = train_scores[safe_x.size:]
    if safe_scores.size == 0 or failure_scores.size == 0:
        return None
    threshold = 0.5 * (
        float(np.mean(safe_scores))
        + float(np.mean(failure_scores))
    )

    grid_features = np.column_stack([
        np.ones_like(np.asarray(grid_x, dtype=float).ravel(), dtype=float),
        np.asarray(grid_x, dtype=float).ravel(),
        np.asarray(grid_y, dtype=float).ravel()
    ])
    score = (grid_features @ coefficients) - threshold
    if not np.any(np.isfinite(score)):
        return None
    return np.asarray(score.reshape(grid_x.shape), dtype=float)


def estimate_failure_surface_grid_from_centroid_separator(failure_x: np.ndarray,
                                                          failure_y: np.ndarray,
                                                          safe_x: np.ndarray,
                                                          safe_y: np.ndarray,
                                                          grid_x: np.ndarray,
                                                          grid_y: np.ndarray) -> Optional[np.ndarray]:
    """Fallback final: garis pemisah berbasis centroid kelas safe/failed."""
    if min(int(failure_x.size), int(safe_x.size)) < 1:
        return None

    safe_centroid = np.asarray([
        float(np.mean(safe_x)),
        float(np.mean(safe_y))
    ], dtype=float)
    failure_centroid = np.asarray([
        float(np.mean(failure_x)),
        float(np.mean(failure_y))
    ], dtype=float)
    direction = failure_centroid - safe_centroid
    direction_norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction_norm) or direction_norm <= 1e-12:
        direction = np.asarray([1.0, 0.0], dtype=float)
        direction_norm = 1.0
    unit_direction = direction / direction_norm

    safe_projection = (
        unit_direction[0] * np.asarray(safe_x, dtype=float)
        + unit_direction[1] * np.asarray(safe_y, dtype=float)
    )
    failure_projection = (
        unit_direction[0] * np.asarray(failure_x, dtype=float)
        + unit_direction[1] * np.asarray(failure_y, dtype=float)
    )
    threshold = 0.5 * (
        float(np.mean(safe_projection))
        + float(np.mean(failure_projection))
    )

    score = (
        unit_direction[0] * np.asarray(grid_x, dtype=float)
        + unit_direction[1] * np.asarray(grid_y, dtype=float)
        - threshold
    )
    if not np.any(np.isfinite(score)):
        return None
    return np.asarray(score, dtype=float)


def estimate_failure_surface_grid(surface_plot_data: Dict[str, Any],
                                  grid_size: int = FAILURE_SURFACE_GRID_SIZE,
                                  max_class_points: int = FAILURE_SURFACE_MAX_CLASS_POINTS) -> Optional[Dict[str, Any]]:
    """Estimasi batas gagal `g_hat(u)=0` dari hasil safe/failed Monte Carlo."""
    u_x = np.asarray(surface_plot_data.get('u_x', []), dtype=float)
    u_y = np.asarray(surface_plot_data.get('u_y', []), dtype=float)
    failure_mask = np.asarray(surface_plot_data.get('failure_mask', []), dtype=bool)
    safe_mask = np.asarray(surface_plot_data.get('safe_mask', []), dtype=bool)

    failure_x = u_x[failure_mask]
    failure_y = u_y[failure_mask]
    safe_x = u_x[safe_mask]
    safe_y = u_y[safe_mask]
    if min(int(failure_x.size), int(safe_x.size), int(failure_y.size), int(safe_y.size)) < 2:
        return None

    failure_x, failure_y = downsample_xy_series(
        failure_x,
        failure_y,
        max_points=int(max_class_points)
    )
    safe_x, safe_y = downsample_xy_series(
        safe_x,
        safe_y,
        max_points=int(max_class_points)
    )
    if min(int(failure_x.size), int(safe_x.size)) < 2:
        return None

    x_limits = get_failure_surface_axis_limits(u_x)
    y_limits = get_failure_surface_axis_limits(u_y)
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_limits[0], x_limits[1], int(grid_size)),
        np.linspace(y_limits[0], y_limits[1], int(grid_size))
    )
    score_grid = None
    estimator_method = None

    if min(int(failure_x.size), int(safe_x.size)) >= 6:
        score_grid = estimate_failure_surface_grid_from_kde(
            failure_x,
            failure_y,
            safe_x,
            safe_y,
            grid_x,
            grid_y
        )
        if score_grid is not None:
            estimator_method = 'kde'

    if score_grid is None:
        score_grid = estimate_failure_surface_grid_from_quadratic_surrogate(
            failure_x,
            failure_y,
            safe_x,
            safe_y,
            grid_x,
            grid_y
        )
        if score_grid is not None:
            estimator_method = 'quadratic'

    if score_grid is None:
        score_grid = estimate_failure_surface_grid_from_linear_surrogate(
            failure_x,
            failure_y,
            safe_x,
            safe_y,
            grid_x,
            grid_y
        )
        if score_grid is not None:
            estimator_method = 'linear'

    if score_grid is None:
        score_grid = estimate_failure_surface_grid_from_centroid_separator(
            failure_x,
            failure_y,
            safe_x,
            safe_y,
            grid_x,
            grid_y
        )
        if score_grid is not None:
            estimator_method = 'centroid'

    if score_grid is None:
        return None

    return {
        'grid_x': grid_x,
        'grid_y': grid_y,
        'score_grid': np.asarray(score_grid, dtype=float),
        'x_limits': x_limits,
        'y_limits': y_limits,
        'estimator_method': str(estimator_method or 'unknown')
    }


def build_failure_surface_figure_and_metadata(failure_cloud_data: Dict[str, Any],
                                              x_variable_name: str,
                                              y_variable_name: str) -> Tuple[Optional[plt.Figure], Dict[str, Any]]:
    """Bangun plot surface failure cloud 2D beserta metadata render."""
    surface_plot_data = prepare_failure_surface_plot_data(
        failure_cloud_data,
        x_variable_name,
        y_variable_name
    )
    if surface_plot_data is None:
        return None, {}

    surface_grid = estimate_failure_surface_grid(surface_plot_data)
    if surface_grid is None:
        return None, {}

    x_record = surface_plot_data['x_record']
    y_record = surface_plot_data['y_record']
    u_x = np.asarray(surface_plot_data.get('u_x', []), dtype=float)
    u_y = np.asarray(surface_plot_data.get('u_y', []), dtype=float)
    safe_mask = np.asarray(surface_plot_data.get('safe_mask', []), dtype=bool)
    failure_mask = np.asarray(surface_plot_data.get('failure_mask', []), dtype=bool)
    grid_x = np.asarray(surface_grid['grid_x'], dtype=float)
    grid_y = np.asarray(surface_grid['grid_y'], dtype=float)
    score_grid = np.asarray(surface_grid['score_grid'], dtype=float)
    signed_margin_grid = -score_grid
    estimator_method = str(surface_grid.get('estimator_method', 'unknown') or 'unknown')
    score_min = float(np.nanmin(score_grid))
    score_max = float(np.nanmax(score_grid))
    estimated_mpp = estimate_failure_surface_2d_mpp_from_grid(
        grid_x,
        grid_y,
        score_grid
    )
    safe_count_full = int(surface_plot_data.get('safe_count', 0) or 0)
    failure_count_full = int(surface_plot_data.get('failure_count', 0) or 0)
    if not (score_min <= 0.0 <= score_max):
        return None, {
            'x_variable_name': str(x_variable_name),
            'y_variable_name': str(y_variable_name),
            'estimator_method': estimator_method,
            'score_min': score_min,
            'score_max': score_max
        }

    fig, axis = plt.subplots(figsize=(9.6, 7.2), dpi=180)
    axis.set_facecolor('#ffffff')

    finite_margin = signed_margin_grid[np.isfinite(signed_margin_grid)]
    max_abs_margin = max(
        float(np.max(np.abs(finite_margin))) if finite_margin.size else 0.0,
        1e-6
    )
    fill_levels = np.linspace(-max_abs_margin, max_abs_margin, 19)
    contour_fill = axis.contourf(
        grid_x,
        grid_y,
        signed_margin_grid,
        levels=fill_levels,
        cmap='RdYlBu',
        alpha=0.16,
        antialiased=True,
        zorder=0
    )
    contour_step = build_nice_contour_step(max_abs_margin)
    nonzero_levels = np.asarray(
        [
            value for value in (
                contour_step * np.arange(-4, 5, dtype=float)
            )
            if (
                not np.isclose(value, 0.0, atol=max(contour_step * 1e-6, 1e-12))
                and abs(float(value)) <= max_abs_margin * 1.02
            )
        ],
        dtype=float
    )
    if nonzero_levels.size > 0:
        axis.contour(
            grid_x,
            grid_y,
            signed_margin_grid,
            levels=np.sort(nonzero_levels),
            colors='#64748b',
            linewidths=0.65,
            alpha=0.48,
            linestyles='dashed',
            zorder=1
        )

    beta_circle_values = build_beta_circle_values(
        surface_grid['x_limits'],
        surface_grid['y_limits']
    )
    for beta_value in beta_circle_values:
        circle = plt.Circle(
            (0.0, 0.0),
            radius=float(beta_value),
            fill=False,
            linestyle='--',
            linewidth=0.8,
            edgecolor='#cbd5e1',
            alpha=0.75
        )
        axis.add_patch(circle)
        axis.text(
            float(beta_value) * 0.70,
            float(beta_value) * 0.72,
            rf"$\beta={beta_value:.0f}$",
            color='#475569',
            fontsize=8.5
        )

    # Tampilkan seluruh sampel valid pada scatter agar total safe/failed
    # di plot mengikuti N simulasi tanpa pembatas visual.
    safe_x = np.asarray(u_x[safe_mask], dtype=float)
    safe_y = np.asarray(u_y[safe_mask], dtype=float)
    failure_x = np.asarray(u_x[failure_mask], dtype=float)
    failure_y = np.asarray(u_y[failure_mask], dtype=float)

    if safe_x.size > 0:
        axis.scatter(
            safe_x,
            safe_y,
            s=12,
            color='#1d4ed8',
            alpha=0.88,
            edgecolors='none',
            label=f"Safe Samples, g(u) > 0 ({safe_count_full:,})",
            zorder=2
        )
    if failure_x.size > 0:
        axis.scatter(
            failure_x,
            failure_y,
            s=12,
            color='#ef4444',
            alpha=0.88,
            edgecolors='none',
            label=f"Failed Samples, g(u) < 0 ({failure_count_full:,})",
            zorder=2
        )

    zero_contour = axis.contour(
        grid_x,
        grid_y,
        signed_margin_grid,
        levels=[0.0],
        colors=['#111827'],
        linewidths=2.4,
        zorder=3
    )
    zero_segments = list((zero_contour.allsegs or [[]])[0])
    zero_contour.remove()
    zero_segment_info = select_primary_zero_contour_segment(
        zero_segments,
        reference_x=u_x,
        reference_y=u_y
    )
    selected_zero_segment = zero_segment_info.get('selected_segment')
    if selected_zero_segment is not None:
        selected_zero_segment_array = np.asarray(selected_zero_segment, dtype=float)
        if (
            selected_zero_segment_array.ndim == 2
            and selected_zero_segment_array.shape[0] >= 2
        ):
            axis.plot(
                selected_zero_segment_array[:, 0],
                selected_zero_segment_array[:, 1],
                color='#111827',
                linewidth=2.4,
                zorder=3
            )
    axis.plot(
        [],
        [],
        color='#111827',
        linewidth=2.4,
        label='Nonlinear contour, g_hat(u)=0'
    )

    if estimated_mpp is not None:
        mpp_point = np.asarray(estimated_mpp.get('point', []), dtype=float).reshape(-1)
        mpp_beta = coerce_finite_float(estimated_mpp.get('beta'))
        if mpp_point.size == 2 and mpp_beta is not None:
            axis.plot(
                [0.0, float(mpp_point[0])],
                [0.0, float(mpp_point[1])],
                linestyle='--',
                linewidth=1.3,
                color='#a21caf',
                label=f"Estimated Beta Line, beta={mpp_beta:.2f}",
                zorder=3
            )
            axis.scatter(
                [float(mpp_point[0])],
                [float(mpp_point[1])],
                marker='o',
                s=52,
                color='#111827',
                edgecolors='#ffffff',
                linewidths=0.5,
                label=f"Estimated MPP, beta={mpp_beta:.2f}",
                zorder=4
            )
            axis.annotate(
                f"MPP\nbeta={mpp_beta:.2f}",
                xy=(float(mpp_point[0]), float(mpp_point[1])),
                xytext=(10, -12),
                textcoords='offset points',
                fontsize=8.7,
                color='#111827',
                ha='left',
                va='top',
                bbox=dict(
                    boxstyle='round,pad=0.18',
                    facecolor='white',
                    edgecolor='#cbd5e1',
                    alpha=0.92
                )
            )

    axis.scatter(
        [0.0],
        [0.0],
        marker='x',
        s=62,
        color='#111827',
        linewidths=1.4,
        label='Origin',
        zorder=4
    )
    axis.text(
        0.10,
        -0.14,
        "O (Origin)",
        fontsize=9,
        color='#111827'
    )

    axis.axhline(0.0, color='#94a3b8', linewidth=0.8, linestyle='--', alpha=0.9)
    axis.axvline(0.0, color='#94a3b8', linewidth=0.8, linestyle='--', alpha=0.9)
    axis.set_xlim(*surface_grid['x_limits'])
    axis.set_ylim(*surface_grid['y_limits'])
    axis.set_aspect('equal', adjustable='box')
    axis.set_xlabel(build_failure_surface_axis_label(x_record))
    axis.set_ylabel(build_failure_surface_axis_label(y_record))
    axis.set_title(
        (
            "Failure Cloud and Nonlinear Contour g_hat(u)=0 "
            f"| Method: {estimator_method.upper()}"
        ),
        fontsize=12,
        pad=12
    )
    axis.grid(True, alpha=0.18, linestyle='--')
    axis.text(
        0.04,
        0.93,
        "DAERAH AMAN\n g(u) > 0",
        transform=axis.transAxes,
        ha='left',
        va='top',
        fontsize=10,
        color='#1d4ed8',
        fontweight='700'
    )
    axis.text(
        0.96,
        0.10,
        "DAERAH GAGAL\n g(u) < 0",
        transform=axis.transAxes,
        ha='right',
        va='bottom',
        fontsize=10,
        color='#dc2626',
        fontweight='700'
    )
    fig.colorbar(
        contour_fill,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label='Estimated signed margin g_hat(u) (-)'
    )
    axis.text(
        0.82,
        0.80,
        "g_hat(u) = 0",
        transform=axis.transAxes,
        fontsize=10,
        color='#111827',
        fontweight='700'
    )
    axis.legend(loc='upper right', fontsize=8.2, frameon=True)

    fig.suptitle(
        "Failure Cloud in Standard Normal Space (U-space)",
        fontsize=13,
        y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, {
        'x_variable_name': str(x_variable_name),
        'y_variable_name': str(y_variable_name),
        'estimator_method': estimator_method,
        'score_min': score_min,
        'score_max': score_max,
        'estimated_mpp_beta': (
            coerce_finite_float((estimated_mpp or {}).get('beta'))
            if estimated_mpp is not None else
            None
        )
    }


def build_failure_surface_figure(failure_cloud_data: Dict[str, Any],
                                 x_variable_name: str,
                                 y_variable_name: str) -> Optional[plt.Figure]:
    """Bangun plot surface failure cloud 2D pada ruang normal baku `U`."""
    figure, _ = build_failure_surface_figure_and_metadata(
        failure_cloud_data,
        x_variable_name,
        y_variable_name
    )
    return figure


def fit_failure_surface_3d_quadratic_model(surface_plot_data: Dict[str, Any],
                                           max_points_per_class: int = FAILURE_SURFACE_MAX_CLASS_POINTS) -> Optional[Dict[str, Any]]:
    """Fit surrogate kuadratik implisit untuk batas `g(u)=0` di ruang 3D."""
    u_x = np.asarray(surface_plot_data.get('u_x', []), dtype=float)
    u_y = np.asarray(surface_plot_data.get('u_y', []), dtype=float)
    u_z = np.asarray(surface_plot_data.get('u_z', []), dtype=float)
    failure_mask = np.asarray(surface_plot_data.get('failure_mask', []), dtype=bool)
    safe_mask = np.asarray(surface_plot_data.get('safe_mask', []), dtype=bool)

    failure_x, failure_y, failure_z = downsample_xyz_series(
        u_x[failure_mask],
        u_y[failure_mask],
        u_z[failure_mask],
        max_points=int(max_points_per_class)
    )
    safe_x, safe_y, safe_z = downsample_xyz_series(
        u_x[safe_mask],
        u_y[safe_mask],
        u_z[safe_mask],
        max_points=int(max_points_per_class)
    )
    if min(int(failure_x.size), int(safe_x.size)) < 4:
        return None

    safe_points = np.column_stack([safe_x, safe_y, safe_z]).astype(float)
    failure_points = np.column_stack([failure_x, failure_y, failure_z]).astype(float)
    train_points = np.vstack([safe_points, failure_points])
    labels = np.concatenate([
        -np.ones(int(safe_points.shape[0]), dtype=float),
        np.ones(int(failure_points.shape[0]), dtype=float)
    ]).astype(float)

    features = build_failure_surface_3d_quadratic_features(
        train_points[:, 0],
        train_points[:, 1],
        train_points[:, 2]
    )
    if features.shape[0] < features.shape[1]:
        return None

    try:
        ridge_lambda = 1e-3
        lhs = features.T @ features + ridge_lambda * np.eye(features.shape[1], dtype=float)
        rhs = features.T @ labels
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        try:
            coefficients = np.linalg.lstsq(features, labels, rcond=None)[0]
        except Exception:
            return None

    train_scores = np.asarray(features @ coefficients, dtype=float)
    safe_scores = train_scores[:safe_points.shape[0]]
    failure_scores = train_scores[safe_points.shape[0]:]
    if safe_scores.size == 0 or failure_scores.size == 0:
        return None

    threshold = 0.5 * (
        float(np.mean(safe_scores))
        + float(np.mean(failure_scores))
    )
    adjusted_coefficients = np.asarray(coefficients, dtype=float).copy()
    adjusted_coefficients[0] = float(adjusted_coefficients[0] - threshold)

    safe_centroid = np.mean(safe_points, axis=0)
    failure_centroid = np.mean(failure_points, axis=0)
    reference_point = 0.5 * (safe_centroid + failure_centroid)

    return {
        'type': 'quadratic',
        'coefficients': adjusted_coefficients,
        'reference_point': np.asarray(reference_point, dtype=float),
        'method': 'quadratic'
    }


def fit_failure_surface_3d_plane(surface_plot_data: Dict[str, Any],
                                 max_points_per_class: int = FAILURE_SURFACE_MAX_CLASS_POINTS) -> Optional[Dict[str, Any]]:
    """Fit bidang batas gagal linear pada ruang `U` tiga dimensi."""
    u_x = np.asarray(surface_plot_data.get('u_x', []), dtype=float)
    u_y = np.asarray(surface_plot_data.get('u_y', []), dtype=float)
    u_z = np.asarray(surface_plot_data.get('u_z', []), dtype=float)
    failure_mask = np.asarray(surface_plot_data.get('failure_mask', []), dtype=bool)
    safe_mask = np.asarray(surface_plot_data.get('safe_mask', []), dtype=bool)

    failure_x = u_x[failure_mask]
    failure_y = u_y[failure_mask]
    failure_z = u_z[failure_mask]
    safe_x = u_x[safe_mask]
    safe_y = u_y[safe_mask]
    safe_z = u_z[safe_mask]

    if min(int(failure_x.size), int(safe_x.size)) < 1:
        return None

    failure_x, failure_y, failure_z = downsample_xyz_series(
        failure_x,
        failure_y,
        failure_z,
        max_points=int(max_points_per_class)
    )
    safe_x, safe_y, safe_z = downsample_xyz_series(
        safe_x,
        safe_y,
        safe_z,
        max_points=int(max_points_per_class)
    )
    if min(int(failure_x.size), int(safe_x.size)) < 1:
        return None

    safe_points = np.column_stack([safe_x, safe_y, safe_z])
    failure_points = np.column_stack([failure_x, failure_y, failure_z])
    train_points = np.vstack([safe_points, failure_points]).astype(float)
    labels = np.concatenate([
        -np.ones(int(safe_points.shape[0]), dtype=float),
        np.ones(int(failure_points.shape[0]), dtype=float)
    ]).astype(float)

    features = np.column_stack([
        np.ones(int(train_points.shape[0]), dtype=float),
        train_points
    ])
    method_name = 'linear'
    try:
        ridge_lambda = 1e-3
        lhs = features.T @ features + ridge_lambda * np.eye(features.shape[1], dtype=float)
        rhs = features.T @ labels
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        try:
            coefficients = np.linalg.lstsq(features, labels, rcond=None)[0]
        except Exception:
            coefficients = None

    normal_vector = None
    intercept_value = None
    if coefficients is not None:
        train_scores = np.asarray(features @ coefficients, dtype=float)
        safe_scores = train_scores[:safe_points.shape[0]]
        failure_scores = train_scores[safe_points.shape[0]:]
        if safe_scores.size > 0 and failure_scores.size > 0:
            threshold = 0.5 * (
                float(np.mean(safe_scores))
                + float(np.mean(failure_scores))
            )
            intercept_value = float(coefficients[0] - threshold)
            normal_vector = np.asarray(coefficients[1:4], dtype=float)

    if normal_vector is None or not np.all(np.isfinite(normal_vector)) or float(np.linalg.norm(normal_vector)) <= 1e-12:
        safe_centroid = np.mean(safe_points, axis=0)
        failure_centroid = np.mean(failure_points, axis=0)
        normal_vector = np.asarray(failure_centroid - safe_centroid, dtype=float)
        if not np.all(np.isfinite(normal_vector)) or float(np.linalg.norm(normal_vector)) <= 1e-12:
            return None
        midpoint = 0.5 * (safe_centroid + failure_centroid)
        intercept_value = float(-np.dot(normal_vector, midpoint))
        method_name = 'centroid'

    return {
        'normal': np.asarray(normal_vector, dtype=float),
        'intercept': float(intercept_value),
        'type': 'plane',
        'method': str(method_name)
    }


def build_failure_surface_3d_mesh(plane_model: Dict[str, Any],
                                  x_limits: Tuple[float, float],
                                  y_limits: Tuple[float, float],
                                  z_limits: Tuple[float, float],
                                  grid_size: int = FAILURE_SURFACE_3D_GRID_SIZE) -> Optional[Dict[str, Any]]:
    """Bangun mesh bidang `g_hat(u)=0` pada ruang tiga dimensi."""
    normal = np.asarray((plane_model or {}).get('normal', []), dtype=float).reshape(-1)
    if normal.size != 3 or not np.all(np.isfinite(normal)):
        return None
    intercept = coerce_finite_float((plane_model or {}).get('intercept'))
    if intercept is None:
        return None

    axis_limits = [tuple(x_limits), tuple(y_limits), tuple(z_limits)]
    dependent_axis = int(np.argmax(np.abs(normal)))
    if abs(float(normal[dependent_axis])) <= 1e-12:
        return None
    independent_axes = [axis_index for axis_index in range(3) if axis_index != dependent_axis]

    grid_a, grid_b = np.meshgrid(
        np.linspace(axis_limits[independent_axes[0]][0], axis_limits[independent_axes[0]][1], int(grid_size)),
        np.linspace(axis_limits[independent_axes[1]][0], axis_limits[independent_axes[1]][1], int(grid_size))
    )
    dependent_values = -(
        float(intercept)
        + float(normal[independent_axes[0]]) * grid_a
        + float(normal[independent_axes[1]]) * grid_b
    ) / float(normal[dependent_axis])

    mesh_axes = [None, None, None]
    mesh_axes[independent_axes[0]] = np.asarray(grid_a, dtype=float)
    mesh_axes[independent_axes[1]] = np.asarray(grid_b, dtype=float)
    mesh_axes[dependent_axis] = np.asarray(dependent_values, dtype=float)

    lower_limit, upper_limit = axis_limits[dependent_axis]
    valid_mask = (
        np.isfinite(mesh_axes[0])
        & np.isfinite(mesh_axes[1])
        & np.isfinite(mesh_axes[2])
        & (mesh_axes[dependent_axis] >= float(lower_limit))
        & (mesh_axes[dependent_axis] <= float(upper_limit))
    )
    if not np.any(valid_mask):
        return None

    for axis_index in range(3):
        axis_array = np.asarray(mesh_axes[axis_index], dtype=float)
        axis_array = np.where(valid_mask, axis_array, np.nan)
        mesh_axes[axis_index] = axis_array

    return {
        'X': np.asarray(mesh_axes[0], dtype=float),
        'Y': np.asarray(mesh_axes[1], dtype=float),
        'Z': np.asarray(mesh_axes[2], dtype=float),
        'dependent_axis': int(dependent_axis)
    }


def build_failure_surface_3d_quadratic_mesh(quadratic_model: Dict[str, Any],
                                            x_limits: Tuple[float, float],
                                            y_limits: Tuple[float, float],
                                            z_limits: Tuple[float, float],
                                            grid_size: int = FAILURE_SURFACE_3D_GRID_SIZE) -> Optional[Dict[str, Any]]:
    """Bangun mesh permukaan kuadratik implisit `g_hat(u)=0` pada ruang 3D."""
    coefficients = np.asarray(
        (quadratic_model or {}).get('coefficients', []),
        dtype=float
    ).reshape(-1)
    if coefficients.size != 10 or not np.all(np.isfinite(coefficients)):
        return None

    reference_point = np.asarray(
        (quadratic_model or {}).get('reference_point', [0.0, 0.0, 0.0]),
        dtype=float
    ).reshape(-1)
    reference_z = float(reference_point[2]) if reference_point.size >= 3 else 0.0

    grid_x, grid_y = np.meshgrid(
        np.linspace(float(x_limits[0]), float(x_limits[1]), int(grid_size)),
        np.linspace(float(y_limits[0]), float(y_limits[1]), int(grid_size))
    )

    a_value = float(coefficients[9])
    b_value = (
        float(coefficients[3])
        + float(coefficients[7]) * grid_x
        + float(coefficients[8]) * grid_y
    )
    c_value = (
        float(coefficients[0])
        + float(coefficients[1]) * grid_x
        + float(coefficients[2]) * grid_y
        + float(coefficients[4]) * (grid_x ** 2)
        + float(coefficients[5]) * (grid_x * grid_y)
        + float(coefficients[6]) * (grid_y ** 2)
    )

    z_surface = np.full_like(grid_x, np.nan, dtype=float)
    if abs(a_value) <= 1e-12:
        valid_linear = np.abs(b_value) > 1e-12
        z_linear = np.full_like(grid_x, np.nan, dtype=float)
        z_linear[valid_linear] = -c_value[valid_linear] / b_value[valid_linear]
        z_surface = z_linear
    else:
        discriminant = (b_value ** 2) - (4.0 * a_value * c_value)
        valid_quadratic = discriminant >= 0.0
        sqrt_discriminant = np.full_like(grid_x, np.nan, dtype=float)
        sqrt_discriminant[valid_quadratic] = np.sqrt(discriminant[valid_quadratic])
        z_root_1 = np.full_like(grid_x, np.nan, dtype=float)
        z_root_2 = np.full_like(grid_x, np.nan, dtype=float)
        denominator = 2.0 * a_value
        z_root_1[valid_quadratic] = (
            -b_value[valid_quadratic] + sqrt_discriminant[valid_quadratic]
        ) / denominator
        z_root_2[valid_quadratic] = (
            -b_value[valid_quadratic] - sqrt_discriminant[valid_quadratic]
        ) / denominator

        root_1_valid = (
            np.isfinite(z_root_1)
            & (z_root_1 >= float(z_limits[0]))
            & (z_root_1 <= float(z_limits[1]))
        )
        root_2_valid = (
            np.isfinite(z_root_2)
            & (z_root_2 >= float(z_limits[0]))
            & (z_root_2 <= float(z_limits[1]))
        )

        choose_root_1 = np.abs(z_root_1 - reference_z) <= np.abs(z_root_2 - reference_z)
        z_surface = np.where(
            choose_root_1 & root_1_valid,
            z_root_1,
            z_surface
        )
        z_surface = np.where(
            (~choose_root_1) & root_2_valid,
            z_root_2,
            z_surface
        )
        z_surface = np.where(
            np.isnan(z_surface) & root_1_valid,
            z_root_1,
            z_surface
        )
        z_surface = np.where(
            np.isnan(z_surface) & root_2_valid,
            z_root_2,
            z_surface
        )

    valid_mask = (
        np.isfinite(grid_x)
        & np.isfinite(grid_y)
        & np.isfinite(z_surface)
        & (z_surface >= float(z_limits[0]))
        & (z_surface <= float(z_limits[1]))
    )
    if not np.any(valid_mask):
        return None

    grid_z = np.where(valid_mask, z_surface, np.nan)
    return {
        'X': np.asarray(grid_x, dtype=float),
        'Y': np.asarray(grid_y, dtype=float),
        'Z': np.asarray(grid_z, dtype=float),
        'dependent_axis': 2
    }


def estimate_failure_surface_3d_mpp_from_mesh(mesh: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Estimasi titik MPP sebagai titik pada mesh yang paling dekat ke origin."""
    mesh_x = np.asarray((mesh or {}).get('X', []), dtype=float)
    mesh_y = np.asarray((mesh or {}).get('Y', []), dtype=float)
    mesh_z = np.asarray((mesh or {}).get('Z', []), dtype=float)
    if mesh_x.size == 0 or mesh_y.size == 0 or mesh_z.size == 0:
        return None

    valid_mask = np.isfinite(mesh_x) & np.isfinite(mesh_y) & np.isfinite(mesh_z)
    if not np.any(valid_mask):
        return None

    valid_points = np.column_stack([
        mesh_x[valid_mask],
        mesh_y[valid_mask],
        mesh_z[valid_mask]
    ]).astype(float)
    radii = np.linalg.norm(valid_points, axis=1)
    if radii.size == 0 or not np.any(np.isfinite(radii)):
        return None

    best_index = int(np.nanargmin(radii))
    best_point = np.asarray(valid_points[best_index], dtype=float)
    beta_value = float(radii[best_index])
    return {
        'point': best_point,
        'beta': beta_value
    }


def build_failure_surface_3d_figure(failure_cloud_data: Dict[str, Any],
                                    x_variable_name: str,
                                    y_variable_name: str,
                                    z_variable_name: str) -> Optional[plt.Figure]:
    """Bangun plot 3D failure cloud dan permukaan batas `g(u)=0`."""
    surface_plot_data = prepare_failure_surface_3d_plot_data(
        failure_cloud_data,
        x_variable_name,
        y_variable_name,
        z_variable_name
    )
    if surface_plot_data is None:
        return None

    u_x = np.asarray(surface_plot_data.get('u_x', []), dtype=float)
    u_y = np.asarray(surface_plot_data.get('u_y', []), dtype=float)
    u_z = np.asarray(surface_plot_data.get('u_z', []), dtype=float)
    safe_mask = np.asarray(surface_plot_data.get('safe_mask', []), dtype=bool)
    failure_mask = np.asarray(surface_plot_data.get('failure_mask', []), dtype=bool)
    x_limits = get_failure_surface_axis_limits(u_x)
    y_limits = get_failure_surface_axis_limits(u_y)
    z_limits = get_failure_surface_axis_limits(u_z)
    safe_count_full = int(surface_plot_data.get('safe_count', 0) or 0)
    failure_count_full = int(surface_plot_data.get('failure_count', 0) or 0)

    surface_model = fit_failure_surface_3d_quadratic_model(surface_plot_data)
    mesh = None
    if surface_model is not None:
        mesh = build_failure_surface_3d_quadratic_mesh(
            surface_model,
            x_limits=x_limits,
            y_limits=y_limits,
            z_limits=z_limits
        )

    if mesh is None:
        surface_model = fit_failure_surface_3d_plane(surface_plot_data)
        if surface_model is None:
            return None
        mesh = build_failure_surface_3d_mesh(
            surface_model,
            x_limits=x_limits,
            y_limits=y_limits,
            z_limits=z_limits
        )
    if mesh is None or surface_model is None:
        return None

    # Tampilkan seluruh sampel valid pada scatter 3D agar total safe/failed
    # di plot mengikuti N simulasi tanpa pembatas visual.
    safe_x = np.asarray(u_x[safe_mask], dtype=float)
    safe_y = np.asarray(u_y[safe_mask], dtype=float)
    safe_z = np.asarray(u_z[safe_mask], dtype=float)
    failure_x = np.asarray(u_x[failure_mask], dtype=float)
    failure_y = np.asarray(u_y[failure_mask], dtype=float)
    failure_z = np.asarray(u_z[failure_mask], dtype=float)

    x_record = surface_plot_data['x_record']
    y_record = surface_plot_data['y_record']
    z_record = surface_plot_data['z_record']
    method_name = str((surface_model or {}).get('method', 'linear') or 'linear').upper()
    estimated_mpp = estimate_failure_surface_3d_mpp_from_mesh(mesh)

    fig = plt.figure(figsize=(10.0, 7.8), dpi=180)
    axis = fig.add_subplot(111, projection='3d')
    axis.set_facecolor('#ffffff')

    if safe_x.size > 0:
        axis.scatter(
            safe_x,
            safe_y,
            safe_z,
            s=10,
            color='#1d4ed8',
            alpha=0.86,
            depthshade=False,
            label=f"Safe Samples ({safe_count_full:,})"
        )
    if failure_x.size > 0:
        axis.scatter(
            failure_x,
            failure_y,
            failure_z,
            s=10,
            color='#ef4444',
            alpha=0.86,
            depthshade=False,
            label=f"Failed Samples ({failure_count_full:,})"
        )

    axis.plot_surface(
        np.asarray(mesh['X'], dtype=float),
        np.asarray(mesh['Y'], dtype=float),
        np.asarray(mesh['Z'], dtype=float),
        color='#f59e0b',
        alpha=0.30,
        linewidth=0.3,
        edgecolor='#b45309',
        antialiased=True,
        shade=False
    )

    try:
        axis.contour(
            np.asarray(mesh['X'], dtype=float),
            np.asarray(mesh['Y'], dtype=float),
            np.asarray(mesh['Z'], dtype=float),
            zdir='z',
            offset=float(z_limits[0]),
            levels=10,
            colors='#cbd5e1',
            linewidths=0.7
        )
    except Exception:
        pass

    axis.scatter(
        [0.0],
        [0.0],
        [0.0],
        marker='o',
        s=118,
        color='#ffffff',
        edgecolors='#ffffff',
        linewidths=0.0,
        depthshade=False,
        label='_nolegend_'
    )
    axis.scatter(
        [0.0],
        [0.0],
        [0.0],
        marker='x',
        s=74,
        color='#111827',
        linewidths=2.2,
        depthshade=False,
        label='Origin'
    )

    mpp_point = None
    mpp_beta = None
    if estimated_mpp is not None:
        mpp_point = np.asarray(estimated_mpp.get('point', []), dtype=float).reshape(-1)
        mpp_beta = coerce_finite_float(estimated_mpp.get('beta'))
        if mpp_point.size == 3 and mpp_beta is not None:
            axis.plot(
                [0.0, float(mpp_point[0])],
                [0.0, float(mpp_point[1])],
                [0.0, float(mpp_point[2])],
                linestyle='-',
                linewidth=3.8,
                color='#ffffff',
                alpha=0.98
            )
            axis.scatter(
                [float(mpp_point[0])],
                [float(mpp_point[1])],
                [float(mpp_point[2])],
                marker='o',
                s=118,
                color='#ffffff',
                edgecolors='#ffffff',
                linewidths=0.0,
                depthshade=False,
                label='_nolegend_'
            )
            axis.scatter(
                [float(mpp_point[0])],
                [float(mpp_point[1])],
                [float(mpp_point[2])],
                marker='o',
                s=58,
                color='#111827',
                edgecolors='#ffffff',
                linewidths=1.0,
                depthshade=False,
                label=f"Estimated MPP, beta={mpp_beta:.2f}"
            )
            axis.plot(
                [0.0, float(mpp_point[0])],
                [0.0, float(mpp_point[1])],
                [0.0, float(mpp_point[2])],
                linestyle='--',
                linewidth=1.8,
                color='#a21caf'
            )

    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_zlim(*z_limits)
    axis.set_xlabel(build_failure_surface_axis_label(x_record), labelpad=8)
    axis.set_ylabel(build_failure_surface_axis_label(y_record), labelpad=8)
    axis.set_zlabel(build_failure_surface_axis_label(z_record), labelpad=8)
    axis.view_init(elev=24, azim=-58)
    axis.grid(True, alpha=0.16)
    axis.set_title(
        f"3D Failure Surface g(u)=0 | Method: {method_name}",
        fontsize=12,
        pad=12
    )

    def annotate_projected_point(point_xyz: Tuple[float, float, float],
                                 text: str,
                                 text_offset: Tuple[float, float],
                                 arrow_color: str = '#111827',
                                 text_color: str = '#111827') -> None:
        projected_x, projected_y, _ = proj3d.proj_transform(
            float(point_xyz[0]),
            float(point_xyz[1]),
            float(point_xyz[2]),
            axis.get_proj()
        )
        annotation = axis.annotate(
            text,
            xy=(projected_x, projected_y),
            xytext=text_offset,
            textcoords='offset points',
            xycoords='data',
            ha='left',
            va='top',
            fontsize=8.7,
            color=text_color,
            bbox=dict(
                boxstyle='round,pad=0.22',
                facecolor='white',
                edgecolor='#cbd5e1',
                alpha=0.96
            ),
            arrowprops=dict(
                arrowstyle='-',
                color=arrow_color,
                linewidth=1.2,
                alpha=0.92
            )
        )
        annotation.set_zorder(12)

    def draw_projected_overlay_line(start_xyz: Tuple[float, float, float],
                                    end_xyz: Tuple[float, float, float]) -> None:
        start_x, start_y, _ = proj3d.proj_transform(
            float(start_xyz[0]),
            float(start_xyz[1]),
            float(start_xyz[2]),
            axis.get_proj()
        )
        end_x, end_y, _ = proj3d.proj_transform(
            float(end_xyz[0]),
            float(end_xyz[1]),
            float(end_xyz[2]),
            axis.get_proj()
        )
        background_line = axis.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            xycoords='data',
            textcoords='data',
            arrowprops=dict(
                arrowstyle='-',
                color='#ffffff',
                linewidth=4.0,
                alpha=0.98
            )
        )
        background_line.set_zorder(10)
        foreground_line = axis.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            xycoords='data',
            textcoords='data',
            arrowprops=dict(
                arrowstyle='-',
                linestyle='--',
                color='#a21caf',
                linewidth=1.8,
                alpha=0.98
            )
        )
        foreground_line.set_zorder(11)

    try:
        fig.canvas.draw()
    except Exception:
        pass

    annotate_projected_point(
        (0.0, 0.0, 0.0),
        "Origin",
        text_offset=(10, 10)
    )
    if mpp_point is not None and mpp_point.size == 3 and mpp_beta is not None:
        draw_projected_overlay_line(
            (0.0, 0.0, 0.0),
            (float(mpp_point[0]), float(mpp_point[1]), float(mpp_point[2]))
        )
        annotate_projected_point(
            (float(mpp_point[0]), float(mpp_point[1]), float(mpp_point[2])),
            f"MPP\nbeta={mpp_beta:.2f}",
            text_offset=(12, -12),
            arrow_color='#a21caf'
        )

    legend_handles = [
        Line2D(
            [],
            [],
            linestyle='none',
            marker='o',
            color='#1d4ed8',
            markersize=5,
            label=f"Safe Samples ({safe_count_full:,})"
        ),
        Line2D(
            [],
            [],
            linestyle='none',
            marker='o',
            color='#ef4444',
            markersize=5,
            label=f"Failed Samples ({failure_count_full:,})"
        ),
        Patch(
            facecolor='#f59e0b',
            edgecolor='#b45309',
            alpha=0.30,
            label='Estimated Failure Surface'
        ),
        Line2D([], [], linestyle='none', marker='x', color='#111827', markersize=6, label='Origin')
    ]
    if estimated_mpp is not None:
        mpp_beta = coerce_finite_float(estimated_mpp.get('beta'))
        mpp_label = (
            f"Estimated MPP, beta={mpp_beta:.2f}"
            if mpp_beta is not None else
            'Estimated MPP'
        )
        legend_handles.extend([
            Line2D([], [], linestyle='none', marker='o', color='#111827', markersize=6, label=mpp_label),
            Line2D([], [], linestyle='--', color='#a21caf', linewidth=1.2, label='Estimated Beta Line')
        ])
    axis.legend(
        handles=legend_handles,
        loc='upper right',
        fontsize=8.2,
        frameon=True
    )

    fig.suptitle(
        "3D Failure Cloud and Estimated Surface in Standard Normal Space (U-space)",
        fontsize=13,
        y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig


def build_failure_surface_3d_plotly_figure(failure_cloud_data: Dict[str, Any],
                                           x_variable_name: str,
                                           y_variable_name: str,
                                           z_variable_name: str) -> Optional[Any]:
    """Bangun plot 3D interaktif yang dapat diputar dengan mouse."""
    if go is None:
        return None

    surface_plot_data = prepare_failure_surface_3d_plot_data(
        failure_cloud_data,
        x_variable_name,
        y_variable_name,
        z_variable_name
    )
    if surface_plot_data is None:
        return None

    u_x = np.asarray(surface_plot_data.get('u_x', []), dtype=float)
    u_y = np.asarray(surface_plot_data.get('u_y', []), dtype=float)
    u_z = np.asarray(surface_plot_data.get('u_z', []), dtype=float)
    safe_mask = np.asarray(surface_plot_data.get('safe_mask', []), dtype=bool)
    failure_mask = np.asarray(surface_plot_data.get('failure_mask', []), dtype=bool)
    x_limits = get_failure_surface_axis_limits(u_x)
    y_limits = get_failure_surface_axis_limits(u_y)
    z_limits = get_failure_surface_axis_limits(u_z)
    safe_count_full = int(surface_plot_data.get('safe_count', 0) or 0)
    failure_count_full = int(surface_plot_data.get('failure_count', 0) or 0)

    surface_model = fit_failure_surface_3d_quadratic_model(surface_plot_data)
    mesh = None
    if surface_model is not None:
        mesh = build_failure_surface_3d_quadratic_mesh(
            surface_model,
            x_limits=x_limits,
            y_limits=y_limits,
            z_limits=z_limits
        )

    if mesh is None:
        surface_model = fit_failure_surface_3d_plane(surface_plot_data)
        if surface_model is None:
            return None
        mesh = build_failure_surface_3d_mesh(
            surface_model,
            x_limits=x_limits,
            y_limits=y_limits,
            z_limits=z_limits
        )
    if mesh is None or surface_model is None:
        return None

    safe_x = np.asarray(u_x[safe_mask], dtype=float)
    safe_y = np.asarray(u_y[safe_mask], dtype=float)
    safe_z = np.asarray(u_z[safe_mask], dtype=float)
    failure_x = np.asarray(u_x[failure_mask], dtype=float)
    failure_y = np.asarray(u_y[failure_mask], dtype=float)
    failure_z = np.asarray(u_z[failure_mask], dtype=float)

    x_record = surface_plot_data['x_record']
    y_record = surface_plot_data['y_record']
    z_record = surface_plot_data['z_record']
    x_axis_label = build_failure_surface_axis_label(x_record)
    y_axis_label = build_failure_surface_axis_label(y_record)
    z_axis_label = build_failure_surface_axis_label(z_record)
    method_name = str((surface_model or {}).get('method', 'linear') or 'linear').upper()
    estimated_mpp = estimate_failure_surface_3d_mpp_from_mesh(mesh)
    plot_count = int(surface_plot_data.get('plot_count', 0) or 0)
    marker_size = 3.0 if plot_count > 7000 else 3.6

    figure = go.Figure()

    if safe_x.size > 0:
        figure.add_trace(go.Scatter3d(
            x=safe_x,
            y=safe_y,
            z=safe_z,
            mode='markers',
            name=f"Safe Samples ({safe_count_full:,})",
            marker=dict(
                size=marker_size,
                color='#1d4ed8',
                opacity=0.72
            ),
            hovertemplate=(
                f"{x_axis_label}: %{{x:.3f}}<br>"
                f"{y_axis_label}: %{{y:.3f}}<br>"
                f"{z_axis_label}: %{{z:.3f}}"
                "<extra>Safe</extra>"
            )
        ))
    if failure_x.size > 0:
        figure.add_trace(go.Scatter3d(
            x=failure_x,
            y=failure_y,
            z=failure_z,
            mode='markers',
            name=f"Failed Samples ({failure_count_full:,})",
            marker=dict(
                size=marker_size,
                color='#ef4444',
                opacity=0.74
            ),
            hovertemplate=(
                f"{x_axis_label}: %{{x:.3f}}<br>"
                f"{y_axis_label}: %{{y:.3f}}<br>"
                f"{z_axis_label}: %{{z:.3f}}"
                "<extra>Failed</extra>"
            )
        ))

    figure.add_trace(go.Surface(
        x=np.asarray(mesh['X'], dtype=float),
        y=np.asarray(mesh['Y'], dtype=float),
        z=np.asarray(mesh['Z'], dtype=float),
        surfacecolor=np.zeros_like(np.asarray(mesh['Z'], dtype=float)),
        colorscale=[
            [0.0, '#f59e0b'],
            [1.0, '#f59e0b']
        ],
        showscale=False,
        opacity=0.34,
        hoverinfo='skip',
        showlegend=False
    ))
    figure.add_trace(go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode='markers',
        name='Estimated Failure Surface',
        marker=dict(
            size=8,
            color='#f59e0b',
            opacity=0.78
        ),
        hoverinfo='skip'
    ))

    figure.add_trace(go.Scatter3d(
        x=[0.0],
        y=[0.0],
        z=[0.0],
        mode='markers+text',
        name='Origin',
        text=['Origin'],
        textposition='top center',
        textfont=dict(
            size=12,
            color='#111827'
        ),
        marker=dict(
            size=7,
            color='#111827',
            line=dict(
                color='#ffffff',
                width=4
            )
        ),
        hovertemplate=(
            f"{x_axis_label}: 0.000<br>"
            f"{y_axis_label}: 0.000<br>"
            f"{z_axis_label}: 0.000"
            "<extra>Origin</extra>"
        )
    ))

    if estimated_mpp is not None:
        mpp_point = np.asarray(estimated_mpp.get('point', []), dtype=float).reshape(-1)
        mpp_beta = coerce_finite_float(estimated_mpp.get('beta'))
        if mpp_point.size == 3 and mpp_beta is not None:
            figure.add_trace(go.Scatter3d(
                x=[0.0, float(mpp_point[0])],
                y=[0.0, float(mpp_point[1])],
                z=[0.0, float(mpp_point[2])],
                mode='lines',
                name='Estimated Beta Line',
                line=dict(
                    color='#a21caf',
                    width=8,
                    dash='dash'
                ),
                hovertemplate=(
                    f"{x_axis_label}: %{{x:.3f}}<br>"
                    f"{y_axis_label}: %{{y:.3f}}<br>"
                    f"{z_axis_label}: %{{z:.3f}}"
                    "<extra>Estimated Beta Line</extra>"
                )
            ))
            figure.add_trace(go.Scatter3d(
                x=[float(mpp_point[0])],
                y=[float(mpp_point[1])],
                z=[float(mpp_point[2])],
                mode='markers+text',
                name=f"Estimated MPP, beta={mpp_beta:.2f}",
                text=[f"MPP<br>beta={mpp_beta:.2f}"],
                textposition='top center',
                textfont=dict(
                    size=12,
                    color='#111827'
                ),
                marker=dict(
                    size=8,
                    color='#111827',
                    line=dict(
                        color='#ffffff',
                        width=4
                    )
                ),
                hovertemplate=(
                    f"{x_axis_label}: %{{x:.3f}}<br>"
                    f"{y_axis_label}: %{{y:.3f}}<br>"
                    f"{z_axis_label}: %{{z:.3f}}<br>"
                    f"Beta: {mpp_beta:.3f}"
                    "<extra>Estimated MPP</extra>"
                )
            ))

    scene_axis_style = dict(
        showbackground=True,
        backgroundcolor='#f8fafc',
        gridcolor='#dbe4f0',
        zerolinecolor='#94a3b8',
        showspikes=False
    )
    figure.update_layout(
        title=dict(
            text=(
                "3D Failure Cloud and Estimated Surface in Standard Normal Space (U-space)"
                f"<br><sup>Method: {method_name}</sup>"
            ),
            x=0.5
        ),
        margin=dict(l=0, r=0, t=72, b=0),
        legend=dict(
            yanchor='top',
            y=0.98,
            xanchor='left',
            x=0.02,
            bgcolor='rgba(255,255,255,0.88)'
        ),
        scene=dict(
            xaxis=dict(
                title=x_axis_label,
                range=[float(x_limits[0]), float(x_limits[1])],
                **scene_axis_style
            ),
            yaxis=dict(
                title=y_axis_label,
                range=[float(y_limits[0]), float(y_limits[1])],
                **scene_axis_style
            ),
            zaxis=dict(
                title=z_axis_label,
                range=[float(z_limits[0]), float(z_limits[1])],
                **scene_axis_style
            ),
            aspectmode='cube',
            dragmode='orbit',
            camera=dict(
                eye=dict(x=1.55, y=-1.75, z=1.15)
            )
        ),
        uirevision=(
            "failure-surface-3d-"
            f"{sanitize_dom_id(str(x_variable_name))}-"
            f"{sanitize_dom_id(str(y_variable_name))}-"
            f"{sanitize_dom_id(str(z_variable_name))}"
        )
    )
    return figure


def get_failure_surface_candidate_pairs(variable_names: List[str],
                                        preferred_pair: Optional[Tuple[str, str]] = None,
                                        max_variables: int = 10,
                                        max_pairs: int = 30) -> List[Tuple[str, str]]:
    """Bangun daftar pasangan kandidat untuk auto-fallback failure surface."""
    ordered_names = [
        str(var_name)
        for var_name in (variable_names or [])[:max(2, int(max_variables))]
    ]
    candidate_pairs: List[Tuple[str, str]] = []
    seen_pairs = set()

    def append_pair(left_name: str, right_name: str) -> None:
        if not left_name or not right_name or str(left_name) == str(right_name):
            return
        pair = (str(left_name), str(right_name))
        if pair in seen_pairs:
            return
        seen_pairs.add(pair)
        candidate_pairs.append(pair)

    if preferred_pair is not None:
        append_pair(preferred_pair[0], preferred_pair[1])

    for left_index in range(len(ordered_names)):
        for right_index in range(left_index + 1, len(ordered_names)):
            append_pair(ordered_names[left_index], ordered_names[right_index])
            if len(candidate_pairs) >= int(max_pairs):
                return candidate_pairs

    return candidate_pairs


def resolve_failure_surface_figure_with_fallback(failure_cloud_data: Dict[str, Any],
                                                 ordered_variable_names: List[str],
                                                 x_variable_name: str,
                                                 y_variable_name: str) -> Tuple[Optional[plt.Figure], Dict[str, Any]]:
    """Cari pasangan yang berhasil dirender bila pasangan awal gagal."""
    candidate_pairs = get_failure_surface_candidate_pairs(
        ordered_variable_names,
        preferred_pair=(str(x_variable_name), str(y_variable_name))
    )
    last_metadata: Dict[str, Any] = {}
    requested_pair = (str(x_variable_name), str(y_variable_name))

    for candidate_x, candidate_y in candidate_pairs:
        figure, metadata = build_failure_surface_figure_and_metadata(
            failure_cloud_data,
            candidate_x,
            candidate_y
        )
        if figure is not None:
            resolved_metadata = dict(metadata or {})
            resolved_metadata['requested_pair'] = requested_pair
            resolved_metadata['used_requested_pair'] = (
                str(candidate_x) == requested_pair[0]
                and str(candidate_y) == requested_pair[1]
            )
            return figure, resolved_metadata
        if metadata:
            last_metadata = dict(metadata)

    if last_metadata:
        last_metadata['requested_pair'] = requested_pair
        last_metadata['used_requested_pair'] = False
    return None, last_metadata


def build_failure_surface_unavailable_message(results_bundle: Dict[str, Any],
                                              failure_cloud_data: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Bangun pesan unavailable yang membedakan kasus `Pf ~ 0` dan `Pf ~ 1`."""
    summary = (results_bundle or {}).get('summary', {}) or {}
    num_simulations = int(
        summary.get('num_simulations')
        or failure_cloud_data.get('num_simulations')
        or 0
    )
    failures = int(
        summary.get('failures')
        or failure_cloud_data.get('failures')
        or 0
    )
    failures = max(failures, 0)
    safe_samples = max(num_simulations - failures, 0)
    pf_value = coerce_finite_float(summary.get('Pf'))

    if num_simulations > 0:
        pf_granularity = 2.0 / float(num_simulations)
    else:
        pf_granularity = 0.0
    pf_low_threshold = max(pf_granularity, 0.01)
    pf_high_threshold = 1.0 - pf_low_threshold

    if (
        num_simulations > 0 and (
            failures < 2
            or (pf_value is not None and pf_value <= pf_low_threshold)
        )
    ):
        return (
            "Failure surface tidak ada karena `Pf ~ 0`, sehingga kejadian gagal "
            "hampir tidak ada pada hasil Monte Carlo.",
            (
                f"Detail: `Pf = {format_metric(pf_value, 6)}` | "
                f"`failure = {failures:,}` dari `N = {num_simulations:,}`"
            )
        )

    if (
        num_simulations > 0 and (
            safe_samples < 2
            or (pf_value is not None and pf_value >= pf_high_threshold)
        )
    ):
        return (
            "Failure surface tidak ada karena `Pf ~ 1`, sehingga sampel aman "
            "hampir tidak ada pada hasil Monte Carlo.",
            (
                f"Detail: `Pf = {format_metric(pf_value, 6)}` | "
                f"`safe = {safe_samples:,}` dari `N = {num_simulations:,}`"
            )
        )

    return (
        "Failure surface belum dapat dibentuk untuk pasangan variabel ini. "
        "Biasanya ini terjadi bila sampel `safe/failed` terlalu sedikit atau "
        "sebaran data terlalu degenerat untuk estimasi batas 2D.",
        None
    )


def render_probabilistic_failure_cloud_output_section(failure_cloud_data: Dict[str, Any],
                                                      results_bundle: Dict[str, Any],
                                                      heading_level: str = "####") -> None:
    """Tampilkan tab failure surface pada ruang normal baku."""
    st.markdown(f"{heading_level} Failure Surface Ruang Normal Baku")
    st.caption(
        "Bagian berikut adalah visualisasi `failure surface` pada ruang normal baku `U`, "
        "yakni interpretasi geometrik yang paling dekat dengan konsep `beta` klasik."
    )
    if not failure_cloud_data:
        st.info(
            "Dataset failure surface ruang normal baku belum tersedia. Jalankan analisis "
            "probabilistik agar sampel Monte Carlo dapat diproyeksikan ke ruang dua variabel."
        )
        return

    variable_records = (failure_cloud_data or {}).get('variables', {}) or {}
    if len(variable_records) < 2:
        st.info("Failure surface ruang normal baku memerlukan minimal dua variabel acak yang tersimpan.")
        return

    sensitivity_results = (results_bundle or {}).get('sensitivity_results', {}) or {}
    ordered_variable_names = get_failure_cloud_variable_sort_order(
        failure_cloud_data,
        sensitivity_results=sensitivity_results
    )
    default_x, default_y = get_failure_cloud_default_variable_names(
        failure_cloud_data,
        sensitivity_results=sensitivity_results
    )
    default_x_3d, default_y_3d, default_z_3d = get_failure_cloud_default_three_variable_names(
        failure_cloud_data,
        sensitivity_results=sensitivity_results
    )
    if default_x is None or default_y is None:
        st.info("Variabel default failure cloud belum bisa ditentukan dari hasil saat ini.")
        return

    st.markdown(f"{heading_level} Nonlinear Contour g(x)=0")
    st.caption(
        "Plot ini menampilkan `failure cloud` hasil Monte Carlo pada pasangan dua variabel "
        "acak yang dipilih, setelah ditransformasikan ke ruang normal baku `U`."
    )
    st.caption(
        "Titik `biru` adalah sampel `safe`, titik `merah` adalah sampel `failed`, "
        "dan garis hitam tebal merepresentasikan `nonlinear contour g_hat(u) = 0` "
        "yang dibentuk dari pemisahan kedua kelompok tersebut."
    )
    st.caption(
        "Warna latar dan garis kontur tipis menunjukkan `signed margin g_hat(u)` hasil "
        "aproksimasi berbasis data SMC. Lingkar putus-putus `beta` dan titik origin "
        "ditampilkan agar pembacaan posisi cloud pada `standard normal space` lebih mudah."
    )

    if failure_cloud_data.get('used_downsampling'):
        st.caption(
            "Untuk menjaga performa dashboard, estimasi surface ini dibangun dari subset "
            "sampel Monte Carlo yang dipilih merata dari hasil simulasi."
        )
    if failure_cloud_data.get('failed_points_truncated'):
        st.caption(
            "Jumlah sampel gagal yang dipakai untuk estimasi surface sudah dipadatkan karena "
            "jumlah kejadian gagal melebihi batas penyimpanan visual."
        )

    x_index = ordered_variable_names.index(default_x)
    y_index = ordered_variable_names.index(default_y)
    selector_cols = st.columns(3)
    x_variable_name = selector_cols[0].selectbox(
        "Sumbu X",
        options=ordered_variable_names,
        index=x_index,
        format_func=lambda var_name: format_failure_cloud_variable_label(
            variable_records.get(var_name, {}),
            short=False
        ),
        key="failure_cloud_x_selector"
    )
    y_variable_name = selector_cols[1].selectbox(
        "Sumbu Y",
        options=ordered_variable_names,
        index=y_index,
        format_func=lambda var_name: format_failure_cloud_variable_label(
            variable_records.get(var_name, {}),
            short=False
        ),
        key="failure_cloud_y_selector"
    )
    z_variable_name = None
    if default_z_3d is not None and default_z_3d in ordered_variable_names:
        z_index = ordered_variable_names.index(default_z_3d)
    else:
        z_index = min(2, len(ordered_variable_names) - 1)
    if len(ordered_variable_names) >= 3:
        z_variable_name = selector_cols[2].selectbox(
            "Sumbu Z",
            options=ordered_variable_names,
            index=z_index,
            format_func=lambda var_name: format_failure_cloud_variable_label(
                variable_records.get(var_name, {}),
                short=False
            ),
            key="failure_cloud_z_selector"
        )
    else:
        selector_cols[2].info("Minimal 3 variabel acak diperlukan untuk plot 3D.")

    if str(x_variable_name) == str(y_variable_name):
        st.warning("Pilih variabel `X` dan `Y` yang berbeda agar failure surface informatif.")
        return

    failure_surface_fig, failure_surface_meta = resolve_failure_surface_figure_with_fallback(
        failure_cloud_data,
        ordered_variable_names=ordered_variable_names,
        x_variable_name=str(x_variable_name),
        y_variable_name=str(y_variable_name)
    )
    if failure_surface_fig is not None:
        used_requested_pair = bool(
            (failure_surface_meta or {}).get('used_requested_pair', True)
        )
        resolved_x_name = str(
            (failure_surface_meta or {}).get('x_variable_name', str(x_variable_name))
        )
        resolved_y_name = str(
            (failure_surface_meta or {}).get('y_variable_name', str(y_variable_name))
        )
        if not used_requested_pair:
            resolved_x_label = format_failure_cloud_variable_label(
                variable_records.get(resolved_x_name, {}),
                short=False
            )
            resolved_y_label = format_failure_cloud_variable_label(
                variable_records.get(resolved_y_name, {}),
                short=False
            )
            estimator_method = str(
                (failure_surface_meta or {}).get('estimator_method', 'unknown')
            ).upper()
            st.caption(
                "Pasangan variabel yang dipilih tidak menghasilkan contour yang stabil, "
                f"sehingga surface dialihkan otomatis ke pasangan `{resolved_x_label}` vs "
                f"`{resolved_y_label}` dengan estimator `{estimator_method}`."
            )
        render_plot(
            failure_surface_fig,
            interactive=True,
            viewer_key=f"failure-surface-{sanitize_dom_id(str(resolved_x_name))}-{sanitize_dom_id(str(resolved_y_name))}",
            alt_text="Failure surface probabilistik Monte Carlo",
            viewer_height=700,
            download_basename=f"failure-surface-{resolved_x_name}-vs-{resolved_y_name}"
        )
    else:
        diagnostic_parts = []
        estimator_method = str(
            (failure_surface_meta or {}).get('estimator_method', '')
        ).strip()
        if estimator_method:
            diagnostic_parts.append(f"estimator terakhir: `{estimator_method.upper()}`")
        score_min = coerce_finite_float((failure_surface_meta or {}).get('score_min'))
        score_max = coerce_finite_float((failure_surface_meta or {}).get('score_max'))
        if score_min is not None and score_max is not None:
            diagnostic_parts.append(
                f"rentang score grid: `{score_min:.4f}` s.d. `{score_max:.4f}`"
            )
        unavailable_message, unavailable_detail = build_failure_surface_unavailable_message(
            results_bundle,
            failure_cloud_data
        )
        st.info(unavailable_message)
        if unavailable_detail:
            st.caption(unavailable_detail)
        if diagnostic_parts:
            st.caption("Diagnostik: " + " | ".join(diagnostic_parts))

    st.markdown(f"{heading_level} Failure Surface 3D")
    st.caption(
        "Panel ini menampilkan `failure cloud` tiga dimensi pada `(u1, u2, u3)` "
        "beserta permukaan aproksimasi batas `g(u)=0`, seperti inset perspektif 3D pada contoh."
    )
    st.caption(
        "Jika data memadai, permukaan 3D dibentuk sebagai `surface kuadratik/nonlinear`. "
        "Jika tidak, plot otomatis fallback ke bidang linear. Titik `MPP` estimasi dan "
        "garis `beta` dari origin ke MPP juga ditampilkan pada panel ini."
    )
    if len(ordered_variable_names) < 3 or z_variable_name is None:
        st.info("Plot 3D belum tersedia karena variabel acak yang tersimpan kurang dari 3.")
        return

    resolved_z_variable_name = str(z_variable_name)
    if resolved_z_variable_name in {str(x_variable_name), str(y_variable_name)}:
        auto_z_variable_name = get_first_distinct_failure_cloud_variable_name(
            ordered_variable_names,
            excluded_names=[str(x_variable_name), str(y_variable_name)]
        )
        if auto_z_variable_name is None:
            st.info("Plot 3D belum dapat dibentuk karena tidak ada variabel `Z` yang berbeda.")
            return
        resolved_z_variable_name = str(auto_z_variable_name)
        resolved_z_label = format_failure_cloud_variable_label(
            variable_records.get(resolved_z_variable_name, {}),
            short=False
        )
        st.caption(
            f"Sumbu `Z` otomatis dialihkan ke `{resolved_z_label}` agar berbeda dari `X` dan `Y`."
        )

    plotly_figure_key = (
        "failure-surface-3d-"
        f"{sanitize_dom_id(str(x_variable_name))}-"
        f"{sanitize_dom_id(str(y_variable_name))}-"
        f"{sanitize_dom_id(str(resolved_z_variable_name))}"
    )
    failure_surface_3d_plotly_fig = build_failure_surface_3d_plotly_figure(
        failure_cloud_data,
        x_variable_name=str(x_variable_name),
        y_variable_name=str(y_variable_name),
        z_variable_name=str(resolved_z_variable_name)
    )
    if failure_surface_3d_plotly_fig is not None:
        st.caption(
            "Gunakan mouse untuk eksplorasi: `drag` untuk memutar, `scroll` untuk zoom, "
            "dan gunakan toolbar grafik untuk reset atau mode kamera lainnya."
        )
        st.plotly_chart(
            failure_surface_3d_plotly_fig,
            use_container_width=True,
            key=plotly_figure_key,
            config={
                'displaylogo': False,
                'scrollZoom': True,
                'responsive': True
            }
        )
    else:
        if go is None:
            st.caption(
                "Viewer 3D interaktif belum aktif karena `plotly` tidak tersedia pada "
                "environment ini, sehingga panel memakai fallback gambar statis."
            )
        failure_surface_3d_fig = build_failure_surface_3d_figure(
            failure_cloud_data,
            x_variable_name=str(x_variable_name),
            y_variable_name=str(y_variable_name),
            z_variable_name=str(resolved_z_variable_name)
        )
        if failure_surface_3d_fig is not None:
            render_plot(
                failure_surface_3d_fig,
                interactive=True,
                viewer_key=plotly_figure_key,
                alt_text="Failure surface 3D probabilistik Monte Carlo",
                viewer_height=760,
                tight_bbox=False,
                download_basename=(
                    f"failure-surface-3d-{x_variable_name}-"
                    f"{y_variable_name}-{resolved_z_variable_name}"
                )
            )
        else:
            unavailable_message, unavailable_detail = build_failure_surface_unavailable_message(
                results_bundle,
                failure_cloud_data
            )
            st.info(unavailable_message)
            if unavailable_detail:
                st.caption(unavailable_detail)


def get_probabilistic_mc_convergence_state_specs() -> List[Dict[str, str]]:
    """Spesifikasi warna dan label limit-state pada tab Simulasi MC."""
    return [
        {
            'key': 'moment',
            'label': 'Lentur',
            'plot_label': 'Flexure',
            'unit': 'kN.m',
            'color': '#f59e0b'
        },
        {
            'key': 'shear',
            'label': 'Geser',
            'plot_label': 'Shear',
            'unit': 'kN',
            'color': '#16a34a'
        },
        {
            'key': 'axial',
            'label': 'Aksial',
            'plot_label': 'Axial',
            'unit': 'kN',
            'color': '#2563eb'
        },
        {
            'key': 'axial_moment',
            'label': 'Aksial+Lentur',
            'plot_label': 'Axial-Flexure Interaction',
            'unit': '(-)',
            'color': '#7c3aed'
        }
    ]


def get_probabilistic_mc_convergence_element_record(convergence_data: Dict,
                                                    elem_id: int) -> Dict[str, Any]:
    """Ambil record konvergensi per elemen dari hasil MC."""
    elements_data = (convergence_data or {}).get('elements', {}) or {}
    return (
        elements_data.get(str(int(elem_id)))
        or elements_data.get(int(elem_id))
        or {}
    )


def build_probabilistic_mc_system_convergence_figure(
    convergence_data: Dict
) -> Optional[plt.Figure]:
    """Bangun figure konvergensi Pf dan Beta sistem portal."""
    sample_counts = np.asarray(
        (convergence_data or {}).get('sample_counts', []),
        dtype=float
    )
    system_record = (convergence_data or {}).get('system', {}) or {}
    if sample_counts.size == 0 or not system_record:
        return None

    pf_values = np.asarray(system_record.get('pf', []), dtype=float)
    beta_values = np.asarray(system_record.get('beta', []), dtype=float)
    if pf_values.size == 0 and beta_values.size == 0:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.8), dpi=180, sharex=True)
    axes_list = list(np.asarray(axes).reshape(-1))
    marker_style = 'o' if sample_counts.size <= 32 else None
    system_color = '#0f4c81'

    panel_specs = [
        {
            'axis': axes_list[0],
            'values': pf_values,
            'title': 'Cumulative System Pf Convergence',
            'ylabel': 'System Pf (-)',
            'ylim': (-0.02, 1.02),
            'legend_label': 'System Pf'
        },
        {
            'axis': axes_list[1],
            'values': beta_values,
            'title': 'Cumulative System Beta Convergence',
            'ylabel': 'System Beta (-)',
            'ylim': None,
            'legend_label': 'System Beta'
        }
    ]

    for panel_spec in panel_specs:
        axis = panel_spec['axis']
        values = panel_spec['values']
        if values.size == 0 or np.all(np.isnan(values)):
            axis.text(
                0.5,
                0.5,
                'Data is unavailable for this panel.',
                ha='center',
                va='center',
                transform=axis.transAxes,
                fontsize=10,
                color='#64748b',
                bbox=dict(
                    boxstyle='round,pad=0.25',
                    facecolor='#f8fafc',
                    edgecolor='#cbd5e1'
                )
            )
        else:
            x_values = sample_counts[:values.size]
            axis.plot(
                x_values,
                values,
                color=system_color,
                linewidth=2.2,
                marker=marker_style,
                markersize=3.4 if marker_style else 0.0,
                label=panel_spec['legend_label']
            )
        axis.set_title(panel_spec['title'], fontsize=11, pad=10)
        axis.set_ylabel(panel_spec['ylabel'])
        axis.grid(True, alpha=0.24, linestyle='--')
        if panel_spec['ylim'] is not None:
            axis.set_ylim(*panel_spec['ylim'])
        if values.size != 0 and not np.all(np.isnan(values)):
            axis.legend(loc='best', fontsize=8)

    axes_list[-1].set_xlabel('Number of Simulations, N (-)')
    axes_list[0].text(
        0.98,
        0.08,
        (
            f"Failure Count = {int(system_record.get('final_failures', 0))}\n"
            f"Final Pf = {format_metric(system_record.get('pf_final'), 6)}\n"
            f"Final Beta = {format_metric(system_record.get('beta_final'), 4)}"
        ),
        transform=axes_list[0].transAxes,
        ha='right',
        va='bottom',
        fontsize=8.5,
        bbox=dict(
            boxstyle='round,pad=0.25',
            facecolor='white',
            alpha=0.88,
            edgecolor='#cbd5e1'
        )
    )
    fig.suptitle(
        "Monte Carlo Simulation Convergence | Structural System",
        fontsize=13,
        y=0.985
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig


def build_probabilistic_mc_convergence_summary_df(convergence_data: Dict,
                                                  elem_id: int) -> pd.DataFrame:
    """Bangun ringkasan akhir konvergensi MC untuk satu elemen."""
    element_record = get_probabilistic_mc_convergence_element_record(
        convergence_data,
        elem_id
    )
    if not element_record:
        return pd.DataFrame()

    rows = []
    element_states = element_record.get('states', {}) or {}
    for spec in get_probabilistic_mc_convergence_state_specs():
        state_record = element_states.get(spec['key'], {}) or {}
        if not state_record.get('applicable'):
            continue

        rows.append({
            'Limit State': state_record.get('label', spec['label']),
            'g Mean Kumulatif': state_record.get('g_mean_final'),
            'Pf Akhir (-)': state_record.get('pf_final'),
            'Beta Akhir (-)': state_record.get('beta_final'),
            'Jumlah Gagal (-)': state_record.get('final_failures'),
            'Sampel g Valid (-)': state_record.get('g_valid_samples'),
            'Satuan g': state_record.get('unit', spec['unit'])
        })

    return pd.DataFrame(rows)


def build_probabilistic_mc_convergence_figure(convergence_data: Dict,
                                              input_data: Optional[Dict],
                                              elem_id: int) -> Optional[plt.Figure]:
    """Bangun figure konvergensi MC per elemen."""
    sample_counts = np.asarray(
        (convergence_data or {}).get('sample_counts', []),
        dtype=float
    )
    if sample_counts.size == 0:
        return None

    element_record = get_probabilistic_mc_convergence_element_record(
        convergence_data,
        elem_id
    )
    if not element_record:
        return None

    state_specs = get_probabilistic_mc_convergence_state_specs()
    series_specs = [
        {
            'series_key': 'g_running_mean',
            'title': 'Cumulative Mean g(x) Convergence',
            'ylabel': 'Cumulative Mean g(x)'
        },
        {
            'series_key': 'pf',
            'title': 'Cumulative Pf Convergence',
            'ylabel': 'Pf (-)'
        },
        {
            'series_key': 'beta',
            'title': 'Cumulative Beta Convergence',
            'ylabel': 'Beta (-)'
        }
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 10.2), dpi=180, sharex=True)
    axes_list = list(np.asarray(axes).reshape(-1))
    element_states = element_record.get('states', {}) or {}
    marker_style = 'o' if sample_counts.size <= 32 else None

    for axis, series_spec in zip(axes_list, series_specs):
        plotted_any = False
        for state_spec in state_specs:
            state_record = element_states.get(state_spec['key'], {}) or {}
            if not state_record.get('applicable'):
                continue

            series_values = np.asarray([
                np.nan if value is None else float(value)
                for value in state_record.get(series_spec['series_key'], [])
            ], dtype=float)
            if series_values.size == 0 or np.all(np.isnan(series_values)):
                continue

            x_values = sample_counts[:series_values.size]
            axis.plot(
                x_values,
                series_values,
                color=state_spec['color'],
                linewidth=2.0,
                marker=marker_style,
                markersize=3.2 if marker_style else 0.0,
                label=state_spec.get('plot_label', state_spec['label'])
            )
            plotted_any = True

        axis.set_title(series_spec['title'], fontsize=11, pad=10)
        axis.set_ylabel(series_spec['ylabel'])
        axis.grid(True, alpha=0.24, linestyle='--')
        if series_spec['series_key'] == 'pf':
            axis.set_ylim(-0.02, 1.02)

        if plotted_any:
            axis.legend(loc='best', fontsize=8, ncol=2)
        else:
            axis.text(
                0.5,
                0.5,
                'Data is unavailable for this panel.',
                ha='center',
                va='center',
                transform=axis.transAxes,
                fontsize=10,
                color='#64748b',
                bbox=dict(
                    boxstyle='round,pad=0.25',
                    facecolor='#f8fafc',
                    edgecolor='#cbd5e1'
                )
            )

    axes_list[-1].set_xlabel('Number of Simulations, N (-)')

    element_code = str(
        element_record.get('code') or get_element_code_from_input(input_data, int(elem_id))
    ).strip().upper()
    element_type = get_element_type_label(element_code)
    element_type_plot = {
        'Balok': 'Beam',
        'Kolom': 'Column'
    }.get(element_type, element_type)
    fig.suptitle(
        f"Monte Carlo Simulation Convergence | E{int(elem_id)} | {element_type_plot}",
        fontsize=13,
        y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig


def render_probabilistic_mc_convergence_output_section(results_bundle: Dict,
                                                       input_data: Optional[Dict],
                                                       heading_level: str = "####") -> None:
    """Tampilkan tab Simulasi MC untuk mode probabilistik."""
    convergence_data = (
        (results_bundle or {}).get('probabilistic_mc_convergence_data', {}) or {}
    )
    if not convergence_data:
        st.info(
            "Data konvergensi Monte Carlo belum tersedia. Jalankan analisis probabilistik "
            "agar tab `Simulasi MC` dapat menampilkan kurva per elemen."
        )
        return

    sample_counts = list(convergence_data.get('sample_counts', []) or [])
    beam_ids = [
        int(elem_id)
        for elem_id in (convergence_data.get('element_groups', {}).get('beam', []) or [])
    ]
    column_ids = [
        int(elem_id)
        for elem_id in (convergence_data.get('element_groups', {}).get('column', []) or [])
    ]

    st.markdown(f"{heading_level} Simulasi Monte Carlo per Elemen")
    st.caption(
        "Sumbu `X` menunjukkan jumlah simulasi `N`. Kurva `g` ditampilkan sebagai "
        "rata-rata kumulatif nilai `g(x)` per elemen, sedangkan `Pf/Beta` dihitung "
        "kumulatif per elemen untuk limit state lentur, geser, aksial, dan aksial-lentur."
    )
    st.caption(
        "Setiap elemen ditampilkan dalam `expander` terpisah. Grafik mendukung zoom "
        "dengan scroll mouse atau pinch."
    )
    beta_plot_cap = coerce_finite_float(convergence_data.get('beta_plot_cap'))
    if beta_plot_cap is not None and beta_plot_cap > 0.0:
        st.caption(
            f"Jika `Pf = 0` atau `Pf = 1`, maka `Beta` teoritis menjadi tak hingga. "
            f"Untuk kebutuhan visual grafik, nilai tersebut dipotong pada `+/-{beta_plot_cap:.1f}`, "
            "sedangkan tabel ringkasan tetap menampilkan nilai akhir aslinya."
        )

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Jumlah Simulasi",
        format_metric_comma(convergence_data.get('num_simulations'), 0)
    )
    metric_cols[1].metric("Jumlah Checkpoint", format_metric_comma(len(sample_counts), 0))
    metric_cols[2].metric("Jumlah Balok", format_metric_comma(len(beam_ids), 0))
    metric_cols[3].metric("Jumlah Kolom", format_metric_comma(len(column_ids), 0))

    analysis_failures = int(convergence_data.get('analysis_failures', 0) or 0)
    if analysis_failures > 0:
        st.warning(
            f"{analysis_failures:,} simulasi tidak menghasilkan output struktur valid. "
            "Simulasi gagal tetap diperhitungkan konservatif pada kurva `Pf/Beta`, "
            "sedangkan kurva `g` hanya dirata-ratakan dari simulasi yang valid."
        )

    system_record = (convergence_data or {}).get('system', {}) or {}
    st.markdown("##### Sistem Portal")
    st.caption(
        "Grafik berikut menunjukkan konvergensi `Pf system` dan `Beta system` portal "
        "secara kumulatif terhadap jumlah simulasi `N`."
    )
    system_metric_cols = st.columns(3)
    system_metric_cols[0].metric(
        "Jumlah Gagal Sistem",
        format_metric_comma(system_record.get('final_failures'), 0)
    )
    system_metric_cols[1].metric(
        "Pf System Akhir",
        format_metric(system_record.get('pf_final'), 6)
    )
    system_metric_cols[2].metric(
        "Beta System Akhir",
        format_metric(system_record.get('beta_final'), 4)
    )

    system_fig = build_probabilistic_mc_system_convergence_figure(convergence_data)
    if system_fig is not None:
        render_plot(
            system_fig,
            interactive=True,
            viewer_key="probabilistic-mc-convergence-system",
            alt_text="Konvergensi Monte Carlo sistem portal",
            viewer_height=640,
            download_basename="simulasi-mc-sistem-portal"
        )
    else:
        st.info("Grafik konvergensi sistem portal belum tersedia.")

    group_specs = [
        ('Balok', beam_ids),
        ('Kolom', column_ids)
    ]
    for group_label, element_ids in group_specs:
        st.markdown(f"##### {group_label}")
        if not element_ids:
            st.info(f"Tidak ada elemen {group_label.lower()} yang dapat ditampilkan.")
            continue

        for index, elem_id in enumerate(element_ids):
            element_record = get_probabilistic_mc_convergence_element_record(
                convergence_data,
                elem_id
            )
            if not element_record:
                continue

            with st.expander(
                f"E{int(elem_id)} | {group_label}",
                expanded=(index == 0)
            ):
                convergence_fig = build_probabilistic_mc_convergence_figure(
                    convergence_data,
                    input_data=input_data,
                    elem_id=int(elem_id)
                )
                if convergence_fig is not None:
                    render_plot(
                        convergence_fig,
                        interactive=True,
                        viewer_key=f"probabilistic-mc-convergence-e{int(elem_id)}",
                        alt_text=f"Konvergensi Monte Carlo elemen {int(elem_id)}",
                        viewer_height=760,
                        download_basename=f"simulasi-mc-e{int(elem_id)}"
                    )
                else:
                    st.info("Grafik konvergensi belum bisa dibentuk untuk elemen ini.")

                summary_df = build_probabilistic_mc_convergence_summary_df(
                    convergence_data,
                    elem_id=int(elem_id)
                )
                st.caption(
                    "Ringkasan berikut menampilkan nilai akhir pada checkpoint terakhir "
                    "untuk setiap limit state yang relevan pada elemen ini."
                )
                if summary_df.empty:
                    st.info("Ringkasan konvergensi untuk elemen ini belum tersedia.")
                else:
                    render_input_table(
                        summary_df,
                        styler=style_input_dataframe(
                            summary_df,
                            table_min_width_px=1100
                        )
                    )


def build_sensitivity_df(sensitivity_results: Dict,
                         beta_system: Optional[float] = None) -> pd.DataFrame:
    rows = build_sensitivity_rows(
        sensitivity_results,
        beta_system=beta_system
    )
    return pd.DataFrame(rows)


def build_sensitivity_rows(sensitivity_results: Dict,
                           beta_system: Optional[float] = None) -> List[Dict[str, Any]]:
    """Bangun baris tabel sensitivitas dan kontribusi terhadap beta sistem."""
    ranked_items = []
    for variable, values in (sensitivity_results or {}).items():
        try:
            sensitivity_index = float(values.get('sensitivity_index', 0.0))
        except (TypeError, ValueError):
            sensitivity_index = 0.0
        if not np.isfinite(sensitivity_index):
            sensitivity_index = 0.0

        ranked_items.append({
            'variable': str(variable),
            'sensitivity_index': sensitivity_index,
            'mean_in_failure': values.get('mean_in_failure'),
            'mean_overall': values.get('mean_overall'),
            'std_overall': values.get('std_overall')
        })

    ranked_items.sort(
        key=lambda item: (abs(float(item['sensitivity_index'])), item['variable']),
        reverse=True
    )

    total_sensitivity = float(sum(abs(float(item['sensitivity_index'])) for item in ranked_items))
    rows = []
    for rank, item in enumerate(ranked_items, 1):
        absolute_sensitivity = abs(float(item['sensitivity_index']))
        importance_factor = (
            (absolute_sensitivity / total_sensitivity) * 100.0
            if total_sensitivity > 0.0 else
            0.0
        )
        delta_mean = compute_sensitivity_delta(
            item['mean_in_failure'],
            item['mean_overall']
        )
        beta_contribution = compute_beta_contribution(
            beta_system,
            importance_factor,
            item['variable'],
            delta_mean
        )
        rows.append({
            'Peringkat (-)': int(rank),
            'Variabel Acak (-)': item['variable'],
            'Indeks Sensitivitas (-)': item['sensitivity_index'],
            'Faktor Kepentingan (%)': importance_factor,
            'Kontribusi terhadap Beta, beta_i (-)': beta_contribution,
            'Rata-rata pada Sampel Gagal': item['mean_in_failure'],
            'Rata-rata Keseluruhan': item['mean_overall'],
            'Δ = μ_gagal - μ_total': delta_mean,
            'Simpangan Baku Keseluruhan': item['std_overall'],
            'Interpretasi Teknis': build_sensitivity_interpretation(
                item['variable'],
                delta_mean
            )
        })

    return rows


def compute_sensitivity_delta(mean_in_failure, mean_overall) -> Optional[float]:
    """Hitung selisih rata-rata sampel gagal terhadap rata-rata keseluruhan."""
    try:
        failure_value = float(mean_in_failure)
        overall_value = float(mean_overall)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(failure_value) or not np.isfinite(overall_value):
        return None
    return float(failure_value - overall_value)


def get_beta_direction_sign(variable_name: str) -> Optional[float]:
    """Arah pengaruh kenaikan variabel terhadap beta."""
    prefix = str(variable_name or '').split('_E', 1)[0]
    if prefix in {'fb', 'fc', 'fy_tarik', 'fy_tekan', 'fy_geser'}:
        return 1.0
    if prefix in {'qDL', 'qLL'}:
        return -1.0
    return None


def compute_beta_contribution(beta_system: Optional[float],
                              importance_factor: Optional[float],
                              variable_name: str,
                              delta_mean: Optional[float]) -> Optional[float]:
    """
    Hitung kontribusi kuantitatif tiap variabel terhadap beta sistem.

    Nilai dihitung dengan mengalokasikan beta sistem secara proporsional terhadap
    faktor kepentingan sensitivitas absolut, lalu diberi tanda sesuai arah
    pengaruh kenaikan variabel terhadap beta.
    """
    try:
        beta_value = abs(float(beta_system))
        importance_value = float(importance_factor)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(beta_value) or not np.isfinite(importance_value):
        return None

    beta_magnitude = beta_value * max(importance_value, 0.0) / 100.0
    direction_sign = get_beta_direction_sign(variable_name)

    if direction_sign is None:
        if delta_mean is None:
            return beta_magnitude
        try:
            delta_value = float(delta_mean)
        except (TypeError, ValueError):
            return beta_magnitude
        if not np.isfinite(delta_value) or np.isclose(delta_value, 0.0):
            return beta_magnitude
        direction_sign = -1.0 if delta_value > 0.0 else 1.0

    return float(direction_sign * beta_magnitude)


def get_sensitivity_variable_label(variable_name: str) -> str:
    """Ubah nama variabel sensitivitas menjadi label yang lebih deskriptif."""
    variable_text = str(variable_name or '').strip()
    match = re.fullmatch(r'([A-Za-z_]+)_E(\d+)', variable_text)
    if not match:
        return variable_text or "variabel ini"

    prefix, elem_id = match.groups()
    mapping = {
        'fb': f"faktor bias modulus elastisitas beton elemen {elem_id}",
        'E': f"modulus elastisitas beton elemen {elem_id}",
        'fc': f"kuat tekan beton elemen {elem_id}",
        'fy_tarik': f"tegangan leleh baja tarik elemen {elem_id}",
        'fy_tekan': f"tegangan leleh baja tekan elemen {elem_id}",
        'fy_geser': f"tegangan leleh baja geser elemen {elem_id}",
        'qDL': f"beban mati terdistribusi elemen {elem_id}",
        'qLL': f"beban hidup terdistribusi elemen {elem_id}"
    }
    return mapping.get(prefix, variable_text)


def build_sensitivity_interpretation(variable_name: str,
                                     delta_mean: Optional[float]) -> str:
    """Bangun interpretasi teknis berdasarkan arah delta pada sampel gagal."""
    if delta_mean is None or not np.isfinite(float(delta_mean)):
        return "Interpretasi teknis belum dapat ditentukan karena data rata-rata tidak lengkap."

    variable_text = get_sensitivity_variable_label(variable_name)
    prefix = str(variable_name or '').split('_E', 1)[0]

    if np.isclose(float(delta_mean), 0.0, atol=1e-12, rtol=1e-9):
        return (
            f"Pada sampel gagal, {variable_text} tidak menunjukkan pergeseran rata-rata "
            "yang berarti terhadap populasi keseluruhan."
        )

    if prefix in {'fb', 'fc', 'fy_tarik', 'fy_tekan', 'fy_geser'}:
        if float(delta_mean) < 0.0:
            return (
                f"Pada sampel gagal, {variable_text} cenderung lebih rendah dari rata-rata "
                "keseluruhan; hal ini mengindikasikan penurunan kapasitas atau kekakuan "
                "berkorelasi dengan kejadian gagal."
            )
        return (
            f"Pada sampel gagal, {variable_text} cenderung lebih tinggi dari rata-rata "
            "keseluruhan; hal ini menunjukkan kondisi gagal dipengaruhi interaksi sistem "
            "yang perlu dibaca bersama variabel lain."
        )

    if prefix in {'qDL', 'qLL'}:
        if float(delta_mean) > 0.0:
            return (
                f"Pada sampel gagal, {variable_text} cenderung lebih tinggi dari rata-rata "
                "keseluruhan; hal ini mengindikasikan peningkatan demand berkontribusi "
                "terhadap kejadian gagal."
            )
        return (
            f"Pada sampel gagal, {variable_text} cenderung lebih rendah dari rata-rata "
            "keseluruhan; kondisi ini menunjukkan pengaruhnya perlu dibaca bersama "
            "kombinasi variabel acak lainnya."
        )

    if float(delta_mean) > 0.0:
        return (
            f"Pada sampel gagal, {variable_text} cenderung lebih tinggi dari rata-rata "
            "keseluruhan."
        )
    return (
        f"Pada sampel gagal, {variable_text} cenderung lebih rendah dari rata-rata "
        "keseluruhan."
    )


def build_sensitivity_tornado_figure(sensitivity_results: Dict,
                                     beta_system: Optional[float] = None,
                                     top_n: int = 15) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Bangun diagram Analisis Sensitivitas kuantitatif kontribusi variabel terhadap beta."""
    sensitivity_rows = build_sensitivity_rows(
        sensitivity_results,
        beta_system=beta_system
    )
    if not sensitivity_rows:
        return None

    display_rows = sensitivity_rows[:max(int(top_n), 1)]
    labels = [row['Variabel Acak (-)'] for row in display_rows]
    beta_contributions = [
        row.get('Kontribusi terhadap Beta, beta_i (-)')
        for row in display_rows
    ]
    if any(value is None for value in beta_contributions):
        return None
    beta_contributions = [float(value) for value in beta_contributions]
    importance_values = [float(row['Faktor Kepentingan (%)']) for row in display_rows]

    fig_height = float(min(max(4.8, 0.55 * len(display_rows) + 1.8), 12.5))
    fig, ax = plt.subplots(figsize=(11, fig_height))

    color_scale = [
        '#2563eb' if value >= 0.0 else '#dc2626'
        for value in beta_contributions
    ]
    y_positions = np.arange(len(display_rows))
    bars = ax.barh(
        y_positions,
        beta_contributions,
        color=color_scale,
        edgecolor='#1f2937',
        linewidth=0.8
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0.0, color='#111827', linewidth=1.0, alpha=0.85)
    ax.set_xlabel('Kontribusi terhadap Indeks Keandalan, beta_i (-)')
    ax.set_title('Diagram Analisis Sensitivitas Kuantitatif Kontribusi terhadap Beta')
    ax.grid(True, axis='x', alpha=0.25, linestyle='--')

    max_abs_value = max(abs(value) for value in beta_contributions) if beta_contributions else 0.0
    x_padding = max(0.05 * max_abs_value, 0.02)
    axis_limit = max_abs_value + 4.5 * x_padding if max_abs_value > 0.0 else 1.0
    ax.set_xlim(-axis_limit, axis_limit)

    for bar, importance_value, beta_value in zip(bars, importance_values, beta_contributions):
        x_coord = float(bar.get_width())
        y_coord = float(bar.get_y() + (bar.get_height() / 2.0))
        text_offset = x_padding if beta_value >= 0.0 else -x_padding
        horizontal_alignment = 'left' if beta_value >= 0.0 else 'right'
        ax.text(
            x_coord + text_offset,
            y_coord,
            f"{beta_value:+.3f} | {importance_value:.1f}%",
            va='center',
            ha=horizontal_alignment,
            fontsize=9,
            color='#111827'
        )

    fig.tight_layout()
    return fig, ax


def render_sensitivity_output_section(results_bundle: Dict,
                                      is_probabilistic: bool,
                                      heading_level: str = "####") -> None:
    """Tampilkan tornado diagram dan tabel sensitivitas di dashboard."""
    sensitivity_results = results_bundle.get('sensitivity_results', {}) if results_bundle else {}
    beta_system = (
        (results_bundle or {}).get('summary', {}).get('Beta')
        if results_bundle else
        None
    )
    sensitivity_df = build_sensitivity_df(
        sensitivity_results,
        beta_system=beta_system
    )

    if not is_probabilistic:
        st.info(
            "Pada mode deterministik tidak tersedia hasil analisis sensitivitas, "
            "karena tidak dilakukan pengambilan sampel variabel acak."
        )
        return

    if sensitivity_df.empty:
        st.info(
            "Data sensitivitas belum tersedia atau tidak terdapat sampel gagal "
            "yang memadai untuk dilakukan pemeringkatan."
        )
        return

    st.markdown(f"{heading_level} Diagram Analisis Sensitivitas Kuantitatif")
    st.caption(
        "Diagram Analisis Sensitivitas menampilkan hingga 15 variabel acak paling dominan berdasarkan "
        "besar kontribusinya terhadap indeks keandalan sistem. Batang berarah positif "
        "menunjukkan variabel yang kenaikannya cenderung menaikkan beta, sedangkan "
        "batang berarah negatif menunjukkan variabel yang kenaikannya cenderung "
        "menurunkan beta."
    )
    tornado_plot = build_sensitivity_tornado_figure(
        sensitivity_results,
        beta_system=beta_system,
        top_n=15
    )
    if tornado_plot is not None:
        tornado_fig, _ = tornado_plot
        render_plot(
            tornado_fig,
            interactive=False,
            alt_text="Diagram Analisis Sensitivitas kuantitatif kontribusi terhadap beta",
            download_basename="output-sensitivitas-probabilistik"
        )
    else:
        st.info(
            "Diagram Analisis Sensitivitas kuantitatif belum dapat dibentuk karena `Beta` sistem "
            "tidak tersedia dalam nilai terhingga."
        )

    st.markdown(f"{heading_level} Tabel Sensitivitas Variabel Acak")
    st.caption(
        "`Kontribusi terhadap Beta, beta_i (-)` dihitung dengan mengalokasikan "
        "`Beta` sistem secara proporsional terhadap `Faktor Kepentingan (%)` tiap "
        "variabel. Dengan demikian, jumlah magnitudo kontribusi seluruh variabel "
        "setara dengan beta sistem hasil simulasi."
    )
    st.caption(
        "Tanda positif pada `beta_i` menunjukkan bahwa kenaikan variabel tersebut "
        "secara teknis cenderung meningkatkan keandalan, sedangkan tanda negatif "
        "menunjukkan kecenderungan menurunkan keandalan."
    )
    st.caption(
        "Kolom `Δ = μ_gagal - μ_total` menyatakan selisih rata-rata nilai variabel "
        "pada sampel gagal terhadap rata-rata keseluruhan, sedangkan `Interpretasi Teknis` "
        "menjelaskan arah kecenderungan pengaruhnya terhadap kegagalan."
    )
    render_input_table(sensitivity_df)


def describe_deterministic_g_effect(delta_g_value: Optional[float]) -> str:
    """Ringkas arah pengaruh kenaikan +sigma terhadap margin keamanan."""
    if delta_g_value is None:
        return "-"

    try:
        numeric_value = float(delta_g_value)
    except (TypeError, ValueError):
        return "-"

    if not np.isfinite(numeric_value):
        return "-"
    if np.isclose(numeric_value, 0.0, atol=1e-12, rtol=1e-9):
        return "Hampir tidak mengubah margin keamanan"
    return (
        "Meningkatkan margin keamanan"
        if numeric_value > 0.0 else
        "Mengurangi margin keamanan"
    )


def _legacy_build_deterministic_sensitivity_interpretation(variable_name: str,
                                                           delta_g_plus: Optional[float],
                                                           perturbation_ratio: float,
                                                           limit_state_label: str) -> str:
    """Interpretasi teknis sensitivitas deterministik lokal berdasarkan skenario kenaikan."""
    if delta_g_plus is None or not np.isfinite(float(delta_g_plus)):
        return (
            "Interpretasi teknis belum dapat ditentukan karena skenario perturbasi "
            "tidak menghasilkan nilai g yang valid."
        )

    variable_text = get_sensitivity_variable_label(variable_name)
    perturbation_percent = abs(float(perturbation_ratio) * 100.0)
    delta_value = float(delta_g_plus)

    if np.isclose(delta_value, 0.0, atol=1e-12, rtol=1e-9):
        return (
            f"Jika {variable_text} dinaikkan {perturbation_percent:.0f}%, nilai g pada "
            f"limit state kontrol `{limit_state_label}` hampir tidak berubah."
        )

    if delta_value < 0.0:
        return (
            f"Jika {variable_text} dinaikkan {perturbation_percent:.0f}%, nilai g pada "
            f"limit state kontrol `{limit_state_label}` turun {abs(delta_value):.4f}; "
            "artinya perubahan ini cenderung mengurangi margin keamanan."
        )

    return (
        f"Jika {variable_text} dinaikkan {perturbation_percent:.0f}%, nilai g pada "
        f"limit state kontrol `{limit_state_label}` naik {abs(delta_value):.4f}; "
        "artinya perubahan ini cenderung meningkatkan margin keamanan."
    )


def _legacy_build_deterministic_sensitivity_rows(
    deterministic_sensitivity_results: Dict
) -> List[Dict[str, Any]]:
    """Bangun baris tabel sensitivitas deterministik lokal."""
    if not deterministic_sensitivity_results:
        return []

    baseline_info = deterministic_sensitivity_results.get('baseline', {}) or {}
    perturbation_ratio = float(
        deterministic_sensitivity_results.get('perturbation_ratio', 0.10) or 0.10
    )
    limit_state_label = str(
        baseline_info.get('limit_state_label', 'Kontrol')
    )

    ranked_items = list((deterministic_sensitivity_results.get('results') or {}).items())
    rows = []
    for rank, (variable_name, values) in enumerate(ranked_items, 1):
        rows.append({
            'Peringkat (-)': int(rank),
            'Variabel Deterministik (-)': variable_name,
            'Nilai Acuan': values.get('baseline_value'),
            'Satuan': values.get('unit'),
            'g jika +10%': values.get('g_plus'),
            'Δg jika +10%': values.get('delta_g_plus'),
            'g jika -10%': values.get('g_minus'),
            'Δg jika -10%': values.get('delta_g_minus'),
            'Efek Maksimum pada g Kontrol': values.get('sensitivity_index'),
            'Pengaruh terhadap Fungsi Kinerja (g(x))': describe_deterministic_g_effect(
                values.get('delta_g_plus')
            ),
            'Interpretasi Teknis': build_deterministic_sensitivity_interpretation(
                variable_name,
                values.get('delta_g_plus'),
                perturbation_ratio,
                limit_state_label
            )
        })

    return rows


def _legacy_build_deterministic_sensitivity_df(
    deterministic_sensitivity_results: Dict
) -> pd.DataFrame:
    """Bangun dataframe sensitivitas deterministik."""
    rows = build_deterministic_sensitivity_rows(deterministic_sensitivity_results)
    df = pd.DataFrame(rows)
    rename_map = {}
    for column in df.columns:
        column_text = str(column)
        if (
            ('g jika +10%' in column_text)
            and ('Delta' not in column_text)
            and (column_text != 'g jika +10%')
        ):
            rename_map[column] = 'Delta g jika +10%'
        elif (
            ('g jika -10%' in column_text)
            and ('Delta' not in column_text)
            and (column_text != 'g jika -10%')
        ):
            rename_map[column] = 'Delta g jika -10%'
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _legacy_build_deterministic_sensitivity_tornado_figure(
    deterministic_sensitivity_results: Dict,
    top_n: int = 15
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Bangun tornado chart sensitivitas deterministik +/ - perturbasi."""
    rows = build_deterministic_sensitivity_rows(deterministic_sensitivity_results)
    if not rows:
        return None

    display_rows = rows[:max(int(top_n), 1)]
    labels = [row['Variabel Deterministik (-)'] for row in display_rows]
    delta_plus = [
        float(value) if value is not None and np.isfinite(float(value)) else 0.0
        for value in (row.get('Δg jika +10%') for row in display_rows)
    ]
    delta_minus = [
        float(value) if value is not None and np.isfinite(float(value)) else 0.0
        for value in (row.get('Δg jika -10%') for row in display_rows)
    ]

    baseline_info = deterministic_sensitivity_results.get('baseline', {}) or {}
    perturbation_ratio = float(
        deterministic_sensitivity_results.get('perturbation_ratio', 0.10) or 0.10
    )
    perturbation_percent = abs(perturbation_ratio * 100.0)
    target_unit = str(baseline_info.get('unit', '-'))
    target_label = str(baseline_info.get('limit_state_label', 'Kontrol'))

    fig_height = float(min(max(4.8, 0.62 * len(display_rows) + 1.8), 13.0))
    fig, ax = plt.subplots(figsize=(11, fig_height))

    y_positions = np.arange(len(display_rows))
    bar_height = 0.34
    ax.barh(
        y_positions - (bar_height / 2.0),
        delta_minus,
        height=bar_height,
        color='#2563eb',
        edgecolor='#1f2937',
        linewidth=0.8,
        label=f"-{perturbation_percent:.0f}%"
    )
    ax.barh(
        y_positions + (bar_height / 2.0),
        delta_plus,
        height=bar_height,
        color='#dc2626',
        edgecolor='#1f2937',
        linewidth=0.8,
        label=f"+{perturbation_percent:.0f}%"
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0.0, color='#111827', linewidth=1.0, alpha=0.85)
    ax.set_xlabel(f"Perubahan g minimum pada limit state kontrol ({target_unit})")
    ax.set_title(
        f"Diagram Sensitivitas Deterministik terhadap g minimum limit state kontrol: {target_label}"
    )
    ax.grid(True, axis='x', alpha=0.25, linestyle='--')
    ax.legend(loc='lower right')

    max_abs_value = max(
        [abs(value) for value in delta_plus + delta_minus],
        default=0.0
    )
    x_padding = max(0.05 * max_abs_value, 0.02)
    axis_limit = max_abs_value + 4.0 * x_padding if max_abs_value > 0.0 else 1.0
    ax.set_xlim(-axis_limit, axis_limit)

    fig.tight_layout()
    return fig, ax


def _legacy_render_deterministic_sensitivity_output_section(results_bundle: Dict,
                                                            heading_level: str = "####") -> None:
    """Tampilkan sensitivitas deterministik lokal pada tab khusus."""
    deterministic_results = (
        results_bundle.get('deterministic_sensitivity_results', {})
        if results_bundle else
        {}
    )
    sensitivity_df = build_deterministic_sensitivity_df(deterministic_results)

    if sensitivity_df.empty:
        st.info(
            "Data sensitivitas deterministik belum tersedia. Jalankan analisis "
            "deterministik untuk membentuk hasil perturbasi one-at-a-time."
        )
        return

    baseline_info = deterministic_results.get('baseline', {}) or {}
    perturbation_ratio = float(
        deterministic_results.get('perturbation_ratio', 0.10) or 0.10
    )
    perturbation_percent = abs(perturbation_ratio * 100.0)
    target_label = str(baseline_info.get('limit_state_label', '-'))
    target_unit = str(baseline_info.get('unit', '-'))
    baseline_g = baseline_info.get('g_value')
    analysis_failures = int(deterministic_results.get('analysis_failures', 0) or 0)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Limit State Kontrol", target_label)
    metric_cols[1].metric("g Acuan", format_metric(baseline_g, 4))
    metric_cols[2].metric("Perturbasi", f"+/-{perturbation_percent:.0f}%")
    metric_cols[3].metric("Analisis Gagal", str(analysis_failures))

    st.markdown(f"{heading_level} Diagram Sensitivitas Deterministik")
    st.caption(
        "Diagram tornado berikut dibentuk dari pendekatan `one-at-a-time`: setiap variabel "
        f"deterministik digeser sendiri sebesar `+/-{perturbation_percent:.0f}%` terhadap nilai acuan, "
        f"lalu perubahan `g minimum` pada jenis limit state kontrol baseline `{target_label}` dibandingkan. "
        "Batang bernilai positif berarti margin keamanan meningkat, sedangkan batang negatif "
        "berarti margin keamanan menurun."
    )
    st.caption(
        "Jenis limit state yang dibandingkan tetap sama pada semua skenario perturbasi. "
        "Yang dapat berubah adalah elemen pengontrol yang menghasilkan nilai `g minimum` tersebut."
    )
    tornado_plot = build_deterministic_sensitivity_tornado_figure(
        deterministic_results,
        top_n=15
    )
    if tornado_plot is not None:
        tornado_fig, _ = tornado_plot
        render_plot(
            tornado_fig,
            interactive=False,
            alt_text="Diagram sensitivitas deterministik one-at-a-time"
        )
    else:
        st.info("Diagram sensitivitas deterministik belum dapat dibentuk.")

    st.markdown(f"{heading_level} Tabel Sensitivitas Deterministik")
    st.caption(
        f"Kolom `g jika +/-10%` menunjukkan nilai `g minimum` pada limit state kontrol "
        f"`{target_label}` ({target_unit}) setelah satu variabel digeser dan variabel lain "
        "dipertahankan pada baseline."
    )
    st.caption(
        "Kolom `Efek Maksimum pada g Kontrol` adalah magnitudo perubahan terbesar dari dua skenario "
        "perturbasi. Pengaruh terhadap Fungsi Kinerja g(x) menunjukkan arah perubahan yang menghasilkan efek terbesar itu."
    )
    render_input_table(sensitivity_df)


def build_deterministic_sensitivity_interpretation(variable_name: str,
                                                   delta_g_plus: Optional[float],
                                                   sigma_value: Optional[float],
                                                   perturbation_ratio: Optional[float],
                                                   limit_state_label: str) -> str:
    """Interpretasi teknis sensitivitas deterministik lokal berbasis COV."""
    if delta_g_plus is None or not np.isfinite(float(delta_g_plus)):
        return (
            "Interpretasi teknis belum dapat ditentukan karena skenario perturbasi "
            "tidak menghasilkan nilai g yang valid."
        )

    variable_text = get_sensitivity_variable_label(variable_name)
    sigma_numeric = float(sigma_value or 0.0)
    perturbation_percent = abs(float(perturbation_ratio or 0.0) * 100.0)
    delta_value = float(delta_g_plus)

    if np.isclose(delta_value, 0.0, atol=1e-12, rtol=1e-9):
        return (
            f"Jika {variable_text} dinaikkan sebesar `+sigma` "
            f"({sigma_numeric:.4f}; {perturbation_percent:.2f}%), nilai g pada limit state kontrol "
            f"`{limit_state_label}` hampir tidak berubah."
        )

    if delta_value < 0.0:
        return (
            f"Jika {variable_text} dinaikkan sebesar `+sigma` "
            f"({sigma_numeric:.4f}; {perturbation_percent:.2f}%), nilai g pada limit state kontrol "
            f"`{limit_state_label}` turun {abs(delta_value):.4f}; artinya perubahan ini "
            "cenderung mengurangi margin keamanan."
        )

    return (
        f"Jika {variable_text} dinaikkan sebesar `+sigma` "
        f"({sigma_numeric:.4f}; {perturbation_percent:.2f}%), nilai g pada limit state kontrol "
        f"`{limit_state_label}` naik {abs(delta_value):.4f}; artinya perubahan ini "
        "cenderung meningkatkan margin keamanan."
    )


def build_deterministic_sensitivity_rows(
    deterministic_sensitivity_results: Dict
) -> List[Dict[str, Any]]:
    """Bangun baris tabel sensitivitas deterministik lokal berbasis COV."""
    if not deterministic_sensitivity_results:
        return []

    baseline_info = deterministic_sensitivity_results.get('baseline', {}) or {}
    cov_scale = float(
        deterministic_sensitivity_results.get('cov_scale', 1.0) or 1.0
    )
    limit_state_label = str(baseline_info.get('limit_state_label', 'Kontrol'))

    ranked_items = list((deterministic_sensitivity_results.get('results') or {}).items())
    rows = []
    for rank, (variable_name, values) in enumerate(ranked_items, 1):
        rows.append({
            'Peringkat (-)': int(rank),
            'Variabel Deterministik (-)': variable_name,
            'Nilai Acuan Deterministik': values.get('baseline_value'),
            'Mean Acuan': values.get('mean_value'),
            'StdDev Acuan': values.get('stddev_value'),
            'COV (-)': values.get('cov_value'),
            'Satuan': values.get('unit'),
            'SF Awal (-)': values.get('sf_baseline'),
            'SF jika +sigma (-)': values.get('sf_plus'),
            'SF jika -sigma (-)': values.get('sf_minus'),
            'g jika +sigma': values.get('g_plus'),
            'Delta g jika +sigma': values.get('delta_g_plus'),
            'g jika -sigma': values.get('g_minus'),
            'Delta g jika -sigma': values.get('delta_g_minus'),
            'Efek Maksimum pada g Kontrol': values.get('sensitivity_index'),
            'Pengaruh terhadap Fungsi Kinerja (g(x))': describe_deterministic_g_effect(
                values.get('delta_g_plus')
            ),
            'Interpretasi Teknis': build_deterministic_sensitivity_interpretation(
                variable_name,
                values.get('delta_g_plus'),
                values.get('sigma_value'),
                values.get('perturbation_ratio'),
                limit_state_label
            )
        })

    return rows


def build_deterministic_sensitivity_df(
    deterministic_sensitivity_results: Dict
) -> pd.DataFrame:
    """Bangun dataframe sensitivitas deterministik berbasis COV."""
    rows = build_deterministic_sensitivity_rows(deterministic_sensitivity_results)
    return pd.DataFrame(rows)


def build_deterministic_sensitivity_tornado_figure(
    deterministic_sensitivity_results: Dict,
    top_n: int = 15
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Bangun tornado chart sensitivitas deterministik untuk skenario +/- sigma."""
    results = list((deterministic_sensitivity_results.get('results') or {}).items())
    if not results:
        return None

    display_items = results[:max(int(top_n), 1)]
    labels = [variable_name for variable_name, _ in display_items]
    delta_plus = [
        (
            float(values.get('delta_g_plus'))
            if values.get('delta_g_plus') is not None and np.isfinite(float(values.get('delta_g_plus')))
            else 0.0
        )
        for _, values in display_items
    ]
    delta_minus = [
        (
            float(values.get('delta_g_minus'))
            if values.get('delta_g_minus') is not None and np.isfinite(float(values.get('delta_g_minus')))
            else 0.0
        )
        for _, values in display_items
    ]

    baseline_info = deterministic_sensitivity_results.get('baseline', {}) or {}
    target_unit = str(baseline_info.get('unit', '-'))
    target_label = str(baseline_info.get('limit_state_label', 'Kontrol'))
    cov_scale = float(
        deterministic_sensitivity_results.get('cov_scale', 1.0) or 1.0
    )

    fig_height = float(min(max(4.8, 0.62 * len(display_items) + 1.8), 13.0))
    fig, ax = plt.subplots(figsize=(11, fig_height))

    y_positions = np.arange(len(display_items))
    bar_height = 0.34
    minus_bars = ax.barh(
        y_positions - (bar_height / 2.0),
        delta_minus,
        height=bar_height,
        color='#2563eb',
        edgecolor='#1f2937',
        linewidth=0.8,
        label=f"-{cov_scale:.1f} sigma"
    )
    plus_bars = ax.barh(
        y_positions + (bar_height / 2.0),
        delta_plus,
        height=bar_height,
        color='#dc2626',
        edgecolor='#1f2937',
        linewidth=0.8,
        label=f"+{cov_scale:.1f} sigma"
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0.0, color='#111827', linewidth=1.0, alpha=0.85)
    ax.set_xlabel(f"Perubahan g minimum pada limit state kontrol ({target_unit})")
    ax.set_title(
        f"Diagram Sensitivitas Deterministik terhadap g minimum limit state kontrol: {target_label}"
    )
    ax.grid(True, axis='x', alpha=0.25, linestyle='--')
    ax.legend(loc='lower right')

    max_abs_value = max(
        [abs(value) for value in delta_plus + delta_minus],
        default=0.0
    )
    x_padding = max(0.05 * max_abs_value, 0.02)
    axis_limit = max_abs_value + 5.5 * x_padding if max_abs_value > 0.0 else 1.0
    ax.set_xlim(-axis_limit, axis_limit)

    for bars, values in ((minus_bars, delta_minus), (plus_bars, delta_plus)):
        for bar, value in zip(bars, values):
            x_coord = float(bar.get_width())
            y_coord = float(bar.get_y() + (bar.get_height() / 2.0))
            text_offset = x_padding if value >= 0.0 else -x_padding
            horizontal_alignment = 'left' if value >= 0.0 else 'right'
            ax.text(
                x_coord + text_offset,
                y_coord,
                f"{value:+.3f}",
                va='center',
                ha=horizontal_alignment,
                fontsize=8.5,
                color='#111827'
            )

    fig.tight_layout()
    return fig, ax


def render_deterministic_sensitivity_output_section(results_bundle: Dict,
                                                    heading_level: str = "####") -> None:
    """Tampilkan sensitivitas deterministik lokal berbasis COV pada tab khusus."""
    deterministic_results = (
        results_bundle.get('deterministic_sensitivity_results', {})
        if results_bundle else
        {}
    )
    sensitivity_df = build_deterministic_sensitivity_df(deterministic_results)

    if sensitivity_df.empty:
        st.info(
            "Data sensitivitas deterministik berbasis COV belum tersedia. "
            "Jalankan analisis deterministik untuk membentuk hasil perturbasi one-at-a-time."
        )
        return

    baseline_info = deterministic_results.get('baseline', {}) or {}
    cov_scale = float(
        deterministic_results.get('cov_scale', 1.0) or 1.0
    )
    target_label = str(baseline_info.get('limit_state_label', '-'))
    target_unit = str(baseline_info.get('unit', '-'))
    baseline_g = baseline_info.get('g_value')
    analysis_failures = int(deterministic_results.get('analysis_failures', 0) or 0)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Limit State Kontrol", target_label)
    metric_cols[1].metric("g Acuan", format_metric(baseline_g, 4))
    metric_cols[2].metric("Skala Perturbasi", f"+/-{cov_scale:.1f} sigma")
    metric_cols[3].metric("Analisis Gagal", str(analysis_failures))

    st.markdown(f"{heading_level} Diagram Sensitivitas Deterministik")
    st.caption(
        "Diagram tornado berikut dibentuk dari pendekatan `one-at-a-time` berbasis COV: "
        "setiap variabel deterministik digeser sendiri sebesar `+/-sigma`, dengan "
        "`sigma = COV x Mean`, "
        f"lalu perubahan `g minimum` pada jenis limit state kontrol baseline `{target_label}` dibandingkan. "
        "Batang bernilai positif berarti margin keamanan meningkat, sedangkan batang negatif "
        "berarti margin keamanan menurun."
    )
    st.caption(
        "Jenis limit state yang dibandingkan tetap sama pada semua skenario perturbasi. "
        "Yang dapat berubah adalah elemen pengontrol yang menghasilkan nilai `g minimum` tersebut."
    )
    tornado_plot = build_deterministic_sensitivity_tornado_figure(
        deterministic_results,
        top_n=15
    )
    if tornado_plot is not None:
        tornado_fig, _ = tornado_plot
        render_plot(
            tornado_fig,
            interactive=False,
            alt_text="Diagram sensitivitas deterministik dengan perturbasi sigma",
            download_basename="output-sensitivitas-deterministik"
        )
    else:
        st.info("Diagram sensitivitas deterministik berbasis COV belum dapat dibentuk.")

    st.markdown(f"{heading_level} Tabel Sensitivitas Deterministik")
    st.caption(
        f"Kolom `g jika +/-sigma` menunjukkan nilai `g minimum` pada limit state kontrol "
        f"`{target_label}` ({target_unit}) setelah satu variabel digeser sebesar `+/-sigma` "
        "dan variabel lain dipertahankan pada baseline."
    )
    st.caption(
        "Kolom `COV (-)` menyatakan `StdDev / |Mean|` tiap variabel. "
        "`Efek Maksimum pada g Kontrol` adalah magnitudo perubahan terbesar dari dua skenario "
        "perturbasi berbasis COV, sedangkan `Pengaruh terhadap Fungsi Kinerja (g(x))` "
        "merangkum apakah perubahan tersebut meningkatkan atau mengurangi margin keamanan."
    )
    st.caption(
        "Kolom `Pengaruh terhadap Fungsi Kinerja (g(x))` dan `Interpretasi Teknis` "
        "mengikuti skenario kenaikan `+sigma`, agar konsisten dengan kolom "
        "`Delta g jika +sigma`. Peringkat dan `Efek Maksimum pada g Kontrol` tetap "
        "ditentukan dari magnitudo terbesar antara skenario `+sigma` dan `-sigma`."
    )
    st.caption(
        "Kolom `SF` memakai definisi `SF = R / S`. Khusus cek aksial-lentur, "
        "`SF` setara dengan `lambda` karena `S = 1.0`."
    )
    render_input_table(
        sensitivity_df,
        styler=style_input_dataframe(
            sensitivity_df,
            table_min_width_px=2600
        )
    )


def run_analysis_dashboard(input_file: str,
                           analysis_mode: str,
                           num_simulations: int,
                           progress_container=None):
    """Jalankan pipeline analisis dengan progress indicator Streamlit."""
    analysis = PortalReliabilityAnalysis(
        input_file,
        num_mc_simulations=num_simulations,
        analysis_mode=analysis_mode
    )
    progress_host = progress_container if progress_container is not None else st.container()
    with progress_host:
        st.markdown("**Proses Analisis**")
        progress_text = st.empty()
        progress = st.progress(0)
        status = st.empty()
        simulation_progress_text = st.empty()
    if analysis.is_probabilistic:
        steps = [
            ("Membaca data input", analysis.read_input),
            ("Inisialisasi portal", analysis.initialize_portal),
            ("Menyiapkan Monte Carlo", analysis.setup_monte_carlo),
            ("Menjalankan simulasi Monte Carlo", analysis.run_monte_carlo),
            ("Analisis keandalan", analysis.reliability_analysis),
            ("Menyusun laporan", analysis.generate_report),
            ("Menyimpan hasil", analysis.save_results),
        ]
    else:
        steps = [
            ("Membaca data input", analysis.read_input),
            ("Inisialisasi portal", analysis.initialize_portal),
            ("Menjalankan analisis deterministik", analysis.run_deterministic_analysis),
            ("Analisis sensitivitas deterministik berbasis COV", analysis.deterministic_sensitivity_analysis),
            ("Menyusun laporan", analysis.generate_report),
            ("Menyimpan hasil", analysis.save_results),
        ]

    total_steps = len(steps)
    render_progress_percentage(progress_text, 0.0, 0, total_steps)

    for index, (label, callback) in enumerate(steps, start=1):
        status.markdown(f"**Tahap:** {label}")
        if analysis.is_probabilistic and label == "Menjalankan simulasi Monte Carlo":
            simulation_progress_text.markdown(
                (
                    "<div style='font-size:0.9rem; color:#e5e7eb; "
                    "margin-top:0.15rem;'>Menyiapkan progress simulasi Monte Carlo...</div>"
                ),
                unsafe_allow_html=True
            )

            def monte_carlo_progress_callback(progress_info: Dict[str, Any]) -> None:
                completed_fraction = float(progress_info.get('completed_fraction', 0.0) or 0.0)
                overall_progress_fraction = ((index - 1) + completed_fraction) / total_steps
                progress.progress(overall_progress_fraction)
                render_progress_percentage(
                    progress_text,
                    overall_progress_fraction,
                    index,
                    total_steps
                )
                render_simulation_progress(
                    simulation_progress_text,
                    completed_simulations=int(progress_info.get('completed_simulations', 0) or 0),
                    total_simulations=int(progress_info.get('total_simulations', num_simulations) or num_simulations),
                    analysis_failures=int(progress_info.get('analysis_failures', 0) or 0)
                )

            callback(
                progress_callback=monte_carlo_progress_callback,
                progress_interval=max(1, int(num_simulations) // 100)
            )
            progress_fraction = index / total_steps
        else:
            simulation_progress_text.empty()
            callback()
            progress_fraction = index / total_steps
        progress.progress(progress_fraction)
        render_progress_percentage(progress_text, progress_fraction, index, total_steps)

    simulation_progress_text.empty()
    status.markdown("**Tahap:** Selesai")
    return analysis


def sanitize_dom_id(value: str) -> str:
    """Ubah key menjadi DOM id yang aman untuk HTML/JS."""
    sanitized = re.sub(r'[^a-zA-Z0-9_-]+', '-', str(value).strip())
    sanitized = sanitized.strip('-')
    return sanitized or "zoomable-plot"


def sanitize_download_filename(value: str) -> str:
    """Ubah nama file unduhan menjadi aman untuk berbagai OS/browser."""
    sanitized = re.sub(r'[^a-zA-Z0-9._-]+', '-', str(value).strip())
    sanitized = sanitized.strip('-.')
    return sanitized or "plot"


def save_figure_to_image_bytes(fig,
                               image_format: str = 'png',
                               image_dpi: int = 220,
                               tight_bbox: bool = True,
                               jpeg_quality: int = PLOT_DOWNLOAD_JPEG_QUALITY) -> bytes:
    """Simpan figure matplotlib ke bytes gambar untuk preview dan unduhan."""
    normalized_format = str(image_format or 'png').strip().lower()
    if normalized_format == 'jpg':
        normalized_format = 'jpeg'

    base_save_kwargs = {
        'format': normalized_format,
        'dpi': int(image_dpi),
        'facecolor': 'white',
        'edgecolor': 'white'
    }
    if tight_bbox:
        base_save_kwargs['bbox_inches'] = 'tight'

    def save_with_kwargs(save_kwargs: Dict[str, Any]) -> bytes:
        image_buffer = io.BytesIO()
        fig.savefig(image_buffer, **save_kwargs)
        image_buffer.seek(0)
        return image_buffer.getvalue()

    if normalized_format == 'jpeg':
        pil_kwargs = {
            'quality': int(max(1, min(int(jpeg_quality), 100))),
            'optimize': True,
            'subsampling': 0
        }
        try:
            return save_with_kwargs({
                **base_save_kwargs,
                'pil_kwargs': pil_kwargs
            })
        except TypeError:
            try:
                return save_with_kwargs({
                    **base_save_kwargs,
                    'pil_kwargs': {
                        'quality': pil_kwargs['quality'],
                        'optimize': pil_kwargs['optimize']
                    }
                })
            except TypeError:
                return save_with_kwargs(base_save_kwargs)

    return save_with_kwargs(base_save_kwargs)


def figure_to_png_data_uri(fig,
                           image_dpi: int = 220,
                           tight_bbox: bool = True) -> str:
    """Konversi figure matplotlib menjadi PNG data URI resolusi tinggi."""
    image_bytes = save_figure_to_image_bytes(
        fig,
        image_format='png',
        image_dpi=image_dpi,
        tight_bbox=tight_bbox
    )
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode('ascii')


def image_file_to_data_uri(image_path: Path,
                           mime_type: str = "image/png") -> str:
    """Konversi file gambar lokal menjadi data URI untuk dirender via HTML."""
    return (
        f"data:{mime_type};base64,"
        + base64.b64encode(image_path.read_bytes()).decode('ascii')
    )


def render_zoomable_plot(fig,
                         viewer_key: str,
                         alt_text: str = "Plot simulasi terakhir",
                         viewer_height: int = ZOOMABLE_PLOT_VIEWER_HEIGHT,
                         tight_bbox: bool = True) -> None:
    """Render plot sebagai viewer interaktif dengan zoom/pan untuk desktop dan HP."""
    viewer_id = sanitize_dom_id(viewer_key)
    image_src = figure_to_png_data_uri(fig, tight_bbox=tight_bbox)
    safe_alt_text = html.escape(alt_text, quote=True)
    viewer_markup = f"""
    <!DOCTYPE html>
    <html lang="id">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        html, body {{
          margin: 0;
          padding: 0;
          height: 100%;
          overflow: hidden;
          background: #ffffff;
          font-family: "Segoe UI", sans-serif;
        }}
        .plot-viewer-root {{
          height: 100%;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          padding: 0.1rem 0;
          box-sizing: border-box;
          color: #111827;
        }}
        .plot-viewer-toolbar {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.5rem;
          font-size: 0.88rem;
        }}
        .plot-viewer-toolbar strong {{
          font-weight: 700;
        }}
        .plot-viewer-buttons {{
          display: flex;
          align-items: center;
          gap: 0.35rem;
          flex-wrap: wrap;
          justify-content: flex-end;
        }}
        .plot-viewer-button {{
          border: 1px solid #cbd5e1;
          background: #f8fafc;
          color: #0f172a;
          border-radius: 0.6rem;
          min-width: 2.2rem;
          height: 2.1rem;
          padding: 0 0.7rem;
          font-size: 0.92rem;
          font-weight: 600;
          cursor: pointer;
        }}
        .plot-viewer-button:active {{
          background: #e2e8f0;
        }}
        .plot-viewer-viewport {{
          position: relative;
          flex: 1 1 auto;
          min-height: 0;
          overflow: hidden;
          border: 1px solid #d1d5db;
          border-radius: 0.9rem;
          background:
            linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%);
          touch-action: none;
          user-select: none;
          cursor: grab;
        }}
        .plot-viewer-viewport.is-dragging {{
          cursor: grabbing;
        }}
        .plot-viewer-anchor {{
          position: absolute;
          left: 50%;
          top: 50%;
          transform: translate(-50%, -50%);
        }}
        .plot-viewer-stage {{
          transform-origin: center center;
          will-change: transform;
        }}
        .plot-viewer-stage img {{
          display: block;
          max-width: none;
          width: auto;
          height: auto;
          user-select: none;
          -webkit-user-drag: none;
          pointer-events: none;
        }}
        .plot-viewer-hint {{
          font-size: 0.76rem;
          color: #6b7280;
          line-height: 1.35;
        }}
      </style>
    </head>
    <body>
      <div id="{viewer_id}" class="plot-viewer-root">
        <div class="plot-viewer-toolbar">
          <div>Zoom: <strong class="plot-viewer-zoom-label">100%</strong></div>
          <div class="plot-viewer-buttons">
            <button type="button" class="plot-viewer-button" data-action="zoom-out">-</button>
            <button type="button" class="plot-viewer-button" data-action="zoom-in">+</button>
            <button type="button" class="plot-viewer-button" data-action="reset">Reset</button>
          </div>
        </div>
        <div class="plot-viewer-viewport">
          <div class="plot-viewer-anchor">
            <div class="plot-viewer-stage">
              <img src="{image_src}" alt="{safe_alt_text}" draggable="false" />
            </div>
          </div>
        </div>
        <div class="plot-viewer-hint">
          Desktop: scroll mouse untuk zoom, drag untuk geser, double click untuk reset.
          HP: pinch untuk zoom, geser untuk pan, tombol Reset untuk kembali ke ukuran awal.
        </div>
      </div>
      <script>
        (() => {{
          const root = document.getElementById({json.dumps(viewer_id)});
          if (!root) return;

          const viewport = root.querySelector('.plot-viewer-viewport');
          const stage = root.querySelector('.plot-viewer-stage');
          const img = root.querySelector('img');
          const zoomLabel = root.querySelector('.plot-viewer-zoom-label');
          const zoomInButton = root.querySelector('[data-action="zoom-in"]');
          const zoomOutButton = root.querySelector('[data-action="zoom-out"]');
          const resetButton = root.querySelector('[data-action="reset"]');
          const maxUserScale = 8;

          let fitScale = 1;
          let userScale = 1;
          let translateX = 0;
          let translateY = 0;
          let isMouseDragging = false;
          let mouseStartX = 0;
          let mouseStartY = 0;
          let mouseStartTranslateX = 0;
          let mouseStartTranslateY = 0;
          let lastTouchX = 0;
          let lastTouchY = 0;
          let pinchDistance = null;
          let pinchCenter = null;

          function clamp(value, minValue, maxValue) {{
            return Math.min(maxValue, Math.max(minValue, value));
          }}

          function getDistance(touchA, touchB) {{
            return Math.hypot(
              touchB.clientX - touchA.clientX,
              touchB.clientY - touchA.clientY
            );
          }}

          function getTouchCenter(touchA, touchB) {{
            return {{
              x: (touchA.clientX + touchB.clientX) / 2,
              y: (touchA.clientY + touchB.clientY) / 2
            }};
          }}

          function getViewportCenter() {{
            const rect = viewport.getBoundingClientRect();
            return {{
              x: rect.left + rect.width / 2,
              y: rect.top + rect.height / 2
            }};
          }}

          function getDisplayedMetrics(scaleValue = userScale) {{
            const rect = viewport.getBoundingClientRect();
            const naturalWidth = img.naturalWidth || 1;
            const naturalHeight = img.naturalHeight || 1;
            const totalScale = fitScale * scaleValue;
            return {{
              rect,
              width: naturalWidth * totalScale,
              height: naturalHeight * totalScale,
              totalScale
            }};
          }}

          function constrainTranslation() {{
            const metrics = getDisplayedMetrics();
            const maxX = Math.max((metrics.width - metrics.rect.width) / 2, 0);
            const maxY = Math.max((metrics.height - metrics.rect.height) / 2, 0);
            translateX = clamp(translateX, -maxX, maxX);
            translateY = clamp(translateY, -maxY, maxY);
          }}

          function updateStage() {{
            constrainTranslation();
            const totalScale = fitScale * userScale;
            stage.style.transform = `translate(${{translateX}}px, ${{translateY}}px) scale(${{totalScale}})`;
            zoomLabel.textContent = `${{Math.round(userScale * 100)}}%`;
          }}

          function recomputeFitScale() {{
            const rect = viewport.getBoundingClientRect();
            const naturalWidth = img.naturalWidth || 1;
            const naturalHeight = img.naturalHeight || 1;
            if (rect.width <= 0 || rect.height <= 0) {{
              return;
            }}
            fitScale = Math.min(
              rect.width / naturalWidth,
              rect.height / naturalHeight
            );
          }}

          function resetView() {{
            recomputeFitScale();
            userScale = 1;
            translateX = 0;
            translateY = 0;
            updateStage();
          }}

          function setScaleAt(nextScale, clientX, clientY) {{
            const rect = viewport.getBoundingClientRect();
            const clampedScale = clamp(nextScale, 1, maxUserScale);
            const previousScale = userScale;
            if (Math.abs(clampedScale - previousScale) < 1e-6) {{
              return;
            }}

            const previousTotalScale = fitScale * previousScale;
            const nextTotalScale = fitScale * clampedScale;
            const offsetX = clientX - rect.left - (rect.width / 2) - translateX;
            const offsetY = clientY - rect.top - (rect.height / 2) - translateY;
            const scaleRatio = nextTotalScale / previousTotalScale;

            translateX -= offsetX * (scaleRatio - 1);
            translateY -= offsetY * (scaleRatio - 1);
            userScale = clampedScale;
            updateStage();
          }}

          function stopMouseDrag() {{
            isMouseDragging = false;
            viewport.classList.remove('is-dragging');
          }}

          img.addEventListener('load', resetView);
          zoomInButton.addEventListener('click', () => {{
            const center = getViewportCenter();
            setScaleAt(userScale * 1.25, center.x, center.y);
          }});
          zoomOutButton.addEventListener('click', () => {{
            const center = getViewportCenter();
            setScaleAt(userScale / 1.25, center.x, center.y);
          }});
          resetButton.addEventListener('click', resetView);

          viewport.addEventListener('wheel', (event) => {{
            event.preventDefault();
            const wheelFactor = event.deltaY < 0 ? 1.12 : (1 / 1.12);
            setScaleAt(userScale * wheelFactor, event.clientX, event.clientY);
          }}, {{ passive: false }});

          viewport.addEventListener('mousedown', (event) => {{
            event.preventDefault();
            isMouseDragging = true;
            mouseStartX = event.clientX;
            mouseStartY = event.clientY;
            mouseStartTranslateX = translateX;
            mouseStartTranslateY = translateY;
            viewport.classList.add('is-dragging');
          }});

          window.addEventListener('mousemove', (event) => {{
            if (!isMouseDragging) {{
              return;
            }}
            translateX = mouseStartTranslateX + (event.clientX - mouseStartX);
            translateY = mouseStartTranslateY + (event.clientY - mouseStartY);
            updateStage();
          }});

          window.addEventListener('mouseup', stopMouseDrag);
          viewport.addEventListener('mouseleave', stopMouseDrag);
          viewport.addEventListener('dblclick', resetView);

          viewport.addEventListener('touchstart', (event) => {{
            if (event.touches.length === 1) {{
              const touch = event.touches[0];
              lastTouchX = touch.clientX;
              lastTouchY = touch.clientY;
              pinchDistance = null;
              pinchCenter = null;
            }} else if (event.touches.length >= 2) {{
              const [touchA, touchB] = event.touches;
              pinchDistance = getDistance(touchA, touchB);
              pinchCenter = getTouchCenter(touchA, touchB);
            }}
          }}, {{ passive: true }});

          viewport.addEventListener('touchmove', (event) => {{
            event.preventDefault();

            if (event.touches.length === 1) {{
              const touch = event.touches[0];
              translateX += touch.clientX - lastTouchX;
              translateY += touch.clientY - lastTouchY;
              lastTouchX = touch.clientX;
              lastTouchY = touch.clientY;
              updateStage();
              return;
            }}

            if (event.touches.length >= 2) {{
              const [touchA, touchB] = event.touches;
              const center = getTouchCenter(touchA, touchB);
              const nextDistance = getDistance(touchA, touchB);

              if (pinchDistance !== null && pinchCenter !== null && pinchDistance > 0) {{
                const pinchFactor = nextDistance / pinchDistance;
                setScaleAt(userScale * pinchFactor, center.x, center.y);
                translateX += center.x - pinchCenter.x;
                translateY += center.y - pinchCenter.y;
                updateStage();
              }}

              pinchDistance = nextDistance;
              pinchCenter = center;
            }}
          }}, {{ passive: false }});

          viewport.addEventListener('touchend', (event) => {{
            if (event.touches.length === 1) {{
              const touch = event.touches[0];
              lastTouchX = touch.clientX;
              lastTouchY = touch.clientY;
            }} else if (event.touches.length === 0) {{
              pinchDistance = null;
              pinchCenter = null;
            }}
          }});

          if (window.ResizeObserver) {{
            const resizeObserver = new ResizeObserver(() => {{
              const previousUserScale = userScale;
              recomputeFitScale();
              userScale = previousUserScale;
              updateStage();
            }});
            resizeObserver.observe(viewport);
          }}

          if (img.complete) {{
            resetView();
          }}
        }})();
      </script>
    </body>
    </html>
    """
    components.html(viewer_markup, height=viewer_height, scrolling=False)


def render_plot(fig,
                interactive: bool = False,
                viewer_key: Optional[str] = None,
                alt_text: str = "Plot simulasi terakhir",
                viewer_height: int = ZOOMABLE_PLOT_VIEWER_HEIGHT,
                tight_bbox: bool = True,
                download_basename: Optional[str] = None,
                download_dpi: int = PLOT_DOWNLOAD_IMAGE_DPI) -> None:
    """Tampilkan plot matplotlib dan tutup figure setelah dirender."""
    download_bytes = None
    download_file_name = None
    download_error = None
    download_key = None

    if download_basename:
        try:
            download_bytes = save_figure_to_image_bytes(
                fig,
                image_format='jpeg',
                image_dpi=download_dpi,
                tight_bbox=tight_bbox,
                jpeg_quality=PLOT_DOWNLOAD_JPEG_QUALITY
            )
            download_file_name = (
                f"{sanitize_download_filename(download_basename)}.jpg"
            )
            download_key = (
                f"{sanitize_dom_id(viewer_key or download_basename or f'plot-{id(fig)}')}"
                "-download-jpg"
            )
        except Exception as exc:
            download_error = format_error_message(exc)

    if interactive:
        render_zoomable_plot(
            fig,
            viewer_key=viewer_key or f"plot-{id(fig)}",
            alt_text=alt_text,
            viewer_height=viewer_height,
            tight_bbox=tight_bbox
        )
    else:
        st.pyplot(fig, clear_figure=True, use_container_width=True)

    if download_bytes is not None and download_file_name and download_key:
        st.download_button(
            label=f"Unduh JPG HD ({int(download_dpi)} DPI)",
            data=download_bytes,
            file_name=download_file_name,
            mime='image/jpeg',
            key=download_key
        )
    elif download_error:
        st.caption(f"File JPG belum dapat disiapkan: {download_error}")

    plt.close(fig)


def get_by_element_value(source: Optional[Dict], elem_id: int, default=None):
    """Ambil nilai dict dengan key elemen int/string."""
    if not isinstance(source, dict):
        return default
    if elem_id in source:
        return source[elem_id]
    elem_id_str = str(int(elem_id))
    if elem_id_str in source:
        return source[elem_id_str]
    return default


def read_positive_number(value) -> float:
    """Baca angka positif, atau 0 jika invalid."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        return 0.0
    return float(numeric_value)


def get_rebar_area_from_count(bar_count: float, bar_diameter: float) -> float:
    """Hitung luas total tulangan dari jumlah batang dan diameter."""
    count_value = max(float(bar_count or 0.0), 0.0)
    diameter_value = max(float(bar_diameter or 0.0), 0.0)
    if count_value <= 0.0 or diameter_value <= 0.0:
        return 0.0
    return float(count_value * np.pi * (diameter_value ** 2) / 4.0)


def get_effective_depth_from_cover(section_height: float,
                                   cover: float,
                                   stirrup_diameter: float,
                                   bar_diameter: float,
                                   from_compression_face: bool = False) -> float:
    """Turunkan d atau d' dari selimut beton dan diameter tulangan."""
    h_value = max(float(section_height or 0.0), 0.0)
    cover_value = max(float(cover or 0.0), 0.0)
    stirrup_value = max(float(stirrup_diameter or 0.0), 0.0)
    bar_value = max(float(bar_diameter or 0.0), 0.0)
    centroid_depth = cover_value + stirrup_value + (0.5 * bar_value)

    if from_compression_face:
        return float(min(centroid_depth, h_value))
    return float(max(h_value - centroid_depth, 0.0))


def get_section_capacity_inputs_from_input(input_data: Dict, elem_id: int) -> Dict[str, Dict]:
    """Ambil geometri dan tulangan penampang dari input Streamlit."""
    geometry_lookup = input_data.get('geometry', {}).get('properties_by_element', {})
    reinforcement_lookup = input_data.get('reinforcement', {}).get('by_element', {})
    geometry_props = (
        get_by_element_value(geometry_lookup, elem_id, {}) or {}
    )
    reinforcement_props = (
        get_by_element_value(reinforcement_lookup, elem_id, {}) or {}
    )

    if not geometry_props:
        raise ValueError(f"Data geometri elemen {int(elem_id)} tidak ditemukan.")
    if not reinforcement_props:
        raise ValueError(f"Data tulangan elemen {int(elem_id)} tidak ditemukan.")

    b_value = read_positive_number(geometry_props.get('b'))
    h_value = read_positive_number(geometry_props.get('h'))
    area_value = read_positive_number(geometry_props.get('area'))
    if min(b_value, h_value, area_value) <= 0.0:
        raise ValueError(
            f"Data penampang elemen {int(elem_id)} belum lengkap pada sheet Geometri."
        )

    ds_tarik = float(reinforcement_props.get('ds_tarik', 0.0) or 0.0)
    ds_tekan = float(reinforcement_props.get('ds_tekan', 0.0) or 0.0)
    d_tarik_input = float(reinforcement_props.get('d_tarik', 0.0) or 0.0)
    d_tekan_input = float(reinforcement_props.get('d_tekan', 0.0) or 0.0)
    n_tarik = float(reinforcement_props.get('n_tarik', 0.0) or 0.0)
    du_tarik = float(reinforcement_props.get('du_tarik', 0.0) or 0.0)
    n_tekan = float(reinforcement_props.get('n_tekan', 0.0) or 0.0)
    du_tekan = float(reinforcement_props.get('du_tekan', 0.0) or 0.0)
    n_geser = float(reinforcement_props.get('n_geser', 0.0) or 0.0)
    du_geser = float(reinforcement_props.get('du_geser', 0.0) or 0.0)

    d_tarik = (
        d_tarik_input
        if d_tarik_input > 0.0 else
        get_effective_depth_from_cover(h_value, ds_tarik, du_geser, du_tarik)
    )
    d_tekan = (
        d_tekan_input
        if d_tekan_input > 0.0 else
        get_effective_depth_from_cover(
            h_value,
            ds_tekan,
            du_geser,
            du_tekan,
            from_compression_face=True
        )
    )

    as_tarik_input = float(reinforcement_props.get('As_tarik', 0.0) or 0.0)
    as_tekan_input = float(reinforcement_props.get('As_tekan', 0.0) or 0.0)
    as_geser_input = float(reinforcement_props.get('As_geser', 0.0) or 0.0)
    as_tarik = (
        as_tarik_input
        if as_tarik_input > 0.0 else
        get_rebar_area_from_count(n_tarik, du_tarik)
    )
    as_tekan = (
        as_tekan_input
        if as_tekan_input > 0.0 else
        get_rebar_area_from_count(n_tekan, du_tekan)
    )
    as_geser = (
        as_geser_input
        if as_geser_input > 0.0 else
        get_rebar_area_from_count(n_geser, du_geser)
    )

    return {
        'section_geometry': {
            'b': b_value,
            'h': h_value,
            'd': d_tarik,
            'd_prime': d_tekan,
            'area': area_value,
            'Ag': area_value,
            'element_id': int(elem_id)
        },
        'steel_area': {
            'As': as_tarik,
            'As_prime': as_tekan,
            'As_shear': as_geser,
            'd_prime': d_tekan,
            'shear_spacing': read_positive_number(reinforcement_props.get('Spasi_geser')),
            'element_id': int(elem_id)
        }
    }


def get_element_material_snapshot(input_data: Dict,
                                  latest_simulation: Optional[Dict],
                                  is_probabilistic: bool,
                                  elem_id: int) -> Dict[str, float]:
    """Ambil snapshot material elemen dari sampel aktif atau fallback input."""
    random_sample = (latest_simulation or {}).get('random_sample') or {}
    concrete_lookup = input_data.get('concrete', {}).get('by_element', {})
    steel_lookup = input_data.get('steel', {}).get('by_element', {})
    concrete_props = get_by_element_value(concrete_lookup, elem_id, {}) or {}
    steel_props = get_by_element_value(steel_lookup, elem_id, {}) or {}

    def resolve_value(sample_key: str,
                      props: Dict,
                      primary_key: str,
                      fallback_key: str) -> float:
        value = random_sample.get(sample_key)
        if value is None:
            value = props.get(primary_key)
        if value is None:
            value = props.get(fallback_key)
        return read_positive_number(value)

    concrete_primary = 'mean' if is_probabilistic else 'deterministic'
    steel_primary = 'tarik_mean' if is_probabilistic else 'tarik_deterministic'
    steel_primary_compression = 'tekan_mean' if is_probabilistic else 'tekan_deterministic'

    fc_value = resolve_value(f'fc_E{int(elem_id)}', concrete_props, concrete_primary, 'mean')
    fy_tarik = resolve_value(
        f'fy_tarik_E{int(elem_id)}',
        steel_props,
        steel_primary,
        'tarik_mean'
    )
    fy_tekan = resolve_value(
        f'fy_tekan_E{int(elem_id)}',
        steel_props,
        steel_primary_compression,
        'tekan_mean'
    )

    if fc_value <= 0.0 or fy_tarik <= 0.0 or fy_tekan <= 0.0:
        raise ValueError(
            f"Snapshot material elemen {int(elem_id)} belum lengkap untuk membentuk kurva interaksi."
        )

    return {
        'fc': fc_value,
        'fy_tarik': fy_tarik,
        'fy_tekan': fy_tekan
    }


def get_max_forces_entry_by_element(latest_result: Dict, elem_id: int) -> Dict:
    """Ambil hasil gaya maksimum satu elemen dari output analisis."""
    max_forces_entry = get_by_element_value(latest_result.get('max_forces', {}), elem_id, {})
    if max_forces_entry:
        return max_forces_entry

    for force_data in latest_result.get('element_forces', []):
        try:
            force_elem_id = int(force_data.get('elem_id'))
        except (TypeError, ValueError):
            continue
        if force_elem_id != int(elem_id):
            continue
        return {
            'max_moment': force_data.get('max_moment'),
            'max_shear': force_data.get('max_shear'),
            'max_axial': force_data.get('max_axial'),
            'forces': force_data
        }
    return {}


def get_current_axial_moment_physical_overlay_from_latest_result(
    input_data: Optional[Dict[str, Any]],
    latest_simulation: Optional[Dict[str, Any]],
    latest_result: Optional[Dict[str, Any]],
    elem_id: int
) -> Dict[str, Any]:
    """Ambil demand, boundary aktif, dan kurva interaksi sampel aktif untuk overlay ruang fisik."""
    if not latest_result:
        return {}

    max_forces_entry = get_max_forces_entry_by_element(latest_result, int(elem_id))
    if not max_forces_entry:
        return {}

    demand_moment = coerce_finite_float(max_forces_entry.get('max_moment'))
    if demand_moment is None:
        return {}

    axial_moment_meta = (
        get_by_element_value(
            latest_result.get('performance_axial_moment_metadata', {}),
            int(elem_id),
            {}
        ) or {}
    )
    force_data = (max_forces_entry.get('forces') or {}) if isinstance(max_forces_entry, dict) else {}
    demand_axial = resolve_axial_moment_plot_demand_axial(
        force_data,
        axial_moment_meta.get('controlling_state')
    )

    overlay_data = {
        'demand_moment': abs(float(demand_moment)),
        'demand_axial': float(demand_axial)
    }
    boundary_moment = coerce_finite_float(axial_moment_meta.get('phi_Mn'))
    boundary_axial = coerce_finite_float(axial_moment_meta.get('phi_Pn'))
    if boundary_moment is not None:
        overlay_data['boundary_moment_interp'] = float(boundary_moment)
    if boundary_axial is not None:
        overlay_data['boundary_axial_interp'] = float(boundary_axial)
    if boundary_moment is not None:
        overlay_data['boundary_moment'] = float(boundary_moment)
    if boundary_axial is not None:
        overlay_data['boundary_axial'] = float(boundary_axial)

    if not input_data:
        return overlay_data

    try:
        section_inputs = get_section_capacity_inputs_from_input(input_data, int(elem_id))
        material_snapshot = get_element_material_snapshot(
            input_data,
            latest_simulation,
            True,
            int(elem_id)
        )
        active_curve = PerformanceFunction._get_column_interaction_curve(
            material_snapshot['fc'],
            material_snapshot['fy_tarik'],
            section_inputs['section_geometry'],
            section_inputs['steel_area'],
            fy_tekan=material_snapshot['fy_tekan'],
            use_code_phi=False
        )
        active_curve_moment = [
            float(point['phi_Mn'])
            for point in (active_curve or [])
            if coerce_finite_float(point.get('phi_Mn')) is not None
            and coerce_finite_float(point.get('phi_Pn')) is not None
        ]
        active_curve_axial = [
            float(point['phi_Pn'])
            for point in (active_curve or [])
            if coerce_finite_float(point.get('phi_Mn')) is not None
            and coerce_finite_float(point.get('phi_Pn')) is not None
        ]
        if len(active_curve_moment) > 1 and len(active_curve_axial) > 1:
            overlay_data['active_curve_moment'] = list(active_curve_moment)
            overlay_data['active_curve_axial'] = list(active_curve_axial)

        exact_boundary = find_exact_interaction_boundary_state(
            material_snapshot['fc'],
            material_snapshot['fy_tarik'],
            material_snapshot['fy_tekan'],
            section_inputs['section_geometry'],
            section_inputs['steel_area'],
            float(demand_axial),
            abs(float(demand_moment)),
            use_code_phi=False
        )
        if exact_boundary is not None:
            exact_boundary_moment = coerce_finite_float(exact_boundary.get('phi_Mn'))
            exact_boundary_axial = coerce_finite_float(exact_boundary.get('phi_Pn'))
            if exact_boundary_moment is not None and exact_boundary_axial is not None:
                overlay_data['boundary_moment_exact'] = float(exact_boundary_moment)
                overlay_data['boundary_axial_exact'] = float(exact_boundary_axial)
                overlay_data['boundary_moment'] = float(exact_boundary_moment)
                overlay_data['boundary_axial'] = float(exact_boundary_axial)
                overlay_data['boundary_source'] = 'exact'
        else:
            overlay_data['boundary_source'] = 'interp'
    except Exception:
        overlay_data.setdefault('boundary_source', 'interp')
    return overlay_data


def get_current_axial_moment_physical_demand_from_latest_result(
    input_data: Optional[Dict[str, Any]],
    latest_simulation: Optional[Dict[str, Any]],
    latest_result: Optional[Dict[str, Any]],
    elem_id: int
) -> Dict[str, float]:
    """Ambil titik demand aksial-lentur dari simulasi aktif untuk overlay ruang fisik."""
    overlay_data = get_current_axial_moment_physical_overlay_from_latest_result(
        input_data,
        latest_simulation,
        latest_result,
        int(elem_id)
    )
    if not overlay_data:
        return {}
    return {
        'moment': float(overlay_data['demand_moment']),
        'axial': float(overlay_data['demand_axial'])
    }


def get_axial_demands_from_force_data(force_data: Dict) -> Dict[str, float]:
    """Ekstrak demand aksial tekan dan tarik maksimum dari gaya elemen."""
    axial_values = [
        float(force_data.get('axial_start', 0.0)),
        float(force_data.get(
            'axial_end_internal',
            force_data.get('axial_end', 0.0)
        ))
    ]
    axial_values = [
        0.0 if abs(value) <= AXIAL_DEMAND_TOLERANCE_KN else value
        for value in axial_values
    ]
    compression_demand = max([0.0] + axial_values)
    tension_demand = max(0.0, -min(axial_values))
    return {
        'compression': float(compression_demand),
        'tension': float(tension_demand)
    }


def collect_interaction_element_ids(input_data: Optional[Dict],
                                    latest_result: Optional[Dict]) -> list[int]:
    """Daftar elemen kolom yang bisa diplot kurva interaksinya."""
    element_ids = set()

    if latest_result:
        for source in (
            latest_result.get('performance_axial_moment', {}),
            latest_result.get('performance_axial_moment_metadata', {})
        ):
            for elem_id in (source or {}).keys():
                try:
                    element_ids.add(int(elem_id))
                except (TypeError, ValueError):
                    continue

    if input_data:
        geometry_lookup = input_data.get('geometry', {}).get('properties_by_element', {})
        for raw_elem_id in geometry_lookup.keys():
            try:
                elem_id = int(raw_elem_id)
            except (TypeError, ValueError):
                continue
            if get_element_code_from_input(input_data, elem_id) == 'K':
                element_ids.add(elem_id)

    return sorted(element_ids)


def _find_interaction_segment(demand_axial: float,
                              demand_moment: float,
                              interaction_curve: List[Dict[str, float]]) -> Optional[Dict[str, Any]]:
    """Cari segmen kurva yang memotong sinar demand."""
    if abs(float(demand_moment)) <= 1e-12:
        return None

    candidates = []
    for first, second in zip(interaction_curve, interaction_curve[1:]):
        solution = PerformanceFunction._solve_interaction_ray_segment(
            demand_axial,
            demand_moment,
            float(first['phi_Pn']),
            float(first['phi_Mn']),
            float(second['phi_Pn']),
            float(second['phi_Mn'])
        )
        if solution is None:
            continue

        lambda_value, segment_ratio = solution
        if lambda_value < -1e-9:
            continue
        if segment_ratio < -1e-6 or segment_ratio > 1.0 + 1e-6:
            continue

        candidates.append({
            'lambda': float(max(lambda_value, 0.0)),
            'segment_ratio': float(np.clip(segment_ratio, 0.0, 1.0)),
            'first': first,
            'second': second
        })

    if not candidates:
        return None

    return min(candidates, key=lambda candidate: candidate['lambda'])


def _get_exact_boundary_lambda(response: Dict[str, Any],
                               demand_axial: float,
                               demand_moment: float) -> float:
    """Hitung lambda exact dari response penampang."""
    if abs(float(demand_moment)) > 1e-12:
        return float(response['phi_Mn']) / float(demand_moment)
    if abs(float(demand_axial)) > AXIAL_DEMAND_TOLERANCE_KN:
        return float(response['phi_Pn']) / float(demand_axial)
    return float('inf')


def find_exact_interaction_boundary_state(fc: float,
                                          fy_tarik: float,
                                          fy_tekan: float,
                                          section_geometry: Dict,
                                          steel_area: Dict,
                                          demand_axial: float,
                                          demand_moment: float,
                                          use_code_phi: bool = False,
                                          num_scan_points: int = 360,
                                          tolerance: float = 1e-9) -> Optional[Dict[str, Any]]:
    """Cari c_boundary exact dari persamaan kontinu Md*Pn(c) - Pd*Mn(c) = 0."""
    demand_axial = float(demand_axial)
    demand_moment = abs(float(demand_moment))
    if abs(demand_axial) <= AXIAL_DEMAND_TOLERANCE_KN and demand_moment <= 1e-12:
        return None
    if demand_moment <= 1e-12:
        return None

    h_value = max(float(section_geometry.get('h', 0.0) or 0.0), 1.0)
    c_values = np.geomspace(
        max(1e-4 * h_value, 1e-4),
        max(25.0 * h_value, 1.0),
        num=max(int(num_scan_points), 80)
    )

    def evaluate_response(c_value: float) -> Tuple[Dict[str, Any], float]:
        response = PerformanceFunction._get_column_section_response_at_c(
            fc,
            fy_tarik,
            section_geometry,
            steel_area,
            c_value,
            fy_tekan=fy_tekan,
            use_code_phi=use_code_phi
        )
        residual = float(
            demand_moment * float(response['phi_Pn'])
            - demand_axial * float(response['phi_Mn'])
        )
        return response, residual

    candidates = []
    previous_c = None
    previous_response = None
    previous_residual = None

    for c_value in c_values:
        response, residual = evaluate_response(float(c_value))
        if not np.isfinite(residual):
            continue

        lambda_value = _get_exact_boundary_lambda(response, demand_axial, demand_moment)
        if abs(residual) <= tolerance and lambda_value >= 0.0:
            candidates.append({
                **response,
                'lambda': float(lambda_value),
                'residual': float(residual)
            })

        if (
            previous_c is not None
            and previous_response is not None
            and previous_residual is not None
            and np.isfinite(previous_residual)
            and previous_residual * residual < 0.0
        ):
            low_c = float(previous_c)
            high_c = float(c_value)
            low_residual = float(previous_residual)

            for _ in range(80):
                mid_c = 0.5 * (low_c + high_c)
                _, mid_residual = evaluate_response(mid_c)
                if abs(mid_residual) <= tolerance:
                    low_c = high_c = mid_c
                    break
                if low_residual * mid_residual <= 0.0:
                    high_c = mid_c
                else:
                    low_c = mid_c
                    low_residual = float(mid_residual)

            exact_c = 0.5 * (low_c + high_c)
            exact_response, exact_residual = evaluate_response(exact_c)
            lambda_value = _get_exact_boundary_lambda(
                exact_response,
                demand_axial,
                demand_moment
            )
            if lambda_value >= 0.0:
                candidates.append({
                    **exact_response,
                    'lambda': float(lambda_value),
                    'residual': float(exact_residual)
                })

        previous_c = float(c_value)
        previous_response = response
        previous_residual = float(residual)

    if not candidates:
        return None

    return min(candidates, key=lambda candidate: candidate['lambda'])


def _estimate_interaction_label_box(text: str,
                                    x_span: float,
                                    y_span: float) -> Tuple[float, float]:
    """Estimasi ukuran box label dalam koordinat data untuk menghindari overlap."""
    lines = [str(line) for line in str(text).splitlines()]
    if not lines:
        lines = [str(text)]

    max_line_length = max(len(line) for line in lines)
    line_count = max(len(lines), 1)
    box_width = max(0.12 * x_span, min(0.34 * x_span, 0.0075 * x_span * max_line_length))
    box_height = max(0.10 * y_span, min(0.30 * y_span, 0.060 * y_span * line_count))
    return float(box_width), float(box_height)


def _interaction_label_bounds(anchor_x: float,
                              anchor_y: float,
                              box_width: float,
                              box_height: float,
                              ha: str,
                              va: str) -> Tuple[float, float, float, float]:
    """Hitung bounds box label berdasarkan titik anchor dan alignment."""
    if ha == 'left':
        x_min = anchor_x
        x_max = anchor_x + box_width
    elif ha == 'right':
        x_min = anchor_x - box_width
        x_max = anchor_x
    else:
        x_min = anchor_x - (0.5 * box_width)
        x_max = anchor_x + (0.5 * box_width)

    if va == 'bottom':
        y_min = anchor_y
        y_max = anchor_y + box_height
    elif va == 'top':
        y_min = anchor_y - box_height
        y_max = anchor_y
    else:
        y_min = anchor_y - (0.5 * box_height)
        y_max = anchor_y + (0.5 * box_height)

    return float(x_min), float(y_min), float(x_max), float(y_max)


def _interaction_box_point_distance(bounds: Tuple[float, float, float, float],
                                    point: Tuple[float, float],
                                    x_span: float,
                                    y_span: float) -> float:
    """Hitung jarak ternormalisasi antara box label dan titik tertentu."""
    x_min, y_min, x_max, y_max = bounds
    point_x, point_y = point
    delta_x = max(x_min - point_x, 0.0, point_x - x_max)
    delta_y = max(y_min - point_y, 0.0, point_y - y_max)
    return float(np.hypot(
        delta_x / max(x_span, 1e-9),
        delta_y / max(y_span, 1e-9)
    ))


def _interaction_box_overlap_ratio(left_bounds: Tuple[float, float, float, float],
                                   right_bounds: Tuple[float, float, float, float]) -> float:
    """Hitung rasio overlap dua box label."""
    left_x_min, left_y_min, left_x_max, left_y_max = left_bounds
    right_x_min, right_y_min, right_x_max, right_y_max = right_bounds
    overlap_width = max(0.0, min(left_x_max, right_x_max) - max(left_x_min, right_x_min))
    overlap_height = max(0.0, min(left_y_max, right_y_max) - max(left_y_min, right_y_min))
    overlap_area = overlap_width * overlap_height
    left_area = max((left_x_max - left_x_min) * (left_y_max - left_y_min), 1e-9)
    return float(overlap_area / left_area)


def _build_interaction_label_candidates(base_x: float,
                                        base_y: float,
                                        x_span: float,
                                        y_span: float,
                                        radial_scale: float = 1.0) -> List[Dict[str, Any]]:
    """Bangun kandidat posisi label di sekitar titik acuan."""
    offset_x = max(0.10 * x_span, 8.0) * float(radial_scale)
    offset_y = max(0.11 * y_span, 18.0) * float(radial_scale)
    far_offset_x = 1.35 * offset_x
    far_offset_y = 1.30 * offset_y

    return [
        {'x': base_x + offset_x, 'y': base_y + offset_y, 'ha': 'left', 'va': 'bottom'},
        {'x': base_x - offset_x, 'y': base_y + offset_y, 'ha': 'right', 'va': 'bottom'},
        {'x': base_x + offset_x, 'y': base_y - offset_y, 'ha': 'left', 'va': 'top'},
        {'x': base_x - offset_x, 'y': base_y - offset_y, 'ha': 'right', 'va': 'top'},
        {'x': base_x + far_offset_x, 'y': base_y, 'ha': 'left', 'va': 'center'},
        {'x': base_x - far_offset_x, 'y': base_y, 'ha': 'right', 'va': 'center'},
        {'x': base_x, 'y': base_y + far_offset_y, 'ha': 'center', 'va': 'bottom'},
        {'x': base_x, 'y': base_y - far_offset_y, 'ha': 'center', 'va': 'top'}
    ]


def _build_interaction_line_label_candidates(target_x: float,
                                             target_y: float,
                                             x_span: float,
                                             y_span: float) -> List[Dict[str, Any]]:
    """Bangun kandidat label garis lambda di beberapa titik sepanjang garis."""
    candidates: List[Dict[str, Any]] = []
    for fraction in (0.16, 0.26, 0.38, 0.62):
        base_x = float(target_x) * float(fraction)
        base_y = float(target_y) * float(fraction)
        candidates.extend(
            _build_interaction_label_candidates(
                base_x,
                base_y,
                x_span,
                y_span,
                radial_scale=0.75
            )
        )
    return candidates


def _choose_interaction_label_position(axis,
                                       text: str,
                                       point_x: float,
                                       point_y: float,
                                       avoid_points: Optional[List[Tuple[float, float]]] = None,
                                       avoid_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
                                       candidate_positions: Optional[List[Dict[str, Any]]] = None,
                                       radial_scale: float = 1.0) -> Dict[str, Any]:
    """Pilih kandidat posisi label terbaik agar box tetap rapi dan minim overlap."""
    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()
    x_span = max(float(x_max - x_min), 1.0)
    y_span = max(float(y_max - y_min), 1.0)
    box_width, box_height = _estimate_interaction_label_box(text, x_span, y_span)

    candidates = candidate_positions or _build_interaction_label_candidates(
        point_x,
        point_y,
        x_span,
        y_span,
        radial_scale=radial_scale
    )
    avoid_points = list(avoid_points or [])
    avoid_boxes = list(avoid_boxes or [])
    base_point = (float(point_x), float(point_y))

    best_choice = None
    best_score = -float('inf')
    for candidate in candidates:
        bounds = _interaction_label_bounds(
            float(candidate['x']),
            float(candidate['y']),
            box_width,
            box_height,
            str(candidate['ha']),
            str(candidate['va'])
        )

        overflow_penalty = (
            max(x_min - bounds[0], 0.0)
            + max(bounds[2] - x_max, 0.0)
        ) / x_span + (
            max(y_min - bounds[1], 0.0)
            + max(bounds[3] - y_max, 0.0)
        ) / y_span

        overlap_penalty = sum(
            _interaction_box_overlap_ratio(bounds, other_bounds)
            for other_bounds in avoid_boxes
        )

        point_clearances = [
            _interaction_box_point_distance(bounds, other_point, x_span, y_span)
            for other_point in avoid_points
        ]
        min_point_clearance = min(point_clearances) if point_clearances else 1.0
        base_point_clearance = _interaction_box_point_distance(
            bounds,
            base_point,
            x_span,
            y_span
        )

        score = (
            3.0 * min_point_clearance
            + 0.8 * base_point_clearance
            - 8.0 * overflow_penalty
            - 14.0 * overlap_penalty
        )

        if score > best_score:
            best_score = score
            best_choice = {
                'x': float(candidate['x']),
                'y': float(candidate['y']),
                'ha': str(candidate['ha']),
                'va': str(candidate['va']),
                'bounds': bounds
            }

    if best_choice is None:
        best_choice = {
            'x': float(point_x),
            'y': float(point_y),
            'ha': 'left',
            'va': 'bottom',
            'bounds': _interaction_label_bounds(
                float(point_x),
                float(point_y),
                box_width,
                box_height,
                'left',
                'bottom'
            )
        }
    return best_choice


def _build_interaction_offset_candidates(scale: float = 1.0) -> List[Dict[str, Any]]:
    """Kandidat offset label dalam satuan points untuk anotasi titik."""
    near = 14.0 * float(scale)
    mid = 28.0 * float(scale)
    far = 42.0 * float(scale)
    return [
        {'dx': near, 'dy': near, 'ha': 'left', 'va': 'bottom'},
        {'dx': -near, 'dy': near, 'ha': 'right', 'va': 'bottom'},
        {'dx': near, 'dy': -near, 'ha': 'left', 'va': 'top'},
        {'dx': -near, 'dy': -near, 'ha': 'right', 'va': 'top'},
        {'dx': mid, 'dy': 0.0, 'ha': 'left', 'va': 'center'},
        {'dx': -mid, 'dy': 0.0, 'ha': 'right', 'va': 'center'},
        {'dx': 0.0, 'dy': mid, 'ha': 'center', 'va': 'bottom'},
        {'dx': 0.0, 'dy': -mid, 'ha': 'center', 'va': 'top'},
        {'dx': far, 'dy': near, 'ha': 'left', 'va': 'bottom'},
        {'dx': -far, 'dy': near, 'ha': 'right', 'va': 'bottom'},
        {'dx': far, 'dy': -near, 'ha': 'left', 'va': 'top'},
        {'dx': -far, 'dy': -near, 'ha': 'right', 'va': 'top'}
    ]


def _build_interaction_line_annotation_candidates(target_x: float,
                                                  target_y: float,
                                                  scale: float = 1.0) -> List[Dict[str, Any]]:
    """Kandidat anotasi untuk label garis lambda di beberapa titik sepanjang garis."""
    candidates: List[Dict[str, Any]] = []
    for fraction in (0.18, 0.30, 0.44, 0.60):
        anchor_xy = (float(target_x) * float(fraction), float(target_y) * float(fraction))
        for offset_candidate in _build_interaction_offset_candidates(scale=scale):
            candidates.append({
                'anchor_xy': anchor_xy,
                'dx': float(offset_candidate['dx']),
                'dy': float(offset_candidate['dy']),
                'ha': str(offset_candidate['ha']),
                'va': str(offset_candidate['va'])
            })
    return candidates


def _interaction_bbox_overlap_ratio_display(left_bbox,
                                            right_bbox) -> float:
    """Rasio overlap dua bbox dalam koordinat display."""
    overlap_width = max(0.0, min(left_bbox.x1, right_bbox.x1) - max(left_bbox.x0, right_bbox.x0))
    overlap_height = max(0.0, min(left_bbox.y1, right_bbox.y1) - max(left_bbox.y0, right_bbox.y0))
    overlap_area = overlap_width * overlap_height
    left_area = max(left_bbox.width * left_bbox.height, 1e-9)
    return float(overlap_area / left_area)


def _interaction_bbox_point_clearance_display(bbox,
                                              point_display: Tuple[float, float],
                                              axes_bbox) -> float:
    """Jarak ternormalisasi antara bbox label dan titik dalam koordinat display."""
    point_x, point_y = point_display
    delta_x = max(bbox.x0 - point_x, 0.0, point_x - bbox.x1)
    delta_y = max(bbox.y0 - point_y, 0.0, point_y - bbox.y1)
    axes_diagonal = max(
        float(np.hypot(float(axes_bbox.width), float(axes_bbox.height))),
        1e-9
    )
    return float(np.hypot(delta_x, delta_y) / axes_diagonal)


def _choose_interaction_annotation_spec(axis,
                                        renderer,
                                        text: str,
                                        target_xy: Tuple[float, float],
                                        occupied_bboxes: Optional[List[Any]] = None,
                                        avoid_points: Optional[List[Tuple[float, float]]] = None,
                                        candidate_specs: Optional[List[Dict[str, Any]]] = None,
                                        with_arrow: bool = True,
                                        fontsize: float = 8.0) -> Dict[str, Any]:
    """Pilih spesifikasi anotasi terbaik berdasarkan bbox render aktual Matplotlib."""
    axes_bbox = axis.get_window_extent(renderer)
    occupied_bboxes = list(occupied_bboxes or [])
    avoid_points_display = [
        axis.transData.transform((float(point[0]), float(point[1])))
        for point in (avoid_points or [])
    ]
    target_xy = (float(target_xy[0]), float(target_xy[1]))
    candidates = list(candidate_specs or [])
    if not candidates:
        candidates = [
            {
                'anchor_xy': target_xy,
                'dx': float(offset_candidate['dx']),
                'dy': float(offset_candidate['dy']),
                'ha': str(offset_candidate['ha']),
                'va': str(offset_candidate['va'])
            }
            for offset_candidate in _build_interaction_offset_candidates()
        ]

    best_spec = None
    best_score = -float('inf')
    for candidate in candidates:
        anchor_xy = tuple(candidate.get('anchor_xy', target_xy))
        temporary_annotation = axis.annotate(
            text,
            xy=anchor_xy,
            xytext=(float(candidate['dx']), float(candidate['dy'])),
            textcoords='offset points',
            ha=str(candidate['ha']),
            va=str(candidate['va']),
            fontsize=float(fontsize),
            annotation_clip=False,
            bbox=dict(
                boxstyle='round,pad=0.22',
                facecolor='white',
                alpha=0.90,
                edgecolor='#cbd5e1'
            ),
            arrowprops=(
                dict(arrowstyle='->', color='#9ca3af', lw=0.9)
                if with_arrow else
                None
            )
        )
        candidate_bbox = temporary_annotation.get_window_extent(renderer).expanded(1.03, 1.08)
        temporary_annotation.remove()

        overflow_penalty = (
            max(float(axes_bbox.x0) - float(candidate_bbox.x0), 0.0)
            + max(float(candidate_bbox.x1) - float(axes_bbox.x1), 0.0)
            + max(float(axes_bbox.y0) - float(candidate_bbox.y0), 0.0)
            + max(float(candidate_bbox.y1) - float(axes_bbox.y1), 0.0)
        ) / max(float(axes_bbox.width + axes_bbox.height), 1e-9)
        overlap_penalty = sum(
            _interaction_bbox_overlap_ratio_display(candidate_bbox, other_bbox)
            for other_bbox in occupied_bboxes
        )
        point_clearance = min(
            (
                _interaction_bbox_point_clearance_display(candidate_bbox, point_display, axes_bbox)
                for point_display in avoid_points_display
            ),
            default=1.0
        )
        offset_length = float(np.hypot(float(candidate['dx']), float(candidate['dy'])))
        offset_penalty = offset_length / 180.0

        score = (
            4.0 * point_clearance
            - 12.0 * overlap_penalty
            - 8.0 * overflow_penalty
            - 0.25 * offset_penalty
        )
        if score > best_score:
            best_score = score
            best_spec = {
                'anchor_xy': anchor_xy,
                'dx': float(candidate['dx']),
                'dy': float(candidate['dy']),
                'ha': str(candidate['ha']),
                'va': str(candidate['va'])
            }

    if best_spec is None:
        best_spec = {
            'anchor_xy': target_xy,
            'dx': 16.0,
            'dy': 16.0,
            'ha': 'left',
            'va': 'bottom'
        }
    return best_spec


def _add_interaction_annotation(axis,
                                renderer,
                                text: str,
                                target_xy: Tuple[float, float],
                                bbox_edgecolor: str,
                                occupied_bboxes: List[Any],
                                avoid_points: Optional[List[Tuple[float, float]]] = None,
                                candidate_specs: Optional[List[Dict[str, Any]]] = None,
                                with_arrow: bool = True,
                                fontsize: float = 8.0,
                                text_color: Optional[str] = None):
    """Tambahkan anotasi final dan simpan bbox-nya agar label berikutnya menghindar."""
    chosen_spec = _choose_interaction_annotation_spec(
        axis,
        renderer,
        text=text,
        target_xy=target_xy,
        occupied_bboxes=occupied_bboxes,
        avoid_points=avoid_points,
        candidate_specs=candidate_specs,
        with_arrow=with_arrow,
        fontsize=fontsize
    )
    annotation = axis.annotate(
        text,
        xy=tuple(chosen_spec['anchor_xy']),
        xytext=(float(chosen_spec['dx']), float(chosen_spec['dy'])),
        textcoords='offset points',
        ha=str(chosen_spec['ha']),
        va=str(chosen_spec['va']),
        fontsize=float(fontsize),
        color=text_color,
        annotation_clip=False,
        bbox=dict(
            boxstyle='round,pad=0.22',
            facecolor='white',
            alpha=0.90,
            edgecolor=bbox_edgecolor
        ),
        arrowprops=(
            dict(arrowstyle='->', color=bbox_edgecolor, lw=0.9)
            if with_arrow else
            None
        )
    )
    occupied_bboxes.append(annotation.get_window_extent(renderer).expanded(1.03, 1.08))
    return annotation


def add_smart_mpp_annotation(axis,
                             text: str,
                             target_xy: Tuple[float, float],
                             avoid_points: Optional[List[Tuple[float, float]]] = None,
                             occupied_bboxes: Optional[List[Any]] = None,
                             bbox_edgecolor: str = '#cbd5e1',
                             text_color: str = '#111827',
                             fontsize: float = 8.0,
                             zorder: float = 7.0,
                             with_arrow: bool = True):
    """Tambahkan anotasi MPP yang tetap berada di dalam area axis dan minim overlap."""
    figure = getattr(axis, 'figure', None)
    renderer = None
    if figure is not None and getattr(figure, 'canvas', None) is not None:
        try:
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
        except Exception:
            renderer = None

    if renderer is None:
        return axis.annotate(
            text,
            xy=(float(target_xy[0]), float(target_xy[1])),
            xytext=(10, -12),
            textcoords='offset points',
            ha='left',
            va='top',
            fontsize=float(fontsize),
            color=text_color,
            annotation_clip=False,
            bbox=dict(
                boxstyle='round,pad=0.22',
                facecolor='white',
                edgecolor=bbox_edgecolor,
                alpha=0.92
            ),
            arrowprops=(
                dict(arrowstyle='->', color=bbox_edgecolor, lw=0.9)
                if with_arrow else
                None
            ),
            zorder=float(zorder)
        )

    candidate_specs = []
    for scale in (0.65, 0.85, 1.05, 1.30, 1.55):
        for offset_candidate in _build_interaction_offset_candidates(scale=scale):
            candidate_specs.append({
                'anchor_xy': (float(target_xy[0]), float(target_xy[1])),
                'dx': float(offset_candidate['dx']),
                'dy': float(offset_candidate['dy']),
                'ha': str(offset_candidate['ha']),
                'va': str(offset_candidate['va'])
            })

    chosen_spec = _choose_interaction_annotation_spec(
        axis,
        renderer,
        text=text,
        target_xy=(float(target_xy[0]), float(target_xy[1])),
        occupied_bboxes=list(occupied_bboxes or []),
        avoid_points=list(avoid_points or []),
        candidate_specs=candidate_specs,
        with_arrow=with_arrow,
        fontsize=float(fontsize)
    )
    annotation = axis.annotate(
        text,
        xy=tuple(chosen_spec['anchor_xy']),
        xytext=(float(chosen_spec['dx']), float(chosen_spec['dy'])),
        textcoords='offset points',
        ha=str(chosen_spec['ha']),
        va=str(chosen_spec['va']),
        fontsize=float(fontsize),
        color=text_color,
        annotation_clip=False,
        bbox=dict(
            boxstyle='round,pad=0.22',
            facecolor='white',
            edgecolor=bbox_edgecolor,
            alpha=0.92
        ),
        arrowprops=(
            dict(arrowstyle='->', color=bbox_edgecolor, lw=0.9)
            if with_arrow else
            None
        ),
        zorder=float(zorder)
    )
    return annotation


def build_interaction_diagram_figure(input_data: Dict,
                                     latest_simulation: Dict,
                                     latest_result: Dict,
                                     is_probabilistic: bool,
                                     elem_id: int) -> Dict[str, Any]:
    """Bangun figure kurva interaksi dan data ringkasnya untuk satu elemen."""
    elem_id = int(elem_id)
    if get_element_code_from_input(input_data, elem_id) != 'K':
        raise ValueError(
            f"Elemen {elem_id} bukan kolom, sehingga tidak punya cek aksial-lentur pada dashboard."
        )

    section_inputs = get_section_capacity_inputs_from_input(input_data, elem_id)
    material_snapshot = get_element_material_snapshot(
        input_data,
        latest_simulation,
        is_probabilistic,
        elem_id
    )
    max_forces_entry = get_max_forces_entry_by_element(latest_result, elem_id)
    if not max_forces_entry:
        raise ValueError(f"Gaya dalam elemen {elem_id} tidak ditemukan pada hasil analisis.")

    force_data = (max_forces_entry.get('forces') or {}) if isinstance(max_forces_entry, dict) else {}
    axial_demands = get_axial_demands_from_force_data(force_data)
    max_moment = abs(float(max_forces_entry.get('max_moment', 0.0) or 0.0))
    interaction_curve = PerformanceFunction._get_column_interaction_curve(
        material_snapshot['fc'],
        material_snapshot['fy_tarik'],
        section_inputs['section_geometry'],
        section_inputs['steel_area'],
        fy_tekan=material_snapshot['fy_tekan'],
        use_code_phi=not is_probabilistic
    )

    axial_moment_meta = (
        get_by_element_value(
            latest_result.get('performance_axial_moment_metadata', {}),
            elem_id,
            {}
        ) or {}
    )
    controlling_state = str(axial_moment_meta.get('controlling_state', '') or '').strip().lower()
    if controlling_state == 'tension' and axial_demands['tension'] > AXIAL_DEMAND_TOLERANCE_KN:
        demand_axial = -float(axial_demands['tension'])
    elif controlling_state == 'pure-bending':
        demand_axial = 0.0
    elif axial_demands['compression'] > AXIAL_DEMAND_TOLERANCE_KN:
        demand_axial = float(axial_demands['compression'])
    elif axial_demands['tension'] > AXIAL_DEMAND_TOLERANCE_KN:
        demand_axial = -float(axial_demands['tension'])
    else:
        demand_axial = 0.0

    boundary = PerformanceFunction._get_interaction_boundary_state(
        demand_axial,
        max_moment,
        interaction_curve
    )
    moment_values = [float(point['phi_Mn']) for point in interaction_curve]
    axial_values = [float(point['phi_Pn']) for point in interaction_curve]
    boundary_moment = float(boundary.get('phi_Mn', 0.0) or 0.0)
    boundary_axial = float(boundary.get('phi_Pn', 0.0) or 0.0)
    lambda_value = float(boundary.get('lambda', 0.0) or 0.0)
    g_value = float(lambda_value - 1.0) if np.isfinite(lambda_value) else float('inf')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    for axis in axes:
        axis.plot(moment_values, axial_values, color='#0f4c81', lw=2.2, label='Kurva interaksi')
        axis.scatter(
            [max_moment],
            [demand_axial],
            color='#d62828',
            s=55,
            zorder=5,
            label='Demand'
        )
        axis.scatter(
            [boundary_moment],
            [boundary_axial],
            color='#2a9d8f',
            s=60,
            zorder=6,
            label='Boundary'
        )
        axis.plot(
            [0.0, boundary_moment],
            [0.0, boundary_axial],
            color='#6c757d',
            ls='--',
            lw=1.5,
            label='Garis lambda'
        )
        axis.grid(True, alpha=0.25)
        axis.set_xlabel('Momen, phiMn (kN.m)')
        axis.set_ylabel('Aksial, phiPn (kN)')

    if segment is not None:
        first = segment['first']
        second = segment['second']
        for axis in axes:
            axis.plot(
                [float(first['phi_Mn']), float(second['phi_Mn'])],
                [float(first['phi_Pn']), float(second['phi_Pn'])],
                color='#ffb703',
                lw=3.0,
                label='Segmen kontrol'
            )
            axis.scatter(
                [float(first['phi_Mn']), float(second['phi_Mn'])],
                [float(first['phi_Pn']), float(second['phi_Pn'])],
                color='#ffb703',
                s=28,
                zorder=6
            )

    line_mid_x = 0.5 * boundary_moment
    line_mid_y = 0.5 * boundary_axial
    axes[0].annotate(
        f"Demand\nM={max_moment:.2f} kN.m\nP={demand_axial:.2f} kN",
        xy=(max_moment, demand_axial),
        xytext=(max_moment + max(12.0, 0.08 * max(moment_values or [1.0])), demand_axial),
        arrowprops=dict(arrowstyle='->', color='#d62828', lw=1.0),
        fontsize=8
    )
    axes[0].annotate(
        f"Boundary\nM={boundary_moment:.2f} kN.m\nP={boundary_axial:.2f} kN",
        xy=(boundary_moment, boundary_axial),
        xytext=(
            boundary_moment + max(12.0, 0.08 * max(moment_values or [1.0])),
            boundary_axial + max(30.0, 0.05 * max(abs(value) for value in axial_values or [1.0]))
        ),
        arrowprops=dict(arrowstyle='->', color='#2a9d8f', lw=1.0),
        fontsize=8
    )
    axes[0].text(
        line_mid_x,
        line_mid_y,
        f"Garis lambda\nλ = {lambda_value:.4f}",
        fontsize=8,
        ha='left',
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='#9ca3af')
    )
    axes[0].set_title(f'Diagram Interaksi Elemen {elem_id}')
    axes[0].legend(loc='best', fontsize=8)

    zoom_x_values = [max_moment, boundary_moment]
    zoom_y_values = [demand_axial, boundary_axial]
    if segment is not None:
        zoom_x_values.extend([float(segment['first']['phi_Mn']), float(segment['second']['phi_Mn'])])
        zoom_y_values.extend([float(segment['first']['phi_Pn']), float(segment['second']['phi_Pn'])])

    x_span = max(zoom_x_values) - min(zoom_x_values)
    y_span = max(zoom_y_values) - min(zoom_y_values)
    x_padding = max(10.0, 0.2 * max(x_span, 1.0))
    y_padding = max(40.0, 0.2 * max(y_span, 1.0))
    axes[1].set_xlim(min(zoom_x_values) - x_padding, max(zoom_x_values) + x_padding)
    axes[1].set_ylim(min(zoom_y_values) - y_padding, max(zoom_y_values) + y_padding)
    axes[1].set_title('Zoom Titik Kontrol')
    axes[1].annotate(
        'Demand',
        xy=(max_moment, demand_axial),
        xytext=(6, 6),
        textcoords='offset points',
        fontsize=8,
        color='#d62828'
    )
    axes[1].annotate(
        'Boundary',
        xy=(boundary_moment, boundary_axial),
        xytext=(6, 6),
        textcoords='offset points',
        fontsize=8,
        color='#2a9d8f'
    )
    axes[1].text(
        line_mid_x,
        line_mid_y,
        f"λ = {lambda_value:.4f}",
        fontsize=8,
        ha='left',
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='#9ca3af')
    )

    fig.suptitle(
        (
            f"Kurva Interaksi Aksial-Lentur | Elemen {elem_id} | "
            f"fc'={material_snapshot['fc']:.2f} MPa | "
            f"fy={material_snapshot['fy_tarik']:.2f} MPa | "
            f"g(x)={g_value:.4f}"
        ),
        fontsize=11,
        y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return {
        'figure': fig,
        'lambda': lambda_value,
        'g': g_value,
        'demand_axial': demand_axial,
        'demand_moment': max_moment,
        'boundary_axial': boundary_axial,
        'boundary_moment': boundary_moment,
        'controlling_state': controlling_state or '-',
        'material_snapshot': material_snapshot
    }


def build_interaction_diagram_figure(input_data: Dict,
                                     latest_simulation: Dict,
                                     latest_result: Dict,
                                     is_probabilistic: bool,
                                     elem_id: int) -> Dict[str, Any]:
    """Versi override dengan boundary interpolasi dan boundary exact."""
    elem_id = int(elem_id)
    if get_element_code_from_input(input_data, elem_id) != 'K':
        raise ValueError(
            f"Elemen {elem_id} bukan kolom, sehingga tidak punya cek aksial-lentur pada dashboard."
        )

    section_inputs = get_section_capacity_inputs_from_input(input_data, elem_id)
    material_snapshot = get_element_material_snapshot(
        input_data,
        latest_simulation,
        is_probabilistic,
        elem_id
    )
    max_forces_entry = get_max_forces_entry_by_element(latest_result, elem_id)
    if not max_forces_entry:
        raise ValueError(f"Gaya dalam elemen {elem_id} tidak ditemukan pada hasil analisis.")

    force_data = (max_forces_entry.get('forces') or {}) if isinstance(max_forces_entry, dict) else {}
    axial_demands = get_axial_demands_from_force_data(force_data)
    max_moment = abs(float(max_forces_entry.get('max_moment', 0.0) or 0.0))
    interaction_curve = PerformanceFunction._get_column_interaction_curve(
        material_snapshot['fc'],
        material_snapshot['fy_tarik'],
        section_inputs['section_geometry'],
        section_inputs['steel_area'],
        fy_tekan=material_snapshot['fy_tekan'],
        use_code_phi=not is_probabilistic
    )

    axial_moment_meta = (
        get_by_element_value(
            latest_result.get('performance_axial_moment_metadata', {}),
            elem_id,
            {}
        ) or {}
    )
    controlling_state = str(axial_moment_meta.get('controlling_state', '') or '').strip().lower()
    if controlling_state == 'tension' and axial_demands['tension'] > AXIAL_DEMAND_TOLERANCE_KN:
        demand_axial = -float(axial_demands['tension'])
    elif controlling_state == 'pure-bending':
        demand_axial = 0.0
    elif axial_demands['compression'] > AXIAL_DEMAND_TOLERANCE_KN:
        demand_axial = float(axial_demands['compression'])
    elif axial_demands['tension'] > AXIAL_DEMAND_TOLERANCE_KN:
        demand_axial = -float(axial_demands['tension'])
    else:
        demand_axial = 0.0

    boundary = PerformanceFunction._get_interaction_boundary_state(
        demand_axial,
        max_moment,
        interaction_curve
    )
    exact_boundary = find_exact_interaction_boundary_state(
        material_snapshot['fc'],
        material_snapshot['fy_tarik'],
        material_snapshot['fy_tekan'],
        section_inputs['section_geometry'],
        section_inputs['steel_area'],
        demand_axial,
        max_moment,
        use_code_phi=not is_probabilistic
    )

    moment_values = [float(point['phi_Mn']) for point in interaction_curve]
    axial_values = [float(point['phi_Pn']) for point in interaction_curve]
    boundary_moment = float(boundary.get('phi_Mn', 0.0) or 0.0)
    boundary_axial = float(boundary.get('phi_Pn', 0.0) or 0.0)
    lambda_interp = float(boundary.get('lambda', 0.0) or 0.0)
    g_interp = float(lambda_interp - 1.0) if np.isfinite(lambda_interp) else float('inf')

    exact_boundary_moment = None
    exact_boundary_axial = None
    lambda_exact = None
    g_exact = None
    c_boundary_exact = None
    if exact_boundary is not None:
        exact_boundary_moment = float(exact_boundary.get('phi_Mn', 0.0) or 0.0)
        exact_boundary_axial = float(exact_boundary.get('phi_Pn', 0.0) or 0.0)
        lambda_exact = float(exact_boundary.get('lambda', 0.0) or 0.0)
        g_exact = float(lambda_exact - 1.0) if np.isfinite(lambda_exact) else float('inf')
        c_boundary_exact = float(exact_boundary.get('neutral_axis_depth', 0.0) or 0.0)

    interp_boundary_available = bool(
        np.isfinite(boundary_moment)
        and np.isfinite(boundary_axial)
    )
    exact_boundary_found = bool(
        exact_boundary_moment is not None
        and exact_boundary_axial is not None
        and np.isfinite(float(exact_boundary_moment))
        and np.isfinite(float(exact_boundary_axial))
    )
    moment_span_reference = max(
        max(moment_values) - min(moment_values) if moment_values else 0.0,
        abs(max_moment),
        abs(boundary_moment),
        abs(exact_boundary_moment) if exact_boundary_moment is not None else 0.0,
        1.0
    )
    axial_span_reference = max(
        max(axial_values) - min(axial_values) if axial_values else 0.0,
        abs(demand_axial),
        abs(boundary_axial),
        abs(exact_boundary_axial) if exact_boundary_axial is not None else 0.0,
        1.0
    )
    exact_boundary_overlaps_interp = bool(
        interp_boundary_available
        and exact_boundary_found
        and abs(float(exact_boundary_moment) - float(boundary_moment)) <= 0.015 * moment_span_reference
        and abs(float(exact_boundary_axial) - float(boundary_axial)) <= 0.015 * axial_span_reference
    )
    if not exact_boundary_found:
        boundary_exact_status = 'Tidak ditemukan'
    elif exact_boundary_overlaps_interp:
        boundary_exact_status = 'Ditemukan, hampir berimpit'
    else:
        boundary_exact_status = 'Ditemukan'

    line_target_moment = exact_boundary_moment if exact_boundary_moment is not None else boundary_moment
    line_target_axial = exact_boundary_axial if exact_boundary_axial is not None else boundary_axial
    line_target_lambda = lambda_exact if lambda_exact is not None else lambda_interp

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    for axis in axes:
        axis.plot(moment_values, axial_values, color='#0f4c81', lw=2.2, label='Kurva interaksi')
        axis.scatter([max_moment], [demand_axial], color='#d62828', s=55, zorder=5, label='Demand')
        if interp_boundary_available:
            axis.scatter(
                [boundary_moment],
                [boundary_axial],
                marker='D',
                s=155 if exact_boundary_overlaps_interp else 70,
                facecolors='none' if exact_boundary_overlaps_interp else '#14b8a6',
                edgecolors='#0f766e' if exact_boundary_overlaps_interp else '#ffffff',
                linewidths=1.8 if exact_boundary_overlaps_interp else 0.9,
                zorder=6.6,
                label='Boundary interp'
            )
        axis.plot(
            [0.0, line_target_moment],
            [0.0, line_target_axial],
            color='#6c757d',
            ls='--',
            lw=1.5,
            label='Garis lambda'
        )
        axis.grid(True, alpha=0.25)
        axis.set_xlabel('Momen, phiMn (kN.m)')
        axis.set_ylabel('Aksial, phiPn (kN)')

    if exact_boundary_moment is not None and exact_boundary_axial is not None:
        for axis in axes:
            axis.scatter(
                [exact_boundary_moment],
                [exact_boundary_axial],
                color='#ffffff',
                marker='*',
                s=300,
                linewidths=0.0,
                zorder=7.8,
                label='_nolegend_'
            )
            axis.scatter(
                [exact_boundary_moment],
                [exact_boundary_axial],
                color='#7c3aed',
                marker='*',
                s=210,
                edgecolors='#7c3aed',
                linewidths=0.6,
                zorder=8.0,
                label='Boundary exact (c)'
            )
            axis.scatter(
                [exact_boundary_moment],
                [exact_boundary_axial],
                color='#581c87',
                marker='o',
                s=30,
                edgecolors='#ffffff',
                linewidths=0.75,
                zorder=8.3,
                label='_nolegend_'
            )

    full_view_x_values = moment_values + [max_moment, line_target_moment]
    full_view_y_values = axial_values + [demand_axial, line_target_axial]
    if exact_boundary_moment is not None and exact_boundary_axial is not None:
        full_view_x_values.append(exact_boundary_moment)
        full_view_y_values.append(exact_boundary_axial)

    full_x_min = min(0.0, min(full_view_x_values))
    full_x_max = max(full_view_x_values)
    full_y_min = min(0.0, min(full_view_y_values))
    full_y_max = max(full_view_y_values)
    full_x_span = max(full_x_max - full_x_min, 1.0)
    full_y_span = max(full_y_max - full_y_min, 1.0)
    full_x_padding = max(15.0, 0.08 * full_x_span)
    full_y_padding = max(60.0, 0.08 * full_y_span)
    axes[0].set_xlim(max(0.0, full_x_min - full_x_padding), full_x_max + full_x_padding)
    axes[0].set_ylim(full_y_min - full_y_padding, full_y_max + full_y_padding)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    demand_full_text = f"Demand\nM={max_moment:.2f} kN.m\nP={demand_axial:.2f} kN"
    boundary_full_text = None
    if exact_boundary_moment is not None and exact_boundary_axial is not None and c_boundary_exact is not None:
        boundary_full_text = (
            "Boundary exact\n"
            f"M={exact_boundary_moment:.2f} kN.m\n"
            f"P={exact_boundary_axial:.2f} kN\n"
            f"c={c_boundary_exact:.3f} mm"
        )
    lambda_full_text = f"Garis lambda\nlambda = {line_target_lambda:.4f}"

    full_label_boxes: List[Any] = []
    if boundary_full_text is not None:
        _add_interaction_annotation(
            axes[0],
            renderer,
            boundary_full_text,
            target_xy=(exact_boundary_moment, exact_boundary_axial),
            bbox_edgecolor='#7c3aed',
            occupied_bboxes=full_label_boxes,
            avoid_points=[
                (max_moment, demand_axial),
                (line_target_moment, line_target_axial),
                (0.0, 0.0)
            ],
            candidate_specs=[
                {
                    'anchor_xy': (exact_boundary_moment, exact_boundary_axial),
                    'dx': float(offset_candidate['dx']),
                    'dy': float(offset_candidate['dy']),
                    'ha': str(offset_candidate['ha']),
                    'va': str(offset_candidate['va'])
                }
                for offset_candidate in _build_interaction_offset_candidates(scale=1.0)
            ],
            with_arrow=True,
            fontsize=8.0,
            text_color='#7c3aed'
        )

    _add_interaction_annotation(
        axes[0],
        renderer,
        demand_full_text,
        target_xy=(max_moment, demand_axial),
        bbox_edgecolor='#d62828',
        occupied_bboxes=full_label_boxes,
        avoid_points=[
            (line_target_moment, line_target_axial),
            (exact_boundary_moment if exact_boundary_moment is not None else boundary_moment,
             exact_boundary_axial if exact_boundary_axial is not None else boundary_axial),
            (0.0, 0.0)
        ],
        candidate_specs=[
            {
                'anchor_xy': (max_moment, demand_axial),
                'dx': float(offset_candidate['dx']),
                'dy': float(offset_candidate['dy']),
                'ha': str(offset_candidate['ha']),
                'va': str(offset_candidate['va'])
            }
            for offset_candidate in _build_interaction_offset_candidates(scale=0.95)
        ],
        with_arrow=True,
        fontsize=8.0,
        text_color='#d62828'
    )

    _add_interaction_annotation(
        axes[0],
        renderer,
        lambda_full_text,
        target_xy=(0.5 * line_target_moment, 0.5 * line_target_axial),
        bbox_edgecolor='#9ca3af',
        occupied_bboxes=full_label_boxes,
        avoid_points=[
            (max_moment, demand_axial),
            (line_target_moment, line_target_axial),
            (0.0, 0.0)
        ],
        candidate_specs=_build_interaction_line_annotation_candidates(
            line_target_moment,
            line_target_axial,
            scale=0.82
        ),
        with_arrow=False,
        fontsize=8.0
    )

    axes[0].set_title(f'Diagram Interaksi Elemen {elem_id}')
    axes[0].legend(loc='upper right', fontsize=8)

    zoom_x_values = [max_moment, line_target_moment]
    zoom_y_values = [demand_axial, line_target_axial]
    if exact_boundary_moment is not None and exact_boundary_axial is not None:
        zoom_x_values.append(exact_boundary_moment)
        zoom_y_values.append(exact_boundary_axial)

    x_span = max(zoom_x_values) - min(zoom_x_values)
    y_span = max(zoom_y_values) - min(zoom_y_values)
    x_padding = max(10.0, 0.2 * max(x_span, 1.0))
    y_padding = max(40.0, 0.2 * max(y_span, 1.0))
    axes[1].set_xlim(min(zoom_x_values) - x_padding, max(zoom_x_values) + x_padding)
    axes[1].set_ylim(min(zoom_y_values) - y_padding, max(zoom_y_values) + y_padding)
    axes[1].set_title('Zoom Titik Kontrol')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    zoom_label_boxes: List[Any] = []
    if boundary_full_text is not None:
        _add_interaction_annotation(
            axes[1],
            renderer,
            boundary_full_text,
            target_xy=(exact_boundary_moment, exact_boundary_axial),
            bbox_edgecolor='#7c3aed',
            occupied_bboxes=zoom_label_boxes,
            avoid_points=[
                (max_moment, demand_axial),
                (line_target_moment, line_target_axial),
                (0.0, 0.0)
            ],
            candidate_specs=[
                {
                    'anchor_xy': (exact_boundary_moment, exact_boundary_axial),
                    'dx': float(offset_candidate['dx']),
                    'dy': float(offset_candidate['dy']),
                    'ha': str(offset_candidate['ha']),
                    'va': str(offset_candidate['va'])
                }
                for offset_candidate in _build_interaction_offset_candidates(scale=0.88)
            ],
            with_arrow=True,
            fontsize=8.0,
            text_color='#7c3aed'
        )

    _add_interaction_annotation(
        axes[1],
        renderer,
        demand_full_text,
        target_xy=(max_moment, demand_axial),
        bbox_edgecolor='#d62828',
        occupied_bboxes=zoom_label_boxes,
        avoid_points=[
            (line_target_moment, line_target_axial),
            (exact_boundary_moment if exact_boundary_moment is not None else boundary_moment,
             exact_boundary_axial if exact_boundary_axial is not None else boundary_axial),
            (0.0, 0.0)
        ],
        candidate_specs=[
            {
                'anchor_xy': (max_moment, demand_axial),
                'dx': float(offset_candidate['dx']),
                'dy': float(offset_candidate['dy']),
                'ha': str(offset_candidate['ha']),
                'va': str(offset_candidate['va'])
            }
            for offset_candidate in _build_interaction_offset_candidates(scale=0.84)
        ],
        with_arrow=True,
        fontsize=8.0,
        text_color='#d62828'
    )

    lambda_zoom_text = f"lambda = {line_target_lambda:.4f}"
    _add_interaction_annotation(
        axes[1],
        renderer,
        lambda_zoom_text,
        target_xy=(0.5 * line_target_moment, 0.5 * line_target_axial),
        bbox_edgecolor='#9ca3af',
        occupied_bboxes=zoom_label_boxes,
        avoid_points=[
            (max_moment, demand_axial),
            (line_target_moment, line_target_axial),
            (0.0, 0.0)
        ],
        candidate_specs=_build_interaction_line_annotation_candidates(
            line_target_moment,
            line_target_axial,
            scale=0.72
        ),
        with_arrow=False,
        fontsize=8.0
    )

    fig.suptitle(
        (
            f"Kurva Interaksi Aksial-Lentur | Elemen {elem_id} | "
            f"fc'={material_snapshot['fc']:.2f} MPa | "
            f"fy={material_snapshot['fy_tarik']:.2f} MPa | "
            f"g exact={((g_exact if g_exact is not None else g_interp)):.4f}"
        ),
        fontsize=11,
        y=0.98
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.11, top=0.89, wspace=0.18)

    return {
        'figure': fig,
        'lambda_interp': lambda_interp,
        'g_interp': g_interp,
        'lambda_exact': lambda_exact,
        'g_exact': g_exact,
        'c_boundary_exact': c_boundary_exact,
        'demand_axial': demand_axial,
        'demand_moment': max_moment,
        'boundary_axial': boundary_axial,
        'boundary_moment': boundary_moment,
        'boundary_axial_exact': exact_boundary_axial,
        'boundary_moment_exact': exact_boundary_moment,
        'exact_boundary_found': exact_boundary_found,
        'exact_boundary_overlaps_interp': exact_boundary_overlaps_interp,
        'boundary_exact_status': boundary_exact_status,
        'controlling_state': controlling_state or '-',
        'material_snapshot': material_snapshot
    }


def render_progress_percentage(target,
                               progress_fraction: float,
                               current_step: int,
                               total_steps: int) -> None:
    """Tampilkan progress persentase dan angka langkah dengan font lebih tegas."""
    percent_value = int(round(max(0.0, min(1.0, float(progress_fraction))) * 100.0))
    target.markdown(
        (
            "<div style='margin-top:0.35rem; margin-bottom:0.35rem; "
            "padding:0.55rem 0.7rem; border-radius:0.7rem; "
            "background:rgba(255,255,255,0.08); "
            "border:1px solid rgba(255,255,255,0.16);'>"
            "<div style='font-size:1.02rem; font-weight:700; color:#f9fafb; "
            "line-height:1.25;'>"
            f"Progress Analisis: {percent_value}%"
            "</div>"
            "<div style='font-size:0.98rem; font-weight:700; color:#fecaca; "
            "line-height:1.2; margin-top:0.18rem;'>"
            f"Langkah {current_step}/{total_steps}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True
    )


def render_simulation_progress(target,
                               completed_simulations: int,
                               total_simulations: int,
                               analysis_failures: int = 0) -> None:
    """Tampilkan progres Monte Carlo berjalan agar sisa simulasi mudah dipantau."""
    total_value = max(int(total_simulations or 0), 0)
    completed_value = min(max(int(completed_simulations or 0), 0), total_value)
    remaining_value = max(total_value - completed_value, 0)
    completed_percent = (
        (completed_value / total_value) * 100.0 if total_value > 0 else 0.0
    )
    remaining_percent = max(100.0 - completed_percent, 0.0)

    analysis_failures_html = ""
    if int(analysis_failures or 0) > 0:
        analysis_failures_html = (
            "<div style='font-size:0.86rem; color:#fecaca; margin-top:0.2rem;'>"
            f"Gagal dieksekusi: {int(analysis_failures):,} simulasi"
            "</div>"
        )

    target.markdown(
        (
            "<div style='margin-top:0.35rem; margin-bottom:0.35rem; "
            "padding:0.55rem 0.7rem; border-radius:0.7rem; "
            "background:rgba(255,255,255,0.08); "
            "border:1px solid rgba(255,255,255,0.16);'>"
            "<div style='font-size:0.98rem; font-weight:700; color:#f9fafb;'>"
            "Progress Monte Carlo"
            "</div>"
            "<div style='font-size:0.9rem; color:#e5e7eb; margin-top:0.18rem;'>"
            f"Selesai: {completed_value:,}/{total_value:,} simulasi"
            "</div>"
            "<div style='font-size:0.9rem; color:#bbf7d0; margin-top:0.14rem;'>"
            f"Persen selesai: {completed_percent:.2f}%"
            "</div>"
            "<div style='font-size:0.9rem; color:#fde68a; margin-top:0.14rem;'>"
            f"Sisa: {remaining_value:,} simulasi ({remaining_percent:.2f}%)"
            "</div>"
            f"{analysis_failures_html}"
            "</div>"
        ),
        unsafe_allow_html=True
    )


st.set_page_config(
    page_title="Analisis Keandalan Struktur Portal 2D",
    layout="wide"
)

header_logo_col, header_text_col = st.columns([1.1, 8.9], gap="medium")

with header_logo_col:
    if ULM_LOGO_PATH.exists():
        ulm_logo_data_uri = image_file_to_data_uri(ULM_LOGO_PATH)
        st.markdown(
            f"""
            <div style="padding-top: 1cm; padding-left: 1cm;">
                <img
                    src="{ulm_logo_data_uri}"
                    alt="Logo Universitas Lambung Mangkurat"
                    style="width: 115px; display: block;"
                />
            </div>
            """,
            unsafe_allow_html=True,
        )

with header_text_col:
    st.title("Analisis Keandalan Struktur Portal 2D")
    st.markdown(
        """
        <p style="font-size: 2rem; font-weight: 600; margin-bottom: 0.1rem;">
            Pengembang: Ir. Darmansyah Tjitradi, M.T., IPU
        </p>
        <p style="font-size: 2rem; margin-top: 0;">
            Fakultas Teknik Universitas Lambung Mangkurat
        </p>
        """,
        unsafe_allow_html=True,
    )
st.caption("Menampilkan input, output simulasi terakhir, Pf, Beta, deformasi, dan gaya dalam.")

preview_input_data = None
preview_source_label = None
preview_portal_nodes = None
preview_portal_elements = None
preview_error = None

with st.sidebar:
    st.header("Parameter")
    input_path = st.text_input("Path file Excel", value=DEFAULT_INPUT_FILE)
    uploaded_file = st.file_uploader("Atau upload file Excel", type=["xlsx", "xlsm"])
    analysis_mode_label = st.radio(
        "Mode Analisis",
        options=[DETERMINISTIC_MODE_LABEL, PROBABILISTIC_MODE_LABEL],
        index=1
    )
    is_sidebar_probabilistic = analysis_mode_label == PROBABILISTIC_MODE_LABEL
    if is_sidebar_probabilistic:
        num_simulations = st.number_input(
            "Jumlah simulasi Monte Carlo",
            min_value=100,
            max_value=1000000,
            value=10000,
            step=100
        )
    else:
        num_simulations = 1
        st.caption("Mode deterministik memakai nilai deterministic input per elemen tanpa sampling acak.")

    try:
        preview_input_data, preview_source_label = load_preview_input_data(input_path, uploaded_file)
        if preview_input_data is not None:
            estimated_seconds = estimate_analysis_runtime_seconds(
                preview_input_data,
                is_sidebar_probabilistic,
                int(num_simulations)
            )
            estimated_elements = len(
                get_geometry_elements_for_mode(
                    preview_input_data,
                    is_sidebar_probabilistic
                )
            )
            estimated_random_variables = count_probabilistic_random_variables(preview_input_data)
            st.caption(
                "Perkiraan waktu analisis: "
                f"`{format_duration_text(estimated_seconds)}`"
            )
            if is_sidebar_probabilistic:
                st.caption(
                    f"Basis estimasi lokal: {estimated_elements} elemen, "
                    f"{estimated_random_variables} variabel random, "
                    f"N={int(num_simulations):,}."
                )
                st.caption(
                    "Kalibrasi acuan saat ini: 400 detik pada mode probabilistik "
                    "dan 5 detik pada mode deterministik untuk kasus referensi."
                )
            else:
                st.caption(
                    f"Basis estimasi lokal: {estimated_elements} elemen, "
                    "1 kali analisis deterministik."
                )
                st.caption(
                    "Kalibrasi acuan saat ini: 400 detik pada mode probabilistik "
                    "dan 5 detik pada mode deterministik untuk kasus referensi."
                )
        elif input_path or uploaded_file is not None:
            st.caption("Perkiraan waktu akan muncul setelah file input valid terbaca.")
    except Exception as exc:
        preview_error = exc
        st.caption("Perkiraan waktu belum tersedia karena file input belum bisa dibaca.")
        st.caption(f"Detail input: {format_error_message(exc)}")

    run_analysis = st.button("Jalankan Analisis", type="primary", use_container_width=True)
    progress_container = st.container()

if run_analysis:
    resolved_file, temporary_file = prepare_input_file(input_path, uploaded_file)

    if resolved_file is None:
        st.error("File Excel belum ditemukan. Isi path yang valid atau upload file input.")
    else:
        try:
            with st.spinner("Analisis sedang dijalankan..."):
                analysis = run_analysis_dashboard(
                    resolved_file,
                    'probabilistic' if is_sidebar_probabilistic else 'deterministic',
                    int(num_simulations),
                    progress_container=progress_container
                )

            st.session_state['results_bundle'] = analysis.get_results_bundle()
            st.session_state['portal_elements'] = analysis.portal.elements
            st.session_state['portal_nodes'] = analysis.data['geometry']['nodes'].astype(float)
            st.session_state['failure_cloud_data'] = (
                # `0` menandakan semua sampel Monte Carlo disimpan tanpa downsampling,
                # agar total `safe + fail` pada failure surface mengikuti N simulasi penuh.
                build_probabilistic_failure_cloud_data(
                    analysis.mc_results,
                    analysis.random_variables,
                    analysis.data,
                    max_points=0,
                    max_failed_points=0
                )
                if analysis.is_probabilistic else
                {}
            )
            st.session_state['limit_state_physical_cloud_data'] = (
                # `0` menandakan semua sampel valid disimpan tanpa downsampling,
                # sehingga legend `safe/fail` mengikuti N simulasi valid penuh.
                build_probabilistic_limit_state_physical_cloud_data(
                    analysis,
                    max_points_per_state=0,
                    max_failed_points=0
                )
                if analysis.is_probabilistic else
                {}
            )
            st.session_state['axial_moment_pm_cloud_data'] = (
                # Terapkan aturan yang sama untuk contour aksial-lentur di ruang fisik.
                build_probabilistic_axial_moment_pm_cloud_data(
                    analysis,
                    max_points=0,
                    max_failed_points=0
                )
                if analysis.is_probabilistic else
                {}
            )
            st.session_state['analysis_source'] = uploaded_file.name if uploaded_file else resolved_file
            st.success("Analisis selesai. Dashboard di bawah sudah diperbarui.")
        except Exception as exc:
            st.error(f"Analisis gagal dijalankan: {format_error_message(exc)}")
        finally:
            cleanup_temporary_file(temporary_file)

if preview_input_data is None and preview_error is None:
    try:
        preview_input_data, preview_source_label = load_preview_input_data(input_path, uploaded_file)
    except Exception as exc:
        preview_error = exc

if preview_input_data is not None:
    try:
        preview_portal_nodes, preview_portal_elements = build_preview_portal(
            preview_input_data,
            is_sidebar_probabilistic
        )
    except Exception as exc:
        preview_error = exc


results_bundle = st.session_state.get('results_bundle')
portal_elements = st.session_state.get('portal_elements')
portal_nodes = st.session_state.get('portal_nodes')
failure_cloud_data = st.session_state.get('failure_cloud_data', {})
limit_state_physical_cloud_data = st.session_state.get('limit_state_physical_cloud_data', {})
axial_moment_pm_cloud_data = st.session_state.get('axial_moment_pm_cloud_data', {})
analysis_source = st.session_state.get('analysis_source')
selected_analysis_mode = (
    'probabilistic' if is_sidebar_probabilistic else 'deterministic'
)

selected_input_data = preview_input_data
selected_input_source = preview_source_label
selected_is_probabilistic = selected_analysis_mode == 'probabilistic'

if results_bundle:
    summary = results_bundle['summary']
    latest_simulation = results_bundle['latest_simulation']
    latest_result = latest_simulation.get('analysis_result')
    is_probabilistic = summary.get('analysis_mode') == 'probabilistic'
    if selected_input_data is None:
        selected_input_data = results_bundle['input_data']
    if selected_input_source is None:
        selected_input_source = analysis_source

    if not latest_simulation.get('is_last_simulation_valid', True):
        st.warning(
            "Simulasi terakhir tidak menghasilkan output valid. "
            "Dashboard menampilkan simulasi valid terakhir yang tersedia."
        )

    st.subheader("Ringkasan Hasil Analisis")
    if is_probabilistic:
        metric_cols = st.columns(5)
        metric_cols[0].metric("Prob. Kegagalan, Pf", format_metric(summary.get('Pf'), 6))
        metric_cols[1].metric("Indeks Keandalan, Beta", format_metric(summary.get('Beta'), 4))
        metric_cols[2].metric("Jumlah Kegagalan", str(summary.get('failures', 0)))
        metric_cols[3].metric("Kelas Keandalan", summary.get('safety_class') or "-")
        metric_cols[4].metric(
            "Status Keamanan",
            "-" if summary.get('is_safe') is None else ("AMAN" if summary.get('is_safe') else "TIDAK AMAN")
        )
        limit_cols = st.columns(4)
        limit_cols[0].metric("g Minimum Lentur (kN.m)", format_metric(summary.get('min_g_moment'), 4))
        limit_cols[1].metric("g Minimum Geser (kN)", format_metric(summary.get('min_g_shear'), 4))
        limit_cols[2].metric("g Minimum Aksial (kN)", format_metric(summary.get('min_g_axial'), 4))
        limit_cols[3].metric("g Minimum Aksial-Lentur (-)", format_metric(summary.get('min_g_axial_moment'), 4))
        st.caption(
            f"Sumber berkas input: `{analysis_source or DEFAULT_INPUT_FILE}` | "
            f"Mode analisis: {summary.get('analysis_mode_label', '-')} | "
            f"Simulasi yang ditampilkan: ke-{(latest_simulation.get('display_index') or 0) + 1}"
        )
        if summary.get('analysis_failures', 0):
            st.warning(
                f"{summary.get('analysis_failures', 0)} simulasi tidak berhasil dieksekusi. "
                "Kegagalan eksekusi tersebut tetap diperhitungkan secara konservatif sebagai kejadian gagal."
            )
    else:
        metric_cols = st.columns(8)
        metric_cols[0].metric("Mode Analisis", summary.get('analysis_mode_label') or "-")
        metric_cols[1].metric("Prob. Kegagalan, Pf", "-")
        metric_cols[2].metric("Indeks Keandalan, Beta", "-")
        metric_cols[3].metric("g Minimum Lentur (kN.m)", format_metric(summary.get('min_g_moment'), 4))
        metric_cols[4].metric("g Minimum Geser (kN)", format_metric(summary.get('min_g_shear'), 4))
        metric_cols[5].metric("g Minimum Aksial (kN)", format_metric(summary.get('min_g_axial'), 4))
        metric_cols[6].metric("g Minimum Aksial-Lentur (-)", format_metric(summary.get('min_g_axial_moment'), 4))
        metric_cols[7].metric(
            "Status Keamanan",
            "-" if summary.get('is_safe') is None else ("AMAN" if summary.get('is_safe') else "TIDAK AMAN")
        )
        st.caption(
            f"Sumber berkas input: `{analysis_source or DEFAULT_INPUT_FILE}` | "
            f"Mode analisis: {summary.get('analysis_mode_label', '-')} | "
            "Hasil yang ditampilkan berasal dari satu kali analisis dengan parameter deterministik pada setiap elemen."
        )
else:
    latest_simulation = {}
    latest_result = None
    is_probabilistic = selected_is_probabilistic

    if selected_input_data is None:
        if preview_error is not None:
            st.error(
                "Input belum bisa dipreview karena file gagal dibaca: "
                f"{format_error_message(preview_error)}"
            )
        else:
            st.info("Isi path file Excel yang valid atau upload file input untuk menampilkan preview.")
        st.stop()

    st.info(
        "Preview input sudah ditampilkan sebelum analisis. "
        "Tekan `Jalankan Analisis` untuk menghitung output struktur."
    )

input_data = selected_input_data
input_preview_differs_from_results = bool(
    results_bundle
    and (
        analysis_source != selected_input_source
        or summary.get('analysis_mode') != selected_analysis_mode
    )
)

if input_preview_differs_from_results:
    st.warning(
        "Preview input sudah mengikuti file/mode yang dipilih di sidebar, "
        "tetapi ringkasan, output, dan plot masih berasal dari analisis terakhir. "
        "Tekan `Jalankan Analisis` untuk memperbarui hasil perhitungan."
    )

analysis_input_data = (
    results_bundle.get('input_data')
    if results_bundle and results_bundle.get('input_data') is not None else
    input_data
)

dashboard_tabs = [
    "Input Data",
    "Output Analisis Struktur",
    "Plot Simulasi Terakhir",
    "Output Reliability",
    "Kurva Interasi P-M",
    "Output Sensitivitas Deterministik",
    "Output Sensitivitas Probabilistik",
    "Histogram",
    "Simulasi MC",
    "Failure Cloud Fisik",
    "Failure Cloud U-Space",
    "Risk Map",
    "Laporan"
]
active_dashboard_tab = st.radio(
    "Navigasi Dashboard",
    options=dashboard_tabs,
    index=0,
    horizontal=False,
    key="active_dashboard_tab",
    label_visibility="collapsed"
)

if active_dashboard_tab == "Input Data":
    input_plot_nodes = preview_portal_nodes if preview_portal_nodes is not None else portal_nodes
    input_plot_elements = preview_portal_elements if preview_portal_elements is not None else portal_elements

    if input_plot_nodes is None or input_plot_elements is None:
        input_plot_nodes, input_plot_elements = build_preview_portal(
            input_data,
            selected_is_probabilistic
        )

    preview_distributed_loads = build_preview_distributed_loads(input_data, selected_is_probabilistic)

    if not (results_bundle and not input_preview_differs_from_results):
        mode_caption = "mean" if selected_is_probabilistic else "deterministic"
        st.caption(
            "Preview gambar input ditampilkan sebelum analisis. "
            f"Beban merata pada gambar memakai nilai `{mode_caption}` sesuai mode yang dipilih."
        )

    st.markdown("#### Preview Geometri dan Pembebanan")
    geometry_fig, _ = PortalPlotter.plot_portal_geometry(
        input_plot_nodes,
        input_plot_elements,
        boundary_conditions=input_data['boundary'],
        distributed_loads=preview_distributed_loads,
        nodal_loads=input_data['nodal_loads'],
        title="Preview Input Struktur"
    )
    render_plot(
        geometry_fig,
        download_basename="input-data-preview-geometri-dan-pembebanan"
    )

    if results_bundle and not input_preview_differs_from_results:
        sample_expander_title = (
            "Input Acak Simulasi Terakhir"
            if is_probabilistic else
            "Input Acuan Deterministik"
        )
        with st.expander(sample_expander_title, expanded=True):
            latest_sample_df = build_latest_sample_df(latest_simulation)
            if is_probabilistic:
                if latest_sample_df.empty:
                    st.info("Tidak ada sampel acak yang perlu ditampilkan.")
                else:
                    render_input_table(latest_sample_df)
            else:
                sample_col, mode_col = st.columns([1.1, 1.9])
                with sample_col:
                    if latest_sample_df.empty:
                        st.info("Tidak ada sampel acuan yang perlu ditampilkan.")
                    else:
                        render_input_table(latest_sample_df)
                with mode_col:
                    st.markdown("#### Keterangan Mode")
                    st.info(
                        "Mode deterministik tidak melakukan sampling Monte Carlo. "
                        "Analisis dijalankan satu kali dengan nilai deterministic tiap elemen."
                    )

        if is_probabilistic:
            with st.expander("Definisi Variabel Random", expanded=False):
                render_input_table(build_random_variable_df(results_bundle['random_variables']))

        if is_probabilistic:
            effective_modulus_df = build_effective_modulus_snapshot_df(
                input_data,
                latest_simulation,
                is_probabilistic
            )
            if not effective_modulus_df.empty:
                with st.expander("Snapshot E Dipakai DSM", expanded=False):
                    st.caption(
                        "`E_dipakai_DSM (MPa)` adalah nilai yang benar-benar dipakai solver DSM "
                        "untuk simulasi yang sedang ditampilkan. "
                        "`E_acuan_mean (MPa)` hanya nilai acuan hasil `E_mean x fb_mean`."
                    )
                    render_input_table(effective_modulus_df)

    with st.expander("Geometri Portal", expanded=True):
        geom_col_1, geom_col_2 = st.columns(2)
        with geom_col_1:
            render_input_table(build_nodes_df(input_data))
        with geom_col_2:
            element_df = build_elements_df(input_data, selected_is_probabilistic)
            render_input_table(element_df)
        if 'E_acuan_mean (MPa)' in element_df.columns:
            st.caption(
                "Kolom `E_acuan_mean (MPa)` menunjukkan hasil `E_mean x fb_mean` "
                "per elemen sebagai nilai acuan mean. Ini bukan selalu nilai yang dipakai DSM "
                "pada tiap simulasi."
            )
        if (
            'E_mean (MPa)' not in element_df.columns
            and 'E_deterministic (MPa)' not in element_df.columns
        ):
            st.caption(f"E mean: {format_metric_comma(input_data['geometry']['E_mean'], 2)} MPa")

    with st.expander("Boundary Condition dan Beban Nodal", expanded=False):
        bc_col, load_col = st.columns(2)
        with bc_col:
            render_input_table(build_boundary_df(input_data['boundary']))
        with load_col:
            nodal_load_df = build_nodal_load_df(input_data['nodal_loads'])
            if nodal_load_df.empty:
                st.info("Tidak ada beban nodal pada input.")
            else:
                render_input_table(nodal_load_df)

    with st.expander("Mutu Beton, Mutu Baja, dan Tulangan", expanded=False):
        mat_col_1, mat_col_2 = st.columns(2)
        with mat_col_1:
            concrete_df = build_concrete_input_df(input_data)
            if concrete_df.empty:
                st.info("Data mutu beton tidak tersedia.")
            else:
                render_input_table(concrete_df)
        with mat_col_2:
            steel_df = build_steel_input_df(input_data)
            if steel_df.empty:
                st.info("Data mutu baja tidak tersedia.")
            else:
                render_input_table(steel_df)

        reinforcement_df = build_reinforcement_input_df(input_data)
        if reinforcement_df.empty:
            st.info("Data tulangan tidak tersedia.")
        else:
            render_input_table(reinforcement_df)

    with st.expander("Beban Merata", expanded=False):
        load_col_1, load_col_2 = st.columns(2)
        with load_col_1:
            dead_load_df = build_distributed_load_input_df(input_data.get('dead_load', {}))
            if dead_load_df.empty:
                st.info("Data beban mati merata tidak tersedia.")
            else:
                render_input_table(dead_load_df)
        with load_col_2:
            live_load_df = build_distributed_load_input_df(input_data.get('live_load', {}))
            if live_load_df.empty:
                st.info("Data beban hidup merata tidak tersedia.")
            else:
                render_input_table(live_load_df)

elif active_dashboard_tab == "Output Analisis Struktur":
    if latest_result is None:
        st.info("Output akan tersedia setelah analisis dijalankan.")
    else:
        st.caption(
            "Maksimum gaya dalam dihitung dari distribusi sepanjang batang, "
            "termasuk kontribusi beban merata pada elemen."
        )
        out_col_1, out_col_2 = st.columns(2)
        with out_col_1:
            st.markdown("#### Deformasi Nodal")
            displacement_df = build_displacement_df(portal_nodes, latest_result['displacements'])
            st.caption(
                "Sel merah muda menandai nilai maksimum absolut per komponen deformasi. "
                "Jika ada nilai maksimum yang sama, semuanya ikut ditandai."
            )
            render_input_table(
                displacement_df,
                styler=style_max_abs_dataframe(
                    displacement_df,
                    highlight_columns=['Ux (mm)', 'Uy (mm)', 'Rz (rad)'],
                    identity_columns=['Node_ID (-)']
                )
            )

        with out_col_2:
            st.markdown("#### Reaksi Tumpuan")
            reaction_df = build_reaction_df(portal_nodes, input_data['boundary'], latest_result['reactions'])
            st.caption(
                "Sel merah muda menandai nilai maksimum absolut per komponen reaksi. "
                "Jika ada nilai maksimum yang sama, semuanya ikut ditandai."
            )
            render_input_table(
                reaction_df,
                styler=style_max_abs_dataframe(
                    reaction_df,
                    highlight_columns=['Rx (kN)', 'Ry (kN)', 'Mz (kN.m)'],
                    identity_columns=['Node_ID (-)']
                )
            )

        st.markdown("#### Gaya Dalam Elemen")
        st.caption(
            "`*_End_Joint` adalah gaya/momen ujung elemen pada node untuk cek "
            "keseimbangan joint. `*_End_Internal` adalah gaya/momen internal "
            "tepat sebelum ujung elemen."
        )
        st.caption(
            "Kolom `Kode` memakai `B=Balok` dan `K=Kolom`. Sel oranye menandai "
            "nilai maksimum absolut grup Balok dan sel biru menandai nilai maksimum "
            "absolut grup Kolom per kolom, termasuk identitas elemen yang mengontrol nilai tersebut. "
            "Jika ada nilai maksimum yang sama, semuanya ikut ditandai."
        )
        st.caption(
            "Setiap tabel mempertahankan kolom identitas `Element_ID`, `Kode`, dan "
            "`Jenis_Elemen`, lalu dipisahkan menjadi komponen momen lentur, geser, dan aksial."
        )
        if SPECIAL_BEAM_JOINT_RAW_SIGN_NODE_IDS:
            special_nodes_text = ", ".join(
                str(node_id) for node_id in sorted(SPECIAL_BEAM_JOINT_RAW_SIGN_NODE_IDS)
            )
            st.caption(
                f"Khusus node {special_nodes_text}, momen joint balok di tabel "
                "ditampilkan mengikuti tanda aksi joint solver."
            )
        st.caption(
            "Warna header dan highlight maksimum absolut per grup `B/K` tetap dipertahankan "
            "agar pembacaan tabel lebih cepat."
        )
        internal_force_df = build_internal_force_df(latest_result, input_data=input_data)
        internal_force_tables = [
            ("Tabel Gaya Momen Lentur", 'moment'),
            ("Tabel Gaya Geser", 'shear'),
            ("Tabel Gaya Aksial", 'axial')
        ]
        for table_title, component_key in internal_force_tables:
            st.markdown(f"##### {table_title}")
            component_df = build_internal_force_component_df(
                internal_force_df,
                component_key
            )
            if component_df.empty:
                st.info(f"Data {table_title.lower()} belum tersedia.")
                continue
            render_input_table(
                component_df,
                styler=style_internal_force_df(component_df)
            )
        with st.expander("Panduan Visual Tanda Gaya Dalam", expanded=False):
            st.caption(
                "Panduan ini membantu membaca arti tanda positif dan negatif pada tabel gaya dalam, "
                "terutama perbedaan momen balok `B` dan kolom `K`."
            )
            render_plot(
                build_internal_force_sign_guide_figure(),
                interactive=False
            )
            render_input_table(build_internal_force_sign_guide_df())

        with st.expander("Cek Keseimbangan Momen per Node", expanded=False):
            st.caption(
                "Tabel ini menjumlahkan momen joint dari tabel gaya dalam elemen pada node yang sama. "
                "Kontribusi momen joint dihitung otomatis mengikuti konvensi tanda tampilan aktif pada tiap node."
            )
            st.caption(
                "Residual dihitung dengan rumus "
                "`Sigma_M_Elemen_Cek + Mz_Beban_Nodal - Mz_Reaksi`, "
                f"dan dinyatakan `OK` bila |residual| <= {MOMENT_EQUILIBRIUM_TOLERANCE_KNM:.1e} kN.m."
            )
            joint_moment_equilibrium_df = build_joint_moment_equilibrium_df(
                latest_result,
                input_data=analysis_input_data
            )
            render_input_table(
                joint_moment_equilibrium_df,
                styler=style_joint_moment_equilibrium_df(joint_moment_equilibrium_df)
            )

elif active_dashboard_tab == "Output Reliability":
    if latest_result is None:
        st.info("Tabel output akan tersedia setelah analisis dijalankan.")
    else:
        if is_probabilistic:
            st.markdown("#### Reliabilitas Sistem Portal Gabungan")
            st.caption(
                "Portal dianggap tersusun seri antara subsistem Balok dan subsistem Kolom, "
                "sehingga portal gagal bila salah satu subsistem gagal."
            )
            st.caption(
                "Kasus 1: Balok = sistem paralel, Kolom = sistem seri. "
                "Kasus 2: Balok = sistem seri, Kolom = sistem seri."
            )
            st.caption(
                "`Pf/Beta` dihitung langsung dari seluruh simulasi Monte Carlo berdasarkan "
                "kejadian gagal subsistem per simulasi."
            )
            portal_system_df = build_portal_system_reliability_df(
                results_bundle.get('portal_system_reliability', [])
            )
            if portal_system_df.empty:
                st.info("Data reliabilitas sistem portal gabungan belum tersedia.")
            else:
                render_input_table(
                    portal_system_df,
                    styler=style_portal_system_reliability_df(portal_system_df)
                )

        st.markdown(
            "#### Nilai g per Elemen"
            if is_probabilistic else
            "#### Hasil Analisis Deterministik"
        )
        st.caption(
            "Tabel dipisah per limit state: lentur, geser, aksial, dan aksial-lentur. "
            "Kolom `R`, `S`, `phi`, dan `g(x)` ditampilkan terpisah agar perhitungan tiap cek "
            "lebih mudah dibaca."
        )
        st.caption(
            "Perhitungan `R` pada tabel ini mengikuti mode analisis aktif: "
            "`phi = 1` pada mode probabilistik, sedangkan pada mode deterministik "
            "`phi` dihitung sesuai SNI 2847:2019 untuk masing-masing limit state."
        )
        if is_probabilistic:
            st.caption(
                "`Pf/Beta` per elemen dihitung langsung dari seluruh simulasi Monte Carlo, "
                "bukan dari rata-rata respons struktur."
            )
        else:
            st.caption(
                "Pada mode deterministik, tabel hanya menampilkan field yang relevan "
                "dengan satu kali analisis, termasuk `SF = R/S`, dan status ditentukan "
                "dari tanda `g(x)`. Nilai `R` sudah memakai faktor reduksi kekuatan "
                "sesuai SNI 2847:2019 untuk lentur, geser, aksial, dan aksial-lentur."
            )
        st.caption(
            "Untuk cek aksial-lentur, `R` merepresentasikan `lambda_boundary`, "
            "sedangkan `S` ditetapkan 1.0 sehingga `g(x) = lambda - 1.0`."
        )
        limit_state_tables = build_limit_state_performance_tables(
            latest_result,
            input_data=input_data,
            element_reliability=results_bundle.get('element_reliability', {}),
            is_probabilistic=is_probabilistic
        )
        resume_df = build_limit_state_resume_df(
            limit_state_tables,
            is_probabilistic=is_probabilistic
        )
        st.markdown("##### Resume Pengontrol")
        if is_probabilistic:
            st.caption(
                "Limit state pengontrol per elemen dipilih dengan kriteria `Pf` maksimum, "
                "lalu `Beta` minimum. Jika `Pf/Beta` tidak tersedia, fallback memakai `g(x)` minimum."
            )
        else:
            st.caption(
                "Pada mode deterministik, limit state pengontrol dipilih dari `g(x)` minimum. "
                "Nilai `SF = R/S` ikut ditampilkan; untuk cek aksial-lentur, nilainya sama "
                "dengan `lambda` karena `S = 1.0`."
            )
        if resume_df.empty:
            st.info("Data resume limit state pengontrol belum tersedia.")
        else:
            render_input_table(
                resume_df,
                styler=style_limit_state_resume_df(
                    resume_df,
                    is_probabilistic=is_probabilistic
                )
            )
        limit_state_specs = [
            ("Lentur", 'lentur'),
            ("Geser", 'geser'),
            ("Aksial", 'aksial'),
            ("Aksial-Lentur", 'aksial_lentur')
        ]
        for title, key in limit_state_specs:
            st.markdown(f"##### {title}")
            table_df = limit_state_tables.get(key, pd.DataFrame())
            if table_df.empty:
                st.info(f"Data {title.lower()} belum tersedia.")
                continue
            render_input_table(
                table_df,
                styler=style_limit_state_performance_df(
                    table_df,
                    is_probabilistic=is_probabilistic
                )
            )
        if is_probabilistic:
            render_output_reliability_beta_sketch_section(
                results_bundle=results_bundle or {},
                input_data=input_data,
                heading_level="####"
            )

elif active_dashboard_tab == "Output Sensitivitas Probabilistik":
    if not results_bundle:
        st.info("Output sensitivitas akan tersedia setelah analisis dijalankan.")
    else:
        st.markdown("#### Keluaran Analisis Sensitivitas")
        st.caption(
            "Tab ini menyajikan kontribusi relatif variabel acak terhadap kejadian gagal "
            "berdasarkan hasil simulasi Monte Carlo."
        )
        render_sensitivity_output_section(
            results_bundle,
            is_probabilistic,
            heading_level="####"
        )

elif active_dashboard_tab == "Histogram":
    if not results_bundle:
        st.info("Histogram variabel acak akan tersedia setelah analisis dijalankan.")
    elif not is_probabilistic:
        st.info(
            "Tab `Histogram` khusus untuk mode probabilistik. "
            "Jalankan analisis probabilistik agar distribusi sampel Monte Carlo bisa ditampilkan."
        )
    else:
        render_probabilistic_histogram_output_section(
            results_bundle,
            heading_level="####"
        )

elif active_dashboard_tab == "Simulasi MC":
    if not results_bundle:
        st.info("Grafik konvergensi Monte Carlo akan tersedia setelah analisis dijalankan.")
    elif not is_probabilistic:
        st.info(
            "Tab `Simulasi MC` khusus untuk mode probabilistik. "
            "Jalankan analisis probabilistik agar konvergensi Monte Carlo per elemen dapat ditampilkan."
        )
    else:
        render_probabilistic_mc_convergence_output_section(
            results_bundle,
            input_data=analysis_input_data,
            heading_level="####"
        )

elif active_dashboard_tab == "Failure Cloud Fisik":
    if not results_bundle:
        st.info("Failure cloud ruang fisik akan tersedia setelah analisis dijalankan.")
    elif not is_probabilistic:
        st.info(
            "Tab `Failure Cloud Fisik` khusus untuk mode probabilistik. "
            "Jalankan analisis probabilistik agar sebaran Monte Carlo di ruang fisik dapat ditampilkan."
        )
    else:
        render_probabilistic_limit_state_physical_failure_cloud_section(
            physical_cloud_data=limit_state_physical_cloud_data or {},
            axial_moment_pm_cloud_data=axial_moment_pm_cloud_data or {},
            results_bundle=results_bundle,
            input_data=analysis_input_data,
            heading_level="####"
        )

elif active_dashboard_tab == "Failure Cloud U-Space":
    if not results_bundle:
        st.info("Failure cloud ruang normal baku akan tersedia setelah analisis dijalankan.")
    elif not is_probabilistic:
        st.info(
            "Tab `Failure Cloud U-Space` khusus untuk mode probabilistik. "
            "Jalankan analisis probabilistik agar sebaran safe/failed Monte Carlo pada ruang normal baku dapat ditampilkan."
        )
    else:
        render_probabilistic_failure_cloud_output_section(
            failure_cloud_data=failure_cloud_data or {},
            results_bundle=results_bundle,
            heading_level="####"
        )

elif active_dashboard_tab == "Output Sensitivitas Deterministik":
    if not results_bundle:
        st.info("Output sensitivitas deterministik akan tersedia setelah analisis dijalankan.")
    elif is_probabilistic:
        st.info(
            "Tab ini khusus untuk mode deterministik. Pada mode probabilistik, gunakan "
            "tab `Output Sensitivitas` untuk melihat sensitivitas berbasis Monte Carlo."
        )
    else:
        st.markdown("#### Keluaran Sensitivitas Deterministik")
        st.caption(
            "Tab ini menyajikan analisis sensitivitas lokal `one-at-a-time` berbasis "
            "`COV (Coefficient of Variation)` terhadap hasil deterministik baseline."
        )
        render_deterministic_sensitivity_output_section(
            results_bundle,
            heading_level="####"
        )

elif active_dashboard_tab == "Risk Map":
    if latest_result is None:
        st.info("Risk map akan tersedia setelah analisis dijalankan.")
    else:
        risk_plot_nodes = portal_nodes
        risk_plot_elements = portal_elements
        if risk_plot_nodes is None or risk_plot_elements is None:
            risk_plot_nodes, risk_plot_elements = build_preview_portal(
                analysis_input_data,
                is_probabilistic
            )

        render_risk_map_output_section(
            results_bundle=results_bundle or {},
            latest_result=latest_result,
            input_data=analysis_input_data,
            nodes=risk_plot_nodes,
            elements=risk_plot_elements,
            is_probabilistic=is_probabilistic,
            heading_level="####"
        )

elif active_dashboard_tab == "Plot Simulasi Terakhir":
    if latest_result is None:
        st.info("Plot hasil analisis akan tersedia setelah analisis dijalankan.")
    else:
        auto_scale = PortalPlotter.suggest_deformation_scale(
            portal_nodes,
            portal_elements,
            latest_result['displacements']
        )
        max_displacement = PortalPlotter.get_max_translational_displacement(
            latest_result['displacements']
        )
        persisted_scale_multiplier = float(
            st.session_state.get('plot_scale_multiplier', 1.0)
        )
        scale_multiplier = st.slider(
            "Pengali skala otomatis",
            min_value=0.1,
            max_value=5.0,
            value=persisted_scale_multiplier,
            step=0.1,
            key="plot_scale_multiplier_widget"
        )
        st.session_state['plot_scale_multiplier'] = scale_multiplier
        scale_factor = auto_scale * scale_multiplier
        persisted_show_result_labels = bool(
            st.session_state.get('plot_show_result_labels', True)
        )
        show_result_labels = st.checkbox(
            "Tampilkan label nilai hasil pada gambar",
            value=persisted_show_result_labels,
            key="plot_show_result_labels_widget"
        )
        st.session_state['plot_show_result_labels'] = show_result_labels
        st.caption(
            "Empat gambar simulasi terakhir ditampilkan langsung di bawah. "
            f"Perpindahan maksimum = {max_displacement:.6f} mm, "
            f"skala plot aktual = {scale_factor:,.0f}x. "
            "Gunakan scroll mouse atau pinch untuk memperbesar gambar."
        )

        top_left, top_right = st.columns(2)
        bottom_left, bottom_right = st.columns(2)

        with top_left:
            st.markdown("#### Deformasi")
            deformed_fig, _ = PortalPlotter.plot_deformed_shape(
                portal_nodes,
                portal_elements,
                latest_result['displacements'],
                scale_factor=scale_factor,
                show_result_labels=show_result_labels
            )
            render_plot(
                deformed_fig,
                interactive=True,
                viewer_key="last-simulation-deformation",
                alt_text="Plot deformasi simulasi terakhir",
                download_basename="plot-simulasi-terakhir-deformasi"
            )

        with top_right:
            st.markdown("#### Diagram Axial")
            axial_fig, _ = PortalPlotter.plot_internal_force_diagram(
                portal_elements,
                latest_result['element_forces'],
                force_type='axial',
                show_result_labels=show_result_labels
            )
            render_plot(
                axial_fig,
                interactive=True,
                viewer_key="last-simulation-axial",
                alt_text="Diagram axial simulasi terakhir",
                download_basename="plot-simulasi-terakhir-axial"
            )

        with bottom_left:
            st.markdown("#### Diagram Shear")
            shear_fig, _ = PortalPlotter.plot_internal_force_diagram(
                portal_elements,
                latest_result['element_forces'],
                force_type='shear',
                show_result_labels=show_result_labels
            )
            render_plot(
                shear_fig,
                interactive=True,
                viewer_key="last-simulation-shear",
                alt_text="Diagram shear simulasi terakhir",
                download_basename="plot-simulasi-terakhir-shear"
            )

        with bottom_right:
            st.markdown("#### Diagram Momen")
            moment_fig, _ = PortalPlotter.plot_internal_force_diagram(
                portal_elements,
                latest_result['element_forces'],
                force_type='moment',
                relative_to_chord=False,
                show_result_labels=show_result_labels
            )
            render_plot(
                moment_fig,
                interactive=True,
                viewer_key="last-simulation-moment",
                alt_text="Diagram momen simulasi terakhir",
                download_basename="plot-simulasi-terakhir-momen"
            )

elif active_dashboard_tab == "Kurva Interasi P-M":
    if latest_result is None:
        st.info("Kurva interaksi akan tersedia setelah analisis dijalankan.")
    else:
        interaction_element_ids = collect_interaction_element_ids(
            analysis_input_data,
            latest_result
        )
        interaction_element_ids = [
            elem_id
            for elem_id in interaction_element_ids
            if get_element_code_from_input(analysis_input_data, elem_id) == 'K'
        ]

        if not interaction_element_ids:
            st.info("Tidak ada elemen kolom dengan data aksial-lentur yang bisa diplot.")
        else:
            axial_moment_values = latest_result.get('performance_axial_moment', {}) or {}

            def get_axial_moment_g_value(elem_id: int) -> float:
                raw_value = get_by_element_value(axial_moment_values, elem_id, None)
                try:
                    numeric = float(raw_value)
                except (TypeError, ValueError):
                    return float('inf')
                if np.isnan(numeric):
                    return float('inf')
                return numeric

            default_elem_id = min(
                interaction_element_ids,
                key=lambda elem_id: (
                    get_axial_moment_g_value(elem_id),
                    int(elem_id)
                )
            )

            persisted_interaction_elem = int(
                st.session_state.get('selected_interaction_elem', default_elem_id)
            )
            if persisted_interaction_elem not in interaction_element_ids:
                persisted_interaction_elem = int(default_elem_id)

            selected_interaction_elem = st.selectbox(
                "Pilih elemen kolom",
                options=interaction_element_ids,
                index=interaction_element_ids.index(persisted_interaction_elem),
                format_func=lambda elem_id: (
                    f"E{int(elem_id)} | "
                    f"g={format_metric(get_by_element_value(axial_moment_values, elem_id), 4)}"
                ),
                key="selected_interaction_elem_widget"
            )
            st.session_state['selected_interaction_elem'] = int(selected_interaction_elem)
            st.caption(
                "Kurva dibentuk dari snapshot material simulasi yang sedang ditampilkan. "
                "Panel kiri menampilkan kurva penuh, panel kanan fokus pada titik kontrol. "
                "Titik `Demand`, `Boundary interp`, `Boundary exact (c)`, dan `Garis lambda` "
                "diberi label langsung pada gambar. "
                "Marker `Boundary interp` ditampilkan sebagai diamond toska, sedangkan "
                "`Boundary exact (c)` ditampilkan sebagai bintang ungu berhalo dengan "
                "titik pusat tambahan agar tetap terlihat meskipun menempel pada kurva. "
                "Pada mode probabilistik kurva memakai `phi = 1`, sedangkan pada mode "
                "deterministik kurva memakai `phi` sesuai SNI 2847:2019. "
                "Gunakan scroll mouse atau pinch untuk memperbesar gambar."
            )

            try:
                interaction_plot = build_interaction_diagram_figure(
                    analysis_input_data,
                    latest_simulation,
                    latest_result,
                    is_probabilistic,
                    selected_interaction_elem
                )
                demand_cols = st.columns(6)
                demand_cols[0].metric(
                    "Demand Aksial (kN)",
                    format_metric(interaction_plot['demand_axial'], 3)
                )
                demand_cols[1].metric(
                    "Demand Momen (kN.m)",
                    format_metric(interaction_plot['demand_moment'], 3)
                )
                demand_cols[2].metric(
                    "Boundary Interp P (kN)",
                    format_metric(interaction_plot['boundary_axial'], 3)
                )
                demand_cols[3].metric(
                    "Boundary Interp M (kN.m)",
                    format_metric(interaction_plot['boundary_moment'], 3)
                )
                demand_cols[4].metric(
                    "Boundary Exact P (kN)",
                    format_metric(interaction_plot['boundary_axial_exact'], 3)
                )
                demand_cols[5].metric(
                    "Boundary Exact M (kN.m)",
                    format_metric(interaction_plot['boundary_moment_exact'], 3)
                )

                metric_cols = st.columns(6)
                metric_cols[0].metric(
                    "Lambda Interp (-)",
                    format_metric(interaction_plot['lambda_interp'], 4)
                )
                metric_cols[1].metric(
                    "Lambda Exact (-)",
                    format_metric(interaction_plot['lambda_exact'], 4)
                )
                metric_cols[2].metric(
                    "c Exact (mm)",
                    format_metric(interaction_plot['c_boundary_exact'], 3)
                )
                metric_cols[3].metric(
                    "g Exact (-)",
                    format_metric(interaction_plot['g_exact'], 4)
                )
                metric_cols[4].metric(
                    "Status Exact",
                    str(interaction_plot.get('boundary_exact_status') or '-')
                )
                metric_cols[5].metric(
                    "Kontrol",
                    str(interaction_plot['controlling_state']).replace('-', ' ').title()
                )
                if not bool(interaction_plot.get('exact_boundary_found')):
                    st.warning(
                        "Boundary exact kontinu belum ditemukan untuk kondisi ini. "
                        "Plot tetap menampilkan `Boundary interp` pada kurva dan "
                        "`Garis lambda` menggunakan titik fallback tersebut."
                    )
                elif bool(interaction_plot.get('exact_boundary_overlaps_interp')):
                    st.info(
                        "Boundary exact ditemukan, tetapi posisinya hampir berimpit dengan "
                        "boundary interp dan kurva interaksi. Pada plot, `exact` ditandai "
                        "sebagai bintang ungu berhalo, sedangkan `interp` sebagai diamond toska."
                    )
                render_plot(
                    interaction_plot['figure'],
                    interactive=True,
                    viewer_key=f"interaction-curve-e{int(selected_interaction_elem)}",
                    alt_text=f"Kurva interaksi elemen {int(selected_interaction_elem)}",
                    viewer_height=620,
                    tight_bbox=False,
                    download_basename=f"kurva-interaksi-pm-e{int(selected_interaction_elem)}"
                )
            except Exception as exc:
                st.error(
                    "Kurva interaksi tidak bisa ditampilkan: "
                    f"{format_error_message(exc)}"
                )

elif active_dashboard_tab == "Laporan":
    if not results_bundle:
        st.info("Laporan hasil analisis akan tersedia setelah analisis dijalankan.")
    else:
        st.markdown("#### Ringkasan Laporan Analisis")
        st.text(results_bundle['report'])

        if latest_result is not None:
            report_risk_nodes = portal_nodes
            report_risk_elements = portal_elements
            if report_risk_nodes is None or report_risk_elements is None:
                report_risk_nodes, report_risk_elements = build_preview_portal(
                    analysis_input_data,
                    is_probabilistic
                )

            st.markdown("#### Risk Map")
            render_risk_map_output_section(
                results_bundle=results_bundle or {},
                latest_result=latest_result,
                input_data=analysis_input_data,
                nodes=report_risk_nodes,
                elements=report_risk_elements,
                is_probabilistic=is_probabilistic,
                heading_level="#####"
            )

        if is_probabilistic:
            render_sensitivity_output_section(
                results_bundle,
                is_probabilistic,
                heading_level="####"
            )
        else:
            render_deterministic_sensitivity_output_section(
                results_bundle,
                heading_level="####"
            )
