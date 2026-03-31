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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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
MOMENT_EQUILIBRIUM_TOLERANCE_KNM = 1e-6

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


def style_input_dataframe(df: pd.DataFrame):
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

    table_styles = [
        {
            'selector': 'table',
            'props': [
                ('border-collapse', 'collapse'),
                ('width', '100%'),
                ('font-size', '0.92rem')
            ]
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
        f'<div style="overflow-x:auto;">{table_html}</div>',
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
            'w_local_y (kN/m)': force.get('transverse_local_load'),
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
            'load': ['w_local_y (kN/m)'],
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
                'Beta_Shear (-)'
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


def build_limit_state_performance_tables(latest_result: Dict,
                                         input_data: Optional[Dict] = None,
                                         element_reliability: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
    """Pisahkan tabel nilai g per elemen per limit state."""
    max_forces_values = latest_result.get('max_forces', {})
    moment_values = latest_result.get('performance', {})
    shear_values = latest_result.get('performance_shear', {})
    axial_values = latest_result.get('performance_axial', {})
    axial_moment_values = latest_result.get('performance_axial_moment', {})
    moment_meta_values = latest_result.get('performance_metadata', {})
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
        moment_rows.append({
            'Elemen (-)': int(elem_id),
            'Kode': get_element_code(elem_id),
            'Kapasitas R (kN.m)': capacity,
            'S dari Analisis Struktur (kN.m)': demand,
            'g(x) (kN.m)': g_value,
            'Jumlah Gagal (-)': reliability.get('failures'),
            'Pf (-)': reliability.get('Pf'),
            'Beta (-)': reliability.get('Beta'),
            'Status': get_status(g_value)
        })

    shear_rows = []
    for elem_id in collect_element_ids(
        max_forces_values,
        shear_values,
        element_reliability.get('shear', {})
    ):
        max_forces_entry = get_by_element(max_forces_values, elem_id, {}) or {}
        demand = normalize_numeric(max_forces_entry.get('max_shear'))
        demand = abs(demand) if demand is not None else None
        g_value = normalize_numeric(get_by_element(shear_values, elem_id))
        capacity = (
            None
            if demand is None or g_value is None else
            g_value + demand
        )
        reliability = get_reliability('shear', elem_id)
        shear_rows.append({
            'Elemen (-)': int(elem_id),
            'Kode': get_element_code(elem_id),
            'Kapasitas R (kN)': capacity,
            'S dari Analisis Struktur (kN)': demand,
            'g(x) (kN)': g_value,
            'Jumlah Gagal (-)': reliability.get('failures'),
            'Pf (-)': reliability.get('Pf'),
            'Beta (-)': reliability.get('Beta'),
            'Status': get_status(g_value)
        })

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
        axial_rows.append({
            'Elemen (-)': int(elem_id),
            'Kode': get_element_code(elem_id),
            'Kapasitas R (kN)': capacity,
            'S dari Analisis Struktur (kN)': demand,
            'g(x) (kN)': g_value,
            'Jumlah Gagal (-)': reliability.get('failures'),
            'Pf (-)': reliability.get('Pf'),
            'Beta (-)': reliability.get('Beta'),
            'Status': get_status(g_value)
        })

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
        axial_moment_rows.append({
            'Elemen (-)': int(elem_id),
            'Kode': get_element_code(elem_id),
            'Kapasitas R (-)': capacity,
            'S dari Analisis Struktur (-)': demand,
            'g(x) (-)': g_value,
            'Jumlah Gagal (-)': reliability.get('failures'),
            'Pf (-)': reliability.get('Pf'),
            'Beta (-)': reliability.get('Beta'),
            'Status': get_status(g_value)
        })

    return {
        'lentur': pd.DataFrame(moment_rows),
        'geser': pd.DataFrame(shear_rows),
        'aksial': pd.DataFrame(axial_rows),
        'aksial_lentur': pd.DataFrame(axial_moment_rows)
    }


def style_limit_state_performance_df(df: pd.DataFrame):
    """Styling sederhana untuk tabel limit state terpisah."""
    styler = style_input_dataframe(df)
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
                or column.startswith('g(x)')
            ],
            'reliability': [
                'Jumlah Gagal (-)',
                'Pf (-)',
                'Beta (-)',
                'Status'
            ]
        }
    )


def build_limit_state_resume_df(limit_state_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
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

        rows.append({
            'Elemen (-)': governing.get('Elemen (-)'),
            'Kode': governing.get('Kode'),
            'Limit State Kontrol': governing.get('Limit State Kontrol'),
            'Satuan': governing.get('Satuan'),
            'Kapasitas R': governing.get('Kapasitas R'),
            'S dari Analisis Struktur': governing.get('S dari Analisis Struktur'),
            'g(x)': governing.get('g(x)'),
            'Jumlah Gagal (-)': governing.get('Jumlah Gagal (-)'),
            'Pf (-)': governing.get('Pf (-)'),
            'Beta (-)': governing.get('Beta (-)'),
            'Status': governing.get('Status')
        })

    return pd.DataFrame(rows)


def style_limit_state_resume_df(df: pd.DataFrame):
    """Styling tabel resume limit state pengontrol."""
    styler = style_input_dataframe(df)
    return apply_grouped_header_styles(
        styler,
        df,
        {
            'identity': ['Elemen (-)', 'Kode'],
            'control': ['Limit State Kontrol', 'Satuan'],
            'performance': ['Kapasitas R', 'S dari Analisis Struktur', 'g(x)'],
            'reliability': ['Jumlah Gagal (-)', 'Pf (-)', 'Beta (-)', 'Status']
        }
    )


def build_sensitivity_df(sensitivity_results: Dict) -> pd.DataFrame:
    rows = []
    for variable, values in sensitivity_results.items():
        rows.append({
            'Variable (-)': variable,
            'Sensitivity Index (-)': values.get('sensitivity_index'),
            'Mean in Failure (sesuai variabel)': values.get('mean_in_failure'),
            'Mean Overall (sesuai variabel)': values.get('mean_overall'),
            'Std Overall (sesuai variabel)': values.get('std_overall')
        })
    return pd.DataFrame(rows)


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


def figure_to_png_data_uri(fig,
                           image_dpi: int = 220,
                           tight_bbox: bool = True) -> str:
    """Konversi figure matplotlib menjadi PNG data URI resolusi tinggi."""
    image_buffer = io.BytesIO()
    save_kwargs = {
        'format': 'png',
        'dpi': image_dpi,
        'facecolor': 'white'
    }
    if tight_bbox:
        save_kwargs['bbox_inches'] = 'tight'
    fig.savefig(image_buffer, **save_kwargs)
    image_buffer.seek(0)
    image_bytes = image_buffer.getvalue()
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode('ascii')


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
                tight_bbox: bool = True) -> None:
    """Tampilkan plot matplotlib dan tutup figure setelah dirender."""
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
            fy_tekan=fy_tekan
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
        fy_tekan=material_snapshot['fy_tekan']
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
        fy_tekan=material_snapshot['fy_tekan']
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
        max_moment
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

    line_target_moment = exact_boundary_moment if exact_boundary_moment is not None else boundary_moment
    line_target_axial = exact_boundary_axial if exact_boundary_axial is not None else boundary_axial
    line_target_lambda = lambda_exact if lambda_exact is not None else lambda_interp

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    for axis in axes:
        axis.plot(moment_values, axial_values, color='#0f4c81', lw=2.2, label='Kurva interaksi')
        axis.scatter([max_moment], [demand_axial], color='#d62828', s=55, zorder=5, label='Demand')
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
                color='#7c3aed',
                marker='*',
                s=140,
                zorder=7,
                label='Boundary exact (c)'
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

    line_mid_x = 0.5 * line_target_moment
    line_mid_y = 0.5 * line_target_axial
    axes[0].annotate(
        f"Demand\nM={max_moment:.2f} kN.m\nP={demand_axial:.2f} kN",
        xy=(max_moment, demand_axial),
        xytext=(12, 12),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', color='#d62828', lw=1.0),
        fontsize=8,
        ha='left',
        va='bottom'
    )
    if exact_boundary_moment is not None and exact_boundary_axial is not None and c_boundary_exact is not None:
        axes[0].annotate(
            (
                "Boundary exact\n"
                f"M={exact_boundary_moment:.2f} kN.m\n"
                f"P={exact_boundary_axial:.2f} kN\n"
                f"c={c_boundary_exact:.3f} mm"
            ),
            xy=(exact_boundary_moment, exact_boundary_axial),
            xytext=(-14, 14),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='#7c3aed', lw=1.0),
            fontsize=8,
            ha='right',
            va='bottom'
        )
    axes[0].text(
        line_mid_x,
        line_mid_y,
        f"Garis lambda\nlambda = {line_target_lambda:.4f}",
        fontsize=8,
        ha='left',
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='#9ca3af')
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
    axes[1].annotate(
        f"Demand\nM={max_moment:.2f} kN.m\nP={demand_axial:.2f} kN",
        xy=(max_moment, demand_axial),
        xytext=(8, 8),
        textcoords='offset points',
        fontsize=8,
        color='#d62828',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='#d62828')
    )
    if exact_boundary_moment is not None and exact_boundary_axial is not None:
        axes[1].annotate(
            (
                "Boundary exact\n"
                f"M={exact_boundary_moment:.2f} kN.m\n"
                f"P={exact_boundary_axial:.2f} kN\n"
                f"c={c_boundary_exact:.3f} mm"
            ),
            xy=(exact_boundary_moment, exact_boundary_axial),
            xytext=(8, -52),
            textcoords='offset points',
            fontsize=8,
            color='#7c3aed',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='#7c3aed')
        )
    axes[1].text(
        line_mid_x,
        line_mid_y,
        f"lambda = {line_target_lambda:.4f}",
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
        options=["Deterministik", "Probabilistik"],
        index=1
    )
    if analysis_mode_label == "Probabilistik":
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
                analysis_mode_label == "Probabilistik",
                int(num_simulations)
            )
            estimated_elements = len(
                get_geometry_elements_for_mode(
                    preview_input_data,
                    analysis_mode_label == "Probabilistik"
                )
            )
            estimated_random_variables = count_probabilistic_random_variables(preview_input_data)
            st.caption(
                "Perkiraan waktu analisis: "
                f"`{format_duration_text(estimated_seconds)}`"
            )
            if analysis_mode_label == "Probabilistik":
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
                    analysis_mode_label,
                    int(num_simulations),
                    progress_container=progress_container
                )

            st.session_state['results_bundle'] = analysis.get_results_bundle()
            st.session_state['portal_elements'] = analysis.portal.elements
            st.session_state['portal_nodes'] = analysis.data['geometry']['nodes'].astype(float)
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
            analysis_mode_label == "Probabilistik"
        )
    except Exception as exc:
        preview_error = exc


results_bundle = st.session_state.get('results_bundle')
portal_elements = st.session_state.get('portal_elements')
portal_nodes = st.session_state.get('portal_nodes')
analysis_source = st.session_state.get('analysis_source')
selected_analysis_mode = (
    'probabilistic' if analysis_mode_label == "Probabilistik" else 'deterministic'
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

    st.subheader("Ringkasan")
    if is_probabilistic:
        metric_cols = st.columns(5)
        metric_cols[0].metric("Pf", format_metric(summary.get('Pf'), 6))
        metric_cols[1].metric("Beta", format_metric(summary.get('Beta'), 4))
        metric_cols[2].metric("Failures", str(summary.get('failures', 0)))
        metric_cols[3].metric("Kelas Keamanan", summary.get('safety_class') or "-")
        metric_cols[4].metric(
            "Status",
            "-" if summary.get('is_safe') is None else ("SAFE" if summary.get('is_safe') else "UNSAFE")
        )
        limit_cols = st.columns(4)
        limit_cols[0].metric("g Min Momen (kN.m)", format_metric(summary.get('min_g_moment'), 4))
        limit_cols[1].metric("g Min Geser (kN)", format_metric(summary.get('min_g_shear'), 4))
        limit_cols[2].metric("g Min Aksial (kN)", format_metric(summary.get('min_g_axial'), 4))
        limit_cols[3].metric("g Min Aksial+Momen (-)", format_metric(summary.get('min_g_axial_moment'), 4))
        st.caption(
            f"Sumber input: `{analysis_source or DEFAULT_INPUT_FILE}` | "
            f"Mode: {summary.get('analysis_mode_label', '-')} | "
            f"Simulasi yang ditampilkan: #{(latest_simulation.get('display_index') or 0) + 1}"
        )
        if summary.get('analysis_failures', 0):
            st.warning(
                f"{summary.get('analysis_failures', 0)} simulasi gagal dieksekusi. "
                "Failure tersebut tetap dihitung konservatif sebagai kegagalan."
            )
    else:
        metric_cols = st.columns(8)
        metric_cols[0].metric("Mode", summary.get('analysis_mode_label') or "-")
        metric_cols[1].metric("Pf", "-")
        metric_cols[2].metric("Beta", "-")
        metric_cols[3].metric("g Min Momen (kN.m)", format_metric(summary.get('min_g_moment'), 4))
        metric_cols[4].metric("g Min Geser (kN)", format_metric(summary.get('min_g_shear'), 4))
        metric_cols[5].metric("g Min Aksial (kN)", format_metric(summary.get('min_g_axial'), 4))
        metric_cols[6].metric("g Min Aksial+Momen (-)", format_metric(summary.get('min_g_axial_moment'), 4))
        metric_cols[7].metric(
            "Status",
            "-" if summary.get('is_safe') is None else ("SAFE" if summary.get('is_safe') else "UNSAFE")
        )
        st.caption(
            f"Sumber input: `{analysis_source or DEFAULT_INPUT_FILE}` | "
            f"Mode: {summary.get('analysis_mode_label', '-')} | "
            "Hasil ditampilkan dari satu analisis berbasis nilai deterministic input per elemen."
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
    "Input",
    "Output",
    "Plot Simulasi Terakhir",
    "Kurva Interaksi",
    "Laporan"
]
active_dashboard_tab = st.radio(
    "Navigasi Dashboard",
    options=dashboard_tabs,
    index=0,
    horizontal=True,
    key="active_dashboard_tab",
    label_visibility="collapsed"
)

if active_dashboard_tab == "Input":
    input_plot_nodes = preview_portal_nodes if preview_portal_nodes is not None else portal_nodes
    input_plot_elements = preview_portal_elements if preview_portal_elements is not None else portal_elements

    if input_plot_nodes is None or input_plot_elements is None:
        input_plot_nodes, input_plot_elements = build_preview_portal(
            input_data,
            selected_is_probabilistic
        )

    preview_distributed_loads = build_preview_distributed_loads(input_data, selected_is_probabilistic)

    if results_bundle and not input_preview_differs_from_results:
        col_left, col_right = st.columns(2)

        with col_left:
            sample_title = (
                "#### Input Acak Simulasi Terakhir"
                if is_probabilistic else
                "#### Input Acuan Deterministik"
            )
            st.markdown(sample_title)
            latest_sample_df = build_latest_sample_df(latest_simulation)
            if latest_sample_df.empty:
                st.info("Tidak ada sampel acak yang perlu ditampilkan.")
            else:
                render_input_table(latest_sample_df)

        with col_right:
            if is_probabilistic:
                st.markdown("#### Definisi Variabel Random")
                render_input_table(build_random_variable_df(results_bundle['random_variables']))
            else:
                st.markdown("#### Keterangan Mode")
                st.info(
                    "Mode deterministik tidak melakukan sampling Monte Carlo. "
                    "Analisis dijalankan satu kali dengan nilai deterministic tiap elemen."
                )

        if is_probabilistic:
            effective_modulus_df = build_effective_modulus_snapshot_df(
                input_data,
                latest_simulation,
                is_probabilistic
            )
            if not effective_modulus_df.empty:
                st.markdown("#### Snapshot E Dipakai DSM")
                st.caption(
                    "`E_dipakai_DSM (MPa)` adalah nilai yang benar-benar dipakai solver DSM "
                    "untuk simulasi yang sedang ditampilkan. "
                    "`E_acuan_mean (MPa)` hanya nilai acuan hasil `E_mean x fb_mean`."
                )
                render_input_table(effective_modulus_df)
    else:
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
    render_plot(geometry_fig)

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

elif active_dashboard_tab == "Output":
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
        if SPECIAL_BEAM_JOINT_RAW_SIGN_NODE_IDS:
            special_nodes_text = ", ".join(
                str(node_id) for node_id in sorted(SPECIAL_BEAM_JOINT_RAW_SIGN_NODE_IDS)
            )
            st.caption(
                f"Khusus node {special_nodes_text}, momen joint balok di tabel "
                "ditampilkan mengikuti tanda aksi joint solver."
            )
        st.caption(
            "Warna header membedakan grup beban merata, aksial, geser, dan momen "
            "agar pembacaan tabel lebih cepat."
        )
        internal_force_df = build_internal_force_df(latest_result, input_data=input_data)
        render_input_table(
            internal_force_df,
            styler=style_internal_force_df(internal_force_df)
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

        st.markdown("#### Cek Keseimbangan Momen per Node")
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

        st.markdown("#### Reliabilitas Sistem Portal Gabungan")
        st.caption(
            "Portal dianggap tersusun seri antara subsistem Balok dan subsistem Kolom, "
            "sehingga portal gagal bila salah satu subsistem gagal."
        )
        st.caption(
            "Kasus 1: Balok = sistem paralel, Kolom = sistem seri. "
            "Kasus 2: Balok = sistem seri, Kolom = sistem seri."
        )
        if is_probabilistic:
            st.caption(
                "`Pf/Beta` dihitung langsung dari seluruh simulasi Monte Carlo berdasarkan "
                "kejadian gagal subsistem per simulasi."
            )
        else:
            st.caption(
                "Pada mode deterministik, kolom `Pf/Beta` tidak berlaku dan status diambil "
                "dari hasil satu kali analisis deterministik."
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

        st.markdown("#### Nilai g per Elemen")
        st.caption(
            "Tabel dipisah per limit state: lentur, geser, aksial, dan aksial-lentur. "
            "Kolom `R`, `S`, dan `g(x)` ditampilkan terpisah agar perhitungan tiap cek "
            "lebih mudah dibaca."
        )
        st.caption(
            "Perhitungan `R` pada tabel ini memakai faktor reduksi kekuatan `phi = 1`, "
            "agar kapasitas yang ditampilkan konsisten dengan evaluasi reliabilitas."
        )
        if is_probabilistic:
            st.caption(
                "`Pf/Beta` per elemen dihitung langsung dari seluruh simulasi Monte Carlo, "
                "bukan dari rata-rata respons struktur."
            )
        else:
            st.caption(
                "Pada mode deterministik, kolom `Jumlah Gagal/Pf/Beta` tidak berlaku "
                "dan status ditentukan dari tanda `g(x)`."
            )
        st.caption(
            "Untuk cek aksial-lentur, `R` merepresentasikan `lambda_boundary`, "
            "sedangkan `S` ditetapkan 1.0 sehingga `g(x) = lambda - 1.0`."
        )
        limit_state_tables = build_limit_state_performance_tables(
            latest_result,
            input_data=input_data,
            element_reliability=results_bundle.get('element_reliability', {})
        )
        resume_df = build_limit_state_resume_df(limit_state_tables)
        st.markdown("##### Resume Pengontrol")
        st.caption(
            "Limit state pengontrol per elemen dipilih dengan kriteria `Pf` maksimum, "
            "lalu `Beta` minimum. Jika `Pf/Beta` tidak tersedia, fallback memakai `g(x)` minimum."
        )
        if resume_df.empty:
            st.info("Data resume limit state pengontrol belum tersedia.")
        else:
            render_input_table(
                resume_df,
                styler=style_limit_state_resume_df(resume_df)
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
                styler=style_limit_state_performance_df(table_df)
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
                alt_text="Plot deformasi simulasi terakhir"
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
                alt_text="Diagram axial simulasi terakhir"
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
                alt_text="Diagram shear simulasi terakhir"
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
                alt_text="Diagram momen simulasi terakhir"
            )

elif active_dashboard_tab == "Kurva Interaksi":
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
                "Titik `Demand`, `Boundary exact (c)`, dan `Garis lambda` "
                "diberi label langsung pada gambar. "
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
                demand_cols = st.columns(4)
                demand_cols[0].metric(
                    "Demand Aksial (kN)",
                    format_metric(interaction_plot['demand_axial'], 3)
                )
                demand_cols[1].metric(
                    "Demand Momen (kN.m)",
                    format_metric(interaction_plot['demand_moment'], 3)
                )
                demand_cols[2].metric(
                    "Boundary Exact P (kN)",
                    format_metric(interaction_plot['boundary_axial_exact'], 3)
                )
                demand_cols[3].metric(
                    "Boundary Exact M (kN.m)",
                    format_metric(interaction_plot['boundary_moment_exact'], 3)
                )

                metric_cols = st.columns(5)
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
                    "Kontrol",
                    str(interaction_plot['controlling_state']).replace('-', ' ').title()
                )
                render_plot(
                    interaction_plot['figure'],
                    interactive=True,
                    viewer_key=f"interaction-curve-e{int(selected_interaction_elem)}",
                    alt_text=f"Kurva interaksi elemen {int(selected_interaction_elem)}",
                    viewer_height=620,
                    tight_bbox=False
                )
            except Exception as exc:
                st.error(
                    "Kurva interaksi tidak bisa ditampilkan: "
                    f"{format_error_message(exc)}"
                )

elif active_dashboard_tab == "Laporan":
    if not results_bundle:
        st.info("Laporan baru tersedia setelah analisis dijalankan.")
    else:
        st.markdown("#### Ringkasan Laporan")
        st.text(results_bundle['report'])

        st.markdown("#### Sensitivitas Variabel")
        sensitivity_df = build_sensitivity_df(results_bundle['sensitivity_results'])
        if not is_probabilistic:
            st.info("Analisis deterministik tidak memiliki hasil sensitivitas variabel acak.")
        elif sensitivity_df.empty:
            st.info("Data sensitivitas belum tersedia atau tidak ada failure sample yang dapat diranking.")
        else:
            st.dataframe(sensitivity_df, use_container_width=True, hide_index=True)
