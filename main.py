"""
Aplikasi Utama: Analisis Portal 2D dengan Metode Matriks Kekakuan Langsung
dan Integrasi Monte Carlo untuk Analisis Keandalan

Penggunaan:
    python main.py <path_to_excel_input_file>
"""

import sys
import numpy as np
from typing import Any, Dict, Optional
import os
import json
from datetime import datetime

# Import modules
from modules.excel_reader import ExcelReader
from modules.stiffness_matrix import Portal2D
from modules.analysis import StructuralAnalysis, LoadHandler
from modules.monte_carlo import MonteCarloAnalysis, RandomVariableGenerator
from modules.reliability import (PerformanceFunction, ReliabilityAssessment,
                               SensitivityAnalysis, AXIAL_DEMAND_TOLERANCE_KN)
from modules.plotting import PortalPlotter, ReliabilityPlotter


class PortalReliabilityAnalysis:
    """Main class untuk analisis portal 2D deterministik/probabilistik."""

    def __init__(self, excel_file: str, num_mc_simulations: int = 10000,
                 analysis_mode: str = 'probabilistic'):
        """
        Initialize analisis
        
        Parameters:
        - excel_file: path ke file Excel input
        - num_mc_simulations: jumlah simulasi Monte Carlo
        """
        self.excel_file = excel_file
        self.num_mc_simulations = num_mc_simulations
        self.analysis_mode = self.normalize_analysis_mode(analysis_mode)
        self.data = None
        self.portal = None
        self.analysis = None
        self.mc_results = None
        self.reliability_assessment = None
        self.random_variables = {}
        self.sensitivity_results = {}
        self.latest_simulation_result = None
        self.latest_random_sample = None
        self.latest_simulation_index = None
        self.output_data = {}
        self.portal_system_reliability = {}
        self._section_capacity_inputs_cache = {}
        self._element_design_code_cache = {}
        self._limit_state_applicability_cache = None
        self._node_coordinate_cache = None
        
        print(
            f"Initializing {self.get_analysis_mode_label()} analysis "
            f"with {excel_file}"
        )

    @staticmethod
    def normalize_analysis_mode(analysis_mode: str) -> str:
        """Normalisasi mode analisis dari input UI/CLI."""
        mode = str(analysis_mode or 'probabilistic').strip().lower()
        aliases = {
            'probabilistic': 'probabilistic',
            'probabilistik': 'probabilistic',
            'monte carlo': 'probabilistic',
            'deterministic': 'deterministic',
            'deterministik': 'deterministic'
        }

        normalized = aliases.get(mode)
        if normalized is None:
            raise ValueError(
                "analysis_mode harus 'deterministic/deterministik' "
                "atau 'probabilistic/probabilistik'"
            )

        return normalized

    @property
    def is_probabilistic(self) -> bool:
        """True jika analisis memakai Monte Carlo."""
        return self.analysis_mode == 'probabilistic'

    def get_analysis_mode_label(self) -> str:
        """Label mode untuk UI/report."""
        return 'Probabilistik' if self.is_probabilistic else 'Deterministik'

    @staticmethod
    def _get_scalar_value(value: Any, reducer: str = 'first') -> float:
        """Ambil nilai float dari scalar/array input."""
        if np.isscalar(value):
            return float(value)

        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size == 0:
            return 0.0

        if reducer == 'mean':
            return float(np.mean(array))

        return float(array[0])

    @staticmethod
    def _element_var_name(prefix: str, elem_id: int) -> str:
        """Nama variabel random flat per elemen."""
        return f"{prefix}_E{int(elem_id)}"

    @staticmethod
    def _summarize_values(values_by_element: Dict[int, float], unit: str) -> str:
        """Ringkas nilai per elemen untuk log/report."""
        if not values_by_element:
            return "-"

        values = np.asarray(list(values_by_element.values()), dtype=float)
        if values.size == 1 or np.allclose(values, values[0]):
            return f"{values[0]:.2f} {unit}"
        return f"{values.min():.2f} s.d. {values.max():.2f} {unit}"

    def _build_flat_element_sample(self, source_by_element: Dict[int, Dict],
                                   prefix: str,
                                   deterministic_key: str,
                                   mean_key: str) -> Dict[str, float]:
        """Bangun sample flat `{prefix}_E{id}` dari data per elemen."""
        sample = {}
        for elem_id in sorted(source_by_element):
            props = source_by_element[elem_id]
            value = props.get(deterministic_key)
            if value is None:
                value = props.get(mean_key, 0.0)
            sample[self._element_var_name(prefix, elem_id)] = float(value)
        return sample

    def _get_element_values_from_sample(self, sample: Optional[Dict[str, float]],
                                        source_by_element: Dict[int, Dict],
                                        prefix: str,
                                        fallback_key: str = 'mean') -> Dict[int, float]:
        """Ambil nilai per elemen dari sample flat, fallback ke data sumber."""
        values_by_element = {}
        sample = sample or {}
        for elem_id in sorted(source_by_element):
            key = self._element_var_name(prefix, elem_id)
            if key in sample:
                values_by_element[int(elem_id)] = float(sample[key])
                continue

            props = source_by_element[elem_id]
            value = props.get(fallback_key)
            if value is None and fallback_key != 'mean':
                value = props.get('mean')
            values_by_element[int(elem_id)] = float(0.0 if value is None else value)

        return values_by_element

    def _require_element_material_value(self,
                                        values_by_element: Dict[int, float],
                                        elem_id: int,
                                        field_label: str,
                                        sheet_name: str) -> float:
        """Ambil nilai material wajib per elemen, atau raise jika tidak valid."""
        elem_id = int(elem_id)
        if elem_id not in values_by_element:
            raise ValueError(
                f"Data input elemen {elem_id} pada sheet {sheet_name} belum lengkap: "
                f"{field_label} tidak ditemukan."
            )

        value = self._read_positive_number(values_by_element.get(elem_id))
        if value <= 0.0:
            raise ValueError(
                f"Data input elemen {elem_id} pada sheet {sheet_name} belum lengkap "
                f"atau tidak valid: {field_label}."
            )
        return value

    def _get_group_sample_summary(self, sample: Dict[str, float],
                                  source_by_element: Dict[int, Dict],
                                  prefix: str,
                                  fallback_key: str,
                                  unit: str) -> str:
        """Ringkas grup nilai per elemen dari sample flat."""
        values = self._get_element_values_from_sample(
            sample,
            source_by_element,
            prefix,
            fallback_key=fallback_key
        )
        return self._summarize_values(values, unit)

    def _summarize_random_variable_group(self, prefix: str, unit: str) -> str:
        """Ringkas mean/stddev grup variabel random dengan prefix tertentu."""
        matching = [
            info for name, info in self.random_variables.items()
            if name.startswith(prefix)
        ]
        if not matching:
            return "-"

        means = np.asarray([float(info.get('mean', 0.0)) for info in matching], dtype=float)
        stddevs = np.asarray([float(info.get('stddev', 0.0)) for info in matching], dtype=float)

        mean_text = self._summarize_values(
            {index: value for index, value in enumerate(means)},
            unit
        )
        std_text = self._summarize_values(
            {index: value for index, value in enumerate(stddevs)},
            unit
        )
        return f"mu={mean_text}, sigma={std_text}"

    def _get_random_variable_definitions(self) -> Dict[str, Dict[str, float]]:
        """Bangun definisi variabel random per elemen dari data input."""
        concrete_data = self.data['concrete']
        steel_data = self.data['steel']
        dead_load_data = self.data['dead_load']
        live_load_data = self.data['live_load']

        definitions = {}

        for elem_id, props in concrete_data.get('by_element', {}).items():
            definitions[self._element_var_name('fc', elem_id)] = {
                'distribution': props.get('distribution', 'lognormal'),
                'mean': float(props.get('mean', 0.0)),
                'stddev': float(props.get('stddev', 0.0))
            }

        for elem_id, props in steel_data.get('by_element', {}).items():
            definitions[self._element_var_name('fy_tarik', elem_id)] = {
                'distribution': props.get('tarik_distribution', 'normal'),
                'mean': float(props.get('tarik_mean', 0.0)),
                'stddev': float(props.get('tarik_stddev', 0.0))
            }
            definitions[self._element_var_name('fy_tekan', elem_id)] = {
                'distribution': props.get('tekan_distribution', 'normal'),
                'mean': float(props.get('tekan_mean', 0.0)),
                'stddev': float(props.get('tekan_stddev', 0.0))
            }
            definitions[self._element_var_name('fy_geser', elem_id)] = {
                'distribution': props.get('geser_distribution', 'normal'),
                'mean': float(props.get('geser_mean', props.get('tarik_mean', 0.0))),
                'stddev': float(props.get('geser_stddev', props.get('tarik_stddev', 0.0)))
            }

        for elem_id, props in dead_load_data.get('by_element', {}).items():
            definitions[self._element_var_name('qDL', elem_id)] = {
                'distribution': props.get('distribution', dead_load_data.get('distribution', 'normal')),
                'mean': float(props.get('mean', 0.0)),
                'stddev': float(props.get('stddev', 0.0))
            }

        for elem_id, props in live_load_data.get('by_element', {}).items():
            definitions[self._element_var_name('qLL', elem_id)] = {
                'distribution': props.get('distribution', live_load_data.get('distribution', 'lognormal')),
                'mean': float(props.get('mean', 0.0)),
                'stddev': float(props.get('stddev', 0.0))
            }

        return definitions

    def _build_reference_sample(self) -> Dict[str, float]:
        """Bangun sampel acuan deterministik per elemen."""
        concrete_data = self.data['concrete']
        steel_data = self.data['steel']
        dead_load_data = self.data['dead_load']
        live_load_data = self.data['live_load']

        reference_sample = {}
        reference_sample.update(self._build_flat_element_sample(
            concrete_data.get('by_element', {}),
            prefix='fc',
            deterministic_key='deterministic',
            mean_key='mean'
        ))
        reference_sample.update(self._build_flat_element_sample(
            steel_data.get('by_element', {}),
            prefix='fy_tarik',
            deterministic_key='tarik_deterministic',
            mean_key='tarik_mean'
        ))
        reference_sample.update(self._build_flat_element_sample(
            steel_data.get('by_element', {}),
            prefix='fy_tekan',
            deterministic_key='tekan_deterministic',
            mean_key='tekan_mean'
        ))
        reference_sample.update(self._build_flat_element_sample(
            steel_data.get('by_element', {}),
            prefix='fy_geser',
            deterministic_key='geser_deterministic',
            mean_key='geser_mean'
        ))
        reference_sample.update(self._build_flat_element_sample(
            dead_load_data.get('by_element', {}),
            prefix='qDL',
            deterministic_key='deterministic',
            mean_key='mean'
        ))
        reference_sample.update(self._build_flat_element_sample(
            live_load_data.get('by_element', {}),
            prefix='qLL',
            deterministic_key='deterministic',
            mean_key='mean'
        ))

        return reference_sample

    def _build_mean_load_dict(self, load_data: Dict,
                              value_key: str = 'mean') -> Dict[str, Dict[int, float]]:
        """Bangun beban per elemen dari kolom input yang dipilih."""
        element_ids = [
            int(elem_id)
            for elem_id in load_data.get('elements', [])
        ]
        if not element_ids:
            return {'values': {}}

        load_values = load_data.get(value_key)
        if load_values is None:
            load_values = load_data.get('mean', 0.0)

        if np.isscalar(load_values):
            values = np.full(len(element_ids), float(load_values), dtype=float)
        else:
            values = np.asarray(load_values, dtype=float).reshape(-1)

        if values.size == 0:
            values = np.zeros(len(element_ids), dtype=float)
        elif values.size == 1 and len(element_ids) > 1:
            values = np.full(len(element_ids), float(values[0]), dtype=float)
        elif values.size != len(element_ids):
            raise ValueError(
                "Jumlah nilai beban tidak cocok dengan jumlah elemen "
                "pada input."
            )

        return {
            'values': {
                element_id: float(values[index])
                for index, element_id in enumerate(element_ids)
            }
        }

    @staticmethod
    def _get_rebar_area_from_count(bar_count: float,
                                   bar_diameter: float) -> float:
        """Hitung luas total tulangan dari jumlah batang dan diameter (mm2)."""
        count_value = max(float(bar_count or 0.0), 0.0)
        diameter_value = max(float(bar_diameter or 0.0), 0.0)
        if count_value <= 0.0 or diameter_value <= 0.0:
            return 0.0
        return float(count_value * np.pi * (diameter_value ** 2) / 4.0)

    @staticmethod
    def _get_effective_depth_from_cover(section_height: float,
                                        cover: float,
                                        stirrup_diameter: float,
                                        bar_diameter: float,
                                        from_compression_face: bool = False) -> float:
        """Turunkan d atau d' dari selimut beton dan diameter tulangan (mm)."""
        h_value = max(float(section_height or 0.0), 0.0)
        cover_value = max(float(cover or 0.0), 0.0)
        stirrup_value = max(float(stirrup_diameter or 0.0), 0.0)
        bar_value = max(float(bar_diameter or 0.0), 0.0)
        centroid_depth = cover_value + stirrup_value + (0.5 * bar_value)

        if from_compression_face:
            return float(min(centroid_depth, h_value))
        return float(max(h_value - centroid_depth, 0.0))

    @staticmethod
    def _read_positive_number(value) -> float:
        """Baca angka positif, atau kembalikan 0 jika invalid."""
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return 0.0

        if not np.isfinite(numeric_value) or numeric_value <= 0.0:
            return 0.0
        return float(numeric_value)

    @staticmethod
    def _raise_element_input_error(elem_id: int,
                                   missing_fields,
                                   sheet_names: str) -> None:
        """Raise ValueError dengan daftar field input elemen yang kurang."""
        unique_fields = list(dict.fromkeys(missing_fields))
        fields_text = ", ".join(unique_fields)
        raise ValueError(
            f"Data input elemen {int(elem_id)} pada {sheet_names} belum lengkap "
            f"atau tidak valid: {fields_text}."
        )

    def _get_section_capacity_inputs(self, elem_id: int) -> Dict[str, Dict]:
        """Ambil geometri dan tulangan penampang per elemen."""
        elem_id = int(elem_id)
        cached_inputs = self._section_capacity_inputs_cache.get(elem_id)
        if cached_inputs is not None:
            return cached_inputs

        geometry_lookup = self.data['geometry'].get('properties_by_element', {})
        reinforcement_lookup = self.data.get('reinforcement', {}).get('by_element', {})
        geometry_props = (
            geometry_lookup.get(elem_id)
            or geometry_lookup.get(str(elem_id))
            or {}
        )
        reinforcement_props = (
            reinforcement_lookup.get(elem_id)
            or reinforcement_lookup.get(str(elem_id))
            or {}
        )

        if not geometry_props:
            raise ValueError(
                f"Data geometri elemen {elem_id} pada sheet Geometri tidak ditemukan."
            )
        if not reinforcement_props:
            raise ValueError(
                f"Data tulangan elemen {elem_id} pada sheet Tulangan tidak ditemukan."
            )

        b_value = self._read_positive_number(geometry_props.get('b'))
        h_value = self._read_positive_number(geometry_props.get('h'))
        area_value = self._read_positive_number(geometry_props.get('area'))
        geometry_missing = []
        if b_value <= 0.0:
            geometry_missing.append('b')
        if h_value <= 0.0:
            geometry_missing.append('h')
        if area_value <= 0.0:
            geometry_missing.append('A/Area')
        if geometry_missing:
            self._raise_element_input_error(elem_id, geometry_missing, 'sheet Geometri')

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
            self._get_effective_depth_from_cover(
                h_value,
                ds_tarik,
                du_geser,
                du_tarik
            )
        )

        d_tekan = (
            d_tekan_input
            if d_tekan_input > 0.0 else
            self._get_effective_depth_from_cover(
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
            self._get_rebar_area_from_count(n_tarik, du_tarik)
        )
        as_tekan = (
            as_tekan_input
            if as_tekan_input > 0.0 else
            self._get_rebar_area_from_count(n_tekan, du_tekan)
        )
        as_geser = (
            as_geser_input
            if as_geser_input > 0.0 else
            self._get_rebar_area_from_count(n_geser, du_geser)
        )

        section_geometry = {
            'b': b_value,
            'h': h_value,
            'd': d_tarik,
            'd_prime': d_tekan,
            'area': area_value,
            'Ag': area_value,
            'element_id': elem_id
        }
        steel_area = {
            'As': as_tarik,
            'As_prime': as_tekan,
            'As_shear': as_geser,
            'd_prime': d_tekan,
            'shear_spacing': self._read_positive_number(
                reinforcement_props.get('Spasi_geser')
            ),
            'element_id': elem_id
        }

        section_inputs = {
            'section_geometry': section_geometry,
            'steel_area': steel_area
        }
        self._section_capacity_inputs_cache[elem_id] = section_inputs
        return section_inputs

    def _get_element_design_code(self, elem_id: int) -> str:
        """
        Ambil kode desain elemen.

        Prioritas:
        1. Gunakan `Kode` dari input Excel bila tersedia.
        2. Jika kosong, infer dari orientasi elemen: horizontal -> B, vertikal -> K.
        """
        elem_id = int(elem_id)
        cached_code = self._element_design_code_cache.get(elem_id)
        if cached_code is not None:
            return cached_code

        geometry_lookup = self.data['geometry'].get('properties_by_element', {})
        geometry_props = (
            geometry_lookup.get(elem_id)
            or geometry_lookup.get(str(elem_id))
            or {}
        )
        raw_code = str(geometry_props.get('code', '') or '').strip().upper()
        if raw_code in {'B', 'K'}:
            self._element_design_code_cache[elem_id] = raw_code
            return raw_code

        node_start = int(geometry_props.get('node_start', 0) or 0)
        node_end = int(geometry_props.get('node_end', 0) or 0)
        if self._node_coordinate_cache is None:
            self._node_coordinate_cache = {
                int(row[0]): np.asarray(row[1:3], dtype=float)
                for row in self.data['geometry'].get('nodes', [])
            }
        start = self._node_coordinate_cache.get(node_start)
        end = self._node_coordinate_cache.get(node_end)
        if start is None or end is None:
            return raw_code

        delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        inferred_code = 'B' if abs(float(delta[0])) >= abs(float(delta[1])) else 'K'
        self._element_design_code_cache[elem_id] = inferred_code
        return inferred_code

    @staticmethod
    def _get_axial_demand_components(force_data: Dict) -> Dict[str, float]:
        """Ekstrak demand aksial tekan/tarik maksimum dari gaya ujung elemen."""
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

    @staticmethod
    def _get_performance_keys() -> tuple[str, ...]:
        """Daftar key limit-state yang diperhitungkan dalam status aman/tidak."""
        return (
            'performance',
            'performance_shear',
            'performance_axial',
            'performance_axial_moment'
        )

    @staticmethod
    def _get_limit_state_key_mapping() -> Dict[str, str]:
        """Mapping nama limit state ringkas ke key output analisis."""
        return {
            'moment': 'performance',
            'shear': 'performance_shear',
            'axial': 'performance_axial',
            'axial_moment': 'performance_axial_moment'
        }

    def _get_applicable_element_ids_by_limit_state(self) -> Dict[str, list[int]]:
        """Daftar elemen yang relevan untuk tiap limit state."""
        if self._limit_state_applicability_cache is not None:
            return {
                state_name: list(elem_ids)
                for state_name, elem_ids in self._limit_state_applicability_cache.items()
            }

        geometry_elements = self.data['geometry'].get(
            'elements_mean',
            self.data['geometry']['elements']
        )
        all_element_ids = sorted({
            int(elem[0]) for elem in np.asarray(geometry_elements, dtype=float)
        })
        axial_element_ids = []
        axial_moment_element_ids = []

        for elem_id in all_element_ids:
            element_code = self._get_element_design_code(elem_id)
            if element_code in {'B', 'K'}:
                axial_element_ids.append(int(elem_id))
            if element_code == 'K':
                axial_moment_element_ids.append(int(elem_id))

        self._limit_state_applicability_cache = {
            'moment': list(all_element_ids),
            'shear': list(all_element_ids),
            'axial': list(axial_element_ids),
            'axial_moment': list(axial_moment_element_ids)
        }
        return {
            state_name: list(elem_ids)
            for state_name, elem_ids in self._limit_state_applicability_cache.items()
        }

    def _evaluate_analysis_result_failure(self,
                                          analysis_result: Optional[Dict]) -> Dict[str, Any]:
        """Bangun detail failure global dan per elemen untuk satu simulasi."""
        applicable_by_state = self._get_applicable_element_ids_by_limit_state()

        if analysis_result is None:
            failed_by_state = {
                state_name: list(elem_ids)
                for state_name, elem_ids in applicable_by_state.items()
            }
            failed_elements = sorted({
                int(elem_id)
                for elem_ids in failed_by_state.values()
                for elem_id in elem_ids
            })
            return {
                'is_safe': False,
                'failed_elements': failed_elements,
                'failed_elements_by_state': failed_by_state,
                'applicable_elements_by_state': applicable_by_state
            }

        failed_by_state = {}
        failed_elements = set()
        for state_name, performance_key in self._get_limit_state_key_mapping().items():
            performance_values = analysis_result.get(performance_key, {}) or {}
            failed_ids = sorted(
                int(elem_id)
                for elem_id, g_value in performance_values.items()
                if float(g_value) < 0.0
            )
            failed_by_state[state_name] = failed_ids
            failed_elements.update(failed_ids)

        return {
            'is_safe': not failed_elements,
            'failed_elements': sorted(failed_elements),
            'failed_elements_by_state': failed_by_state,
            'applicable_elements_by_state': applicable_by_state
        }

    def _build_performance_values(self, max_forces_result: Dict,
                                  material_sample: Dict[str, float]) -> Dict[str, Dict]:
        """Hitung fungsi performa untuk setiap elemen relevan."""
        fc_values = self._get_element_values_from_sample(
            material_sample,
            self.data['concrete'].get('by_element', {}),
            prefix='fc',
            fallback_key='deterministic' if not self.is_probabilistic else 'mean'
        )
        fy_tarik_values = self._get_element_values_from_sample(
            material_sample,
            self.data['steel'].get('by_element', {}),
            prefix='fy_tarik',
            fallback_key='tarik_deterministic' if not self.is_probabilistic else 'tarik_mean'
        )
        fy_tekan_values = self._get_element_values_from_sample(
            material_sample,
            self.data['steel'].get('by_element', {}),
            prefix='fy_tekan',
            fallback_key='tekan_deterministic' if not self.is_probabilistic else 'tekan_mean'
        )
        fy_geser_values = self._get_element_values_from_sample(
            material_sample,
            self.data['steel'].get('by_element', {}),
            prefix='fy_geser',
            fallback_key='geser_deterministic' if not self.is_probabilistic else 'geser_mean'
        )

        moment_performance = {}
        shear_performance = {}
        axial_performance = {}
        axial_moment_performance = {}
        moment_metadata = {}
        axial_metadata = {}
        axial_moment_metadata = {}

        def _extract_section_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
            metadata = {
                'phi': result.get('phi'),
                'epsilon_t': result.get('epsilon_t_net'),
                'epsilon_ty': result.get('epsilon_ty'),
                'classification': result.get('classification')
            }
            optional_keys = (
                'beta1',
                'neutral_axis_depth',
                'phi_Pn',
                'phi_Pn_tekan',
                'phi_Pn_tarik',
                'Pn',
                'phi_Mn',
                'Mn',
                'lambda',
                'demand_axial_abs',
                'controlling_state',
                'tension_steel_yielded',
                'compression_steel_yielded'
            )
            for key in optional_keys:
                if key in result:
                    metadata[key] = result.get(key)
            return metadata

        for elem_id, forces_dict in max_forces_result.items():
            max_moment = forces_dict['max_moment']
            max_shear = forces_dict.get('max_shear', 0.0)
            force_data = forces_dict.get('forces', {})
            section_inputs = self._get_section_capacity_inputs(elem_id)
            element_code = self._get_element_design_code(elem_id)
            fy_tarik = self._require_element_material_value(
                fy_tarik_values,
                elem_id,
                'fy_tarik / Mean_tarik',
                'Mutu_Baja'
            )
            fy_tekan = self._require_element_material_value(
                fy_tekan_values,
                elem_id,
                'fy_tekan / Mean_tekan',
                'Mutu_Baja'
            )
            fy_geser = self._require_element_material_value(
                fy_geser_values,
                elem_id,
                'fy_geser / Mean_geser',
                'Mutu_Baja'
            )
            fc_value = self._require_element_material_value(
                fc_values,
                elem_id,
                "fc' / Mean",
                'Mutu_Beton'
            )

            moment_result = PerformanceFunction._get_beam_flexural_response(
                fc_value,
                fy_tarik,
                section_inputs['section_geometry'],
                section_inputs['steel_area'],
                fy_tekan=fy_tekan
            )
            g_moment = float(moment_result['phi_Mn'] - abs(max_moment))
            g_shear = PerformanceFunction.shear_capacity_demand(
                max_shear,
                fc_value,
                fy_geser,
                section_inputs['section_geometry'],
                section_inputs['steel_area'].get('As_shear', 0.0),
                shear_spacing=section_inputs['steel_area'].get('shear_spacing', 0.0)
            )
            moment_performance[elem_id] = g_moment
            shear_performance[elem_id] = g_shear
            moment_metadata[elem_id] = _extract_section_metadata(moment_result)
            axial_demands = self._get_axial_demand_components(force_data)
            interaction_curve = None
            if element_code in {'B', 'K'}:
                interaction_curve = PerformanceFunction._get_column_interaction_curve(
                    fc_value,
                    fy_tarik,
                    section_inputs['section_geometry'],
                    section_inputs['steel_area'],
                    fy_tekan=fy_tekan
                )

            if element_code in {'B', 'K'}:
                axial_result = PerformanceFunction._get_axial_capacity_check_result(
                    axial_demands['compression'],
                    axial_demands['tension'],
                    fc_value,
                    fy_tarik,
                    section_inputs['section_geometry'],
                    section_inputs['steel_area'],
                    fy_tekan=fy_tekan,
                    interaction_curve=interaction_curve
                )
                if not self.is_probabilistic:
                    phi_pn_tekan, phi_pn_tarik = PerformanceFunction._get_axial_capacities(
                        fc_value,
                        fy_tarik,
                        section_inputs['section_geometry'],
                        section_inputs['steel_area'],
                        fy_tekan=fy_tekan
                    )
                    compression_boundary = max(
                        interaction_curve,
                        key=lambda point: float(point['phi_Pn'])
                    ) if interaction_curve else {}
                    demand_axial_abs = max(
                        float(axial_demands['compression']),
                        float(axial_demands['tension'])
                    )
                    axial_result = {
                        'g': float(phi_pn_tekan - demand_axial_abs),
                        'phi': compression_boundary.get('phi', 0.65),
                        'epsilon_t_net': compression_boundary.get('epsilon_t_net', 0.0),
                        'epsilon_ty': compression_boundary.get(
                            'epsilon_ty',
                            PerformanceFunction._get_tension_control_limits(fy_tarik)[0]
                        ),
                        'classification': compression_boundary.get(
                            'classification',
                            'compression-controlled'
                        ),
                        'phi_Pn': float(phi_pn_tekan),
                        'phi_Pn_tekan': float(phi_pn_tekan),
                        'phi_Pn_tarik': float(phi_pn_tarik),
                        'phi_Mn': compression_boundary.get('phi_Mn', 0.0),
                        'neutral_axis_depth': compression_boundary.get(
                            'neutral_axis_depth',
                            float('inf')
                        ),
                        'demand_axial_abs': float(demand_axial_abs),
                        'controlling_state': 'absolute-axial'
                    }
                axial_performance[elem_id] = float(axial_result['g'])
                axial_metadata[elem_id] = _extract_section_metadata(axial_result)

            if element_code == 'K':
                axial_moment_result = PerformanceFunction._get_axial_moment_interaction_result(
                    axial_demands['compression'],
                    axial_demands['tension'],
                    max_moment,
                    fc_value,
                    fy_tarik,
                    section_inputs['section_geometry'],
                    section_inputs['steel_area'],
                    fy_tekan=fy_tekan,
                    interaction_curve=interaction_curve
                )
                axial_moment_performance[elem_id] = float(axial_moment_result['g'])
                axial_moment_metadata[elem_id] = _extract_section_metadata(
                    axial_moment_result
                )

        return {
            'moment': moment_performance,
            'shear': shear_performance,
            'axial': axial_performance,
            'axial_moment': axial_moment_performance,
            'moment_metadata': moment_metadata,
            'axial_metadata': axial_metadata,
            'axial_moment_metadata': axial_moment_metadata
        }

    def _assemble_analysis_output(self, results: Dict,
                                  material_sample: Dict[str, float]) -> Dict:
        """Lengkapi hasil solver dengan gaya maksimum dan nilai g."""
        max_forces_result = self.analysis.extract_maximum_forces(
            results['element_forces'])
        performance_values = self._build_performance_values(
            max_forces_result,
            material_sample
        )

        output = dict(results)
        output['max_forces'] = max_forces_result
        output['performance'] = performance_values.get('moment', {})
        output['performance_shear'] = performance_values.get('shear', {})
        output['performance_axial'] = performance_values.get('axial', {})
        output['performance_axial_moment'] = performance_values.get('axial_moment', {})
        output['performance_metadata'] = performance_values.get('moment_metadata', {})
        output['performance_axial_metadata'] = performance_values.get('axial_metadata', {})
        output['performance_axial_moment_metadata'] = performance_values.get(
            'axial_moment_metadata',
            {}
        )
        return output

    @staticmethod
    def _get_min_performance_value(analysis_result: Optional[Dict],
                                   performance_key: str = 'performance') -> Optional[float]:
        """Ambil nilai g minimum dari hasil analisis untuk limit-state tertentu."""
        if not analysis_result:
            return None

        performance = analysis_result.get(performance_key, {})
        if not performance:
            return None

        return float(min(performance.values()))

    def _is_analysis_result_safe(self, analysis_result: Optional[Dict] = None) -> Optional[bool]:
        """Cek aman/tidak berdasarkan semua nilai g."""
        if analysis_result is None:
            analysis_result = self.latest_simulation_result

        found_limit_state = False
        for performance_key in self._get_performance_keys():
            min_g = self._get_min_performance_value(
                analysis_result,
                performance_key=performance_key
            )
            if min_g is None:
                continue
            found_limit_state = True
            if min_g < 0.0:
                return False

        if not found_limit_state:
            return None

        return True
    
    def read_input(self):
        """Baca data input dari Excel"""
        print("\n[1/6] Reading input data...")
        
        reader = ExcelReader(self.excel_file)
        self.data = reader.get_all_data()
        self._section_capacity_inputs_cache = {}
        self._element_design_code_cache = {}
        self._limit_state_applicability_cache = None
        
        print(f"  [OK] Geometry loaded: {len(self.data['geometry']['nodes'])} nodes")
        print(f"  [OK] Elements loaded: {len(self.data['geometry']['elements'])} elements")
        print(f"  [OK] Random variables loaded")
        
        return self.data
    
    def initialize_portal(self):
        """Initialize struktur portal"""
        print("\n[2/6] Initializing portal structure...")
        
        nodes = self.data['geometry']['nodes'].astype(float)
        if self.is_probabilistic:
            elements = self.data['geometry'].get('elements_mean', self.data['geometry']['elements']).astype(float)
            E = self.data['geometry'].get('E_mean', 30000)
        else:
            elements = self.data['geometry'].get('elements_deterministic', self.data['geometry']['elements']).astype(float)
            E = self.data['geometry'].get('E_deterministic', self.data['geometry'].get('E_mean', 30000))
        
        # Boundary conditions sudah dalam format dict dari reader
        bc = self.data['boundary']
        boundary_conditions = bc  # Already dict format: {node_id: {X, Y, R}}
        
        self.portal = Portal2D(nodes, elements, boundary_conditions, E)
        self.analysis = StructuralAnalysis(self.portal, nodes)
        
        print(f"  [OK] Portal initialized with {self.portal.num_nodes} nodes")
        print(f"  [OK] Total DOFs: {self.portal.num_dof} (Free: {len(self.portal.free_dofs)})")
        
        return self.portal
    
    def setup_monte_carlo(self):
        """Setup konfigurasi Monte Carlo"""
        print("\n[3/6] Setting up Monte Carlo simulation...")

        self.random_variables = self._get_random_variable_definitions()
        
        print(f"  [OK] Random variables defined: {len(self.random_variables)}")
        print(
            "  [OK] Grup variabel: "
            f"fc={len([k for k in self.random_variables if k.startswith('fc_')])}, "
            f"fy_tarik={len([k for k in self.random_variables if k.startswith('fy_tarik_')])}, "
            f"fy_tekan={len([k for k in self.random_variables if k.startswith('fy_tekan_')])}, "
            f"fy_geser={len([k for k in self.random_variables if k.startswith('fy_geser_')])}, "
            f"qDL={len([k for k in self.random_variables if k.startswith('qDL_')])}, "
            f"qLL={len([k for k in self.random_variables if k.startswith('qLL_')])}"
        )
        
        return self.random_variables

    def _get_latest_valid_simulation_index(self) -> Optional[int]:
        """Ambil indeks simulasi terakhir yang menghasilkan output analisis valid."""
        if not self.mc_results:
            return None

        history = self.mc_results.get('max_forces_history', [])
        for idx in range(len(history) - 1, -1, -1):
            if history[idx] is not None:
                return idx

        return None

    def get_latest_simulation_data(self) -> Dict[str, Any]:
        """Ambil paket data simulasi terakhir untuk kebutuhan pelaporan/UI."""
        if not self.mc_results:
            return {}

        analysis_history = self.mc_results.get('max_forces_history', [])
        sample_history = self.mc_results.get('random_samples_history', [])

        if not analysis_history:
            return {}

        last_index = len(analysis_history) - 1
        valid_index = self._get_latest_valid_simulation_index()
        display_index = last_index if analysis_history[last_index] is not None else valid_index

        analysis_result = None
        random_sample = None
        if display_index is not None:
            analysis_result = analysis_history[display_index]
            random_sample = sample_history[display_index]

        if self.latest_simulation_result is not None:
            analysis_result = self.latest_simulation_result

        if self.latest_random_sample is not None:
            random_sample = self.latest_random_sample

        return {
            'last_index': last_index,
            'valid_index': valid_index,
            'display_index': display_index,
            'is_last_simulation_valid': analysis_history[last_index] is not None,
            'random_sample': random_sample,
            'analysis_result': analysis_result
        }

    def get_results_bundle(self) -> Dict[str, Any]:
        """Ringkas seluruh hasil untuk konsumsi Streamlit/UI."""
        latest_simulation = self.get_latest_simulation_data()
        latest_result = latest_simulation.get('analysis_result')

        return {
            'analysis_mode': self.analysis_mode,
            'analysis_mode_label': self.get_analysis_mode_label(),
            'input_data': self.data,
            'random_variables': self.random_variables,
            'portal_system_reliability': self.get_portal_system_reliability_results(),
            'element_reliability': (
                self.mc_results.get('element_reliability', {})
                if self.mc_results else
                {}
            ),
            'summary': {
                'analysis_mode': self.analysis_mode,
                'analysis_mode_label': self.get_analysis_mode_label(),
                'num_simulations': self.mc_results['num_simulations'] if self.mc_results else 0,
                'failures': self.mc_results['failures'] if self.mc_results else 0,
                'analysis_failures': self.mc_results.get('analysis_failures', 0) if self.mc_results else 0,
                'Pf': self.mc_results['Pf'] if self.mc_results else None,
                'Beta': self.mc_results['Beta'] if self.mc_results else None,
                'safety_class': (
                    self.reliability_assessment.get_safety_class()
                    if self.reliability_assessment else
                    ('Deterministik' if latest_result is not None else None)
                ),
                'is_safe': (
                    self.reliability_assessment.is_safe('ultimate')
                    if self.reliability_assessment else
                    self._is_analysis_result_safe(latest_result)
                ),
                'min_g': min([
                    value for value in (
                        self._get_min_performance_value(latest_result, performance_key=key)
                        for key in self._get_performance_keys()
                    ) if value is not None
                ], default=None),
                'min_g_moment': self._get_min_performance_value(
                    latest_result,
                    performance_key='performance'
                ),
                'min_g_shear': self._get_min_performance_value(
                    latest_result,
                    performance_key='performance_shear'
                ),
                'min_g_axial': self._get_min_performance_value(
                    latest_result,
                    performance_key='performance_axial'
                ),
                'min_g_axial_moment': self._get_min_performance_value(
                    latest_result,
                    performance_key='performance_axial_moment'
                )
            },
            'sensitivity_results': self.sensitivity_results,
            'latest_simulation': latest_simulation,
            'report': self.output_data.get('report', '')
        }

    def _get_element_reliability_results(self,
                                         limit_state: str = 'overall') -> Dict[int, Dict[str, float]]:
        """Ambil hasil reliability per elemen untuk limit state tertentu."""
        if not self.mc_results:
            return {}
        return self.mc_results.get('element_reliability', {}).get(limit_state, {})

    @staticmethod
    def _sort_element_reliability_items(element_results: Dict[int, Dict[str, float]]) -> list[tuple[int, Dict[str, float]]]:
        """Urutkan elemen berdasarkan Pf tertinggi lalu Beta terendah."""
        def sort_key(item: tuple[int, Dict[str, float]]) -> tuple[float, float, int]:
            elem_id, values = item
            pf = float(values.get('Pf', 0.0))
            beta = float(values.get('Beta', np.inf))
            if np.isnan(pf):
                pf = -np.inf
            if np.isnan(beta):
                beta = np.inf
            return (-pf, beta, int(elem_id))

        return sorted(
            ((int(elem_id), values) for elem_id, values in element_results.items()),
            key=sort_key
        )

    def _get_top_element_reliability(self,
                                     limit_state: str = 'overall',
                                     top_n: int = 5) -> list[tuple[int, Dict[str, float]]]:
        """Ambil elemen paling kritis berdasarkan Pf."""
        element_results = self._get_element_reliability_results(limit_state)
        return self._sort_element_reliability_items(element_results)[:top_n]

    def _get_element_ids_by_design_group(self) -> Dict[str, list[int]]:
        """Kelompokkan elemen menjadi balok dan kolom berdasarkan kode desain."""
        geometry_elements = self.data['geometry'].get(
            'elements_mean',
            self.data['geometry']['elements']
        )
        all_element_ids = sorted({
            int(elem[0]) for elem in np.asarray(geometry_elements, dtype=float)
        })

        beam_ids = []
        column_ids = []
        for elem_id in all_element_ids:
            element_code = self._get_element_design_code(elem_id)
            if element_code == 'B':
                beam_ids.append(int(elem_id))
            elif element_code == 'K':
                column_ids.append(int(elem_id))

        return {
            'beam': beam_ids,
            'column': column_ids
        }

    @staticmethod
    def _evaluate_subsystem_failure(failed_elements: set[int],
                                    member_ids: set[int],
                                    system_type: str) -> Optional[bool]:
        """Evaluasi kegagalan subsistem seri/paralel untuk satu simulasi."""
        if not member_ids:
            return None

        normalized_type = str(system_type).strip().lower()
        failed_members = member_ids.intersection(failed_elements)

        if normalized_type == 'series':
            return bool(failed_members)
        if normalized_type == 'parallel':
            return member_ids.issubset(failed_elements)

        raise ValueError(f"Tipe sistem tidak dikenali: {system_type}")

    def _get_system_reliability_status(self,
                                       failed: Optional[bool] = None,
                                       beta: Optional[float] = None,
                                       target_beta: float = 3.0) -> str:
        """Status subsistem/sistem untuk mode probabilistik atau deterministik."""
        if self.is_probabilistic:
            if beta is None or np.isnan(beta):
                return '-'
            return 'SAFE' if float(beta) >= float(target_beta) else 'UNSAFE'

        if failed is None:
            return '-'
        return 'SAFE' if not failed else 'UNSAFE'

    def _build_portal_system_reliability_results(self) -> list[Dict[str, Any]]:
        """Hitung reliabilitas sistem portal gabungan berbasis grup balok/kolom."""
        if not self.mc_results:
            return []

        group_ids = self._get_element_ids_by_design_group()
        beam_ids = set(group_ids['beam'])
        column_ids = set(group_ids['column'])
        if not beam_ids and not column_ids:
            return []

        analysis_history = list(self.mc_results.get('max_forces_history', []) or [])
        if not analysis_history and self.latest_simulation_result is not None:
            analysis_history = [self.latest_simulation_result]

        num_samples = int(self.mc_results.get('num_simulations', len(analysis_history) or 1))
        system_cases = [
            {
                'case_name': 'Kasus 1',
                'description': 'Balok Paralel + Kolom Seri',
                'beam_system': 'Paralel',
                'beam_mode': 'parallel',
                'column_system': 'Seri',
                'column_mode': 'series',
                'portal_failure_rule': 'Portal gagal jika semua balok gagal ATAU ada satu kolom gagal'
            },
            {
                'case_name': 'Kasus 2',
                'description': 'Balok Seri + Kolom Seri',
                'beam_system': 'Seri',
                'beam_mode': 'series',
                'column_system': 'Seri',
                'column_mode': 'series',
                'portal_failure_rule': 'Portal gagal jika ada satu balok gagal ATAU ada satu kolom gagal'
            }
        ]

        results = []
        for case in system_cases:
            beam_failures = 0
            column_failures = 0
            portal_failures = 0
            latest_beam_failed = None
            latest_column_failed = None
            latest_portal_failed = None

            for analysis_result in analysis_history:
                failure_details = self._evaluate_analysis_result_failure(analysis_result)
                failed_elements = {
                    int(elem_id) for elem_id in failure_details.get('failed_elements', [])
                }
                beam_failed = self._evaluate_subsystem_failure(
                    failed_elements,
                    beam_ids,
                    case['beam_mode']
                )
                column_failed = self._evaluate_subsystem_failure(
                    failed_elements,
                    column_ids,
                    case['column_mode']
                )

                component_failures = [
                    flag for flag in (beam_failed, column_failed)
                    if flag is not None
                ]
                portal_failed = any(component_failures) if component_failures else None

                beam_failures += int(bool(beam_failed))
                column_failures += int(bool(column_failed))
                portal_failures += int(bool(portal_failed))

                latest_beam_failed = beam_failed
                latest_column_failed = column_failed
                latest_portal_failed = portal_failed

            if self.is_probabilistic:
                beam_pf, beam_beta = MonteCarloAnalysis.calculate_pf_and_beta(
                    beam_failures,
                    num_samples
                )
                column_pf, column_beta = MonteCarloAnalysis.calculate_pf_and_beta(
                    column_failures,
                    num_samples
                )
                portal_pf, portal_beta = MonteCarloAnalysis.calculate_pf_and_beta(
                    portal_failures,
                    num_samples
                )
            else:
                beam_pf = beam_beta = None
                column_pf = column_beta = None
                portal_pf = portal_beta = None

            results.append({
                'case_name': case['case_name'],
                'description': case['description'],
                'beam_system': case['beam_system'],
                'column_system': case['column_system'],
                'portal_failure_rule': case['portal_failure_rule'],
                'num_beams': len(beam_ids),
                'num_columns': len(column_ids),
                'beam_failures': beam_failures,
                'beam_pf': beam_pf,
                'beam_beta': beam_beta,
                'beam_status': self._get_system_reliability_status(
                    failed=latest_beam_failed,
                    beta=beam_beta
                ),
                'column_failures': column_failures,
                'column_pf': column_pf,
                'column_beta': column_beta,
                'column_status': self._get_system_reliability_status(
                    failed=latest_column_failed,
                    beta=column_beta
                ),
                'portal_failures': portal_failures,
                'portal_pf': portal_pf,
                'portal_beta': portal_beta,
                'portal_status': self._get_system_reliability_status(
                    failed=latest_portal_failed,
                    beta=portal_beta
                )
            })

        return results

    def get_portal_system_reliability_results(self) -> list[Dict[str, Any]]:
        """Ambil hasil reliabilitas sistem portal gabungan dari cache/lazy build."""
        if not self.portal_system_reliability:
            self.portal_system_reliability = self._build_portal_system_reliability_results()
        return list(self.portal_system_reliability)

    def _to_serializable(self, value: Any) -> Any:
        """Konversi numpy/scalar non-JSON menjadi tipe Python standar."""
        if isinstance(value, dict):
            return {
                str(key): self._to_serializable(val)
                for key, val in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(item) for item in value]
        if isinstance(value, np.ndarray):
            return self._to_serializable(value.tolist())
        if isinstance(value, (np.floating, np.integer)):
            return self._to_serializable(value.item())
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, float):
            if np.isnan(value):
                return None
            if np.isposinf(value):
                return "Infinity"
            if np.isneginf(value):
                return "-Infinity"
            return value
        return value
    
    def analysis_function(self, random_samples: Dict,
                         include_section_samples: bool = False) -> Dict:
        """
        Function untuk análisis struktural dengan sampled variables
        Digunakan dalam Monte Carlo simulation
        
        Parameters:
        - random_samples: dict dengan sampled values untuk setiap random variable
        
        Returns:
        - dict dengan hasil analisis untuk stochastic case
        """
        dead_load_values = self._get_element_values_from_sample(
            random_samples,
            self.data['dead_load'].get('by_element', {}),
            prefix='qDL',
            fallback_key='mean'
        )
        dead_load_dict = {'values': dead_load_values}

        live_load_values = self._get_element_values_from_sample(
            random_samples,
            self.data['live_load'].get('by_element', {}),
            prefix='qLL',
            fallback_key='mean'
        )
        live_load_dict = {'values': live_load_values}
        
        # Nodal loads (deterministik)
        nodal_loads = self.data['nodal_loads']
        
        # Run analisis
        try:
            results = self.analysis.analyze(
                dead_load_dict,
                live_load_dict,
                nodal_loads,
                include_section_samples=include_section_samples
            )

            return self._assemble_analysis_output(results, random_samples)
        
        except Exception as e:
            print(f"Warning: Analysis failed - {e}")
            return None

    def run_deterministic_analysis(self):
        """Jalankan satu kali analisis dengan input deterministik Excel."""
        print("\n[3/5] Running deterministic structural analysis...")

        reference_sample = self._build_reference_sample()
        dead_load_dict = self._build_mean_load_dict(
            self.data['dead_load'],
            value_key='deterministic'
        )
        live_load_dict = self._build_mean_load_dict(
            self.data['live_load'],
            value_key='deterministic'
        )
        nodal_loads = self.data['nodal_loads']

        results = self.analysis.analyze(
            dead_load_dict,
            live_load_dict,
            nodal_loads,
            include_section_samples=True
        )
        deterministic_result = self._assemble_analysis_output(
            results,
            reference_sample
        )

        self.random_variables = {}
        self.sensitivity_results = {}
        self.latest_random_sample = reference_sample
        self.latest_simulation_result = deterministic_result
        self.latest_simulation_index = 0

        is_safe = self._is_analysis_result_safe(deterministic_result)
        self.mc_results = {
            'num_simulations': 1,
            'failures': 0 if is_safe is not False else 1,
            'analysis_failures': 0,
            'failure_indices': [] if is_safe is not False else [0],
            'Pf': None,
            'Beta': None,
            'max_forces_history': [deterministic_result],
            'random_samples_history': [reference_sample],
            'element_reliability': {}
        }
        self.portal_system_reliability = self._build_portal_system_reliability_results()

        print("  [OK] Deterministic analysis complete")
        print(
            "  [OK] Input deterministik: "
            f"fc'={self._get_group_sample_summary(reference_sample, self.data['concrete'].get('by_element', {}), 'fc', 'deterministic', 'MPa')}, "
            f"fy_tarik={self._get_group_sample_summary(reference_sample, self.data['steel'].get('by_element', {}), 'fy_tarik', 'tarik_deterministic', 'MPa')}, "
            f"fy_tekan={self._get_group_sample_summary(reference_sample, self.data['steel'].get('by_element', {}), 'fy_tekan', 'tekan_deterministic', 'MPa')}, "
            f"fy_geser={self._get_group_sample_summary(reference_sample, self.data['steel'].get('by_element', {}), 'fy_geser', 'geser_deterministic', 'MPa')}, "
            f"qDL={self._get_group_sample_summary(reference_sample, self.data['dead_load'].get('by_element', {}), 'qDL', 'deterministic', 'kN/m')}, "
            f"qLL={self._get_group_sample_summary(reference_sample, self.data['live_load'].get('by_element', {}), 'qLL', 'deterministic', 'kN/m')}"
        )
        min_g_moment = self._get_min_performance_value(
            deterministic_result,
            performance_key='performance'
        )
        min_g_shear = self._get_min_performance_value(
            deterministic_result,
            performance_key='performance_shear'
        )
        min_g_axial = self._get_min_performance_value(
            deterministic_result,
            performance_key='performance_axial'
        )
        min_g_axial_moment = self._get_min_performance_value(
            deterministic_result,
            performance_key='performance_axial_moment'
        )
        if min_g_moment is not None:
            print(f"  [OK] Minimum g momen: {min_g_moment:.4f} kN.m")
        if min_g_shear is not None:
            print(f"  [OK] Minimum g geser: {min_g_shear:.4f} kN")
        if min_g_axial is not None:
            print(f"  [OK] Minimum g aksial: {min_g_axial:.4f} kN")
        if min_g_axial_moment is not None:
            print(f"  [OK] Minimum g aksial+momen kolom: {min_g_axial_moment:.4f} (-)")
        print(f"  [OK] Status: {'SAFE' if is_safe else 'UNSAFE'}")

        return deterministic_result
    
    def run_monte_carlo(self):
        """Jalankan simulasi Monte Carlo"""
        print(f"\n[4/6] Running Monte Carlo Simulation ({self.num_mc_simulations} samples)...")
        
        def performance_function(analysis_result):
            """Bangun detail failure global dan per elemen per sampel."""
            return self._evaluate_analysis_result_failure(analysis_result)
        
        mc = MonteCarloAnalysis(self.num_mc_simulations)
        self.mc_results = mc.run_simulation(
            self.analysis_function,
            self.random_variables,
            performance_function
        )

        latest_simulation = self.get_latest_simulation_data()
        self.latest_simulation_index = latest_simulation.get('display_index')
        self.latest_random_sample = latest_simulation.get('random_sample')

        if self.latest_random_sample is not None:
            detailed_result = self.analysis_function(
                self.latest_random_sample,
                include_section_samples=True
            )
            if detailed_result is not None:
                self.latest_simulation_result = detailed_result
                if self.latest_simulation_index is not None:
                    self.mc_results['max_forces_history'][self.latest_simulation_index] = detailed_result
            else:
                self.latest_simulation_result = latest_simulation.get('analysis_result')
        else:
            self.latest_simulation_result = latest_simulation.get('analysis_result')
        self.portal_system_reliability = self._build_portal_system_reliability_results()
        
        print(f"  [OK] Simulation complete")
        print(f"  [OK] Failures: {self.mc_results['failures']} out of {self.num_mc_simulations}")
        print(f"  [OK] Pf (Probability of Failure): {self.mc_results['Pf']:.6f}")
        print(f"  [OK] Beta (Reliability Index): {self.mc_results['Beta']:.4f}")
        if self.mc_results.get('analysis_failures'):
            print(f"  [OK] Analysis execution failures: {self.mc_results['analysis_failures']}")

        top_elements = self._get_top_element_reliability('overall', top_n=3)
        if top_elements:
            print("  [OK] Critical elements by Pf:")
            for rank, (elem_id, values) in enumerate(top_elements, 1):
                print(
                    f"    {rank}. E{elem_id}: "
                    f"Pf={values['Pf']:.6f}, Beta={values['Beta']:.4f}, "
                    f"failures={values['failures']}"
                )
        
        return self.mc_results
    
    def _legacy_reliability_analysis(self):
        """Lakukan analisis keandalan"""
        print("\n[5/6] Analyzing reliability...")
        
        self.reliability_assessment = ReliabilityAssessment(self.mc_results)
        
        safety_class = self.reliability_assessment.get_safety_class()
        is_safe = self.reliability_assessment.is_safe('ultimate')
        
        print(f"  [OK] Safety Classification: {safety_class}")
        print(f"  [OK] Safety Status: {'SAFE' if is_safe else 'UNSAFE'}")
        
        # Sensitivity analysis
        var_names = list(self.random_variables.keys())
        sensitivities = SensitivityAnalysis.rank_variables(
            self.mc_results, var_names)
        self.sensitivity_results = sensitivities
        
        print(f"  [OK] Sensitivity Analysis:")
        for idx, (var_name, sens_data) in enumerate(sensitivities.items(), 1):
            print(f"    {idx}. {var_name}: {sens_data['sensitivity_index']:.4f}")
        
        return self.reliability_assessment, sensitivities
    
    def _legacy_generate_report(self):
        """Generate laporan lengkap"""
        print("\n[6/6] Generating report...")
        
        report = self.reliability_assessment.get_report()
        
        # Additional statistics
        report += "\nMaterial Properties Statistics:\n"
        report += f"  Concrete: μ={self.random_variables['fc']['mean']:.2f} MPa, "
        report += f"σ={self.random_variables['fc']['stddev']:.2f} MPa\n"
        report += f"  Steel (Tarik): μ={self.random_variables['fy_tarik']['mean']:.2f} MPa, "
        report += f"σ={self.random_variables['fy_tarik']['stddev']:.2f} MPa\n"
        
        report += "\nLoad Statistics:\n"
        report += f"  Dead Load: μ={self.random_variables['dead_load']['mean']:.2f} kN/m, "
        report += f"σ={self.random_variables['dead_load']['stddev']:.2f} kN/m\n"
        report += f"  Live Load: μ={self.random_variables['live_load']['mean']:.2f} kN/m, "
        report += f"σ={self.random_variables['live_load']['stddev']:.2f} kN/m\n"
        
        self.output_data['report'] = report
        
        print("  [OK] Report generated")
        
        return report
    
    def _legacy_save_results(self, output_dir: str = 'output'):
        """Simpan semua hasil ke file"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save report
        report_file = f'{output_dir}/reliability_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self.output_data['report'])
        
        print(f"\n[Report saved to {report_file}]")
        
        # Save numerical results
        results_file = f'{output_dir}/analysis_results.json'

        json_results = self._to_serializable({
            'timestamp': datetime.now().isoformat(),
            'num_simulations': self.mc_results['num_simulations'],
            'failures': self.mc_results['failures'],
            'Pf': self.mc_results['Pf'],
            'Beta': self.mc_results['Beta'],
            'safety_class': self.reliability_assessment.get_safety_class(),
            'is_safe': self.reliability_assessment.is_safe('ultimate'),
            'random_variables': self.random_variables,
            'sensitivity_results': self.sensitivity_results,
            'input_data': {
                'geometry': self.data['geometry'],
                'boundary': self.data['boundary'],
                'nodal_loads': self.data['nodal_loads'],
                'dead_load': self.data['dead_load'],
                'live_load': self.data['live_load']
            },
            'latest_simulation': self.get_latest_simulation_data()
        })

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=4)
        
        print(f"[Results saved to {results_file}]")
        
        # Try to generate plots if matplotlib available
        try:
            print("\nGenerating plots...")
            ReliabilityPlotter.save_all_plots(
                output_dir,
                self.mc_results,
                self.sensitivity_results
            )

            latest = self.get_latest_simulation_data()
            latest_result = latest.get('analysis_result')
            if latest_result:
                deformed_fig, _ = PortalPlotter.plot_deformed_shape(
                    self.data['geometry']['nodes'].astype(float),
                    self.portal.elements,
                    latest_result['displacements']
                )
                deformed_fig.savefig(
                    f'{output_dir}/last_simulation_deformation.png',
                    dpi=150,
                    bbox_inches='tight'
                )
                deformed_fig.clf()

                for force_type in ('axial', 'shear', 'moment'):
                    force_fig, _ = PortalPlotter.plot_internal_force_diagram(
                        self.portal.elements,
                        latest_result['element_forces'],
                        force_type=force_type,
                        relative_to_chord=False
                    )
                    force_fig.savefig(
                        f'{output_dir}/last_simulation_{force_type}.png',
                        dpi=150,
                        bbox_inches='tight'
                    )
                    force_fig.clf()

                beam_moment_fig, _ = PortalPlotter.plot_beam_moment_profiles(
                    self.portal.elements,
                    latest_result['element_forces'],
                    relative_to_chord=True
                )
                beam_moment_fig.savefig(
                    f'{output_dir}/last_simulation_beam_moment_curvature.png',
                    dpi=150,
                    bbox_inches='tight'
                )
                beam_moment_fig.clf()
        except Exception as e:
            print(f"  Note: Could not generate plots ({e})")

    def reliability_analysis(self):
        """Lakukan analisis keandalan."""
        if not self.is_probabilistic:
            self.reliability_assessment = None
            self.sensitivity_results = {}
            return None, {}

        print("\n[5/6] Analyzing reliability...")

        self.reliability_assessment = ReliabilityAssessment(self.mc_results)

        safety_class = self.reliability_assessment.get_safety_class()
        is_safe = self.reliability_assessment.is_safe('ultimate')

        print(f"  [OK] Safety Classification: {safety_class}")
        print(f"  [OK] Safety Status: {'SAFE' if is_safe else 'UNSAFE'}")

        var_names = list(self.random_variables.keys())
        sensitivities = SensitivityAnalysis.rank_variables(
            self.mc_results, var_names)
        self.sensitivity_results = sensitivities

        print("  [OK] Sensitivity Analysis:")
        for idx, (var_name, sens_data) in enumerate(sensitivities.items(), 1):
            print(f"    {idx}. {var_name}: {sens_data['sensitivity_index']:.4f}")

        return self.reliability_assessment, sensitivities

    def generate_report(self):
        """Generate laporan lengkap."""
        step_label = "[6/6]" if self.is_probabilistic else "[4/5]"
        print(f"\n{step_label} Generating report...")

        if self.is_probabilistic:
            report = self.reliability_assessment.get_report()
            latest_result = self.get_latest_simulation_data().get('analysis_result')
            min_g_moment = self._get_min_performance_value(
                latest_result,
                performance_key='performance'
            )
            min_g_shear = self._get_min_performance_value(
                latest_result,
                performance_key='performance_shear'
            )
            min_g_axial = self._get_min_performance_value(
                latest_result,
                performance_key='performance_axial'
            )
            min_g_axial_moment = self._get_min_performance_value(
                latest_result,
                performance_key='performance_axial_moment'
            )

            report += "\nMaterial Properties Statistics:\n"
            report += f"  Concrete per elemen: {self._summarize_random_variable_group('fc_', 'MPa')}\n"
            report += f"  Steel tarik per elemen: {self._summarize_random_variable_group('fy_tarik_', 'MPa')}\n"
            report += f"  Steel tekan per elemen: {self._summarize_random_variable_group('fy_tekan_', 'MPa')}\n"
            report += f"  Steel geser per elemen: {self._summarize_random_variable_group('fy_geser_', 'MPa')}\n"

            report += "\nLoad Statistics:\n"
            report += f"  Dead Load per elemen: {self._summarize_random_variable_group('qDL_', 'kN/m')}\n"
            report += f"  Live Load per elemen: {self._summarize_random_variable_group('qLL_', 'kN/m')}\n"

            report += "\nLatest Simulation Limit States:\n"
            report += (
                f"  - Minimum g momen: {min_g_moment:.4f} kN.m\n"
                if min_g_moment is not None else
                "  - Minimum g momen: -\n"
            )
            report += (
                f"  - Minimum g geser: {min_g_shear:.4f} kN\n"
                if min_g_shear is not None else
                "  - Minimum g geser: -\n"
            )
            report += (
                f"  - Minimum g aksial: {min_g_axial:.4f} kN\n"
                if min_g_axial is not None else
                "  - Minimum g aksial: -\n"
            )
            report += (
                f"  - Minimum g aksial+momen kolom: {min_g_axial_moment:.4f} (-)\n"
                if min_g_axial_moment is not None else
                "  - Minimum g aksial+momen kolom: -\n"
            )

            report += "\nCritical Elements by Overall Reliability:\n"
            top_elements = self._get_top_element_reliability('overall', top_n=5)
            if top_elements:
                for elem_id, values in top_elements:
                    report += (
                        f"  - E{elem_id}: Pf={values['Pf']:.6f}, "
                        f"Beta={values['Beta']:.4f}, failures={values['failures']}\n"
                    )
            else:
                report += "  - Data reliability per elemen belum tersedia\n"
        else:
            latest_result = self.get_latest_simulation_data().get('analysis_result')
            min_g_moment = self._get_min_performance_value(
                latest_result,
                performance_key='performance'
            )
            min_g_shear = self._get_min_performance_value(
                latest_result,
                performance_key='performance_shear'
            )
            min_g_axial = self._get_min_performance_value(
                latest_result,
                performance_key='performance_axial'
            )
            min_g_axial_moment = self._get_min_performance_value(
                latest_result,
                performance_key='performance_axial_moment'
            )
            is_safe = self._is_analysis_result_safe(latest_result)
            reference_sample = self.latest_random_sample or self._build_reference_sample()
            min_g_moment_text = f"{min_g_moment:.4f}" if min_g_moment is not None else "-"
            min_g_shear_text = f"{min_g_shear:.4f}" if min_g_shear is not None else "-"
            min_g_axial_text = f"{min_g_axial:.4f}" if min_g_axial is not None else "-"
            min_g_axial_moment_text = (
                f"{min_g_axial_moment:.4f}"
                if min_g_axial_moment is not None else
                "-"
            )
            status_text = "-" if is_safe is None else ("SAFE" if is_safe else "UNSAFE")

            report = f"""
{'='*60}
DETERMINISTIC STRUCTURAL ANALYSIS REPORT
{'='*60}

Analysis Basis:
  - Analysis mode: Deterministic
  - Structural response computed once using deterministic input values per elemen
  - Probability of Failure (Pf): N/A
  - Reliability Index (Beta): N/A

Results:
  - Number of analyses: 1
  - Minimum g momen (Capacity - Demand): {min_g_moment_text} kN.m
  - Minimum g geser (Capacity - Demand): {min_g_shear_text} kN
  - Minimum g aksial (Capacity - Demand): {min_g_axial_text} kN
  - Minimum g aksial+momen kolom: {min_g_axial_moment_text} (-)
  - Status: {status_text}

Reference Deterministic Values:
  - Concrete strength fc per elemen: {self._get_group_sample_summary(reference_sample, self.data['concrete'].get('by_element', {}), 'fc', 'deterministic', 'MPa')}
  - Steel yield fy tarik per elemen: {self._get_group_sample_summary(reference_sample, self.data['steel'].get('by_element', {}), 'fy_tarik', 'tarik_deterministic', 'MPa')}
  - Steel yield fy tekan per elemen: {self._get_group_sample_summary(reference_sample, self.data['steel'].get('by_element', {}), 'fy_tekan', 'tekan_deterministic', 'MPa')}
  - Steel yield fy geser per elemen: {self._get_group_sample_summary(reference_sample, self.data['steel'].get('by_element', {}), 'fy_geser', 'geser_deterministic', 'MPa')}
  - Dead load per elemen: {self._get_group_sample_summary(reference_sample, self.data['dead_load'].get('by_element', {}), 'qDL', 'deterministic', 'kN/m')}
  - Live load per elemen: {self._get_group_sample_summary(reference_sample, self.data['live_load'].get('by_element', {}), 'qLL', 'deterministic', 'kN/m')}

Interpretation:
  - Deterministic mode does not perform random sampling.
  - Safety status is based on whether all element g momen, g geser, g aksial, dan g aksial+momen kolom remain non-negative.

{'='*60}
"""

        self.output_data['report'] = report

        print("  [OK] Report generated")

        return report

    def save_results(self, output_dir: str = 'output'):
        """Simpan semua hasil ke file."""
        os.makedirs(output_dir, exist_ok=True)

        report_file = f'{output_dir}/reliability_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self.output_data['report'])

        print(f"\n[Report saved to {report_file}]")

        results_file = f'{output_dir}/analysis_results.json'
        latest_result = self.get_latest_simulation_data().get('analysis_result')
        is_safe = (
            self.reliability_assessment.is_safe('ultimate')
            if self.reliability_assessment else
            self._is_analysis_result_safe(latest_result)
        )
        safety_class = (
            self.reliability_assessment.get_safety_class()
            if self.reliability_assessment else
            ('Deterministik' if latest_result is not None else None)
        )

        json_results = self._to_serializable({
            'timestamp': datetime.now().isoformat(),
            'analysis_mode': self.analysis_mode,
            'analysis_mode_label': self.get_analysis_mode_label(),
            'num_simulations': self.mc_results['num_simulations'],
            'failures': self.mc_results['failures'],
            'analysis_failures': self.mc_results.get('analysis_failures', 0),
            'failure_indices': self.mc_results.get('failure_indices', []),
            'Pf': self.mc_results['Pf'],
            'Beta': self.mc_results['Beta'],
            'safety_class': safety_class,
            'is_safe': is_safe,
            'random_variables': self.random_variables,
            'sensitivity_results': self.sensitivity_results,
            'element_reliability': self.mc_results.get('element_reliability', {}),
            'portal_system_reliability': self.get_portal_system_reliability_results(),
            'input_data': {
                'geometry': self.data['geometry'],
                'boundary': self.data['boundary'],
                'nodal_loads': self.data['nodal_loads'],
                'dead_load': self.data['dead_load'],
                'live_load': self.data['live_load']
            },
            'latest_simulation': self.get_latest_simulation_data()
        })

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=4)

        print(f"[Results saved to {results_file}]")

        try:
            print("\nGenerating plots...")
            if self.is_probabilistic:
                ReliabilityPlotter.save_all_plots(
                    output_dir,
                    self.mc_results,
                    self.sensitivity_results
                )

            latest = self.get_latest_simulation_data()
            latest_result = latest.get('analysis_result')
            if latest_result:
                deformed_fig, _ = PortalPlotter.plot_deformed_shape(
                    self.data['geometry']['nodes'].astype(float),
                    self.portal.elements,
                    latest_result['displacements']
                )
                deformed_fig.savefig(
                    f'{output_dir}/last_simulation_deformation.png',
                    dpi=150,
                    bbox_inches='tight'
                )
                deformed_fig.clf()

                for force_type in ('axial', 'shear', 'moment'):
                    force_fig, _ = PortalPlotter.plot_internal_force_diagram(
                        self.portal.elements,
                        latest_result['element_forces'],
                        force_type=force_type,
                        relative_to_chord=False
                    )
                    force_fig.savefig(
                        f'{output_dir}/last_simulation_{force_type}.png',
                        dpi=150,
                        bbox_inches='tight'
                    )
                    force_fig.clf()

                beam_moment_fig, _ = PortalPlotter.plot_beam_moment_profiles(
                    self.portal.elements,
                    latest_result['element_forces'],
                    relative_to_chord=True
                )
                beam_moment_fig.savefig(
                    f'{output_dir}/last_simulation_beam_moment_curvature.png',
                    dpi=150,
                    bbox_inches='tight'
                )
                beam_moment_fig.clf()
        except Exception as e:
            print(f"  Note: Could not generate plots ({e})")

    def run_full_analysis(self):
        """Jalankan analisis lengkap dari awal hingga akhir"""
        try:
            self.read_input()
            self.initialize_portal()
            if self.is_probabilistic:
                self.setup_monte_carlo()
                self.run_monte_carlo()
                self.reliability_analysis()
            else:
                self.run_deterministic_analysis()
            self.generate_report()
            self.save_results()
            
            print("\n" + "="*60)
            print("ANALYSIS COMPLETED SUCCESSFULLY")
            print("="*60 + "\n")
            
            return True
        
        except Exception as e:
            print(f"\n[ERROR] Analysis failed: {e}")
            return False


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_excel_input> [deterministic|probabilistic]")
        print("\nExample: python main.py input_data.xlsx deterministic")
        sys.exit(1)
    
    excel_file = sys.argv[1]
    analysis_mode = sys.argv[2] if len(sys.argv) >= 3 else 'probabilistic'
    
    # Check if file exists
    if not os.path.exists(excel_file):
        print(f"Error: File '{excel_file}' not found")
        sys.exit(1)
    
    # Run analysis
    analysis = PortalReliabilityAnalysis(
        excel_file,
        num_mc_simulations=10000,
        analysis_mode=analysis_mode
    )
    success = analysis.run_full_analysis()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
