"""
Modul untuk analisis keandalan lengkap (FORM, SORM, dan hasil)
"""
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from scipy import stats
import warnings

RELIABILITY_PHI_FACTOR = 1.0
AXIAL_DEMAND_TOLERANCE_KN = 1e-6


class PerformanceFunction:
    """Generator untuk limit state function (performance function)"""

    @staticmethod
    def _get_input_context(section_geometry: Optional[Dict] = None,
                           steel_area: Optional[Dict] = None) -> str:
        """Bangun label konteks elemen untuk pesan error."""
        for source in (section_geometry, steel_area):
            if not isinstance(source, dict):
                continue
            raw_elem_id = source.get('element_id')
            if raw_elem_id is None:
                continue
            try:
                return f"elemen {int(raw_elem_id)}"
            except (TypeError, ValueError):
                return f"elemen {raw_elem_id}"
        return "elemen tidak diketahui"

    @staticmethod
    def _read_positive_input(value,
                             label: str,
                             missing_fields: List[str]) -> float:
        """Baca input numerik positif, atau tandai sebagai kurang/invalid."""
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            missing_fields.append(label)
            return 0.0

        if not np.isfinite(numeric_value) or numeric_value <= 0.0:
            missing_fields.append(label)
            return 0.0
        return float(numeric_value)

    @staticmethod
    def _read_nonnegative_input(value) -> float:
        """Baca input numerik non-negatif, fallback ke 0 bila invalid."""
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return 0.0

        if not np.isfinite(numeric_value) or numeric_value < 0.0:
            return 0.0
        return float(numeric_value)

    @staticmethod
    def _raise_missing_input_error(limit_state_label: str,
                                   missing_fields: List[str],
                                   section_geometry: Optional[Dict] = None,
                                   steel_area: Optional[Dict] = None) -> None:
        """Raise ValueError dengan daftar data input yang kurang."""
        context = PerformanceFunction._get_input_context(
            section_geometry,
            steel_area
        )
        unique_fields = list(dict.fromkeys(missing_fields))
        fields_text = ", ".join(unique_fields)
        raise ValueError(
            f"Data input tidak lengkap atau tidak valid untuk {limit_state_label} "
            f"pada {context}: {fields_text}. "
            "Lengkapi data pada sheet Geometri, Tulangan, Mutu_Beton, atau Mutu_Baja."
        )

    @staticmethod
    def _get_section_design_inputs(fc: float,
                                   fy_tarik: float,
                                   section_geometry: Dict,
                                   steel_area: Dict,
                                   fy_tekan: float = None,
                                   limit_state_label: str = "kapasitas penampang",
                                   require_tension_steel: bool = True
                                   ) -> Dict[str, float]:
        """Validasi dan ekstrak input desain penampang dari hasil baca Excel."""
        missing_fields: List[str] = []

        b = PerformanceFunction._read_positive_input(
            section_geometry.get('b'),
            'b (sheet Geometri)',
            missing_fields
        )
        h = PerformanceFunction._read_positive_input(
            section_geometry.get('h'),
            'h (sheet Geometri)',
            missing_fields
        )
        d = PerformanceFunction._read_positive_input(
            section_geometry.get('d'),
            'd_tarik atau kombinasi ds_tarik + du_geser + du_tarik (sheet Tulangan)',
            missing_fields
        )
        As = PerformanceFunction._read_positive_input(
            steel_area.get('As'),
            'As_tarik atau kombinasi n_tarik + du_tarik (sheet Tulangan)',
            missing_fields
        ) if require_tension_steel else PerformanceFunction._read_nonnegative_input(
            steel_area.get('As')
        )

        d_prime = PerformanceFunction._read_nonnegative_input(
            section_geometry.get('d_prime')
            if section_geometry.get('d_prime') is not None else
            steel_area.get('d_prime')
        )
        As_prime = PerformanceFunction._read_nonnegative_input(
            steel_area.get('As_prime')
        )

        if As_prime > 0.0 and d_prime <= 0.0:
            missing_fields.append(
                'd_tekan atau kombinasi ds_tekan + du_geser + du_tekan (sheet Tulangan)'
            )

        fc_value = PerformanceFunction._read_positive_input(
            fc,
            "fc' / Mean (sheet Mutu_Beton)",
            missing_fields
        )
        fy_tension = PerformanceFunction._read_positive_input(
            fy_tarik,
            'fy_tarik / Mean_tarik (sheet Mutu_Baja)',
            missing_fields
        )

        if fy_tekan is None:
            fy_compression = fy_tension
        else:
            fy_compression = PerformanceFunction._read_positive_input(
                fy_tekan,
                'fy_tekan / Mean_tekan (sheet Mutu_Baja)',
                missing_fields
            )

        if missing_fields:
            PerformanceFunction._raise_missing_input_error(
                limit_state_label,
                missing_fields,
                section_geometry,
                steel_area
            )

        return {
            'b': b,
            'h': h,
            'd': d,
            'd_prime': d_prime,
            'As': As,
            'As_prime': As_prime,
            'fc_value': fc_value,
            'fy_tension': fy_tension,
            'fy_compression': fy_compression
        }

    @staticmethod
    def _get_beta1(fc: float) -> float:
        """
        Nilai beta1 sesuai SNI 2847:2019 Tabel 22.2.2.4.3.
        """
        fc_value = float(fc)
        if fc_value <= 28.0:
            return 0.85
        if fc_value >= 55.0:
            return 0.65
        return float(0.85 - 0.05 * ((fc_value - 28.0) / 7.0))

    @staticmethod
    def _get_tension_control_limits(fy: float,
                                    steel_modulus: float = 200000.0) -> Tuple[float, float]:
        """
        Ambil batas regangan netto untuk klasifikasi penampang lentur.

        Returns:
        - (epsilon_ty, epsilon_tension_controlled)
        """
        fy_value = abs(float(fy))
        epsilon_ty = fy_value / max(float(steel_modulus), 1e-9)
        if abs(fy_value - 420.0) <= 1e-9:
            epsilon_ty = 0.002
        return float(epsilon_ty), 0.005

    @staticmethod
    def _solve_interaction_ray_segment(demand_axial: float,
                                       demand_moment: float,
                                       p1: float,
                                       m1: float,
                                       p2: float,
                                       m2: float,
                                       tolerance: float = 1e-9
                                       ) -> Optional[Tuple[float, float]]:
        """
        Selesaikan perpotongan sinar dari origin dengan satu segmen kurva interaksi.

        Returns:
        - (lambda_value, segment_ratio) jika ada solusi numerik
        - None jika sistem singular
        """
        delta_p = float(p2) - float(p1)
        delta_m = float(m2) - float(m1)
        determinant = float(demand_moment * delta_p - demand_axial * delta_m)
        if abs(determinant) <= tolerance:
            return None

        lambda_value = (float(m1) * delta_p - float(p1) * delta_m) / determinant
        segment_ratio = (
            float(demand_axial) * float(m1) - float(demand_moment) * float(p1)
        ) / determinant
        return float(lambda_value), float(segment_ratio)

    @staticmethod
    def _get_flexural_phi_nonspiral(epsilon_t_net: float,
                                    fy: float,
                                    steel_modulus: float = 200000.0) -> Tuple[float, str]:
        """
        Faktor reduksi kekuatan phi untuk lentur non-spiral sesuai SNI 2847:2019
        Tabel 21.2.2.
        """
        epsilon_ty, epsilon_tc = PerformanceFunction._get_tension_control_limits(
            fy,
            steel_modulus=steel_modulus
        )
        epsilon_t_net = max(float(epsilon_t_net), 0.0)

        if epsilon_t_net <= epsilon_ty:
            return 0.65, 'compression-controlled'
        if epsilon_t_net >= epsilon_tc:
            return 0.90, 'tension-controlled'

        phi = 0.65 + 0.25 * (
            (epsilon_t_net - epsilon_ty) /
            max(epsilon_tc - epsilon_ty, 1e-9)
        )
        return float(phi), 'transition'

    @staticmethod
    def _get_rebar_stress(strain: float, fy_tension: float,
                          fy_compression: float,
                          steel_modulus: float = 200000.0) -> float:
        """
        Hitung tegangan baja dari regangan, dibatasi leleh tarik dan tekan.

        Konvensi tanda:
        - positif: tekan
        - negatif: tarik
        """
        stress = float(steel_modulus) * float(strain)
        fy_tension = abs(float(fy_tension))
        fy_compression = abs(float(fy_compression))

        if stress >= 0.0:
            return float(min(stress, fy_compression))
        return float(max(stress, -fy_tension))

    @staticmethod
    def _solve_neutral_axis_depth(section_force_resultant,
                                  lower_bound: float,
                                  upper_bound: float,
                                  max_iterations: int = 120,
                                  tolerance: float = 1e-9) -> float:
        """
        Selesaikan kedalaman garis netral dengan metode bisection.
        """
        c_low = max(float(lower_bound), 1e-9)
        c_high = max(float(upper_bound), c_low * 2.0)
        f_low = float(section_force_resultant(c_low))
        f_high = float(section_force_resultant(c_high))

        expansion_count = 0
        while f_low * f_high > 0.0 and expansion_count < 60:
            c_high *= 2.0
            f_high = float(section_force_resultant(c_high))
            expansion_count += 1

        if f_low == 0.0:
            return c_low
        if f_high == 0.0:
            return c_high
        if f_low * f_high > 0.0:
            return c_high

        for _ in range(max_iterations):
            c_mid = 0.5 * (c_low + c_high)
            f_mid = float(section_force_resultant(c_mid))

            if abs(f_mid) <= tolerance:
                return c_mid

            if f_low * f_mid <= 0.0:
                c_high = c_mid
                f_high = f_mid
            else:
                c_low = c_mid
                f_low = f_mid

        return 0.5 * (c_low + c_high)

    @staticmethod
    def _get_beam_flexural_response(fc: float, fy: float,
                                    section_geometry: Dict,
                                    steel_area: Dict,
                                    fy_tekan: float = None) -> Dict[str, float]:
        """
        Hitung respons kapasitas lentur balok berbasis kompatibilitas regangan.

        Asumsi:
        - penampang persegi panjang
        - beton tekan ultimit epsilon_cu = 0.003
        - penampang non-spiral untuk klasifikasi phi lentur
        """
        validated_inputs = PerformanceFunction._get_section_design_inputs(
            fc,
            fy,
            section_geometry,
            steel_area,
            fy_tekan=fy_tekan,
            limit_state_label='kapasitas lentur'
        )
        b = validated_inputs['b']
        h = validated_inputs['h']
        d = validated_inputs['d']
        d_prime = validated_inputs['d_prime']
        As = max(validated_inputs['As'], 0.0)
        As_prime = max(validated_inputs['As_prime'], 0.0)
        fc_value = max(validated_inputs['fc_value'], 1e-9)
        fy_tension = abs(validated_inputs['fy_tension'])
        fy_compression = abs(validated_inputs['fy_compression'])
        steel_modulus = 200000.0
        epsilon_cu = 0.003
        beta1 = PerformanceFunction._get_beta1(fc_value)

        def section_force_resultant(c_value: float) -> float:
            c_value = max(float(c_value), 1e-9)
            a_value = min(beta1 * c_value, h)
            concrete_force = 0.85 * fc_value * b * a_value / 1000.0

            tension_strain = epsilon_cu * (1.0 - d / c_value)
            compression_strain = epsilon_cu * (1.0 - d_prime / c_value)

            tension_stress = PerformanceFunction._get_rebar_stress(
                tension_strain,
                fy_tension=fy_tension,
                fy_compression=fy_compression,
                steel_modulus=steel_modulus
            )
            compression_stress = PerformanceFunction._get_rebar_stress(
                compression_strain,
                fy_tension=fy_tension,
                fy_compression=fy_compression,
                steel_modulus=steel_modulus
            )

            steel_force_tension = tension_stress * As / 1000.0
            steel_force_compression = compression_stress * As_prime / 1000.0

            return float(concrete_force + steel_force_tension + steel_force_compression)

        c = PerformanceFunction._solve_neutral_axis_depth(
            section_force_resultant,
            lower_bound=1e-6,
            upper_bound=max(h, d, d_prime, 1.0)
        )

        a = min(beta1 * c, h)
        concrete_force = 0.85 * fc_value * b * a / 1000.0
        tension_strain = epsilon_cu * (1.0 - d / c)
        compression_strain = epsilon_cu * (1.0 - d_prime / c)
        tension_stress = PerformanceFunction._get_rebar_stress(
            tension_strain,
            fy_tension=fy_tension,
            fy_compression=fy_compression,
            steel_modulus=steel_modulus
        )
        compression_stress = PerformanceFunction._get_rebar_stress(
            compression_strain,
            fy_tension=fy_tension,
            fy_compression=fy_compression,
            steel_modulus=steel_modulus
        )
        steel_force_tension = tension_stress * As / 1000.0
        steel_force_compression = compression_stress * As_prime / 1000.0

        epsilon_t_net = max(-tension_strain, 0.0)
        epsilon_ty, _ = PerformanceFunction._get_tension_control_limits(
            fy_tension,
            steel_modulus=steel_modulus
        )
        phi, classification = PerformanceFunction._get_flexural_phi_nonspiral(
            epsilon_t_net,
            fy_tension,
            steel_modulus=steel_modulus
        )

        moment_internal = (
            concrete_force * (a / 2.0)
            + steel_force_compression * d_prime
            + steel_force_tension * d
        )
        nominal_moment = max(-moment_internal / 1000.0, 0.0)
        design_moment = RELIABILITY_PHI_FACTOR * nominal_moment

        tension_steel_yielded = (
            tension_strain < 0.0 and
            abs(tension_stress) >= fy_tension - 1e-6
        )
        compression_steel_yielded = (
            compression_strain > 0.0 and
            abs(compression_stress) >= fy_compression - 1e-6
        )

        return {
            'beta1': float(beta1),
            'neutral_axis_depth': float(c),
            'compression_block_depth': float(a),
            'epsilon_t_net': float(epsilon_t_net),
            'epsilon_ty': float(epsilon_ty),
            'phi': float(phi),
            'classification': classification,
            'Mn': float(nominal_moment),
            'phi_Mn': float(design_moment),
            'tension_steel_strain': float(tension_strain),
            'tension_steel_stress': float(tension_stress),
            'tension_steel_yielded': bool(tension_steel_yielded),
            'compression_steel_strain': float(compression_strain),
            'compression_steel_stress': float(compression_stress),
            'compression_steel_yielded': bool(compression_steel_yielded)
        }

    @staticmethod
    def _get_moment_capacity(fc: float, fy: float,
                             section_geometry: Dict,
                             steel_area: Dict,
                             fy_tekan: float = None) -> float:
        """Hitung kapasitas momen rencana lentur balok (phiMn) dalam kN.m."""
        response = PerformanceFunction._get_beam_flexural_response(
            fc,
            fy,
            section_geometry,
            steel_area,
            fy_tekan=fy_tekan
        )
        return float(response['phi_Mn'])

    @staticmethod
    def _get_gross_section_area(section_geometry: Dict) -> float:
        """Luas bruto penampang beton (mm2)."""
        b = float(section_geometry.get('b', 0.0) or 0.0)
        h = float(section_geometry.get('h', 0.0) or 0.0)
        return float(
            section_geometry.get('Ag')
            or section_geometry.get('area')
            or (b * h)
            or 0.0
        )

    @staticmethod
    def _get_po(fc: float, fy_tarik: float,
                section_geometry: Dict,
                steel_area: Dict,
                fy_tekan: float = None) -> float:
        """
        Kekuatan aksial nominal konsentris Po (kN) untuk komponen nonprategang.
        """
        gross_area = PerformanceFunction._get_gross_section_area(section_geometry)
        As = float(steel_area.get('As', 0.0) or 0.0)
        As_prime = float(steel_area.get('As_prime', 0.0) or 0.0)
        Ast = max(As + As_prime, 0.0)
        fy_tekan_value = abs(float(fy_tarik if fy_tekan is None else fy_tekan))
        concrete_area = max(gross_area - Ast, 0.0)
        return float(
            (
                0.85 * max(float(fc), 15.0) * concrete_area
                + fy_tekan_value * Ast
            ) / 1000.0
        )

    @staticmethod
    def _get_column_section_response_at_c(fc: float,
                                          fy_tarik: float,
                                          section_geometry: Dict,
                                          steel_area: Dict,
                                          c: float,
                                          fy_tekan: float = None) -> Dict[str, float]:
        """
        Respons penampang kolom pada kedalaman garis netral tertentu.

        Konvensi tanda P:
        - positif = tekan
        - negatif = tarik
        """
        validated_inputs = PerformanceFunction._get_section_design_inputs(
            fc,
            fy_tarik,
            section_geometry,
            steel_area,
            fy_tekan=fy_tekan,
            limit_state_label='kapasitas aksial-lentur kolom'
        )
        b = validated_inputs['b']
        h = validated_inputs['h']
        d = validated_inputs['d']
        d_prime = validated_inputs['d_prime']
        As = max(validated_inputs['As'], 0.0)
        As_prime = max(validated_inputs['As_prime'], 0.0)
        fc_value = max(validated_inputs['fc_value'], 1e-9)
        fy_tension = abs(validated_inputs['fy_tension'])
        fy_compression = abs(validated_inputs['fy_compression'])
        steel_modulus = 200000.0
        epsilon_cu = 0.003
        beta1 = PerformanceFunction._get_beta1(fc_value)
        c = max(float(c), 1e-9)

        a = min(beta1 * c, h)
        concrete_force = 0.85 * fc_value * b * a / 1000.0
        tension_strain = epsilon_cu * (1.0 - d / c)
        compression_strain = epsilon_cu * (1.0 - d_prime / c)
        tension_stress = PerformanceFunction._get_rebar_stress(
            tension_strain,
            fy_tension=fy_tension,
            fy_compression=fy_compression,
            steel_modulus=steel_modulus
        )
        compression_stress = PerformanceFunction._get_rebar_stress(
            compression_strain,
            fy_tension=fy_tension,
            fy_compression=fy_compression,
            steel_modulus=steel_modulus
        )

        steel_force_tension = tension_stress * As / 1000.0
        steel_force_compression = compression_stress * As_prime / 1000.0
        nominal_axial = concrete_force + steel_force_tension + steel_force_compression
        nominal_moment = abs(
            concrete_force * ((a / 2.0) - (h / 2.0))
            + steel_force_compression * (d_prime - (h / 2.0))
            + steel_force_tension * (d - (h / 2.0))
        ) / 1000.0

        epsilon_t_net = max(-tension_strain, 0.0)
        epsilon_ty, _ = PerformanceFunction._get_tension_control_limits(
            fy_tension,
            steel_modulus=steel_modulus
        )
        phi, classification = PerformanceFunction._get_flexural_phi_nonspiral(
            epsilon_t_net,
            fy_tension,
            steel_modulus=steel_modulus
        )

        return {
            'beta1': float(beta1),
            'neutral_axis_depth': float(c),
            'compression_block_depth': float(a),
            'epsilon_t_net': float(epsilon_t_net),
            'epsilon_ty': float(epsilon_ty),
            'phi': float(phi),
            'classification': classification,
            'Pn': float(nominal_axial),
            'Mn': float(nominal_moment),
            'phi_Pn': float(RELIABILITY_PHI_FACTOR * nominal_axial),
            'phi_Mn': float(RELIABILITY_PHI_FACTOR * nominal_moment),
            'tension_steel_strain': float(tension_strain),
            'tension_steel_stress': float(tension_stress),
            'tension_steel_yielded': bool(
                tension_strain < 0.0 and abs(tension_stress) >= fy_tension - 1e-6
            ),
            'compression_steel_strain': float(compression_strain),
            'compression_steel_stress': float(compression_stress),
            'compression_steel_yielded': bool(
                compression_strain > 0.0 and abs(compression_stress) >= fy_compression - 1e-6
            )
        }

    @staticmethod
    def _get_column_section_response_arrays(fc: float,
                                            fy_tarik: float,
                                            section_geometry: Dict,
                                            steel_area: Dict,
                                            c_values: np.ndarray,
                                            fy_tekan: float = None) -> Dict[str, np.ndarray]:
        """
        Versi vectorized dari respons penampang kolom untuk banyak nilai c.
        Dipakai untuk mempercepat pembentukan kurva interaksi.
        """
        validated_inputs = PerformanceFunction._get_section_design_inputs(
            fc,
            fy_tarik,
            section_geometry,
            steel_area,
            fy_tekan=fy_tekan,
            limit_state_label='kurva interaksi aksial-lentur kolom'
        )
        b = validated_inputs['b']
        h = validated_inputs['h']
        d = validated_inputs['d']
        d_prime = validated_inputs['d_prime']
        As = max(validated_inputs['As'], 0.0)
        As_prime = max(validated_inputs['As_prime'], 0.0)
        fc_value = max(validated_inputs['fc_value'], 1e-9)
        fy_tension = abs(validated_inputs['fy_tension'])
        fy_compression = abs(validated_inputs['fy_compression'])
        steel_modulus = 200000.0
        epsilon_cu = 0.003
        beta1 = PerformanceFunction._get_beta1(fc_value)
        epsilon_ty, epsilon_tc = PerformanceFunction._get_tension_control_limits(
            fy_tension,
            steel_modulus=steel_modulus
        )

        c_values = np.maximum(np.asarray(c_values, dtype=float), 1e-9)
        a_values = np.minimum(beta1 * c_values, h)
        concrete_force = 0.85 * fc_value * b * a_values / 1000.0

        tension_strain = epsilon_cu * (1.0 - d / c_values)
        compression_strain = epsilon_cu * (1.0 - d_prime / c_values)

        tension_stress_raw = steel_modulus * tension_strain
        compression_stress_raw = steel_modulus * compression_strain
        tension_stress = np.where(
            tension_stress_raw >= 0.0,
            np.minimum(tension_stress_raw, fy_compression),
            np.maximum(tension_stress_raw, -fy_tension)
        )
        compression_stress = np.where(
            compression_stress_raw >= 0.0,
            np.minimum(compression_stress_raw, fy_compression),
            np.maximum(compression_stress_raw, -fy_tension)
        )

        steel_force_tension = tension_stress * As / 1000.0
        steel_force_compression = compression_stress * As_prime / 1000.0
        nominal_axial = concrete_force + steel_force_tension + steel_force_compression
        nominal_moment = np.abs(
            concrete_force * ((a_values / 2.0) - (h / 2.0))
            + steel_force_compression * (d_prime - (h / 2.0))
            + steel_force_tension * (d - (h / 2.0))
        ) / 1000.0

        epsilon_t_net = np.maximum(-tension_strain, 0.0)
        transition_denominator = max(epsilon_tc - epsilon_ty, 1e-9)
        phi = np.where(
            epsilon_t_net <= epsilon_ty,
            0.65,
            np.where(
                epsilon_t_net >= epsilon_tc,
                0.90,
                0.65 + 0.25 * ((epsilon_t_net - epsilon_ty) / transition_denominator)
            )
        )
        classification = np.where(
            epsilon_t_net <= epsilon_ty,
            'compression-controlled',
            np.where(
                epsilon_t_net >= epsilon_tc,
                'tension-controlled',
                'transition'
            )
        )

        return {
            'neutral_axis_depth': c_values,
            'phi_Pn': RELIABILITY_PHI_FACTOR * nominal_axial,
            'phi_Mn': RELIABILITY_PHI_FACTOR * nominal_moment,
            'phi': phi,
            'epsilon_t_net': epsilon_t_net,
            'epsilon_ty': np.full_like(c_values, epsilon_ty, dtype=float),
            'classification': classification
        }

    @staticmethod
    def _get_column_interaction_curve(fc: float,
                                      fy_tarik: float,
                                      section_geometry: Dict,
                                      steel_area: Dict,
                                      fy_tekan: float = None,
                                      num_points: int = 240) -> List[Dict[str, float]]:
        """
        Bangun kurva interaksi desain phiPn-phiMn untuk kolom beton bertulang.
        """
        validated_inputs = PerformanceFunction._get_section_design_inputs(
            fc,
            fy_tarik,
            section_geometry,
            steel_area,
            fy_tekan=fy_tekan,
            limit_state_label='kurva interaksi aksial-lentur kolom'
        )
        h = validated_inputs['h']
        Po = PerformanceFunction._get_po(
            fc,
            fy_tarik,
            section_geometry,
            steel_area,
            fy_tekan=fy_tekan
        )
        phi_pn_max = RELIABILITY_PHI_FACTOR * Po
        As = max(float(steel_area.get('As', 0.0) or 0.0), 0.0)
        As_prime = max(float(steel_area.get('As_prime', 0.0) or 0.0), 0.0)
        Ast = As + As_prime
        phi_pn_tension = RELIABILITY_PHI_FACTOR * abs(float(fy_tarik)) * Ast / 1000.0
        epsilon_ty = PerformanceFunction._get_tension_control_limits(fy_tarik)[0]

        points: List[Dict[str, float]] = [
            {
                'phi_Pn': float(phi_pn_max),
                'phi_Mn': 0.0,
                'phi': 0.65,
                'epsilon_t_net': 0.0,
                'epsilon_ty': epsilon_ty,
                'classification': 'compression-controlled',
                'neutral_axis_depth': float('inf')
            },
            {
                'phi_Pn': float(-phi_pn_tension),
                'phi_Mn': 0.0,
                'phi': 0.90,
                'epsilon_t_net': max(0.005, epsilon_ty),
                'epsilon_ty': epsilon_ty,
                'classification': 'tension-controlled',
                'neutral_axis_depth': 0.0
            }
        ]

        pure_bending = PerformanceFunction._get_beam_flexural_response(
            fc,
            fy_tarik,
            section_geometry,
            steel_area,
            fy_tekan=fy_tekan
        )
        points.append({
            'phi_Pn': 0.0,
            'phi_Mn': float(pure_bending['phi_Mn']),
            'phi': float(pure_bending['phi']),
            'epsilon_t_net': float(pure_bending['epsilon_t_net']),
            'epsilon_ty': float(pure_bending['epsilon_ty']),
            'classification': str(pure_bending['classification']),
            'neutral_axis_depth': float(pure_bending['neutral_axis_depth'])
        })

        c_values = np.geomspace(
            max(1e-4 * h, 1e-4),
            max(25.0 * h, 1.0),
            num=max(int(num_points), 40)
        )
        responses = PerformanceFunction._get_column_section_response_arrays(
            fc,
            fy_tarik,
            section_geometry,
            steel_area,
            c_values,
            fy_tekan=fy_tekan
        )
        valid_mask = responses['phi_Pn'] <= (phi_pn_max + 1e-6)
        valid_indices = np.nonzero(valid_mask)[0]

        for idx in valid_indices:
            points.append({
                'phi_Pn': float(responses['phi_Pn'][idx]),
                'phi_Mn': float(responses['phi_Mn'][idx]),
                'phi': float(responses['phi'][idx]),
                'epsilon_t_net': float(responses['epsilon_t_net'][idx]),
                'epsilon_ty': float(responses['epsilon_ty'][idx]),
                'classification': str(responses['classification'][idx]),
                'neutral_axis_depth': float(responses['neutral_axis_depth'][idx])
            })

        points.sort(key=lambda point: point['phi_Pn'], reverse=True)

        collapsed: List[Dict[str, float]] = []
        for point in points:
            if not collapsed:
                collapsed.append(point)
                continue

            if abs(point['phi_Pn'] - collapsed[-1]['phi_Pn']) <= 1e-6:
                if point['phi_Mn'] > collapsed[-1]['phi_Mn']:
                    collapsed[-1] = point
            else:
                collapsed.append(point)

        return collapsed

    @staticmethod
    def _get_interaction_scale_factor(demand_axial: float,
                                      demand_moment: float,
                                      interaction_curve: List[Dict[str, float]]) -> float:
        """
        Cari faktor skala lambda pada sinar dari origin ke titik demand terhadap
        kurva interaksi. lambda >= 1 berarti demand masih di dalam kurva.
        """
        demand_axial = float(demand_axial)
        demand_moment = abs(float(demand_moment))
        tolerance = 1e-9

        if abs(demand_axial) <= tolerance and demand_moment <= tolerance:
            return float('inf')

        if demand_moment <= tolerance:
            if demand_axial >= 0.0:
                max_axial = max(point['phi_Pn'] for point in interaction_curve)
                return max_axial / max(demand_axial, tolerance)

            min_axial = min(point['phi_Pn'] for point in interaction_curve)
            return min_axial / min(demand_axial, -tolerance)

        lambdas: List[float] = []
        for first, second in zip(interaction_curve, interaction_curve[1:]):
            p1 = float(first['phi_Pn'])
            m1 = float(first['phi_Mn'])
            p2 = float(second['phi_Pn'])
            m2 = float(second['phi_Mn'])

            solution = PerformanceFunction._solve_interaction_ray_segment(
                demand_axial,
                demand_moment,
                p1,
                m1,
                p2,
                m2,
                tolerance=tolerance
            )
            if solution is None:
                continue

            lambda_value, segment_ratio = solution
            if lambda_value < -tolerance:
                continue
            if segment_ratio < -1e-6 or segment_ratio > 1.0 + 1e-6:
                continue

            lambdas.append(float(max(lambda_value, 0.0)))

        if not lambdas:
            return 0.0

        return float(min(lambdas))

    @staticmethod
    def _get_interaction_boundary_state(demand_axial: float,
                                        demand_moment: float,
                                        interaction_curve: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Ambil titik batas pada kurva interaksi yang terpotong oleh sinar dari origin
        ke titik demand.
        """
        demand_axial = float(demand_axial)
        demand_moment = abs(float(demand_moment))
        tolerance = 1e-9

        if abs(demand_axial) <= tolerance and demand_moment <= tolerance:
            return {
                'lambda': float('inf'),
                'phi_Pn': 0.0,
                'phi_Mn': 0.0,
                'phi': None,
                'epsilon_t_net': None,
                'epsilon_ty': None,
                'classification': None,
                'neutral_axis_depth': None
            }

        if demand_moment <= tolerance:
            if demand_axial >= 0.0:
                boundary_point = max(interaction_curve, key=lambda point: point['phi_Pn'])
                lambda_value = boundary_point['phi_Pn'] / max(demand_axial, tolerance)
            else:
                boundary_point = min(interaction_curve, key=lambda point: point['phi_Pn'])
                lambda_value = boundary_point['phi_Pn'] / min(demand_axial, -tolerance)

            result = dict(boundary_point)
            result['lambda'] = float(lambda_value)
            return result

        candidates: List[Dict[str, float]] = []
        for first, second in zip(interaction_curve, interaction_curve[1:]):
            p1 = float(first['phi_Pn'])
            m1 = float(first['phi_Mn'])
            p2 = float(second['phi_Pn'])
            m2 = float(second['phi_Mn'])

            solution = PerformanceFunction._solve_interaction_ray_segment(
                demand_axial,
                demand_moment,
                p1,
                m1,
                p2,
                m2,
                tolerance=tolerance
            )
            if solution is None:
                continue

            lambda_value, segment_ratio = solution
            if lambda_value < -tolerance:
                continue
            if segment_ratio < -1e-6 or segment_ratio > 1.0 + 1e-6:
                continue

            lambda_value = float(max(lambda_value, 0.0))
            segment_ratio = float(np.clip(segment_ratio, 0.0, 1.0))
            anchor = first if segment_ratio <= 0.5 else second
            candidate = {
                'lambda': lambda_value,
                'phi_Pn': float(lambda_value * demand_axial),
                'phi_Mn': float(lambda_value * demand_moment),
                'phi': anchor.get('phi'),
                'epsilon_t_net': anchor.get('epsilon_t_net'),
                'epsilon_ty': anchor.get('epsilon_ty'),
                'classification': anchor.get('classification'),
                'neutral_axis_depth': anchor.get('neutral_axis_depth')
            }
            candidates.append(candidate)

        if not candidates:
            return {
                'lambda': 0.0,
                'phi_Pn': 0.0,
                'phi_Mn': 0.0,
                'phi': None,
                'epsilon_t_net': None,
                'epsilon_ty': None,
                'classification': None,
                'neutral_axis_depth': None
            }

        return min(candidates, key=lambda candidate: candidate['lambda'])

    @staticmethod
    def _get_axial_capacity_check_result(compression_demand: float,
                                         tension_demand: float,
                                         fc: float,
                                         fy_tarik: float,
                                         section_geometry: Dict,
                                         steel_area: Dict,
                                         fy_tekan: float = None,
                                         interaction_curve: Optional[List[Dict[str, float]]] = None
                                         ) -> Dict[str, float]:
        """
        Hasil cek kapasitas aksial murni dengan metadata klasifikasi penampang.
        """
        if interaction_curve is None:
            interaction_curve = PerformanceFunction._get_column_interaction_curve(
                fc,
                fy_tarik,
                section_geometry,
                steel_area,
                fy_tekan=fy_tekan
            )
        compression_boundary = max(
            interaction_curve,
            key=lambda point: float(point['phi_Pn'])
        )
        tension_boundary = min(
            interaction_curve,
            key=lambda point: float(point['phi_Pn'])
        )
        compression_demand = (
            0.0
            if abs(float(compression_demand)) <= AXIAL_DEMAND_TOLERANCE_KN else
            max(float(compression_demand), 0.0)
        )
        tension_demand = (
            0.0
            if abs(float(tension_demand)) <= AXIAL_DEMAND_TOLERANCE_KN else
            max(float(tension_demand), 0.0)
        )

        states: List[Dict[str, float]] = []
        if float(compression_demand) > AXIAL_DEMAND_TOLERANCE_KN:
            states.append({
                'g': float(float(compression_boundary['phi_Pn']) - compression_demand),
                'phi': compression_boundary.get('phi'),
                'epsilon_t_net': compression_boundary.get('epsilon_t_net'),
                'epsilon_ty': compression_boundary.get('epsilon_ty'),
                'classification': compression_boundary.get('classification'),
                'phi_Pn': compression_boundary.get('phi_Pn'),
                'phi_Mn': compression_boundary.get('phi_Mn'),
                'neutral_axis_depth': compression_boundary.get('neutral_axis_depth'),
                'controlling_state': 'compression'
            })
        if float(tension_demand) > AXIAL_DEMAND_TOLERANCE_KN:
            phi_pn_tarik = max(-float(tension_boundary['phi_Pn']), 0.0)
            states.append({
                'g': float(phi_pn_tarik - tension_demand),
                'phi': tension_boundary.get('phi'),
                'epsilon_t_net': tension_boundary.get('epsilon_t_net'),
                'epsilon_ty': tension_boundary.get('epsilon_ty'),
                'classification': tension_boundary.get('classification'),
                'phi_Pn': tension_boundary.get('phi_Pn'),
                'phi_Mn': tension_boundary.get('phi_Mn'),
                'neutral_axis_depth': tension_boundary.get('neutral_axis_depth'),
                'controlling_state': 'tension'
            })

        if not states:
            states.append({
                'g': float(compression_boundary['phi_Pn']),
                'phi': compression_boundary.get('phi'),
                'epsilon_t_net': compression_boundary.get('epsilon_t_net'),
                'epsilon_ty': compression_boundary.get('epsilon_ty'),
                'classification': compression_boundary.get('classification'),
                'phi_Pn': compression_boundary.get('phi_Pn'),
                'phi_Mn': compression_boundary.get('phi_Mn'),
                'neutral_axis_depth': compression_boundary.get('neutral_axis_depth'),
                'controlling_state': 'compression'
            })

        return min(states, key=lambda state: state['g'])

    @staticmethod
    def _get_axial_moment_interaction_result(compression_demand: float,
                                             tension_demand: float,
                                             max_moment_demand: float,
                                             fc: float,
                                             fy_tarik: float,
                                             section_geometry: Dict,
                                             steel_area: Dict,
                                             fy_tekan: float = None,
                                             interaction_curve: Optional[List[Dict[str, float]]] = None
                                             ) -> Dict[str, float]:
        """
        Hasil cek interaksi aksial-momen dengan metadata titik batas yang mengontrol.
        """
        compression_demand = (
            0.0
            if abs(float(compression_demand)) <= AXIAL_DEMAND_TOLERANCE_KN else
            max(float(compression_demand), 0.0)
        )
        tension_demand = (
            0.0
            if abs(float(tension_demand)) <= AXIAL_DEMAND_TOLERANCE_KN else
            max(float(tension_demand), 0.0)
        )
        max_moment_demand = abs(float(max_moment_demand))

        if max_moment_demand <= 1e-12:
            return PerformanceFunction._get_axial_capacity_check_result(
                compression_demand,
                tension_demand,
                fc,
                fy_tarik,
                section_geometry,
                steel_area,
                fy_tekan=fy_tekan,
                interaction_curve=interaction_curve
            )

        if interaction_curve is None:
            interaction_curve = PerformanceFunction._get_column_interaction_curve(
                fc,
                fy_tarik,
                section_geometry,
                steel_area,
                fy_tekan=fy_tekan
            )

        states: List[Dict[str, float]] = []
        if compression_demand > AXIAL_DEMAND_TOLERANCE_KN:
            boundary = PerformanceFunction._get_interaction_boundary_state(
                compression_demand,
                max_moment_demand,
                interaction_curve
            )
            states.append({
                'g': float(boundary['lambda'] - 1.0),
                'phi': boundary.get('phi'),
                'epsilon_t_net': boundary.get('epsilon_t_net'),
                'epsilon_ty': boundary.get('epsilon_ty'),
                'classification': boundary.get('classification'),
                'phi_Pn': boundary.get('phi_Pn'),
                'phi_Mn': boundary.get('phi_Mn'),
                'neutral_axis_depth': boundary.get('neutral_axis_depth'),
                'lambda': boundary.get('lambda'),
                'controlling_state': 'compression'
            })
        if tension_demand > AXIAL_DEMAND_TOLERANCE_KN:
            boundary = PerformanceFunction._get_interaction_boundary_state(
                -tension_demand,
                max_moment_demand,
                interaction_curve
            )
            states.append({
                'g': float(boundary['lambda'] - 1.0),
                'phi': boundary.get('phi'),
                'epsilon_t_net': boundary.get('epsilon_t_net'),
                'epsilon_ty': boundary.get('epsilon_ty'),
                'classification': boundary.get('classification'),
                'phi_Pn': boundary.get('phi_Pn'),
                'phi_Mn': boundary.get('phi_Mn'),
                'neutral_axis_depth': boundary.get('neutral_axis_depth'),
                'lambda': boundary.get('lambda'),
                'controlling_state': 'tension'
            })

        if not states:
            boundary = PerformanceFunction._get_interaction_boundary_state(
                0.0,
                max_moment_demand,
                interaction_curve
            )
            states.append({
                'g': float(boundary['lambda'] - 1.0),
                'phi': boundary.get('phi'),
                'epsilon_t_net': boundary.get('epsilon_t_net'),
                'epsilon_ty': boundary.get('epsilon_ty'),
                'classification': boundary.get('classification'),
                'phi_Pn': boundary.get('phi_Pn'),
                'phi_Mn': boundary.get('phi_Mn'),
                'neutral_axis_depth': boundary.get('neutral_axis_depth'),
                'lambda': boundary.get('lambda'),
                'controlling_state': 'pure-bending'
            })

        return min(states, key=lambda state: state['g'])

    @staticmethod
    def _get_axial_capacities(fc: float, fy_tarik: float,
                              section_geometry: Dict,
                              steel_area: Dict,
                              fy_tekan: float = None) -> Tuple[float, float]:
        """
        Hitung kapasitas aksial (kN).

        Returns:
        - (phiPn_tekan, phiPn_tarik)
        """
        As = float(steel_area.get('As', 0.0) or 0.0)
        As_prime = float(steel_area.get('As_prime', 0.0) or 0.0)
        Ast = max(As + As_prime, 0.0)
        Po = PerformanceFunction._get_po(
            fc,
            fy_tarik,
            section_geometry,
            steel_area,
            fy_tekan=fy_tekan
        )
        phi_pn_tekan = RELIABILITY_PHI_FACTOR * Po
        phi_pn_tarik = RELIABILITY_PHI_FACTOR * fy_tarik * Ast / 1000.0

        return float(phi_pn_tekan), float(phi_pn_tarik)

    @staticmethod
    def moment_capacity_demand(max_moment_demand: float,
                              fc: float, fy: float,
                              section_geometry: Dict,
                              steel_area: Dict,
                              fy_tekan: float = None) -> float:
        """
        Performance function untuk momen:
        g = Capacity - Demand
        Failure jika g < 0

        Parameters:
        - max_moment_demand: maximum moment dari analisis (kN.m)
        - fc: concrete compressive strength (MPa)
        - fy: steel yield strength (MPa)
        - section_geometry: {'b': width, 'h': height, 'd': effective_depth} (mm)
        - steel_area: {'As': tension_steel, 'As_prime': compression_steel} (mm2)

        Returns:
        - safety margin (g-value)
        """
        Mc = PerformanceFunction._get_moment_capacity(
            fc,
            fy,
            section_geometry,
            steel_area,
            fy_tekan=fy_tekan
        )
        Md = abs(max_moment_demand)
        g = Mc - Md

        return g

    @staticmethod
    def axial_capacity_demand(compression_demand: float,
                              tension_demand: float,
                              fc: float,
                              fy_tarik: float,
                              section_geometry: Dict,
                              steel_area: Dict,
                              fy_tekan: float = None,
                              interaction_curve: Optional[List[Dict[str, float]]] = None
                              ) -> float:
        """
        Performance function untuk aksial:
        g = Capacity - Demand

        Solver internal memakai konvensi positif = tekan, negatif = tarik.
        """
        result = PerformanceFunction._get_axial_capacity_check_result(
            compression_demand,
            tension_demand,
            fc,
            fy_tarik,
            section_geometry,
            steel_area,
            fy_tekan=fy_tekan,
            interaction_curve=interaction_curve
        )
        return float(result['g'])

    @staticmethod
    def combined_axial_moment_capacity_demand(compression_demand: float,
                                              tension_demand: float,
                                              max_moment_demand: float,
                                              fc: float,
                                              fy_tarik: float,
                                              section_geometry: Dict,
                                              steel_area: Dict,
                                              fy_tekan: float = None,
                                              interaction_curve: Optional[List[Dict[str, float]]] = None
                                              ) -> float:
        """
        Performance function interaksi aksial + momen berbasis kurva interaksi
        phiPn-phiMn dari hasil kompatibilitas regangan.

        Nilai g = lambda_boundary - 1.
        g >= 0 menandakan titik demand masih berada di dalam kurva interaksi.
        """
        result = PerformanceFunction._get_axial_moment_interaction_result(
            compression_demand,
            tension_demand,
            max_moment_demand,
            fc,
            fy_tarik,
            section_geometry,
            steel_area,
            fy_tekan=fy_tekan,
            interaction_curve=interaction_curve
        )
        return float(result['g'])

    @staticmethod
    def shear_capacity_demand(max_shear_demand: float,
                            fc: float, fy_shear: float,
                            section_geometry: Dict,
                            shear_steel_area: float,
                            shear_spacing: float = 200.0) -> float:
        """
        Performance function untuk geser:
        g = Capacity - Demand
        """
        missing_fields: List[str] = []
        b = PerformanceFunction._read_positive_input(
            section_geometry.get('b'),
            'b (sheet Geometri)',
            missing_fields
        )
        d = PerformanceFunction._read_positive_input(
            section_geometry.get('d'),
            'd_tarik atau kombinasi ds_tarik + du_geser + du_tarik (sheet Tulangan)',
            missing_fields
        )
        fc_value = PerformanceFunction._read_positive_input(
            fc,
            "fc' / Mean (sheet Mutu_Beton)",
            missing_fields
        )
        fy_shear_value = PerformanceFunction._read_positive_input(
            fy_shear,
            'fy_geser / Mean_geser (sheet Mutu_Baja)',
            missing_fields
        )
        Av = PerformanceFunction._read_positive_input(
            shear_steel_area,
            'As_geser atau kombinasi n_geser + du_geser (sheet Tulangan)',
            missing_fields
        )
        s = PerformanceFunction._read_positive_input(
            shear_spacing,
            'Spasi_geser (sheet Tulangan)',
            missing_fields
        )

        if missing_fields:
            PerformanceFunction._raise_missing_input_error(
                'kapasitas geser',
                missing_fields,
                section_geometry
            )

        # Kapasitas geser beton
        Vc = 0.17 * np.sqrt(max(fc_value, 15.0)) * b * d / 1000  # kN

        # Kapasitas geser baja berdasarkan luas dan spasi sengkang aktual.
        Vs = fy_shear_value * Av * d / s / 1000  # kN

        Vc_total = RELIABILITY_PHI_FACTOR * (Vc + Vs)
        Vd = abs(max_shear_demand)
        g = Vc_total - Vd

        return g

    @staticmethod
    def combined_moment_shear(max_moment: float,
                            max_shear: float,
                            fc: float, fy: float,
                            section_geometry: Dict,
                            steel_area: Dict,
                            fy_tekan: float = None,
                            fy_shear: float = None) -> Tuple[float, float, float]:
        """
        Performance function kombinasi untuk momen dan geser

        Returns:
        - (g_moment, g_shear, g_combined)
        """
        g_m = PerformanceFunction.moment_capacity_demand(
            max_moment,
            fc,
            fy,
            section_geometry,
            steel_area,
            fy_tekan=fy_tekan
        )

        g_s = PerformanceFunction.shear_capacity_demand(
            max_shear,
            fc,
            fy if fy_shear is None else fy_shear,
            section_geometry,
            steel_area.get('As_shear', 0.0),
            shear_spacing=steel_area.get('shear_spacing', 0.0)
        )

        # Tetap dipertahankan untuk kompatibilitas pemanggilan lama.
        g_combined = min(g_m, g_s)

        return g_m, g_s, g_combined


class ReliabilityAssessment:
    """Penilaian keandalan struktur berdasarkan hasil Monte Carlo"""

    def __init__(self, mc_results: Dict):
        """
        Parameters:
        - mc_results: hasil dari MonteCarloAnalysis.run_simulation()
        """
        self.mc_results = mc_results
        self.Pf = mc_results['Pf']
        self.Beta = mc_results['Beta']

    def get_reliability_index(self) -> float:
        """Dapatkan reliability index (Beta)"""
        return self.Beta

    def get_probability_of_failure(self) -> float:
        """Dapatkan probability of failure"""
        return self.Pf

    def get_element_reliability(self) -> Dict:
        """Dapatkan hasil reliability per elemen jika tersedia."""
        return self.mc_results.get('element_reliability', {})

    def get_safety_class(self) -> str:
        """
        Klasifikasi keamanan berdasarkan Beta:
        - Beta >= 3.5: Very Safe (Pf < 0.02%)
        - Beta >= 2.5: Safe (Pf < 0.6%)
        - Beta >= 1.5: Acceptable (Pf < 7%)
        - Beta < 1.5: Risky (Pf > 7%)
        """
        Beta = self.Beta

        if Beta >= 3.5:
            return "Very Safe"
        elif Beta >= 2.5:
            return "Safe"
        elif Beta >= 1.5:
            return "Acceptable"
        else:
            return "Risky"

    def get_safety_class_formal_id(self) -> str:
        """Padanan bahasa Indonesia formal untuk kelas keandalan."""
        mapping = {
            "Very Safe": "Sangat Aman",
            "Safe": "Aman",
            "Acceptable": "Dapat Diterima",
            "Risky": "Berisiko"
        }
        return mapping.get(self.get_safety_class(), self.get_safety_class())

    def get_target_reliability(self, limit_state: str = "ultimate") -> float:
        """
        Dapatkan target reliability index berdasarkan limit state

        Parameters:
        - limit_state: "ultimate" (LSC) atau "serviceability" (SLS)

        Returns:
        - target Beta
        """
        if limit_state == "ultimate":
            return 3.0  # untuk ULS
        elif limit_state == "serviceability":
            return 1.5  # untuk SLS
        else:
            return 2.5  # default

    def is_safe(self, limit_state: str = "ultimate", margin: float = 0) -> bool:
        """
        Check apakah struktur aman

        Parameters:
        - limit_state: "ultimate" atau "serviceability"
        - margin: safety margin untuk Beta (default 0)

        Returns:
        - True jika Beta > target
        """
        target = self.get_target_reliability(limit_state)
        return self.Beta > (target + margin)

    def get_report(self) -> str:
        """Generate laporan keandalan"""
        safety_status = "AMAN" if self.is_safe('ultimate') else "TIDAK AMAN"
        report = f"""
{'='*60}
LAPORAN PENILAIAN KEANDALAN STRUKTUR
{'='*60}

Ikhtisar Simulasi Monte Carlo:
  - Jumlah sampel simulasi: {self.mc_results['num_simulations']}
  - Jumlah kejadian gagal teramati: {self.mc_results['failures']}

Ringkasan Hasil Penilaian:
  - Probabilitas kegagalan, Pf: {self.Pf:.6f} ({self.Pf*100:.4f}%)
  - Indeks keandalan, Beta: {self.Beta:.4f}
  - Kelas keandalan struktur: {self.get_safety_class_formal_id()}
  - Nilai target indeks keandalan untuk kondisi batas ultimit (ULS): {self.get_target_reliability('ultimate')}
  - Status kinerja keamanan struktur: {safety_status}

Interpretasi Rekayasa:
"""

        if self.Pf == 0:
            report += (
                "  - Tidak teramati kejadian gagal pada seluruh sampel simulasi Monte Carlo, "
                "sehingga struktur menunjukkan tingkat keandalan yang sangat tinggi dalam ruang "
                "acak yang dianalisis.\n"
            )
        else:
            report += (
                f"  - Secara statistik, kejadian gagal diperkirakan terjadi sekitar "
                f"satu kali pada setiap {1/self.Pf:.0f} realisasi struktur dengan "
                f"karakteristik acak yang sebanding.\n"
            )

        report += (
            f"  - Berdasarkan perbandingan terhadap target keandalan yang ditetapkan, "
            f"tingkat keandalan struktur berada dalam kategori "
            f"{self.get_safety_class_formal_id()}.\n"
        )
        report += f"\n{'='*60}\n"

        return report


class SensitivityAnalysis:
    """Analisis sensitivitas terhadap variabel random"""

    @staticmethod
    def rank_variables(mc_results: Dict,
                      variable_names: List[str]) -> Dict:
        """
        Ranking variabel berdasarkan kontribusi terhadap failure

        Parameters:
        - mc_results: hasil simulasi
        - variable_names: list nama variabel

        Returns:
        - dict dengan ranking dan sensitivity indices
        """
        sensitivities = {}

        failure_indices = {
            int(index) for index in mc_results.get('failure_indices', [])
            if 0 <= int(index) < len(mc_results.get('random_samples_history', []))
        }

        for var_name in variable_names:
            var_failures = []
            var_all = []

            for i, samples in enumerate(mc_results['random_samples_history']):
                if var_name in samples:
                    var_all.append(samples[var_name])
                    if i in failure_indices:
                        var_failures.append(samples[var_name])

            if var_all and var_failures:
                mean_failure = np.mean(var_failures)
                mean_all = np.mean(var_all)
                std_all = np.std(var_all)

                # Sensitivity index (normalized difference in means)
                sensitivity = abs(mean_failure - mean_all) / (std_all + 1e-6)

                sensitivities[var_name] = {
                    'sensitivity_index': sensitivity,
                    'mean_in_failure': mean_failure,
                    'mean_overall': mean_all,
                    'std_overall': std_all
                }

        # Ranking by sensitivity index
        ranked = sorted(sensitivities.items(),
                       key=lambda x: x[1]['sensitivity_index'],
                       reverse=True)

        return dict(ranked)

    @staticmethod
    def variance_decomposition(mc_results: Dict,
                             variable_names: List[str]) -> Dict:
        """
        Dekomposisi varians untuk mengidentifikasi variabel dominan
        """
        decomposition = {}

        for var_name in variable_names:
            var_values = [
                sample[var_name]
                for sample in mc_results['random_samples_history']
                if var_name in sample
            ]

            if var_values:
                variance = np.var(var_values)
                std = np.std(var_values)
                cov = std / (np.mean(var_values) + 1e-6)  # Coefficient of variation

                decomposition[var_name] = {
                    'variance': variance,
                    'std_dev': std,
                    'cov': cov
                }

        # Normalize variance terhadap total
        total_var = sum([d['variance'] for d in decomposition.values()])
        for var in decomposition:
            decomposition[var]['variance_ratio'] = (
                decomposition[var]['variance'] / (total_var + 1e-6)
            )

        return decomposition
