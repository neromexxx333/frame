"""
Modul untuk analisis struktural dengan portal 2D.
Menangani kombinasi pembebanan dan recovery gaya dalam.
"""
import numpy as np
from typing import Dict, List, Tuple

KILO = 1000.0


class LoadHandler:
    """Menangani pembebanan merata dan nodal."""

    def __init__(self, portal, num_nodes: int):
        self.portal = portal
        self.num_nodes = num_nodes
        self.num_dof = num_nodes * 3
        self._unit_distributed_load_cache = self._build_unit_distributed_load_cache()

    def _build_unit_distributed_load_cache(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Precompute respons beban merata unit 1 kN/m per elemen."""
        cache = {}

        for element in self.portal.elements:
            equivalent_local = self.distributed_load_to_local_equivalent(1.0, element)
            equivalent_global = element.get_transformation_matrix().T @ equivalent_local
            axial_local, transverse_local = self.get_uniform_load_components(1.0, element)

            cache[element.elem_id] = {
                'equivalent_nodal_local_unit': np.asarray(equivalent_local, dtype=float),
                'equivalent_nodal_global_unit': np.asarray(equivalent_global, dtype=float),
                'axial_local_unit': float(axial_local),
                'transverse_local_unit': float(transverse_local)
            }

        return cache

    def get_uniform_load_components(self, load_value: float, element) -> Tuple[float, float]:
        """
        Proyeksikan beban merata global vertikal ke sumbu lokal elemen.

        Returns:
        - axial_local: komponen sepanjang sumbu lokal x (kN/m)
        - transverse_local: komponen pada sumbu lokal y (kN/m)
        """
        global_load = np.array([0.0, -load_value], dtype=float)
        local_x = np.array([element.cos_theta, element.sin_theta], dtype=float)
        local_y = np.array([-element.sin_theta, element.cos_theta], dtype=float)

        axial_local = float(np.dot(global_load, local_x))
        transverse_local = float(np.dot(global_load, local_y))

        return axial_local, transverse_local

    def distributed_load_to_local_equivalent(self, load_value: float, element) -> np.ndarray:
        """
        Konversi beban merata global ke vektor beban nodal ekuivalen lokal.

        Satuan internal solver:
        - gaya: kN
        - panjang: mm
        - momen: kN.mm
        """
        if load_value == 0:
            return np.zeros(6)

        length_mm = float(element.length)
        axial_local, transverse_local = self.get_uniform_load_components(
            load_value, element)
        axial_local_internal = axial_local / KILO
        transverse_local_internal = transverse_local / KILO

        return np.array([
            axial_local_internal * length_mm / 2,
            transverse_local_internal * length_mm / 2,
            transverse_local_internal * length_mm**2 / 12,
            axial_local_internal * length_mm / 2,
            transverse_local_internal * length_mm / 2,
            -transverse_local_internal * length_mm**2 / 12
        ], dtype=float)

    def build_element_distributed_loads(self, dead_load: Dict, live_load: Dict,
                                       elements_list: List) -> Dict[int, Dict]:
        """
        Bangun metadata beban merata per elemen untuk analisis gaya dalam.
        """
        element_loads = {}

        for element in elements_list:
            total_load = 0.0

            if element.elem_id in dead_load.get('values', {}):
                total_load += dead_load['values'][element.elem_id]

            if element.elem_id in live_load.get('values', {}):
                total_load += live_load['values'][element.elem_id]

            if total_load == 0:
                continue

            unit_load = self._unit_distributed_load_cache[element.elem_id]
            equivalent_local = (
                unit_load['equivalent_nodal_local_unit'] * float(total_load)
            )
            equivalent_global = (
                unit_load['equivalent_nodal_global_unit'] * float(total_load)
            )
            axial_local = float(unit_load['axial_local_unit'] * float(total_load))
            transverse_local = float(
                unit_load['transverse_local_unit'] * float(total_load)
            )

            element_loads[element.elem_id] = {
                'load_global_vertical': total_load,
                'axial_local': axial_local,
                'transverse_local': transverse_local,
                'equivalent_nodal_local': equivalent_local,
                'equivalent_nodal_global': equivalent_global
            }

        return element_loads

    def apply_distributed_loads(self, dead_load: Dict, live_load: Dict,
                                elements_list: List) -> Tuple[np.ndarray, Dict[int, Dict]]:
        """
        Apply semua beban merata dan kembalikan vektor global + metadata elemen.
        """
        total_loads = np.zeros(self.num_dof)
        element_loads = self.build_element_distributed_loads(
            dead_load, live_load, elements_list)

        for element in elements_list:
            if element.elem_id not in element_loads:
                continue

            equivalent_global = element_loads[element.elem_id]['equivalent_nodal_global']
            total_loads[element.dofs] += equivalent_global

        return total_loads, element_loads

    def apply_nodal_loads(self, nodal_loads: Dict) -> np.ndarray:
        """
        Apply beban nodal deterministik.

        Input user:
        - Fx, Fy: kN
        - Mz: kN.m

        Satuan internal solver:
        - Fx, Fy: kN
        - Mz: kN.mm
        """
        loads = np.zeros(self.num_dof)

        for node_id, load_dict in nodal_loads.items():
            dof = (node_id - 1) * 3
            loads[dof] += load_dict.get('Fx', 0)
            loads[dof + 1] += load_dict.get('Fy', 0)
            loads[dof + 2] += load_dict.get('Mz', 0) * KILO

        return loads

    def convert_dof_vector_to_output_units(self, values: np.ndarray) -> np.ndarray:
        """
        Konversi vektor DOF dari satuan internal ke satuan output.

        Internal: [Fx(kN), Fy(kN), Mz(kN.mm), ...]
        Output:   [Fx(kN), Fy(kN), Mz(kN.m), ...]
        """
        output = np.asarray(values, dtype=float).copy()
        output[2::3] /= KILO
        return output


class StructuralAnalysis:
    """Melakukan analisis struktural pada portal 2D."""

    def __init__(self, portal, nodes: np.ndarray):
        self.portal = portal
        self.nodes = nodes
        self.load_handler = LoadHandler(portal, len(nodes))

    def analyze(self, dead_load_dict: Dict, live_load_dict: Dict,
               nodal_loads: Dict, include_section_samples: bool = False) -> Dict:
        """
        Perform structural analysis.
        """
        distributed_loads, element_distributed_loads = self.load_handler.apply_distributed_loads(
            dead_load_dict, live_load_dict, self.portal.elements)
        nodal_loads_array = self.load_handler.apply_nodal_loads(nodal_loads)
        total_loads = distributed_loads + nodal_loads_array

        displacements, reactions = self.portal.solve(total_loads)
        element_forces = self.portal.get_element_forces(
            displacements,
            element_loads=element_distributed_loads,
            include_section_samples=include_section_samples
        )

        return {
            'displacements': displacements,
            'reactions': self.load_handler.convert_dof_vector_to_output_units(reactions),
            'element_forces': element_forces,
            'total_loads': self.load_handler.convert_dof_vector_to_output_units(total_loads),
            'distributed_loads': self.load_handler.convert_dof_vector_to_output_units(distributed_loads),
            'nodal_loads': self.load_handler.convert_dof_vector_to_output_units(nodal_loads_array),
            'element_distributed_loads': element_distributed_loads
        }

    def extract_maximum_forces(self, element_forces: List[Dict]) -> Dict:
        """
        Ekstrak gaya maksimum absolut dari setiap elemen.
        """
        max_forces = {}

        for elem_force in element_forces:
            elem_id = elem_force['elem_id']
            max_forces[elem_id] = {
                'max_moment': elem_force.get(
                    'max_moment',
                    max(abs(elem_force['moment_start']), abs(elem_force['moment_end']))
                ),
                'max_shear': elem_force.get(
                    'max_shear',
                    max(abs(elem_force['shear_start']), abs(elem_force['shear_end']))
                ),
                'max_axial': elem_force.get(
                    'max_axial',
                    max(abs(elem_force['axial_start']), abs(elem_force['axial_end']))
                ),
                'x_max_moment': elem_force.get('x_max_moment'),
                'x_max_shear': elem_force.get('x_max_shear'),
                'x_max_axial': elem_force.get('x_max_axial'),
                'forces': elem_force
            }

        return max_forces
