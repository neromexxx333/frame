"""
Modul untuk implementasi metode matriks kekakuan langsung (Direct Stiffness Method)
untuk portal 2D.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.linalg import lu_factor, lu_solve

MPA_TO_KN_PER_MM2 = 1.0 / 1000.0
KNMM_TO_KNM = 1.0 / 1000.0


class Element2D:
    """Elemen balok 2D dengan 6 derajat kebebasan (3 per node)."""

    def __init__(self, elem_id: int, node_start: int, node_end: int,
                 E: float, A: float, I: float,
                 coord_start: np.ndarray, coord_end: np.ndarray):
        """
        Parameters:
        - elem_id: ID elemen
        - node_start, node_end: ID node awal dan akhir
        - E: modulus elastisitas (MPa)
        - A: luas penampang (mm2)
        - I: momen inersia (mm4)
        - coord_start, coord_end: koordinat node [x, y]
        """
        self.elem_id = elem_id
        self.node_start = node_start
        self.node_end = node_end
        self.E = E * MPA_TO_KN_PER_MM2
        self.A = float(A)
        self.I = float(I)
        self.coord_start = np.asarray(coord_start, dtype=float)
        self.coord_end = np.asarray(coord_end, dtype=float)

        self.length = float(np.linalg.norm(self.coord_end - self.coord_start))
        dx = float(self.coord_end[0] - self.coord_start[0])
        dy = float(self.coord_end[1] - self.coord_start[1])
        self.cos_theta = dx / self.length
        self.sin_theta = dy / self.length

        self.k_local = None
        self.k_global = None
        self.T = None
        self.dofs = None

        self._compute_stiffness()

    def _compute_stiffness_local(self) -> np.ndarray:
        """
        Compute local stiffness matrix untuk elemen balok 2D.
        DOF: [u1, v1, theta1, u2, v2, theta2]
        """
        length = self.length
        EA = self.E * self.A
        EI = self.E * self.I

        stiffness = np.zeros((6, 6), dtype=float)

        stiffness[0, 0] = EA / length
        stiffness[0, 3] = -EA / length
        stiffness[3, 0] = -EA / length
        stiffness[3, 3] = EA / length

        stiffness[1, 1] = 12 * EI / length**3
        stiffness[1, 2] = 6 * EI / length**2
        stiffness[1, 4] = -12 * EI / length**3
        stiffness[1, 5] = 6 * EI / length**2

        stiffness[2, 1] = 6 * EI / length**2
        stiffness[2, 2] = 4 * EI / length
        stiffness[2, 4] = -6 * EI / length**2
        stiffness[2, 5] = 2 * EI / length

        stiffness[4, 1] = -12 * EI / length**3
        stiffness[4, 2] = -6 * EI / length**2
        stiffness[4, 4] = 12 * EI / length**3
        stiffness[4, 5] = -6 * EI / length**2

        stiffness[5, 1] = 6 * EI / length**2
        stiffness[5, 2] = 2 * EI / length
        stiffness[5, 4] = -6 * EI / length**2
        stiffness[5, 5] = 4 * EI / length

        return stiffness

    def _compute_transformation_matrix(self) -> np.ndarray:
        """Compute transformation matrix dari lokal ke global."""
        c = self.cos_theta
        s = self.sin_theta

        transform = np.zeros((6, 6), dtype=float)
        transform[0:2, 0:2] = np.array([[c, s], [-s, c]], dtype=float)
        transform[2, 2] = 1.0
        transform[3:5, 3:5] = np.array([[c, s], [-s, c]], dtype=float)
        transform[5, 5] = 1.0

        return transform

    def _compute_stiffness(self):
        """Hitung dan simpan matriks kekakuan lokal dan global."""
        self.k_local = self._compute_stiffness_local()
        self.T = self._compute_transformation_matrix()
        self.k_global = self.T.T @ self.k_local @ self.T

    def get_global_stiffness(self) -> np.ndarray:
        return self.k_global

    def get_local_stiffness(self) -> np.ndarray:
        return self.k_local

    def get_transformation_matrix(self) -> np.ndarray:
        return self.T

    def update_elastic_modulus(self, E: float):
        """Perbarui modulus elastisitas elemen (MPa) lalu hitung ulang kekakuan."""
        self.E = float(E) * MPA_TO_KN_PER_MM2
        self._compute_stiffness()


class Portal2D:
    """Struktur portal 2D dengan multiple elements."""

    def __init__(self, nodes: np.ndarray, elements_data: np.ndarray,
                 boundary_conditions: Dict, E: float = 30000):
        """
        Parameters:
        - nodes: array [node_id, x, y]
        - elements_data: array [elem_id, node_start, node_end, A, I]
        - boundary_conditions: dict {node_id: {'X': bool, 'Y': bool, 'R': bool}}
        - E: modulus elastisitas input (MPa)
        """
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements_data = np.asarray(elements_data, dtype=float)
        self.boundary_conditions = boundary_conditions
        self.E = float(E)

        self.num_nodes = len(self.nodes)
        self.num_dof = self.num_nodes * 3

        self.K_global = np.zeros((self.num_dof, self.num_dof), dtype=float)
        self.elements: List[Element2D] = []
        self.free_dofs: List[int] = []
        self.restrained_dofs: List[int] = []

        self._node_lookup = {
            int(node[0]): np.asarray(node[1:3], dtype=float)
            for node in self.nodes
        }
        self._K_free = None
        self._K_free_factorized = None

        self._build_elements()
        self._assemble_stiffness_matrix()
        self._apply_boundary_conditions()
        self._build_solver_cache()

    def _build_elements(self):
        """Membuat objek Element2D untuk setiap elemen."""
        for elem in self.elements_data:
            elem_id = int(elem[0])
            node_start = int(elem[1])
            node_end = int(elem[2])
            area = float(elem[3])
            inertia = float(elem[4])

            element_modulus = self.E
            if len(elem) >= 6 and not np.isnan(float(elem[5])):
                element_modulus = float(elem[5])

            element = Element2D(
                elem_id,
                node_start,
                node_end,
                element_modulus,
                area,
                inertia,
                self._node_lookup[node_start],
                self._node_lookup[node_end]
            )
            element.dofs = np.array([
                (node_start - 1) * 3,
                (node_start - 1) * 3 + 1,
                (node_start - 1) * 3 + 2,
                (node_end - 1) * 3,
                (node_end - 1) * 3 + 1,
                (node_end - 1) * 3 + 2
            ], dtype=int)
            self.elements.append(element)

    def _assemble_stiffness_matrix(self):
        """Assembly matriks kekakuan global dari semua elemen."""
        for element in self.elements:
            k_global = element.get_global_stiffness()
            dofs = element.dofs.tolist()

            for i, dof_i in enumerate(dofs):
                for j, dof_j in enumerate(dofs):
                    self.K_global[dof_i, dof_j] += k_global[i, j]

    def _apply_boundary_conditions(self):
        """Apply boundary conditions dengan pemisahan DOF bebas dan tertahan."""
        self.free_dofs = []
        self.restrained_dofs = []

        for node in self.nodes:
            node_id = int(node[0])
            conditions = self.boundary_conditions.get(node_id, {})
            dof_base = (node_id - 1) * 3

            if conditions.get('X', False) == 0:
                self.free_dofs.append(dof_base)
            else:
                self.restrained_dofs.append(dof_base)

            if conditions.get('Y', False) == 0:
                self.free_dofs.append(dof_base + 1)
            else:
                self.restrained_dofs.append(dof_base + 1)

            if conditions.get('R', False) == 0:
                self.free_dofs.append(dof_base + 2)
            else:
                self.restrained_dofs.append(dof_base + 2)

        self.free_dofs = sorted(self.free_dofs)
        self.restrained_dofs = sorted(self.restrained_dofs)

    def _build_solver_cache(self):
        """Simpan matriks tereduksi dan faktorisasi solver sekali saja."""
        self._K_free = self.K_global[np.ix_(self.free_dofs, self.free_dofs)]
        self._K_free_factorized = lu_factor(self._K_free)

    def update_element_moduli(self,
                              element_moduli: Dict[int, float],
                              default_E: Optional[float] = None):
        """Perbarui modulus tiap elemen tanpa membangun ulang geometri portal."""
        if default_E is not None:
            self.E = float(default_E)

        self.K_global.fill(0.0)
        for element in self.elements:
            modulus = float(element_moduli.get(element.elem_id, self.E))
            element.update_elastic_modulus(modulus)

        self._assemble_stiffness_matrix()
        self._build_solver_cache()

    def get_reduced_stiffness_matrix(self) -> np.ndarray:
        """Dapatkan matriks kekakuan tereduksi (hanya free DOFs)."""
        return self._K_free

    def solve(self, loads: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve untuk displacement dan reaction forces.

        Parameters:
        - loads: array gaya total (deterministik + stokastik), shape: (num_dof,)

        Returns:
        - displacements: displacement di semua DOF (global)
        - reactions: reaksi tumpuan
        """
        loads = np.asarray(loads, dtype=float)
        free_loads = loads[self.free_dofs]
        free_displacements = lu_solve(self._K_free_factorized, free_loads)

        global_displacements = np.zeros(self.num_dof, dtype=float)
        global_displacements[self.free_dofs] = free_displacements

        reactions = self.K_global @ global_displacements - loads
        return global_displacements, reactions

    def get_element_forces(self, displacements: np.ndarray,
                           element_loads: Dict = None,
                           include_section_samples: bool = False,
                           num_section_points: int = 101) -> List[Dict]:
        """
        Hitung gaya dalam (axial, shear, moment) setiap elemen.
        Beban merata elemen diperhitungkan langsung pada recovery gaya ujung
        dan penentuan maksimum sepanjang batang.
        """
        if element_loads is None:
            element_loads = {}

        displacements = np.asarray(displacements, dtype=float)
        element_forces = []

        for element in self.elements:
            u_elem_global = np.asarray(displacements[element.dofs], dtype=float)

            transform = element.get_transformation_matrix()
            u_elem_local = transform @ u_elem_global

            load_data = element_loads.get(element.elem_id, {})
            equivalent_nodal_local = np.asarray(
                load_data.get('equivalent_nodal_local', np.zeros(6)),
                dtype=float
            )
            axial_local_load = float(load_data.get('axial_local', 0.0))
            transverse_local_load = float(load_data.get('transverse_local', 0.0))

            f_local = element.get_local_stiffness() @ u_elem_local - equivalent_nodal_local

            length_m = element.length / 1000.0
            moment_start = float(f_local[2] * KNMM_TO_KNM)
            moment_end = float(f_local[5] * KNMM_TO_KNM)

            critical_x = [0.0, length_m]
            if abs(transverse_local_load) > 1e-12:
                x_zero_shear = -f_local[1] / transverse_local_load
                if 0.0 < x_zero_shear < length_m:
                    critical_x.append(float(x_zero_shear))

            moment_candidates = [
                moment_start - f_local[1] * x - 0.5 * transverse_local_load * x**2
                for x in critical_x
            ]
            shear_candidates = [
                -f_local[1] - transverse_local_load * x
                for x in (0.0, length_m)
            ]
            axial_candidates = [
                f_local[0] + axial_local_load * x
                for x in (0.0, length_m)
            ]

            x_max_moment = critical_x[int(np.argmax(np.abs(moment_candidates)))]
            x_max_shear = (0.0, length_m)[int(np.argmax(np.abs(shear_candidates)))]
            x_max_axial = (0.0, length_m)[int(np.argmax(np.abs(axial_candidates)))]

            element_result = {
                'elem_id': element.elem_id,
                'axial_start': float(f_local[0]),
                'shear_start': float(f_local[1]),
                'moment_start': moment_start,
                'axial_end': float(f_local[3]),
                'shear_end': float(f_local[4]),
                'moment_end': moment_end,
                'axial_end_internal': float(f_local[0] + axial_local_load * length_m),
                'shear_end_internal': float(-f_local[1] - transverse_local_load * length_m),
                'moment_end_internal': float(
                    moment_start
                    - f_local[1] * length_m
                    - 0.5 * transverse_local_load * length_m**2
                ),
                'axial_local_load': axial_local_load,
                'transverse_local_load': transverse_local_load,
                'length': float(element.length),
                'length_m': float(length_m),
                'max_axial': float(np.max(np.abs(axial_candidates))),
                'max_shear': float(np.max(np.abs(shear_candidates))),
                'max_moment': float(np.max(np.abs(moment_candidates))),
                'x_max_axial': float(x_max_axial),
                'x_max_shear': float(x_max_shear),
                'x_max_moment': float(x_max_moment)
            }

            if include_section_samples:
                x_points = np.linspace(0.0, length_m, num_section_points)
                axial_profile = f_local[0] + axial_local_load * x_points
                shear_profile = -f_local[1] - transverse_local_load * x_points
                moment_profile = (
                    moment_start
                    - f_local[1] * x_points
                    - 0.5 * transverse_local_load * x_points**2
                )
                element_result['section_profile'] = [
                    {
                        'x_m': float(x),
                        'axial': float(axial),
                        'shear': float(shear),
                        'moment': float(moment)
                    }
                    for x, axial, shear, moment in zip(
                        x_points, axial_profile, shear_profile, moment_profile
                    )
                ]

            element_forces.append(element_result)

        return element_forces
