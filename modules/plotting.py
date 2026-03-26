"""
Modul untuk visualisasi hasil analisis dan keandalan
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon, Circle
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class PortalPlotter:
    """Visualisasi struktur portal dan hasil analisis"""

    @staticmethod
    def convert_beam_moment_to_display_convention(moment_values,
                                                  is_beam: bool):
        """
        Untuk tampilan balok, ubah konvensi momen menjadi:
        sagging positif, hogging negatif.

        Solver internal menyimpan konvensi kebalikannya pada balok.
        """
        if not is_beam:
            return moment_values
        return -np.asarray(moment_values, dtype=float)

    @staticmethod
    def _get_element_axis_vectors(element) -> Tuple[np.ndarray, np.ndarray, float]:
        """Ambil sumbu lokal-elemen untuk kebutuhan plotting."""
        start = np.asarray(element.coord_start, dtype=float)
        end = np.asarray(element.coord_end, dtype=float)
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length <= 1e-12:
            return np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0

        local_x = vector / length
        local_y = np.array([-local_x[1], local_x[0]], dtype=float)
        return local_x, local_y, length

    @staticmethod
    def _get_support_face(coord: np.ndarray, bounds: Tuple[float, float, float, float],
                          restraints: Dict) -> str:
        """Tentukan arah bidang tumpuan terdekat untuk orientasi simbol."""
        x_coord = float(coord[0])
        y_coord = float(coord[1])
        x_min, x_max, y_min, y_max = bounds

        restrain_x = int(restraints.get('X', 0)) == 1
        restrain_y = int(restraints.get('Y', 0)) == 1

        if restrain_y and not restrain_x:
            return 'bottom' if abs(y_coord - y_min) <= abs(y_coord - y_max) else 'top'
        if restrain_x and not restrain_y:
            return 'left' if abs(x_coord - x_min) <= abs(x_coord - x_max) else 'right'

        distances = {
            'bottom': abs(y_coord - y_min),
            'top': abs(y_coord - y_max),
            'left': abs(x_coord - x_min),
            'right': abs(x_coord - x_max)
        }
        return min(distances, key=distances.get)

    @staticmethod
    def _get_face_axes(face: str) -> Tuple[np.ndarray, np.ndarray]:
        """Ambil vektor tangent dan normal untuk bidang tumpuan."""
        mapping = {
            'bottom': (np.array([1.0, 0.0]), np.array([0.0, -1.0])),
            'top': (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
            'left': (np.array([0.0, 1.0]), np.array([-1.0, 0.0])),
            'right': (np.array([0.0, 1.0]), np.array([1.0, 0.0]))
        }
        return mapping.get(face, mapping['bottom'])

    @staticmethod
    def _get_support_hatch_segments(center: np.ndarray, tangent: np.ndarray,
                                    normal: np.ndarray, width: float,
                                    hatch_length: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Bangun geometri garis dasar dan arsiran simbol tumpuan."""
        start = center - 0.5 * width * tangent
        end = center + 0.5 * width * tangent
        segments = [(start, end)]

        hatch_count = int(np.clip(np.ceil(width / max(hatch_length * 0.9, 1.0)), 4, 9))
        offsets = np.linspace(-0.42 * width, 0.42 * width, hatch_count)
        slash_direction = normal - 0.55 * tangent
        slash_norm = np.linalg.norm(slash_direction)
        if slash_norm <= 1e-12:
            slash_direction = normal
        else:
            slash_direction = slash_direction / slash_norm

        for offset in offsets:
            hatch_start = center + offset * tangent
            hatch_end = hatch_start + hatch_length * slash_direction
            segments.append((hatch_start, hatch_end))

        return segments

    @staticmethod
    def _draw_support_hatch(ax, center: np.ndarray, tangent: np.ndarray,
                            normal: np.ndarray, width: float,
                            hatch_length: float, color: str) -> None:
        """Gambar garis dasar dan arsiran simbol tumpuan."""
        segments = PortalPlotter._get_support_hatch_segments(
            center, tangent, normal, width, hatch_length
        )
        for segment_index, (start, end) in enumerate(segments):
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                linewidth=1.1 if segment_index == 0 else 1.0
            )

    @staticmethod
    def _get_support_symbol_min_y(coord: np.ndarray, restraints: Dict,
                                  size: float,
                                  bounds: Tuple[float, float, float, float]) -> float:
        """Ambil ordinat paling bawah simbol tumpuan untuk penempatan label."""
        restrain_x = int(restraints.get('X', 0)) == 1
        restrain_y = int(restraints.get('Y', 0)) == 1
        restrain_r = int(restraints.get('R', 0)) == 1

        if not any((restrain_x, restrain_y, restrain_r)):
            return float(coord[1])

        face = PortalPlotter._get_support_face(coord, bounds, restraints)
        tangent, normal = PortalPlotter._get_face_axes(face)
        coord = np.asarray(coord, dtype=float)
        base_center = coord + 1.15 * size * normal
        y_values = [float(coord[1])]

        def add_segment_y(center: np.ndarray, width: float, hatch_length: float) -> None:
            segments = PortalPlotter._get_support_hatch_segments(
                center, tangent, normal, width, hatch_length
            )
            for start, end in segments:
                y_values.append(float(start[1]))
                y_values.append(float(end[1]))

        if restrain_x and restrain_y and restrain_r:
            support_line_center = coord + 0.45 * size * normal
            support_start = support_line_center - 0.95 * size * tangent
            support_end = support_line_center + 0.95 * size * tangent
            y_values.extend([float(support_start[1]), float(support_end[1])])
            add_segment_y(
                support_line_center + 0.25 * size * normal,
                width=2.1 * size,
                hatch_length=0.7 * size
            )
            return min(y_values)

        if restrain_x and restrain_y:
            triangle_points = [
                coord,
                base_center - 0.9 * size * tangent,
                base_center + 0.9 * size * tangent
            ]
            y_values.extend(float(point[1]) for point in triangle_points)
            add_segment_y(
                base_center + 0.18 * size * normal,
                width=2.5 * size,
                hatch_length=0.85 * size
            )
            return min(y_values)

        if restrain_y and not restrain_x:
            triangle_points = [
                coord,
                base_center - 0.78 * size * tangent,
                base_center + 0.78 * size * tangent
            ]
            y_values.extend(float(point[1]) for point in triangle_points)
            for offset in (-0.35 * size, 0.35 * size):
                center = base_center + 0.32 * size * normal + offset * tangent
                radius = 0.16 * size
                y_values.extend([float(center[1] - radius), float(center[1] + radius)])
            add_segment_y(
                base_center + 0.62 * size * normal,
                width=2.2 * size,
                hatch_length=0.8 * size
            )
            return min(y_values)

        if restrain_x and not restrain_y:
            triangle_points = [
                coord,
                base_center - 0.78 * size * tangent,
                base_center + 0.78 * size * tangent
            ]
            y_values.extend(float(point[1]) for point in triangle_points)
            for offset in (-0.35 * size, 0.35 * size):
                center = base_center + 0.32 * size * normal + offset * tangent
                radius = 0.16 * size
                y_values.extend([float(center[1] - radius), float(center[1] + radius)])
            add_segment_y(
                base_center + 0.62 * size * normal,
                width=2.2 * size,
                hatch_length=0.8 * size
            )
            return min(y_values)

        if restrain_r:
            y_values.extend([
                float(coord[1] - 0.28 * size),
                float(coord[1] + 0.28 * size)
            ])

        return min(y_values)

    @staticmethod
    def _draw_support_symbol(ax, coord: np.ndarray, restraints: Dict,
                             size: float, bounds: Tuple[float, float, float, float]) -> None:
        """Gambar simbol tumpuan sederhana pada node."""
        restrain_x = int(restraints.get('X', 0)) == 1
        restrain_y = int(restraints.get('Y', 0)) == 1
        restrain_r = int(restraints.get('R', 0)) == 1

        if not any((restrain_x, restrain_y, restrain_r)):
            return

        edge_color = 'dimgray'
        face_color = '#cfd8dc'
        face = PortalPlotter._get_support_face(coord, bounds, restraints)
        tangent, normal = PortalPlotter._get_face_axes(face)
        coord = np.asarray(coord, dtype=float)
        base_center = coord + 1.15 * size * normal

        if restrain_x and restrain_y and restrain_r:
            support_line_center = coord + 0.45 * size * normal
            support_start = support_line_center - 0.95 * size * tangent
            support_end = support_line_center + 0.95 * size * tangent
            ax.plot(
                [support_start[0], support_end[0]],
                [support_start[1], support_end[1]],
                color=edge_color,
                linewidth=2.4
            )
            PortalPlotter._draw_support_hatch(
                ax,
                support_line_center + 0.25 * size * normal,
                tangent,
                normal,
                width=2.1 * size,
                hatch_length=0.7 * size,
                color=edge_color
            )
            return

        if restrain_x and restrain_y:
            triangle = Polygon(
                [
                    coord,
                    base_center - 0.9 * size * tangent,
                    base_center + 0.9 * size * tangent
                ],
                closed=True,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=1.0,
                alpha=0.95
            )
            ax.add_patch(triangle)
            PortalPlotter._draw_support_hatch(
                ax,
                base_center + 0.18 * size * normal,
                tangent,
                normal,
                width=2.5 * size,
                hatch_length=0.85 * size,
                color=edge_color
            )
            return

        if restrain_y and not restrain_x:
            triangle = Polygon(
                [
                    coord,
                    base_center - 0.78 * size * tangent,
                    base_center + 0.78 * size * tangent
                ],
                closed=True,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=1.0,
                alpha=0.95
            )
            ax.add_patch(triangle)
            for offset in (-0.35 * size, 0.35 * size):
                roller = Circle(
                    base_center + 0.32 * size * normal + offset * tangent,
                    0.16 * size,
                    facecolor='white',
                    edgecolor=edge_color,
                    linewidth=1.0
                )
                ax.add_patch(roller)
            PortalPlotter._draw_support_hatch(
                ax,
                base_center + 0.62 * size * normal,
                tangent,
                normal,
                width=2.2 * size,
                hatch_length=0.8 * size,
                color=edge_color
            )
            return

        if restrain_x and not restrain_y:
            # Translasi horizontal tertahan tetapi vertikal bebas dipetakan
            # sebagai rol pada bidang vertikal.
            triangle = Polygon(
                [
                    coord,
                    base_center - 0.78 * size * tangent,
                    base_center + 0.78 * size * tangent
                ],
                closed=True,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=1.0,
                alpha=0.95
            )
            ax.add_patch(triangle)
            for offset in (-0.35 * size, 0.35 * size):
                roller = Circle(
                    base_center + 0.32 * size * normal + offset * tangent,
                    0.16 * size,
                    facecolor='white',
                    edgecolor=edge_color,
                    linewidth=1.0
                )
                ax.add_patch(roller)
            PortalPlotter._draw_support_hatch(
                ax,
                base_center + 0.62 * size * normal,
                tangent,
                normal,
                width=2.2 * size,
                hatch_length=0.8 * size,
                color=edge_color
            )
            return

        if restrain_r:
            fix_box = Rectangle(
                (coord[0] - 0.28 * size, coord[1] - 0.28 * size),
                0.56 * size,
                0.56 * size,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=1.0,
                alpha=0.95
            )
            ax.add_patch(fix_box)

    @staticmethod
    def _draw_distributed_load_preview(ax, element, magnitude: float,
                                       offset: float, color: str) -> None:
        """Gambar simbol beban merata global vertikal pada elemen."""
        if abs(magnitude) <= 1e-12:
            return

        start = np.asarray(element.coord_start, dtype=float)
        end = np.asarray(element.coord_end, dtype=float)
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length <= 1e-12:
            return

        direction = np.array([0.0, -1.0 if magnitude >= 0.0 else 1.0], dtype=float)
        offset_vector = -direction * offset
        arrow_count = int(np.clip(np.ceil(length / max(offset * 1.8, 1.0)), 4, 10))
        arrow_positions = np.linspace(0.0, 1.0, arrow_count)

        line_start = start + offset_vector
        line_end = end + offset_vector
        ax.plot(
            [line_start[0], line_end[0]],
            [line_start[1], line_end[1]],
            color=color,
            linewidth=1.4,
            alpha=0.9
        )

        for ratio in arrow_positions:
            member_point = start + ratio * vector
            arrow_start = member_point + offset_vector
            ax.annotate(
                '',
                xy=(member_point[0], member_point[1]),
                xytext=(arrow_start[0], arrow_start[1]),
                arrowprops=dict(arrowstyle='-|>', color=color, linewidth=1.2)
            )

        label_point = (start + end) / 2 + 1.2 * offset_vector
        ax.text(
            label_point[0],
            label_point[1],
            f"q={magnitude:.2f} kN/m",
            color=color,
            fontsize=8.5,
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', alpha=0.8)
        )

    @staticmethod
    def _draw_nodal_load_preview(ax, coord: np.ndarray, load_data: Dict,
                                 size: float, color: str) -> None:
        """Gambar simbol beban terpusat/nodal."""
        x_coord = float(coord[0])
        y_coord = float(coord[1])
        arrow_length = 3.0 * size
        labels = []
        x_positions = [x_coord]
        y_positions = [y_coord]
        label_bbox = dict(boxstyle='round,pad=0.18', facecolor='white', alpha=0.8)

        fx = float(load_data.get('Fx', 0.0))
        if abs(fx) > 1e-12:
            direction = np.array([np.sign(fx), 0.0], dtype=float)
            start = np.array([x_coord, y_coord], dtype=float) - direction * arrow_length
            end = np.array([x_coord, y_coord], dtype=float)
            ax.annotate(
                '',
                xy=(x_coord, y_coord),
                xytext=(start[0], start[1]),
                arrowprops=dict(arrowstyle='-|>', color=color, linewidth=1.8)
            )
            label_anchor = start + 0.5 * (end - start)
            horizontal_alignment = 'right' if fx > 0.0 else 'left'
            ax.annotate(
                f"Fx={fx:.2f} kN",
                xy=(float(label_anchor[0]), float(label_anchor[1])),
                xycoords='data',
                xytext=(0, -12),
                textcoords='offset points',
                fontsize=8.2,
                color=color,
                ha=horizontal_alignment,
                va='top',
                bbox=label_bbox
            )

        fy = float(load_data.get('Fy', 0.0))
        if abs(fy) > 1e-12:
            direction = np.array([0.0, np.sign(fy)], dtype=float)
            start = np.array([x_coord, y_coord], dtype=float) - direction * arrow_length
            end = np.array([x_coord, y_coord], dtype=float)
            ax.annotate(
                '',
                xy=(x_coord, y_coord),
                xytext=(start[0], start[1]),
                arrowprops=dict(arrowstyle='-|>', color=color, linewidth=1.8)
            )
            label_anchor = start + 0.5 * (end - start)
            ax.annotate(
                f"Fy={fy:.2f} kN",
                xy=(float(label_anchor[0]), float(label_anchor[1])),
                xycoords='data',
                xytext=(10, 0),
                textcoords='offset points',
                fontsize=8.2,
                color=color,
                ha='left',
                va='center',
                bbox=label_bbox
            )

        mz = float(load_data.get('Mz', 0.0))
        if abs(mz) > 1e-12:
            radius = 1.75 * size
            x_start = x_coord - 0.7 * radius
            x_end = x_coord + 0.7 * radius
            curve_rad = 0.9 if mz > 0.0 else -0.9
            ax.annotate(
                '',
                xy=(x_end, y_coord),
                xytext=(x_start, y_coord),
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=color,
                    linewidth=1.6,
                    connectionstyle=f'arc3,rad={curve_rad}'
                )
            )
            labels.append(f"Mz={mz:.2f} kN.m")
            x_positions.extend([x_start, x_end])
            y_positions.extend([y_coord - radius, y_coord + radius])

        if labels:
            label_anchor_x = 0.5 * (min(x_positions) + max(x_positions))
            label_anchor_y = min(y_positions)
            ax.annotate(
                "\n".join(labels),
                xy=(label_anchor_x, label_anchor_y),
                xycoords='data',
                xytext=(0, -8),
                textcoords='offset points',
                fontsize=8.2,
                color=color,
                ha='center',
                va='top',
                bbox=label_bbox
            )

    @staticmethod
    def expand_axes_for_data_text(ax, pad_fraction: float = 0.02) -> None:
        """
        Perluas batas sumbu agar label berbasis koordinat data tidak terpotong.

        Hanya teks yang memang ditempatkan pada koordinat data yang
        diperhitungkan; teks berbasis `transAxes` atau elemen UI lain diabaikan.
        """
        if ax is None or not getattr(ax, 'texts', None):
            return

        data_texts = []
        for text in ax.texts:
            if not text.get_visible():
                continue
            transform = text.get_transform()
            if hasattr(transform, 'contains_branch') and transform.contains_branch(ax.transData):
                data_texts.append(text)

        if not data_texts:
            return

        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        data_transform_inv = ax.transData.inverted()

        original_xlim = ax.get_xlim()
        original_ylim = ax.get_ylim()
        x_min, x_max = sorted(original_xlim)
        y_min, y_max = sorted(original_ylim)
        expanded_x_min = x_min
        expanded_x_max = x_max
        expanded_y_min = y_min
        expanded_y_max = y_max

        for text in data_texts:
            bbox = text.get_window_extent(renderer=renderer)
            if bbox.width <= 0 and bbox.height <= 0:
                continue

            corners_display = np.array([
                [bbox.x0, bbox.y0],
                [bbox.x0, bbox.y1],
                [bbox.x1, bbox.y0],
                [bbox.x1, bbox.y1]
            ], dtype=float)
            corners_data = data_transform_inv.transform(corners_display)
            expanded_x_min = min(expanded_x_min, float(np.min(corners_data[:, 0])))
            expanded_x_max = max(expanded_x_max, float(np.max(corners_data[:, 0])))
            expanded_y_min = min(expanded_y_min, float(np.min(corners_data[:, 1])))
            expanded_y_max = max(expanded_y_max, float(np.max(corners_data[:, 1])))

        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)
        x_pad = pad_fraction * x_span
        y_pad = pad_fraction * y_span

        new_xlim = (expanded_x_min - x_pad, expanded_x_max + x_pad)
        new_ylim = (expanded_y_min - y_pad, expanded_y_max + y_pad)

        if original_xlim[0] > original_xlim[1]:
            new_xlim = new_xlim[::-1]
        if original_ylim[0] > original_ylim[1]:
            new_ylim = new_ylim[::-1]

        ax.set_xlim(*new_xlim)
        ax.set_ylim(*new_ylim)

    @staticmethod
    def get_structure_span(nodes: np.ndarray, elements: List) -> float:
        """Ambil bentang karakteristik struktur untuk kebutuhan scaling plot."""
        if nodes is not None and len(nodes):
            coords = np.asarray(nodes[:, 1:3], dtype=float)
        else:
            coords = np.vstack([
                np.asarray(element.coord_start, dtype=float)
                for element in elements
            ] + [
                np.asarray(element.coord_end, dtype=float)
                for element in elements
            ])

        if coords.size == 0:
            return 1.0

        return float(max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), 1.0))

    @staticmethod
    def get_max_translational_displacement(displacements: np.ndarray) -> float:
        """Hitung resultan perpindahan translasi maksimum dari semua node."""
        values = np.asarray(displacements, dtype=float)
        if values.size == 0:
            return 0.0

        translational = values.reshape(-1, 3)[:, :2]
        if translational.size == 0:
            return 0.0

        return float(np.max(np.linalg.norm(translational, axis=1)))

    @staticmethod
    def get_force_profile(force_data: Dict, force_type: str = 'moment',
                          num_points: int = 101,
                          prefer_continuous: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ambil profil gaya dalam sepanjang elemen.

        Jika `section_profile` tersedia maka dipakai langsung; jika tidak,
        profil dibangun ulang dari gaya ujung dan beban merata lokal.
        """
        if prefer_continuous and force_type == 'moment':
            length_m = float(force_data.get('length_m', 0.0))
            x_profile = np.linspace(0.0, length_m, num_points)
            start_moment = float(force_data.get('moment_start', 0.0))
            # Konvensi diagram batang: momen internal di ujung kanan
            # diambil dari aksi joint ujung kanan dengan tanda berlawanan.
            end_moment_internal = -float(force_data.get('moment_end', 0.0))
            transverse_load = float(force_data.get('transverse_local_load', 0.0))

            if length_m <= 1e-12:
                force_profile = np.full_like(x_profile, start_moment, dtype=float)
            else:
                linear_term = (
                    end_moment_internal
                    - start_moment
                    + 0.5 * transverse_load * length_m**2
                ) / length_m
                force_profile = (
                    start_moment
                    + linear_term * x_profile
                    - 0.5 * transverse_load * x_profile**2
                )

            return x_profile, force_profile

        if (
            not prefer_continuous
            and 'section_profile' in force_data
            and force_data['section_profile']
        ):
            x_profile = np.array([
                point['x_m']
                for point in force_data['section_profile']
            ], dtype=float)
            force_profile = np.array([
                point[force_type]
                for point in force_data['section_profile']
            ], dtype=float)
            return x_profile, force_profile

        length_m = float(force_data.get('length_m', 0.0))
        x_profile = np.linspace(0.0, length_m, num_points)

        if force_type == 'axial':
            force_profile = (
                float(force_data.get('axial_start', 0.0))
                + float(force_data.get('axial_local_load', 0.0)) * x_profile
            )
        elif force_type == 'shear':
            force_profile = (
                -float(force_data.get('shear_start', 0.0))
                -float(force_data.get('transverse_local_load', 0.0)) * x_profile
            )
        elif force_type == 'moment':
            force_profile = (
                float(force_data.get('moment_start', 0.0))
                - float(force_data.get('shear_start', 0.0)) * x_profile
                - 0.5 * float(force_data.get('transverse_local_load', 0.0)) * x_profile**2
            )
        else:
            raise ValueError(f"force_type '{force_type}' tidak dikenali")

        return x_profile, force_profile

    @staticmethod
    def get_beam_elements(elements: List, slope_tolerance: float = 0.05) -> List:
        """
        Ambil elemen yang bersifat balok/hampir horizontal.

        slope_tolerance dinyatakan sebagai rasio |dy| / |dx|.
        """
        beam_elements = []
        for element in elements:
            start = np.asarray(element.coord_start, dtype=float)
            end = np.asarray(element.coord_end, dtype=float)
            dx = abs(float(end[0] - start[0]))
            dy = abs(float(end[1] - start[1]))

            if dx <= 1e-9:
                continue

            if (dy / dx) <= slope_tolerance:
                beam_elements.append(element)

        return beam_elements

    @staticmethod
    def suggest_deformation_scale(nodes: np.ndarray, elements: List,
                                  displacements: np.ndarray,
                                  target_ratio: float = 0.05,
                                  min_scale: float = 1.0,
                                  max_scale: float = 1e6) -> float:
        """
        Rekomendasikan skala plot agar deformasi terlihat tanpa terlalu berlebihan.

        target_ratio menyatakan fraksi bentang struktur yang ingin dipakai sebagai
        defleksi visual maksimum.
        """
        structure_span = PortalPlotter.get_structure_span(nodes, elements)
        max_displacement = PortalPlotter.get_max_translational_displacement(
            displacements)

        if max_displacement <= 1e-12:
            return float(min_scale)

        suggested = (structure_span * target_ratio) / max_displacement
        return float(np.clip(suggested, min_scale, max_scale))
    
    @staticmethod
    def plot_portal_geometry(nodes: np.ndarray, elements: List,
                            boundary_conditions: Optional[Dict] = None,
                            distributed_loads: Optional[Dict[int, float]] = None,
                            nodal_loads: Optional[Dict[int, Dict]] = None,
                            title: str = "Portal 2D Geometry") -> None:
        """
        Plot geometri portal beserta simbol input utama.
        
        Parameters:
        - nodes: array koordinat nodes [node_id, x, y]
        - elements: list Element2D objects
        - boundary_conditions: dict kondisi tumpuan per node
        - distributed_loads: dict {elem_id: q_total_global_vertical}
        - nodal_loads: dict {node_id: {Fx, Fy, Mz}}
        """
        fig, ax = plt.subplots(1, 1, figsize=(11, 8))
        boundary_conditions = boundary_conditions or {}
        distributed_loads = distributed_loads or {}
        nodal_loads = nodal_loads or {}
        structure_span = PortalPlotter.get_structure_span(nodes, elements)
        symbol_size = 0.025 * structure_span
        load_offset = 0.06 * structure_span
        node_label_offset = 0.018 * structure_span
        x_values = np.asarray(nodes[:, 1], dtype=float)
        y_values = np.asarray(nodes[:, 2], dtype=float)
        support_bounds = (
            float(np.min(x_values)),
            float(np.max(x_values)),
            float(np.min(y_values)),
            float(np.max(y_values))
        )
        
        # Plot elemen
        for element in elements:
            node_start = element.coord_start
            node_end = element.coord_end
            ax.plot([node_start[0], node_end[0]],
                   [node_start[1], node_end[1]],
                   color='#37474f', linewidth=2.2)

            mid_point = (np.asarray(node_start, dtype=float) + np.asarray(node_end, dtype=float)) / 2
            ax.text(
                mid_point[0],
                mid_point[1],
                f"E{int(element.elem_id)}",
                fontsize=9,
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white', alpha=0.8)
            )

            if int(element.elem_id) in distributed_loads:
                PortalPlotter._draw_distributed_load_preview(
                    ax,
                    element,
                    float(distributed_loads[int(element.elem_id)]),
                    load_offset,
                    color='#2e7d32'
                )
        
        # Plot nodal
        for node in nodes:
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
                markeredgecolor='#d32f2f',
                markeredgewidth=1.4
            )

            label_x = coord[0]
            label_y = coord[1] - node_label_offset
            label_ha = 'center'
            label_va = 'top'
            label_bbox = dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8)

            if any(int(restraints.get(axis, 0)) == 1 for axis in ('X', 'Y', 'R')):
                support_min_y = PortalPlotter._get_support_symbol_min_y(
                    coord, restraints, symbol_size, support_bounds
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
                    bbox=label_bbox
                )
            else:
                ax.text(
                    label_x,
                    label_y,
                    f"N{node_id}",
                    fontsize=8.5,
                    ha=label_ha,
                    va=label_va,
                    bbox=label_bbox
                )

            if node_id in nodal_loads:
                PortalPlotter._draw_nodal_load_preview(
                    ax,
                    coord,
                    nodal_loads[node_id],
                    symbol_size,
                    color='#c62828'
                )
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.margins(0.08)
        PortalPlotter.expand_axes_for_data_text(ax)
        
        return fig, ax
    
    @staticmethod
    def plot_deformed_shape(nodes: np.ndarray, elements: List,
                           displacements: np.ndarray,
                           scale_factor: float = None,
                           show_result_labels: bool = True) -> None:
        """
        Plot bentuk deformasi portal
        
        Parameters:
        - nodes: koordinat nodes
        - elements: list elemen
        - displacements: array displacement global
        - scale_factor: faktor pengali untuk visualisasi.
          Jika None, dipilih otomatis berdasarkan bentang struktur dan
          perpindahan maksimum.
        """
        if scale_factor is None:
            scale_factor = PortalPlotter.suggest_deformation_scale(
                nodes, elements, displacements)

        max_displacement = PortalPlotter.get_max_translational_displacement(
            displacements)
        structure_span = PortalPlotter.get_structure_span(nodes, elements)
        node_label_offset = 0.02 * structure_span
        deformed_label_offset = np.array(
            [0.015 * structure_span, 0.015 * structure_span],
            dtype=float
        )
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plotted_undeformed = False
        plotted_deformed = False
        
        # Undeformed
        for element in elements:
            node_start = element.coord_start
            node_end = element.coord_end
            ax.plot([node_start[0], node_end[0]],
                   [node_start[1], node_end[1]],
                   'b--', linewidth=1, alpha=0.5,
                   label='Undeformed' if not plotted_undeformed else '')
            plotted_undeformed = True
        
        # Deformed
        for element in elements:
            node_start_id = element.node_start
            node_end_id = element.node_end
            
            dof_start = (node_start_id - 1) * 3
            dof_end = (node_end_id - 1) * 3
            
            node_start_deformed = element.coord_start + scale_factor * np.array([
                displacements[dof_start],
                displacements[dof_start + 1]
            ])
            
            node_end_deformed = element.coord_end + scale_factor * np.array([
                displacements[dof_end],
                displacements[dof_end + 1]
            ])
            
            ax.plot([node_start_deformed[0], node_end_deformed[0]],
                   [node_start_deformed[1], node_end_deformed[1]],
                   'r-', linewidth=2,
                   label='Deformed' if not plotted_deformed else '')
            plotted_deformed = True

        # Label node pada posisi asli agar tetap konsisten dengan nomor input.
        for node in nodes:
            node_id = int(node[0])
            x_coord = float(node[1])
            y_coord = float(node[2])
            dof = (node_id - 1) * 3
            ux = float(displacements[dof])
            uy = float(displacements[dof + 1])
            deformed_point = np.array([x_coord, y_coord], dtype=float) + (
                scale_factor * np.array([ux, uy], dtype=float)
            )
            ax.text(
                x_coord,
                y_coord - node_label_offset,
                f"N{node_id}",
                fontsize=8.5,
                ha='center',
                va='top',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75)
            )
            if show_result_labels:
                ax.plot(
                    deformed_point[0],
                    deformed_point[1],
                    'ro',
                    markersize=4,
                    alpha=0.9
                )
                ax.text(
                    deformed_point[0] + deformed_label_offset[0],
                    deformed_point[1] + deformed_label_offset[1],
                    f"Ux={ux:.3f} mm\nUy={uy:.3f} mm",
                    fontsize=7.5,
                    color='darkred',
                    ha='left',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.18', facecolor='white', alpha=0.8)
                )

        # Label elemen di titik tengah batang seperti diagram gaya dalam.
        for element in elements:
            start = np.asarray(element.coord_start, dtype=float)
            end = np.asarray(element.coord_end, dtype=float)
            mid_point = (start + end) / 2
            ax.text(
                mid_point[0],
                mid_point[1],
                f"E{int(element.elem_id)}",
                fontsize=9,
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
            )

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title(
         #   f'Deformed Shape (Scale factor: {scale_factor:,.0f}x, '
            f'Max displacement: {max_displacement:.6f} mm)'
        )
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.margins(0.05)
        PortalPlotter.expand_axes_for_data_text(ax)
        ax.legend()

        return fig, ax

    @staticmethod
    def plot_internal_force_diagram(elements: List, element_forces: List[Dict],
                                   force_type: str = 'moment',
                                   scale_factor: float = None,
                                   relative_to_chord: bool = False,
                                   show_result_labels: bool = True):
        """
        Plot diagram gaya dalam pada geometri portal.

        Parameters:
        - elements: list Element2D objects
        - element_forces: list hasil gaya dalam per elemen
        - force_type: 'axial', 'shear', atau 'moment'
        - scale_factor: faktor skala diagram; jika None dihitung otomatis
        - relative_to_chord: jika True untuk momen, profil yang digambar adalah
          deviasi terhadap garis lurus penghubung momen ujung agar kurvatur
          lebih jelas terlihat. Nilai `'mixed'` berarti balok memakai mode
          relatif terhadap chord, sedangkan kolom tetap momen absolut.
        """
        force_config = {
            'axial': {
                'start_key': 'axial_start',
                'end_key': 'axial_end_internal',
                'end_joint_key': 'axial_end',
                'profile_key': 'axial',
                'title': 'Axial Force Diagram',
                'unit': 'kN',
                'color': 'tab:blue'
            },
            'shear': {
                'start_key': 'shear_start',
                'end_key': 'shear_end_internal',
                'end_joint_key': 'shear_end',
                'profile_key': 'shear',
                'title': 'Shear Force Diagram',
                'unit': 'kN',
                'color': 'tab:orange'
            },
            'moment': {
                'start_key': 'moment_start',
                'end_key': 'moment_end_internal',
                'end_joint_key': 'moment_end',
                'profile_key': 'moment',
                'title': 'Bending Moment Diagram',
                'unit': 'kN.m',
                'color': 'tab:green'
            }
        }

        if force_type not in force_config:
            raise ValueError(f"force_type '{force_type}' tidak dikenali")

        config = force_config[force_type]
        profile_num_points = 401 if force_type == 'moment' else 101
        if force_type == 'axial':
            # Konvensi solver aksial: positif = tekan, negatif = tarik.
            positive_color = 'tab:red'
            negative_color = 'tab:blue'
        else:
            positive_color = 'tab:blue'
            negative_color = 'tab:red'
        zero_color = '0.45'
        sign_tolerance = 1e-12
        curvature_tolerance = 1e-9

        def get_signed_color(value: float) -> str:
            if value > sign_tolerance:
                return positive_color
            if value < -sign_tolerance:
                return negative_color
            return zero_color

        def get_positive_peak(x_values: np.ndarray,
                              response_values: np.ndarray,
                              interior_only: bool = False):
            x_values = np.asarray(x_values, dtype=float)
            response_values = np.asarray(response_values, dtype=float)
            if len(x_values) == 0 or len(response_values) == 0:
                return None

            candidate_mask = response_values > sign_tolerance
            if interior_only and len(x_values) > 1:
                candidate_mask &= (
                    (x_values > 1e-9)
                    & (x_values < x_values[-1] - 1e-9)
                )

            if not np.any(candidate_mask):
                return None

            candidate_indices = np.flatnonzero(candidate_mask)
            peak_idx = int(candidate_indices[np.argmax(response_values[candidate_mask])])
            return {
                'index': peak_idx,
                'x': float(x_values[peak_idx]),
                'value': float(response_values[peak_idx])
            }

        beam_element_ids = {
            int(element.elem_id)
            for element in PortalPlotter.get_beam_elements(elements)
        }
        element_lookup = {
            int(element.elem_id): element
            for element in elements
        }
        node_connection_counts = {}
        for element in elements:
            node_connection_counts[int(element.node_start)] = (
                node_connection_counts.get(int(element.node_start), 0) + 1
            )
            node_connection_counts[int(element.node_end)] = (
                node_connection_counts.get(int(element.node_end), 0) + 1
            )

        def is_cantilever_beam(elem_id: Optional[int] = None) -> bool:
            if elem_id not in beam_element_ids or elem_id not in element_lookup:
                return False
            element = element_lookup[elem_id]
            start_count = node_connection_counts.get(int(element.node_start), 0)
            end_count = node_connection_counts.get(int(element.node_end), 0)
            return min(start_count, end_count) <= 1

        def should_use_relative_profile(elem_id: Optional[int] = None) -> bool:
            if force_type != 'moment':
                return False
            if relative_to_chord == 'mixed':
                return elem_id in beam_element_ids
            return bool(relative_to_chord)

        def should_use_blended_beam_profile(elem_id: Optional[int] = None) -> bool:
            return force_type == 'moment' and relative_to_chord == 'mixed' and elem_id in beam_element_ids

        def should_prefer_reconstructed_profile(elem_id: Optional[int] = None) -> bool:
            if force_type != 'moment':
                return False
            return True

        force_lookup = {
            int(force['elem_id']): force
            for force in element_forces
        }

        coords = np.vstack([
            np.asarray(element.coord_start, dtype=float)
            for element in elements
        ] + [
            np.asarray(element.coord_end, dtype=float)
            for element in elements
        ])
        structure_center_x = 0.5 * (
            float(np.min(coords[:, 0])) + float(np.max(coords[:, 0]))
        ) if len(coords) else 0.0
        structure_span = max(
            np.ptp(coords[:, 0]) if len(coords) else 0,
            np.ptp(coords[:, 1]) if len(coords) else 0,
            1.0
        )

        max_force = 0.0
        beam_curvature_max = 0.0
        absolute_end_max = 0.0
        for force in element_forces:
            elem_id = int(force['elem_id'])
            x_profile_m, force_profile = PortalPlotter.get_force_profile(
                force,
                force_type=config['profile_key'],
                num_points=profile_num_points,
                prefer_continuous=should_prefer_reconstructed_profile(elem_id)
            )
            if len(force_profile) == 0:
                continue

            if relative_to_chord == 'mixed' and force_type == 'moment':
                start_end_force = float(force_profile[0])
                end_end_force = float(force_profile[-1])
                absolute_end_max = max(
                    absolute_end_max,
                    abs(start_end_force),
                    abs(end_end_force)
                )
                if elem_id in beam_element_ids:
                    chord_profile = np.linspace(
                        force_profile[0],
                        force_profile[-1],
                        len(force_profile)
                    )
                    curvature_profile = force_profile - chord_profile
                    curvature_profile[np.abs(curvature_profile) <= curvature_tolerance] = 0.0
                    beam_curvature_max = max(
                        beam_curvature_max,
                        float(np.max(np.abs(curvature_profile)))
                    )
                continue

            if should_use_relative_profile(elem_id):
                chord_profile = np.linspace(
                    force_profile[0],
                    force_profile[-1],
                    len(force_profile)
                )
                force_profile = force_profile - chord_profile

            element_max_force = float(np.max(np.abs(force_profile)))
            max_force = max(max_force, element_max_force)

        if relative_to_chord == 'mixed' and force_type == 'moment' and scale_factor is None:
            # Momen ujung balok dan kolom memakai satu skala absolut yang sama.
            # Hanya deviasi parabola balok yang diperjelas dengan skala terpisah.
            absolute_scale_factor = (
                0.12 * structure_span / absolute_end_max
                if absolute_end_max else 0.0
            )
            beam_curvature_scale_factor = (
                0.18 * structure_span / beam_curvature_max
                if beam_curvature_max > curvature_tolerance else 0.0
            )
            beam_scale_factor = beam_curvature_scale_factor
            nonbeam_scale_factor = absolute_scale_factor
        else:
            scale_factor = 0.15 * structure_span / max_force if (
                scale_factor is None and max_force
            ) else (0.0 if scale_factor is None else scale_factor)
            beam_scale_factor = scale_factor
            nonbeam_scale_factor = scale_factor

        figure_dpi = 180 if force_type == 'moment' else 100
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=figure_dpi)
        plotted_member = False
        plotted_diagram = False

        for element in elements:
            start = np.asarray(element.coord_start, dtype=float)
            end = np.asarray(element.coord_end, dtype=float)
            direction = end - start
            length = np.linalg.norm(direction)

            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color='0.65',
                linestyle='--',
                linewidth=1.2,
                label='Portal member' if not plotted_member else ''
            )
            plotted_member = True

            if length == 0 or element.elem_id not in force_lookup:
                continue

            tangent = direction / length
            normal = np.array([-tangent[1], tangent[0]])
            mid_x = float((start[0] + end[0]) / 2.0)
            is_column_like = abs(tangent[1]) > abs(tangent[0])
            force_data = force_lookup[element.elem_id]
            is_beam_element = int(element.elem_id) in beam_element_ids
            x_profile_m, force_profile = PortalPlotter.get_force_profile(
                force_data,
                force_type=config['profile_key'],
                num_points=profile_num_points,
                prefer_continuous=should_prefer_reconstructed_profile(
                    int(element.elem_id)
                )
            )
            if len(force_profile) == 0:
                continue

            if force_type == 'moment':
                force_profile = PortalPlotter.convert_beam_moment_to_display_convention(
                    force_profile,
                    is_beam=is_beam_element
                )
            elif force_type == 'shear':
                # Diagram geser ditampilkan dengan konvensi visual yang dibaca
                # pengguna pada referensi: balok sisi kiri positif, dan kolom
                # mengarah ke dalam bentang portal.
                force_profile = -np.asarray(force_profile, dtype=float)

            if (
                force_type == 'moment'
                and not is_beam_element
                and is_column_like
                and mid_x > structure_center_x
            ):
                force_profile = -force_profile

            x_profile_mm = x_profile_m * 1000.0
            base_points = np.array([
                start + tangent * x_mm
                for x_mm in x_profile_mm
            ])
            start_force = float(force_profile[0])
            end_force = float(force_profile[-1])
            start_joint_force = float(force_data[config['start_key']])
            end_joint_force = float(force_data[config['end_joint_key']])
            if force_type == 'moment' and is_beam_element:
                start_joint_force = -start_joint_force
                end_joint_force = -end_joint_force
            if (
                force_type == 'moment'
                and not is_beam_element
                and is_column_like
                and mid_x > structure_center_x
            ):
                start_joint_force = -start_joint_force
                end_joint_force = -end_joint_force
            display_profile = force_profile.copy()
            use_relative_profile = should_use_relative_profile(int(element.elem_id))
            use_blended_beam_profile = should_use_blended_beam_profile(int(element.elem_id))
            internal_baseline_profile = np.linspace(
                start_force,
                end_force,
                len(force_profile)
            )

            if use_relative_profile:
                chord_profile = np.linspace(
                    start_force,
                    end_force,
                    len(force_profile)
                )
                display_profile = force_profile - chord_profile
                display_profile[np.abs(display_profile) <= curvature_tolerance] = 0.0
            if use_blended_beam_profile:
                # Balok pada diagram utama portal penuh memakai baseline momen ujung
                # ditambah kurvatur yang diperjelas agar parabola tetap terlihat
                # tanpa jatuh ke nol di kedua ujung.
                offset_points = (
                    base_points
                    + np.outer(internal_baseline_profile, normal) * nonbeam_scale_factor
                    + np.outer(display_profile, normal) * beam_scale_factor
                )
            elif relative_to_chord == 'mixed' and force_type == 'moment':
                offset_points = (
                    base_points
                    + np.outer(force_profile, normal) * nonbeam_scale_factor
                )
            else:
                active_scale_factor = (
                    beam_scale_factor if use_relative_profile else nonbeam_scale_factor
                )
                offset_points = (
                    base_points
                    + np.outer(display_profile, normal) * active_scale_factor
                )
            display_signed_ordinates = np.sum(
                (offset_points - base_points) * normal,
                axis=1
            )
            # Warna diagram shear mengikuti tanda gaya geser aktual,
            # sehingga kolom kanan E4-E6 yang bernilai negatif tetap merah.
            if force_type == 'shear':
                color_reference_profile = force_profile
            # Warna momen kolom pada diagram utama mengikuti tanda momen joint
            # di kedua ujung elemen, karena itulah konvensi yang dibaca pengguna
            # saat membandingkan dengan output beam end forces.
            elif force_type == 'moment' and not is_beam_element and relative_to_chord is False:
                color_reference_profile = np.linspace(
                    start_joint_force,
                    end_joint_force,
                    len(display_signed_ordinates)
                )
            # Pada mode campuran, kolom tetap diberi warna berdasarkan pembacaan joint.
            elif force_type == 'moment' and relative_to_chord == 'mixed' and int(element.elem_id) not in beam_element_ids:
                color_reference_profile = np.linspace(
                    start_joint_force,
                    end_joint_force,
                    len(display_signed_ordinates)
                )
            else:
                color_reference_profile = display_signed_ordinates
            line_label = f"{config['title']} ({config['unit']})" if not plotted_diagram else ''
            def get_sign_class(value: float, fallback: int = 0) -> int:
                if value > sign_tolerance:
                    return 1
                if value < -sign_tolerance:
                    return -1
                return fallback

            initial_sign = 0
            for value in color_reference_profile:
                initial_sign = get_sign_class(float(value), initial_sign)
                if initial_sign != 0:
                    break
            if initial_sign == 0:
                initial_sign = 1

            signed_regions = []
            current_sign = initial_sign
            current_base_region = [base_points[0]]
            current_offset_region = [offset_points[0]]

            for idx in range(len(base_points) - 1):
                value_start = float(color_reference_profile[idx])
                value_end = float(color_reference_profile[idx + 1])
                start_sign = get_sign_class(value_start, current_sign)
                end_sign = get_sign_class(value_end, start_sign)
                base_start = base_points[idx]
                base_end = base_points[idx + 1]
                offset_start = offset_points[idx]
                offset_end = offset_points[idx + 1]

                if start_sign == end_sign or abs(value_start - value_end) <= sign_tolerance:
                    current_base_region.append(base_end)
                    current_offset_region.append(offset_end)
                    current_sign = end_sign
                    continue

                denominator = abs(value_start) + abs(value_end)
                interpolation = (
                    0.5 if denominator <= sign_tolerance
                    else abs(value_start) / denominator
                )
                base_cross = base_start + interpolation * (base_end - base_start)
                offset_cross = offset_start + interpolation * (offset_end - offset_start)
                current_base_region.append(base_cross)
                current_offset_region.append(offset_cross)
                signed_regions.append((
                    current_sign,
                    np.asarray(current_base_region, dtype=float),
                    np.asarray(current_offset_region, dtype=float)
                ))
                current_sign = end_sign
                current_base_region = [base_cross, base_end]
                current_offset_region = [offset_cross, offset_end]

            signed_regions.append((
                current_sign,
                np.asarray(current_base_region, dtype=float),
                np.asarray(current_offset_region, dtype=float)
            ))

            for region_sign, region_base_points, region_offset_points in signed_regions:
                run_color = get_signed_color(float(region_sign))
                fill_kwargs = {}
                line_kwargs = {}
                if force_type == 'moment':
                    fill_kwargs = {
                        'edgecolor': 'none',
                        'linewidth': 0.0,
                        'antialiased': True
                    }
                    line_kwargs = {
                        'solid_joinstyle': 'round',
                        'solid_capstyle': 'round'
                    }

                use_fill_between = np.max(
                    np.abs(region_offset_points[:, 0] - region_base_points[:, 0])
                ) <= 1e-9
                use_fill_betweenx = np.max(
                    np.abs(region_offset_points[:, 1] - region_base_points[:, 1])
                ) <= 1e-9

                if use_fill_between:
                    ax.fill_between(
                        region_base_points[:, 0],
                        region_base_points[:, 1],
                        region_offset_points[:, 1],
                        color=run_color,
                        alpha=0.20,
                        **fill_kwargs
                    )
                elif use_fill_betweenx:
                    ax.fill_betweenx(
                        region_base_points[:, 1],
                        region_base_points[:, 0],
                        region_offset_points[:, 0],
                        color=run_color,
                        alpha=0.20,
                        **fill_kwargs
                    )
                else:
                    region_polygon = np.vstack([
                        region_base_points,
                        region_offset_points[::-1]
                    ])
                    ax.fill(
                        region_polygon[:, 0],
                        region_polygon[:, 1],
                        color=run_color,
                        alpha=0.20,
                        **fill_kwargs
                    )
                ax.plot(
                    region_offset_points[:, 0],
                    region_offset_points[:, 1],
                    color=run_color,
                    linewidth=2,
                    label=line_label,
                    **line_kwargs
                )
                line_label = ''

            plotted_diagram = True
            start_segment_color = get_signed_color(float(color_reference_profile[0]))
            end_segment_color = get_signed_color(float(color_reference_profile[-1]))
            ax.plot(
                [base_points[0, 0], offset_points[0, 0]],
                [base_points[0, 1], offset_points[0, 1]],
                color=start_segment_color,
                linewidth=1
            )
            ax.plot(
                [base_points[-1, 0], offset_points[-1, 0]],
                [base_points[-1, 1], offset_points[-1, 1]],
                color=end_segment_color,
                linewidth=1
            )

            label_start_force = start_joint_force
            label_end_force = end_joint_force
            if force_type == 'shear':
                label_start_force = start_force
                label_end_force = end_force
            if relative_to_chord == 'mixed' and force_type == 'moment':
                label_start_force = start_force
                label_end_force = end_force
            label_start_color = get_signed_color(float(label_start_force))
            label_end_color = get_signed_color(float(label_end_force))

            if show_result_labels:
                if use_relative_profile:
                    label_offset = normal * (0.03 * structure_span)
                    ax.text(
                        base_points[0, 0] + label_offset[0],
                        base_points[0, 1] + label_offset[1],
                        f"{label_start_force:.2f}",
                        fontsize=8,
                        color=label_start_color,
                        ha='right',
                        va='bottom',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75)
                    )
                    ax.text(
                        base_points[-1, 0] + label_offset[0],
                        base_points[-1, 1] + label_offset[1],
                        f"{label_end_force:.2f}",
                        fontsize=8,
                        color=label_end_color,
                        ha='left',
                        va='bottom',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75)
                    )
                else:
                    if force_type == 'moment':
                        if not is_beam_element and is_column_like:
                            outward_sign = -1.0 if mid_x <= structure_center_x else 1.0
                            label_offset = np.array(
                                [outward_sign * (0.038 * structure_span), 0.0],
                                dtype=float
                            )
                            tangent_shift = tangent * (0.018 * structure_span)
                            start_anchor = start + label_offset + tangent_shift
                            end_anchor = end + label_offset - tangent_shift
                            start_ha = 'right' if outward_sign < 0 else 'left'
                            end_ha = start_ha
                        elif is_beam_element:
                            label_offset = normal * (0.03 * structure_span)
                            tangent_shift = tangent * (0.02 * structure_span)
                            start_anchor = start + label_offset + tangent_shift
                            end_anchor = end + label_offset - tangent_shift
                            start_ha = 'left'
                            end_ha = 'right'
                        else:
                            label_offset = normal * (0.03 * structure_span)
                            start_anchor = start + label_offset
                            end_anchor = end + label_offset
                            start_ha = 'right'
                            end_ha = 'left'
                    else:
                        label_offset = np.zeros(2, dtype=float)
                        start_anchor = start
                        end_anchor = end
                        start_ha = 'right'
                        end_ha = 'left'
                    ax.text(
                        start_anchor[0],
                        start_anchor[1],
                        f"{label_start_force:.2f}",
                        fontsize=8,
                        color=label_start_color,
                        ha=start_ha,
                        va='bottom',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75)
                    )
                    ax.text(
                        end_anchor[0],
                        end_anchor[1],
                        f"{label_end_force:.2f}",
                        fontsize=8,
                        color=label_end_color,
                        ha=end_ha,
                        va='bottom',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75)
                    )

            if (
                show_result_labels
                and relative_to_chord is False
                and force_type == 'moment'
                and is_beam_element
            ):
                positive_peak = get_positive_peak(
                    x_profile_m,
                    force_profile,
                    interior_only=False
                )
                if positive_peak is not None:
                    peak_idx = int(positive_peak['index'])
                    peak_point = offset_points[peak_idx]
                    tangent_sign = 1.0 if positive_peak['x'] <= 0.5 * x_profile_m[-1] else -1.0
                    label_anchor = (
                        peak_point
                        + tangent * (0.026 * structure_span * tangent_sign)
                        + normal * (0.022 * structure_span)
                    )
                    label_ha = 'left' if tangent_sign > 0 else 'right'
                    peak_color = get_signed_color(positive_peak['value'])
                    ax.scatter(
                        [peak_point[0]],
                        [peak_point[1]],
                        color=peak_color,
                        s=18,
                        zorder=3
                    )
                    ax.text(
                        label_anchor[0],
                        label_anchor[1],
                        f"M+max={positive_peak['value']:.2f} @ x={positive_peak['x']:.2f} m",
                        fontsize=7.5,
                        color=peak_color,
                        ha=label_ha,
                        va='bottom',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8)
                    )

            if (
                show_result_labels
                and relative_to_chord == 'mixed'
                and force_type == 'moment'
                and int(element.elem_id) in beam_element_ids
                and not is_cantilever_beam(int(element.elem_id))
                and abs(float(force_data.get('transverse_local_load', 0.0))) > 1e-12
            ):
                positive_peak = get_positive_peak(
                    x_profile_m,
                    force_profile,
                    interior_only=True
                )
                if positive_peak is not None:
                    point_idx = int(positive_peak['index'])
                    field_point = offset_points[point_idx]
                    ax.scatter(
                        [field_point[0]],
                        [field_point[1]],
                        color=get_signed_color(positive_peak['value']),
                        s=18,
                        zorder=3
                    )
                    ax.text(
                        field_point[0],
                        field_point[1],
                        f"  M+max={positive_peak['value']:.2f} @ x={positive_peak['x']:.2f} m",
                        fontsize=7.5,
                        color=get_signed_color(positive_peak['value']),
                        ha='left',
                        va='bottom',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75)
                    )
            mid_point = (start + end) / 2
            ax.text(mid_point[0], mid_point[1], f"E{int(element.elem_id)}",
                    fontsize=9, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        title = config['title']
        if bool(relative_to_chord) and force_type == 'moment' and relative_to_chord != 'mixed':
            title += ' (Curvature Emphasized)'
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        PortalPlotter.expand_axes_for_data_text(ax)
        ax.legend()

        return fig, ax

    @staticmethod
    def plot_beam_moment_profiles(elements: List, element_forces: List[Dict],
                                  relative_to_chord: bool = True,
                                  element_ids: Optional[List[int]] = None):
        """
        Plot profil momen lokal sepanjang balok agar kelengkungan lebih jelas.

        Parameters:
        - elements: list Element2D objects
        - element_forces: list hasil gaya dalam per elemen
        - relative_to_chord: jika True, tampilkan M(x) dikurangi garis lurus
          penghubung momen ujung sehingga parabola akibat beban merata terlihat jelas
        - element_ids: batasi hanya pada elemen tertentu
        """
        force_lookup = {
            int(force['elem_id']): force
            for force in element_forces
        }

        beam_elements = PortalPlotter.get_beam_elements(elements)
        if element_ids is not None:
            requested_ids = {int(elem_id) for elem_id in element_ids}
            beam_elements = [
                element for element in beam_elements
                if int(element.elem_id) in requested_ids
            ]

        beam_elements = [
            element for element in beam_elements
            if int(element.elem_id) in force_lookup
        ]

        beam_elements.sort(
            key=lambda element: (
                -float(np.mean([element.coord_start[1], element.coord_end[1]])),
                float(np.mean([element.coord_start[0], element.coord_end[0]])),
                int(element.elem_id)
            )
        )

        if not beam_elements:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.text(
                0.5, 0.5,
                'Tidak ada elemen balok horizontal yang dapat diplot.',
                ha='center', va='center', transform=ax.transAxes
            )
            ax.set_axis_off()
            return fig, ax

        fig_height = max(3.2 * len(beam_elements), 4.0)
        fig, axes = plt.subplots(
            len(beam_elements), 1,
            figsize=(10, fig_height),
            squeeze=False
        )
        axes = axes.flatten()

        for idx, (ax, element) in enumerate(zip(axes, beam_elements)):
            force_data = force_lookup[int(element.elem_id)]
            x_profile_m, moment_profile = PortalPlotter.get_force_profile(
                force_data, force_type='moment'
            )
            moment_profile = PortalPlotter.convert_beam_moment_to_display_convention(
                moment_profile,
                is_beam=True
            )

            if len(x_profile_m) == 0:
                continue

            chord_profile = np.linspace(
                moment_profile[0],
                moment_profile[-1],
                len(moment_profile)
            )
            display_profile = (
                moment_profile - chord_profile
                if relative_to_chord else moment_profile
            )

            if relative_to_chord:
                ax.axhline(
                    0.0,
                    color='0.5',
                    linestyle='--',
                    linewidth=1.2,
                    label='Garis penghubung momen ujung' if idx == 0 else ''
                )
                ax.fill_between(
                    x_profile_m,
                    0.0,
                    display_profile,
                    color='tab:blue',
                    alpha=0.18
                )
                ax.plot(
                    x_profile_m,
                    display_profile,
                    color='tab:blue',
                    linewidth=2.4,
                    label='Kurvatur momen relatif' if idx == 0 else ''
                )
                y_label = 'Delta M (kN.m)'
                peak_idx = int(np.argmax(np.abs(display_profile)))
                peak_value = float(display_profile[peak_idx])
                title_suffix = f'Deviasi maksimum = {peak_value:.2f} kN.m'
            else:
                ax.plot(
                    x_profile_m,
                    chord_profile,
                    color='0.5',
                    linestyle='--',
                    linewidth=1.2,
                    label='Garis penghubung momen ujung' if idx == 0 else ''
                )
                ax.fill_between(
                    x_profile_m,
                    chord_profile,
                    moment_profile,
                    color='tab:blue',
                    alpha=0.18
                )
                ax.plot(
                    x_profile_m,
                    moment_profile,
                    color='tab:blue',
                    linewidth=2.4,
                    label='Momen lentur M(x)' if idx == 0 else ''
                )
                y_label = 'M (kN.m)'
                peak_idx = int(np.argmax(np.abs(moment_profile)))
                peak_value = float(moment_profile[peak_idx])
                title_suffix = f'Maksimum absolut = {peak_value:.2f} kN.m'

            ax.scatter(
                [x_profile_m[peak_idx]],
                [display_profile[peak_idx]],
                color='tab:blue',
                s=28,
                zorder=3
            )
            ax.text(
                x_profile_m[peak_idx],
                display_profile[peak_idx],
                f'  x={x_profile_m[peak_idx]:.2f} m',
                fontsize=8,
                va='bottom'
            )

            transverse_load = float(force_data.get('transverse_local_load', 0.0))
            ax.set_title(
                f"E{int(element.elem_id)} | L = {float(force_data.get('length_m', 0.0)):.2f} m | "
                f"w lokal = {transverse_load:.2f} kN/m | {title_suffix}",
                fontsize=10
            )
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(float(x_profile_m[0]), float(x_profile_m[-1]))
            PortalPlotter.expand_axes_for_data_text(ax)

        axes[-1].set_xlabel('x sepanjang balok (m)')
        fig.suptitle(
            'Profil Momen Sepanjang Balok'
            if not relative_to_chord
            else 'Parabola Momen Balok (M(x) dikurangi garis ujung)',
            fontsize=13,
            y=0.995
        )
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(loc='best')
        fig.tight_layout()

        return fig, axes

    @staticmethod
    def plot_beam_moment_focus(elements: List, element_forces: List[Dict],
                               element_id: int):
        """
        Plot fokus momen sepanjang satu balok.

        Panel atas menampilkan M(x) absolut dan garis chord ujung.
        Panel bawah menampilkan Delta M = M(x) - chord agar bentuk parabola
        akibat beban merata lebih mudah dibaca.
        """
        force_lookup = {
            int(force['elem_id']): force
            for force in element_forces
        }
        element_lookup = {
            int(element.elem_id): element
            for element in PortalPlotter.get_beam_elements(elements)
        }

        element_id = int(element_id)
        if element_id not in force_lookup or element_id not in element_lookup:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.text(
                0.5, 0.5,
                f'Balok E{element_id} tidak tersedia untuk diplot.',
                ha='center', va='center', transform=ax.transAxes
            )
            ax.set_axis_off()
            return fig, ax

        force_data = force_lookup[element_id]
        x_profile_m, moment_profile = PortalPlotter.get_force_profile(
            force_data,
            force_type='moment',
            num_points=201,
            prefer_continuous=True
        )
        moment_profile = PortalPlotter.convert_beam_moment_to_display_convention(
            moment_profile,
            is_beam=True
        )
        chord_profile = np.linspace(
            moment_profile[0],
            moment_profile[-1],
            len(moment_profile)
        )
        relative_profile = moment_profile - chord_profile
        length_m = float(force_data.get('length_m', 0.0))
        transverse_load = float(force_data.get('transverse_local_load', 0.0))

        fig, axes = plt.subplots(
            2, 1,
            figsize=(11, 7),
            sharex=True,
            gridspec_kw={'height_ratios': [2.0, 1.25]}
        )
        ax_abs, ax_rel = axes

        ax_abs.plot(
            x_profile_m,
            chord_profile,
            color='0.45',
            linestyle='--',
            linewidth=1.3,
            label='Garis penghubung momen ujung'
        )
        ax_abs.fill_between(
            x_profile_m,
            chord_profile,
            moment_profile,
            color='tab:blue',
            alpha=0.18
        )
        ax_abs.plot(
            x_profile_m,
            moment_profile,
            color='tab:blue',
            linewidth=2.5,
            label='Momen lentur M(x)'
        )

        start_moment = float(moment_profile[0])
        end_moment = float(moment_profile[-1])
        peak_abs_idx = int(np.argmax(np.abs(moment_profile)))
        peak_rel_idx = int(np.argmax(np.abs(relative_profile)))

        ax_abs.scatter(
            [x_profile_m[0], x_profile_m[-1], x_profile_m[peak_abs_idx]],
            [start_moment, end_moment, moment_profile[peak_abs_idx]],
            color='tab:blue',
            s=30,
            zorder=3
        )
        ax_abs.text(
            x_profile_m[0],
            start_moment,
            f'  M0={start_moment:.2f}',
            fontsize=8,
            va='bottom'
        )
        ax_abs.text(
            x_profile_m[-1],
            end_moment,
            f'  ML={end_moment:.2f}',
            fontsize=8,
            ha='right',
            va='bottom'
        )
        ax_abs.text(
            x_profile_m[peak_abs_idx],
            moment_profile[peak_abs_idx],
            f'  Mmax={moment_profile[peak_abs_idx]:.2f} @ x={x_profile_m[peak_abs_idx]:.2f} m',
            fontsize=8,
            va='bottom'
        )
        ax_abs.set_ylabel('M (kN.m)')
        ax_abs.grid(True, alpha=0.3)
        ax_abs.legend(loc='best')
        ax_abs.set_title(
            f'E{element_id} | L = {length_m:.2f} m | w lokal = {transverse_load:.2f} kN/m',
            fontsize=11
        )

        ax_rel.axhline(
            0.0,
            color='0.45',
            linestyle='--',
            linewidth=1.2,
            label='Chord ujung'
        )
        ax_rel.fill_between(
            x_profile_m,
            0.0,
            relative_profile,
            color='tab:blue',
            alpha=0.18
        )
        ax_rel.plot(
            x_profile_m,
            relative_profile,
            color='tab:blue',
            linewidth=2.5,
            label='Delta M = M(x) - chord'
        )
        ax_rel.scatter(
            [x_profile_m[peak_rel_idx]],
            [relative_profile[peak_rel_idx]],
            color='tab:blue',
            s=30,
            zorder=3
        )
        ax_rel.text(
            x_profile_m[peak_rel_idx],
            relative_profile[peak_rel_idx],
            f'  DeltaM={relative_profile[peak_rel_idx]:.2f} @ x={x_profile_m[peak_rel_idx]:.2f} m',
            fontsize=8,
            va='bottom'
        )
        ax_rel.set_ylabel('Delta M (kN.m)')
        ax_rel.set_xlabel('x sepanjang balok (m)')
        ax_rel.grid(True, alpha=0.3)
        ax_rel.legend(loc='best')
        ax_rel.set_xlim(float(x_profile_m[0]), float(x_profile_m[-1]))
        PortalPlotter.expand_axes_for_data_text(ax_abs)
        PortalPlotter.expand_axes_for_data_text(ax_rel)

        note_text = (
            'Beban merata transversal ada, sehingga profil M(x) diharapkan melengkung/parabolik.'
            if abs(transverse_load) > 1e-12
            else 'Tidak ada beban merata transversal; profil M(x) cenderung linear.'
        )
        ax_rel.text(
            0.01,
            0.96,
            note_text,
            transform=ax_rel.transAxes,
            fontsize=8,
            va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
        )

        fig.suptitle('Plot Momen Sepanjang Balok', fontsize=13, y=0.995)
        fig.tight_layout()

        return fig, axes


class ReliabilityPlotter:
    """Visualisasi hasil analisis keandalan"""
    
    @staticmethod
    def plot_failure_probability(mc_results: Dict) -> None:
        """
        Plot histogram dari hasil simulasi dengan indikasi failure
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Extract max moments atau forces
        max_moments = []
        for analysis_result in mc_results['max_forces_history']:
            if not analysis_result:
                continue

            force_dict = analysis_result.get('max_forces', {})
            if not force_dict:
                continue

            elem_momen = max(abs(f['max_moment']) for f in force_dict.values())
            max_moments.append(elem_momen)

        if not max_moments:
            ax.text(0.5, 0.5, 'No valid simulation results',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return fig, ax
        
        ax.hist(max_moments, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(max_moments), color='r', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(max_moments):.2f}')
        ax.set_xlabel('Maximum Moment (kN.m)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Maximum Moments in MC Simulation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_reliability_index_evolution(num_simulations_list: List[int],
                                        beta_evolution: List[float]) -> None:
        """
        Plot evolusi Beta terhadap jumlah simulasi
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(num_simulations_list, beta_evolution, 'b-o', linewidth=2)
        ax.set_xlabel('Number of Simulations')
        ax.set_ylabel('Reliability Index (Beta)')
        ax.set_title('Convergence of Reliability Index')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        return fig, ax
    
    @staticmethod
    def plot_sensitivity_ranking(sensitivities: Dict) -> None:
        """
        Plot ranking sensitivitas variabel
        
        Parameters:
        - sensitivities: dict hasil sensitivity analysis
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        var_names = list(sensitivities.keys())
        sensitivity_indices = [
            sensitivities[v]['sensitivity_index'] for v in var_names
        ]
        
        y_pos = np.arange(len(var_names))
        ax.barh(y_pos, sensitivity_indices)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(var_names)
        ax.set_xlabel('Sensitivity Index')
        ax.set_title('Sensitivity Analysis - Variable Ranking')
        ax.grid(True, alpha=0.3, axis='x')
        
        return fig, ax
    
    @staticmethod
    def plot_safety_margin_distribution(max_forces_list: List[Dict],
                                       capacity: float) -> None:
        """
        Plot distribusi safety margin
        
        Parameters:
        - max_forces_list: list max forces dari MC
        - capacity: kapasitas struktur
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        demands = []
        for item in max_forces_list:
            if not item:
                continue

            forces_dict = item.get('max_forces', item)
            if not forces_dict:
                continue

            max_demands = max([f['max_moment'] for f in forces_dict.values()])
            demands.append(max_demands)

        if not demands:
            ax.text(0.5, 0.5, 'No valid simulation results',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return fig, ax
        
        safety_margins = capacity - np.array(demands)
        
        ax.hist(safety_margins, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Limit State (g=0)')
        ax.axvline(np.mean(safety_margins), color='g', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(safety_margins):.2f}')
        ax.set_xlabel('Safety Margin (Capacity - Demand)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Safety Margins')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Shade failure region
        ax.axvspan(min(safety_margins), 0, alpha=0.2, color='red', label='Failure Region')
        
        return fig, ax
    
    @staticmethod
    def save_all_plots(output_dir: str, mc_results: Dict,
                      sensitivity_data: Dict) -> None:
        """
        Save semua plot ke file
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Failure Probability Distribution
        ReliabilityPlotter.plot_failure_probability(mc_results)
        plt.savefig(f'{output_dir}/failure_probability.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Sensitivity Analysis
        if sensitivity_data:
            ReliabilityPlotter.plot_sensitivity_ranking(sensitivity_data)
            plt.savefig(f'{output_dir}/sensitivity_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Plots saved to {output_dir}")
