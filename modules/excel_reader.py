"""
Modul untuk membaca data input dari file Excel.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class ExcelReader:
    """Membaca dan memproses data input dari file Excel."""

    def __init__(self, excel_file: str):
        self.excel_file = excel_file
        self.data = {}

    def find_column(self, df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
        """
        Cari nama kolom yang mengandung salah satu keyword.

        Parameters:
        - df: DataFrame sumber
        - keywords: daftar keyword lowercase

        Returns:
        - nama kolom jika ada, else None
        """
        for col in df.columns:
            col_lower = str(col).strip().lower()
            if any(keyword in col_lower for keyword in keywords):
                return col
        return None

    @staticmethod
    def _normalize_distribution(value, default: str) -> str:
        """Normalisasi nama distribusi."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        text = str(value).strip().lower()
        if not text:
            return default
        if 'log' in text:
            return 'lognormal'
        if 'norm' in text:
            return 'normal'
        return default

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        """Konversi aman ke float."""
        numeric = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
        if pd.isna(numeric):
            return float(default)
        return float(numeric)

    def _resolve_element_ids(self, preferred_ids: Optional[List[int]] = None) -> List[int]:
        """Ambil daftar elemen referensi, utamakan dari argumen."""
        if preferred_ids:
            return [int(elem_id) for elem_id in preferred_ids]

        if 'Geometri' not in self.data:
            return []

        geom = self.data['Geometri']
        elem_col = self.find_column(geom, ['element_id', 'element'])
        if elem_col is None:
            return []

        element_ids = pd.to_numeric(geom[elem_col], errors='coerce').dropna().astype(int)
        return element_ids.tolist()

    def _get_element_row_lookup(self, df: pd.DataFrame) -> Dict[int, pd.Series]:
        """Bangun lookup row berdasarkan Element_ID bila tersedia."""
        elem_col = self.find_column(df, ['element_id', 'element'])
        if elem_col is None or elem_col not in df.columns:
            return {}

        lookup: Dict[int, pd.Series] = {}
        for _, row in df.iterrows():
            elem_id = pd.to_numeric(pd.Series([row[elem_col]]), errors='coerce').iloc[0]
            if pd.isna(elem_id):
                continue
            lookup[int(elem_id)] = row
        return lookup

    def read_all_sheets(self) -> Dict:
        """Membaca semua sheet dari file Excel."""
        with pd.ExcelFile(self.excel_file) as xls:
            print(f"Sheet yang tersedia: {xls.sheet_names}")
            for sheet in xls.sheet_names:
                self.data[sheet] = pd.read_excel(xls, sheet_name=sheet)
        return self.data

    def get_geometry(self) -> Dict:
        """
        Mengambil data geometri portal dari sheet 'Geometri'.
        Termasuk nodes, elemen, b, h, A, I, kode elemen, dan E per elemen.
        """
        if 'Geometri' not in self.data:
            raise ValueError("Sheet 'Geometri' tidak ditemukan")

        geom = self.data['Geometri']
        columns_lower = {str(col).strip().lower(): col for col in geom.columns}

        if 'Nodes' in self.data:
            nodes_df = self.data['Nodes']
            node_id_col = (
                self.find_column(nodes_df, ['node_id', 'node'])
                or 'Node_ID'
            )
            x_col = self.find_column(nodes_df, ['x (mm)', 'x']) or 'X (mm)'
            y_col = self.find_column(nodes_df, ['y (mm)', 'y']) or 'Y (mm)'
            nodes = nodes_df[[node_id_col, x_col, y_col]].to_numpy(dtype=float)
        else:
            raise ValueError("Sheet 'Nodes' tidak ditemukan")

        elem_id_col = self.find_column(geom, ['element_id', 'element']) or 'Element_ID'
        code_col = (
            columns_lower.get('kode')
            or self.find_column(geom, ['kode', 'code', 'type'])
        )
        node_start_col = self.find_column(geom, ['node_start', 'start']) or 'Node_Start'
        node_end_col = self.find_column(geom, ['node_end', 'end']) or 'Node_End'
        b_col = (
            columns_lower.get('b')
            or columns_lower.get('b(mm)')
            or columns_lower.get('b (mm)')
            or self.find_column(geom, ['b(mm)', 'b (mm)', 'width'])
        )
        h_col = (
            columns_lower.get('h')
            or columns_lower.get('h(mm)')
            or columns_lower.get('h (mm)')
            or self.find_column(geom, ['h(mm)', 'h (mm)', 'height'])
        )
        area_col = (
            columns_lower.get('a')
            or columns_lower.get('area (mm²)')
            or columns_lower.get('area (mm2)')
            or columns_lower.get('area')
            or self.find_column(geom, ['area', 'luas'])
        )
        inertia_col = (
            columns_lower.get('i')
            or columns_lower.get('inertia (mm⁴)')
            or columns_lower.get('inertia (mm4)')
            or columns_lower.get('inertia')
            or self.find_column(geom, ['inertia', 'inersia'])
        )
        e_mean_col = (
            columns_lower.get('e_mean (mpa)')
            or columns_lower.get('e_mean')
            or columns_lower.get('emean')
            or columns_lower.get('mean')
            or columns_lower.get('e (mpa)')
            or columns_lower.get('e')
            or self.find_column(geom, ['e_mean', 'elastic', 'young'])
        )
        e_deterministic_col = (
            columns_lower.get('deterministic')
            or columns_lower.get('deterministic (mpa)')
            or columns_lower.get('e_deterministic (mpa)')
            or columns_lower.get('e_deterministic')
            or self.find_column(geom, ['deterministic'])
        )

        missing_columns = []
        if elem_id_col is None:
            missing_columns.append('Element_ID')
        if node_start_col is None:
            missing_columns.append('Node_Start')
        if node_end_col is None:
            missing_columns.append('Node_End')
        if b_col is None:
            missing_columns.append('b')
        if h_col is None:
            missing_columns.append('h')
        if area_col is None:
            missing_columns.append('A/Area')
        if inertia_col is None:
            missing_columns.append('I/Inertia')
        if e_mean_col is None:
            missing_columns.append('E_mean')
        if e_deterministic_col is None:
            missing_columns.append('Deterministic')

        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(
                "Sheet 'Geometri' belum lengkap. "
                f"Kolom wajib yang belum ditemukan: {missing_text}."
            )

        base_rows = []
        element_properties = {}

        for row_index, row in geom.iterrows():
            excel_row = int(row_index) + 2
            elem_id = int(self._to_float(row.get(elem_id_col), 0))
            node_start = int(self._to_float(row.get(node_start_col), 0))
            node_end = int(self._to_float(row.get(node_end_col), 0))
            b = self._to_float(row.get(b_col), 0.0)
            h = self._to_float(row.get(h_col), 0.0)
            area = self._to_float(row.get(area_col), 0.0)
            inertia = self._to_float(row.get(inertia_col), 0.0)
            e_mean = self._to_float(row.get(e_mean_col), 0.0)
            e_deterministic = self._to_float(row.get(e_deterministic_col), 0.0)

            missing_fields = []
            if elem_id <= 0:
                missing_fields.append('Element_ID')
            if node_start <= 0:
                missing_fields.append('Node_Start')
            if node_end <= 0:
                missing_fields.append('Node_End')
            if b <= 0.0:
                missing_fields.append('b')
            if h <= 0.0:
                missing_fields.append('h')
            if area <= 0.0:
                missing_fields.append('A/Area')
            if inertia <= 0.0:
                missing_fields.append('I/Inertia')
            if e_mean <= 0.0:
                missing_fields.append('E_mean')
            if e_deterministic <= 0.0:
                missing_fields.append('Deterministic')

            if missing_fields:
                elem_label = f"elemen {elem_id}" if elem_id > 0 else f"baris {excel_row}"
                missing_text = ", ".join(missing_fields)
                raise ValueError(
                    f"Data geometri {elem_label} pada sheet 'Geometri' belum lengkap "
                    f"atau tidak valid: {missing_text}."
                )

            code_value = ''
            if code_col and code_col in geom.columns:
                raw_code = row.get(code_col)
                if raw_code is not None and not pd.isna(raw_code):
                    code_value = str(raw_code).strip()

            base_rows.append({
                'Element_ID': elem_id,
                'Kode': code_value,
                'Node_Start': node_start,
                'Node_End': node_end,
                'b (mm)': b,
                'h (mm)': h,
                'Area (mm2)': area,
                'Inertia (mm4)': inertia,
                'E_mean (MPa)': e_mean,
                'E_deterministic (MPa)': e_deterministic
            })

            element_properties[elem_id] = {
                'elem_id': elem_id,
                'code': code_value,
                'node_start': node_start,
                'node_end': node_end,
                'b': b,
                'h': h,
                'area': area,
                'inertia': inertia,
                'E_mean': e_mean,
                'E_deterministic': e_deterministic
            }

        geometry_df = pd.DataFrame(base_rows).sort_values('Element_ID').reset_index(drop=True)
        base_elements = geometry_df[
            ['Element_ID', 'Node_Start', 'Node_End', 'Area (mm2)', 'Inertia (mm4)']
        ].to_numpy(dtype=float)
        elements_mean = np.column_stack([
            base_elements,
            geometry_df['E_mean (MPa)'].to_numpy(dtype=float)
        ])
        elements_deterministic = np.column_stack([
            base_elements,
            geometry_df['E_deterministic (MPa)'].to_numpy(dtype=float)
        ])

        return {
            'nodes': nodes,
            'elements': elements_mean,
            'elements_mean': elements_mean,
            'elements_deterministic': elements_deterministic,
            'E_mean': float(geometry_df['E_mean (MPa)'].mean()) if not geometry_df.empty else 30000.0,
            'E_deterministic': float(geometry_df['E_deterministic (MPa)'].mean()) if not geometry_df.empty else 30000.0,
            'element_ids': geometry_df['Element_ID'].astype(int).tolist(),
            'properties_by_element': element_properties
        }

    def get_concrete_properties(self, element_ids: Optional[List[int]] = None) -> Dict:
        """
        Mengambil mutu beton per elemen dari sheet 'Mutu_Beton'.
        Mendukung format lama satu nilai global maupun format baru per elemen.
        """
        if 'Mutu_Beton' not in self.data:
            raise ValueError("Sheet 'Mutu_Beton' tidak ditemukan")

        df = self.data['Mutu_Beton']
        row_lookup = self._get_element_row_lookup(df)
        resolved_ids = self._resolve_element_ids(element_ids) or sorted(row_lookup.keys())
        default_row = df.iloc[0] if not df.empty else None

        mean_col = (
            self.find_column(df, ['fc_mean', 'mean'])
            or 'fc_Mean (MPa)'
        )
        stddev_col = (
            self.find_column(df, ['fc_stddev', 'stddev', 'std'])
            or 'fc_StdDev (MPa)'
        )
        distribution_col = self.find_column(df, ['distribution'])
        deterministic_col = (
            self.find_column(df, ['fc_deterministic', 'deterministic'])
            or 'fc_Deterministic (MPa)'
        )

        by_element = {}
        means = []
        stddevs = []
        deterministic_values = []
        distributions = []

        for elem_id in resolved_ids:
            row = row_lookup.get(elem_id, default_row)
            if row is None:
                continue

            mean_value = self._to_float(row.get(mean_col), 0.0)
            stddev_value = self._to_float(row.get(stddev_col), 0.0)
            deterministic_value = self._to_float(
                row.get(deterministic_col),
                mean_value
            ) if deterministic_col else mean_value
            distribution_value = self._normalize_distribution(
                row.get(distribution_col) if distribution_col else None,
                'lognormal'
            )

            by_element[int(elem_id)] = {
                'mean': mean_value,
                'stddev': stddev_value,
                'distribution': distribution_value,
                'deterministic': deterministic_value
            }
            means.append(mean_value)
            stddevs.append(stddev_value)
            deterministic_values.append(deterministic_value)
            distributions.append(distribution_value)

        return {
            'element_ids': resolved_ids,
            'mean': np.asarray(means, dtype=float),
            'stddev': np.asarray(stddevs, dtype=float),
            'distribution': distributions[0] if len(set(distributions)) == 1 and distributions else 'mixed',
            'distributions': np.asarray(distributions, dtype=object),
            'deterministic': np.asarray(deterministic_values, dtype=float),
            'by_element': by_element
        }

    def get_steel_properties(self, element_ids: Optional[List[int]] = None) -> Dict:
        """
        Mengambil mutu baja per elemen dari sheet 'Mutu_Baja'.
        Mendukung format lama berbasis tipe dan format baru per elemen.
        """
        if 'Mutu_Baja' not in self.data:
            raise ValueError("Sheet 'Mutu_Baja' tidak ditemukan")

        df = self.data['Mutu_Baja']
        row_lookup = self._get_element_row_lookup(df)
        resolved_ids = self._resolve_element_ids(element_ids) or sorted(row_lookup.keys())

        by_element = {}

        new_format = self.find_column(df, ['mean_tarik']) is not None
        if new_format:
            default_row = df.iloc[0] if not df.empty else None
            mean_tarik_col = self.find_column(df, ['mean_tarik']) or 'Mean_tarik (MPa)'
            stddev_tarik_col = self.find_column(df, ['stddev_tarik', 'std_tarik']) or 'StdDev_tarik (MPa)'
            distribution_tarik_col = self.find_column(df, ['distribution_tarik'])
            mean_tekan_col = self.find_column(df, ['mean_tekan']) or 'Mean_tekan (MPa)'
            stddev_tekan_col = self.find_column(df, ['stddev_tekan', 'std_tekan']) or 'StdDev_tekan (MPa)'
            distribution_tekan_col = self.find_column(df, ['distribution_tekan'])
            mean_geser_col = self.find_column(df, ['mean_geser']) or 'Mean_geser (MPa)'
            stddev_geser_col = self.find_column(df, ['stddev_geser', 'std_geser']) or 'StdDev_geser (MPa)'
            distribution_geser_col = self.find_column(df, ['distribution_geser'])
            deterministic_tarik_col = self.find_column(df, ['deterministic_tarik']) or 'Deterministic_tarik (MPa)'
            deterministic_tekan_col = self.find_column(df, ['deterministic_tekan']) or 'Deterministic_tekan (MPa)'
            deterministic_geser_col = self.find_column(df, ['deterministic_geser']) or 'Deterministic_geser (MPa)'

            for elem_id in resolved_ids:
                row = row_lookup.get(elem_id, default_row)
                if row is None:
                    continue

                by_element[int(elem_id)] = {
                    'tarik_mean': self._to_float(row.get(mean_tarik_col), 0.0),
                    'tarik_stddev': self._to_float(row.get(stddev_tarik_col), 0.0),
                    'tarik_distribution': self._normalize_distribution(
                        row.get(distribution_tarik_col) if distribution_tarik_col else None,
                        'normal'
                    ),
                    'tekan_mean': self._to_float(row.get(mean_tekan_col), 0.0),
                    'tekan_stddev': self._to_float(row.get(stddev_tekan_col), 0.0),
                    'tekan_distribution': self._normalize_distribution(
                        row.get(distribution_tekan_col) if distribution_tekan_col else None,
                        'normal'
                    ),
                    'geser_mean': self._to_float(
                        row.get(mean_geser_col),
                        self._to_float(row.get(mean_tarik_col), 0.0)
                    ) if mean_geser_col else self._to_float(row.get(mean_tarik_col), 0.0),
                    'geser_stddev': self._to_float(
                        row.get(stddev_geser_col),
                        self._to_float(row.get(stddev_tarik_col), 0.0)
                    ) if stddev_geser_col else self._to_float(row.get(stddev_tarik_col), 0.0),
                    'geser_distribution': self._normalize_distribution(
                        row.get(distribution_geser_col) if distribution_geser_col else None,
                        self._normalize_distribution(
                            row.get(distribution_tarik_col) if distribution_tarik_col else None,
                            'normal'
                        )
                    ),
                    'tarik_deterministic': self._to_float(
                        row.get(deterministic_tarik_col),
                        self._to_float(row.get(mean_tarik_col), 0.0)
                    ) if deterministic_tarik_col else self._to_float(row.get(mean_tarik_col), 0.0),
                    'tekan_deterministic': self._to_float(
                        row.get(deterministic_tekan_col),
                        self._to_float(row.get(mean_tekan_col), 0.0)
                    ) if deterministic_tekan_col else self._to_float(row.get(mean_tekan_col), 0.0),
                    'geser_deterministic': self._to_float(
                        row.get(deterministic_geser_col),
                        self._to_float(
                            row.get(mean_geser_col),
                            self._to_float(row.get(mean_tarik_col), 0.0)
                        )
                    ) if deterministic_geser_col else self._to_float(
                        row.get(mean_geser_col),
                        self._to_float(row.get(mean_tarik_col), 0.0)
                    )
                }
        else:
            tipe_col = self.find_column(df, ['tipe', 'type']) or 'Tipe'
            mean_col = self.find_column(df, ['mean']) or 'Mean'
            stddev_col = self.find_column(df, ['stddev', 'std']) or 'StdDev'
            distribution_col = self.find_column(df, ['distribution'])
            deterministic_col = self.find_column(df, ['deterministic'])

            tarik_row = df.loc[df[tipe_col].astype(str).str.lower() == 'tarik']
            tekan_row = df.loc[df[tipe_col].astype(str).str.lower() == 'tekan']
            geser_row = df.loc[df[tipe_col].astype(str).str.lower() == 'geser']
            if tarik_row.empty or tekan_row.empty:
                raise ValueError("Format Mutu_Baja lama harus memiliki baris Tarik dan Tekan")

            tarik_row = tarik_row.iloc[0]
            tekan_row = tekan_row.iloc[0]
            geser_row = geser_row.iloc[0] if not geser_row.empty else tarik_row
            for elem_id in resolved_ids:
                by_element[int(elem_id)] = {
                    'tarik_mean': self._to_float(tarik_row.get(mean_col), 0.0),
                    'tarik_stddev': self._to_float(tarik_row.get(stddev_col), 0.0),
                    'tarik_distribution': self._normalize_distribution(
                        tarik_row.get(distribution_col) if distribution_col else None,
                        'normal'
                    ),
                    'tekan_mean': self._to_float(tekan_row.get(mean_col), 0.0),
                    'tekan_stddev': self._to_float(tekan_row.get(stddev_col), 0.0),
                    'tekan_distribution': self._normalize_distribution(
                        tekan_row.get(distribution_col) if distribution_col else None,
                        'normal'
                    ),
                    'geser_mean': self._to_float(
                        geser_row.get(mean_col),
                        self._to_float(tarik_row.get(mean_col), 0.0)
                    ),
                    'geser_stddev': self._to_float(
                        geser_row.get(stddev_col),
                        self._to_float(tarik_row.get(stddev_col), 0.0)
                    ),
                    'geser_distribution': self._normalize_distribution(
                        geser_row.get(distribution_col) if distribution_col else None,
                        self._normalize_distribution(
                            tarik_row.get(distribution_col) if distribution_col else None,
                            'normal'
                        )
                    ),
                    'tarik_deterministic': self._to_float(
                        tarik_row.get(deterministic_col),
                        self._to_float(tarik_row.get(mean_col), 0.0)
                    ) if deterministic_col else self._to_float(tarik_row.get(mean_col), 0.0),
                    'tekan_deterministic': self._to_float(
                        tekan_row.get(deterministic_col),
                        self._to_float(tekan_row.get(mean_col), 0.0)
                    ) if deterministic_col else self._to_float(tekan_row.get(mean_col), 0.0),
                    'geser_deterministic': self._to_float(
                        geser_row.get(deterministic_col),
                        self._to_float(
                            geser_row.get(mean_col),
                            self._to_float(tarik_row.get(mean_col), 0.0)
                        )
                    ) if deterministic_col else self._to_float(
                        geser_row.get(mean_col),
                        self._to_float(tarik_row.get(mean_col), 0.0)
                    )
                }

        tarik_means = [props['tarik_mean'] for props in by_element.values()]
        tarik_stddevs = [props['tarik_stddev'] for props in by_element.values()]
        tekan_means = [props['tekan_mean'] for props in by_element.values()]
        tekan_stddevs = [props['tekan_stddev'] for props in by_element.values()]
        geser_means = [props['geser_mean'] for props in by_element.values()]
        geser_stddevs = [props['geser_stddev'] for props in by_element.values()]

        return {
            'element_ids': list(by_element.keys()),
            'tarik_mean': np.asarray(tarik_means, dtype=float),
            'tarik_stddev': np.asarray(tarik_stddevs, dtype=float),
            'tekan_mean': np.asarray(tekan_means, dtype=float),
            'tekan_stddev': np.asarray(tekan_stddevs, dtype=float),
            'geser_mean': np.asarray(geser_means, dtype=float),
            'geser_stddev': np.asarray(geser_stddevs, dtype=float),
            'tarik_deterministic': np.asarray(
                [props['tarik_deterministic'] for props in by_element.values()],
                dtype=float
            ),
            'tekan_deterministic': np.asarray(
                [props['tekan_deterministic'] for props in by_element.values()],
                dtype=float
            ),
            'geser_deterministic': np.asarray(
                [props['geser_deterministic'] for props in by_element.values()],
                dtype=float
            ),
            'distribution': 'mixed',
            'by_element': by_element
        }

    def get_reinforcement_properties(self, element_ids: Optional[List[int]] = None) -> Dict:
        """Mengambil data tulangan per elemen dari sheet 'Tulangan'."""
        if 'Tulangan' not in self.data:
            return {'element_ids': [], 'by_element': {}}

        df = self.data['Tulangan']
        row_lookup = self._get_element_row_lookup(df)
        resolved_ids = self._resolve_element_ids(element_ids) or sorted(row_lookup.keys())
        default_row = df.iloc[0] if not df.empty else None

        field_map = {
            'ds_tarik': ['ds_tarik'],
            'ds_tekan': ['ds_tekan'],
            'd_tarik': ['d_tarik'],
            'd_tekan': ['d_tekan'],
            'n_tarik': ['n_tarik'],
            'du_tarik': ['du_tarik'],
            'n_tekan': ['n_tekan'],
            'du_tekan': ['du_tekan'],
            'As_tarik': ['as_tarik'],
            'As_tekan': ['as_tekan'],
            'n_geser': ['n_geser'],
            'du_geser': ['du_geser'],
            'As_geser': ['as_geser'],
            'Spasi_geser': ['spasi_geser']
        }
        resolved_columns = {
            key: self.find_column(df, keywords)
            for key, keywords in field_map.items()
        }

        by_element = {}
        for elem_id in resolved_ids:
            row = row_lookup.get(elem_id, default_row)
            if row is None:
                continue

            by_element[int(elem_id)] = {
                key: self._to_float(row.get(column), 0.0) if column else 0.0
                for key, column in resolved_columns.items()
            }

        return {
            'element_ids': list(by_element.keys()),
            'by_element': by_element
        }

    def _get_load_data(self, sheet_name: str, default_distribution: str) -> Dict:
        """Pembacaan umum untuk beban mati/hidup."""
        if sheet_name not in self.data:
            raise ValueError(f"Sheet '{sheet_name}' tidak ditemukan")

        df = self.data[sheet_name]
        elem_col = self.find_column(df, ['element_id', 'element']) or 'Element_ID'
        mean_col = self.find_column(df, ['mean']) or 'Mean'
        stddev_col = self.find_column(df, ['stddev', 'std']) or 'StdDev'
        distribution_col = self.find_column(df, ['distribution'])
        deterministic_col = self.find_column(df, ['deterministic'])

        elements = pd.to_numeric(df[elem_col], errors='coerce').dropna().astype(int).to_numpy()
        means = pd.to_numeric(df[mean_col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        stddevs = pd.to_numeric(df[stddev_col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        deterministic_values = None
        if deterministic_col and deterministic_col in df.columns:
            deterministic_series = pd.to_numeric(df[deterministic_col], errors='coerce')
            deterministic_series = deterministic_series.where(~deterministic_series.isna(), pd.Series(means, index=df.index))
            deterministic_values = deterministic_series.to_numpy(dtype=float)
        distributions = np.asarray([
            self._normalize_distribution(
                row[distribution_col] if distribution_col and distribution_col in df.columns else None,
                default_distribution
            )
            for _, row in df.iterrows()
        ], dtype=object)

        by_element = {}
        for index, elem_id in enumerate(elements):
            by_element[int(elem_id)] = {
                'mean': float(means[index]),
                'stddev': float(stddevs[index]),
                'distribution': str(distributions[index]),
                'deterministic': (
                    float(deterministic_values[index])
                    if deterministic_values is not None else
                    float(means[index])
                )
            }

        return {
            'mean': means,
            'stddev': stddevs,
            'elements': elements,
            'distribution': default_distribution,
            'distributions': distributions,
            'deterministic': deterministic_values,
            'by_element': by_element
        }

    def get_dead_load(self) -> Dict:
        """Mengambil beban mati merata dari sheet 'Beban_Mati'."""
        return self._get_load_data('Beban_Mati', default_distribution='normal')

    def get_live_load(self) -> Dict:
        """Mengambil beban hidup merata dari sheet 'Beban_Hidup'."""
        return self._get_load_data('Beban_Hidup', default_distribution='lognormal')

    def get_nodal_loads(self) -> Dict:
        """
        Mengambil beban nodal deterministik dari sheet 'Beban_Nodal'.
        Format: Node_ID, Fx, Fy, Mz
        """
        if 'Beban_Nodal' not in self.data:
            return {}

        df = self.data['Beban_Nodal']
        node_col = self.find_column(df, ['node', 'id']) or 'Node_ID'
        fx_col = self.find_column(df, ['fx', 'f_x']) or 'Fx'
        fy_col = self.find_column(df, ['fy', 'f_y']) or 'Fy'
        mz_col = self.find_column(df, ['mz', 'm_z', 'moment']) or 'Mz'

        result = {}
        for _, row in df.iterrows():
            node_id = pd.to_numeric(pd.Series([row[node_col]]), errors='coerce').iloc[0]
            if pd.isna(node_id):
                continue
            node_id = int(node_id)
            result[node_id] = {
                'Fx': self._to_float(row.get(fx_col), 0.0) if fx_col in df.columns else 0.0,
                'Fy': self._to_float(row.get(fy_col), 0.0) if fy_col in df.columns else 0.0,
                'Mz': self._to_float(row.get(mz_col), 0.0) if mz_col in df.columns else 0.0
            }

        return result

    def get_boundary_conditions(self) -> Dict:
        """
        Mengambil kondisi batas dari sheet 'Boundary_Condition'.
        Format: Node_ID, Restrain_X, Restrain_Y, Restrain_Rz
        """
        if 'Boundary_Condition' not in self.data:
            raise ValueError("Sheet 'Boundary_Condition' tidak ditemukan")

        df = self.data['Boundary_Condition']
        columns_lower = {str(col).strip().lower(): col for col in df.columns}

        node_col = (
            columns_lower.get('node_id')
            or columns_lower.get('node')
            or self.find_column(df, ['node_id', 'node'])
            or 'Node_ID'
        )
        x_col = (
            columns_lower.get('restrain_x')
            or columns_lower.get('fix_x')
            or self.find_column(df, ['restrain_x', 'fix_x'])
            or 'Restrain_X'
        )
        y_col = (
            columns_lower.get('restrain_y')
            or columns_lower.get('fix_y')
            or self.find_column(df, ['restrain_y', 'fix_y'])
            or 'Restrain_Y'
        )
        r_col = (
            columns_lower.get('restrain_rz')
            or columns_lower.get('restrain_r')
            or columns_lower.get('fix_rz')
            or columns_lower.get('fix_r')
            or self.find_column(df, ['restrain_rz', 'restrain_r', 'fix_rz', 'fix_r'])
            or 'Restrain_Rz'
        )

        result = {}
        for _, row in df.iterrows():
            node_id = pd.to_numeric(pd.Series([row[node_col]]), errors='coerce').iloc[0]
            if pd.isna(node_id):
                continue
            node_id = int(node_id)
            result[node_id] = {
                'X': int(self._to_float(row.get(x_col), 0.0)) if x_col in df.columns else 0,
                'Y': int(self._to_float(row.get(y_col), 0.0)) if y_col in df.columns else 0,
                'R': int(self._to_float(row.get(r_col), 0.0)) if r_col in df.columns else 0
            }

        return result

    def get_all_data(self) -> Dict:
        """Mengambil semua data yang diperlukan untuk analisis."""
        self.read_all_sheets()

        geometry = self.get_geometry()
        element_ids = geometry.get('element_ids', [])

        return {
            'geometry': geometry,
            'concrete': self.get_concrete_properties(element_ids),
            'steel': self.get_steel_properties(element_ids),
            'reinforcement': self.get_reinforcement_properties(element_ids),
            'dead_load': self.get_dead_load(),
            'live_load': self.get_live_load(),
            'nodal_loads': self.get_nodal_loads(),
            'boundary': self.get_boundary_conditions()
        }
