"""
Modul untuk simulasi Monte Carlo dan analisis keandalan probabilistik.
"""
import numpy as np
from typing import Any, Callable, Dict, List, Tuple
from scipy import stats


class RandomVariableGenerator:
    """Generator untuk variabel random dengan distribusi tertentu."""

    @staticmethod
    def lognormal(mean: float, stddev: float, size: int = 1) -> np.ndarray:
        """
        Generate variabel random dengan distribusi log-normal.

        Parameters:
        - mean: mean dari distribusi
        - stddev: standard deviation dari distribusi
        - size: jumlah samples
        """
        mean = float(mean)
        stddev = float(stddev)

        if size <= 0:
            return np.asarray([], dtype=float)
        if stddev <= 0.0 or mean <= 0.0:
            return np.full(size, mean, dtype=float)

        variance_ratio = (stddev / mean) ** 2
        sigma = np.sqrt(np.log(1.0 + variance_ratio))
        mu = np.log(mean) - 0.5 * sigma**2
        return np.random.lognormal(mu, sigma, size)

    @staticmethod
    def normal(mean: float, stddev: float, size: int = 1) -> np.ndarray:
        """
        Generate variabel random dengan distribusi normal.

        Parameters:
        - mean: mean
        - stddev: standard deviation
        - size: jumlah samples
        """
        mean = float(mean)
        stddev = float(stddev)

        if size <= 0:
            return np.asarray([], dtype=float)
        if stddev <= 0.0:
            return np.full(size, mean, dtype=float)

        return np.random.normal(mean, stddev, size)


class MonteCarloAnalysis:
    """Analisis Monte Carlo untuk keandalan struktural."""

    def __init__(self, num_simulations: int = 10000):
        self.num_simulations = int(num_simulations)
        self.results: Dict[str, Any] = {}

    @staticmethod
    def calculate_pf_and_beta(failures: int, num_simulations: int) -> Tuple[float, float]:
        """Hitung Pf dan Beta dari jumlah failure."""
        if num_simulations <= 0:
            return np.nan, np.nan

        pf = float(failures) / float(num_simulations)
        if 0.0 < pf < 1.0:
            beta = float(stats.norm.ppf(1.0 - pf))
        elif pf == 0.0:
            beta = float(np.inf)
        else:
            beta = float(-np.inf)
        return pf, beta

    def _generate_sample_arrays(self, random_vars: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:
        """Sample semua variabel random sekaligus per variabel."""
        sample_arrays: Dict[str, np.ndarray] = {}

        for var_name, var_info in random_vars.items():
            distribution = str(var_info.get('distribution', 'normal')).strip().lower()
            mean = float(var_info.get('mean', 0.0))
            stddev = float(var_info.get('stddev', 0.0))

            if distribution == 'normal':
                values = RandomVariableGenerator.normal(
                    mean,
                    stddev,
                    size=self.num_simulations
                )
            elif distribution == 'lognormal':
                values = RandomVariableGenerator.lognormal(
                    mean,
                    stddev,
                    size=self.num_simulations
                )
            else:
                raise ValueError(f"Unknown distribution: {distribution}")

            sample_arrays[var_name] = np.asarray(values, dtype=float)

        return sample_arrays

    @staticmethod
    def _normalize_performance_result(result: Any) -> Dict[str, Any]:
        """
        Normalisasi output performance function.

        Bentuk yang didukung:
        - bool
        - dict dengan key `is_safe`, `failed_elements_by_state`,
          `applicable_elements_by_state`
        """
        if isinstance(result, dict):
            is_safe = bool(result.get('is_safe', True))
            failed_by_state = {}
            applicable_by_state = {}

            for state_name, elem_ids in (result.get('failed_elements_by_state') or {}).items():
                normalized_state = str(state_name)
                failed_by_state[normalized_state] = sorted({
                    int(elem_id) for elem_id in (elem_ids or [])
                })

            for state_name, elem_ids in (result.get('applicable_elements_by_state') or {}).items():
                normalized_state = str(state_name)
                applicable_by_state[normalized_state] = sorted({
                    int(elem_id) for elem_id in (elem_ids or [])
                })

            failed_elements = sorted({
                int(elem_id)
                for elem_ids in failed_by_state.values()
                for elem_id in elem_ids
            })
            if result.get('failed_elements') is not None:
                failed_elements = sorted({
                    int(elem_id) for elem_id in (result.get('failed_elements') or [])
                })

            return {
                'is_safe': is_safe,
                'failed_elements': failed_elements,
                'failed_elements_by_state': failed_by_state,
                'applicable_elements_by_state': applicable_by_state
            }

        is_safe = bool(result)
        return {
            'is_safe': is_safe,
            'failed_elements': [],
            'failed_elements_by_state': {},
            'applicable_elements_by_state': {}
        }

    @staticmethod
    def _merge_applicable_elements(target: Dict[str, set], source: Dict[str, List[int]]) -> None:
        """Gabungkan himpunan elemen yang relevan per limit state."""
        for state_name, elem_ids in source.items():
            target.setdefault(state_name, set()).update(int(elem_id) for elem_id in elem_ids)

    @classmethod
    def _build_element_reliability_results(cls,
                                           num_simulations: int,
                                           overall_failure_counts: Dict[int, int],
                                           state_failure_counts: Dict[str, Dict[int, int]],
                                           applicable_by_state: Dict[str, set]) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Bangun Pf/Beta per elemen dan per limit state."""
        element_reliability: Dict[str, Dict[int, Dict[str, float]]] = {}

        overall_applicable = set()
        for elem_ids in applicable_by_state.values():
            overall_applicable.update(int(elem_id) for elem_id in elem_ids)

        overall_results: Dict[int, Dict[str, float]] = {}
        for elem_id in sorted(overall_applicable):
            failures = int(overall_failure_counts.get(elem_id, 0))
            pf, beta = cls.calculate_pf_and_beta(failures, num_simulations)
            overall_results[int(elem_id)] = {
                'failures': failures,
                'Pf': pf,
                'Beta': beta
            }
        element_reliability['overall'] = overall_results

        for state_name in sorted(applicable_by_state):
            state_results: Dict[int, Dict[str, float]] = {}
            state_failures = state_failure_counts.get(state_name, {})
            for elem_id in sorted(applicable_by_state[state_name]):
                failures = int(state_failures.get(elem_id, 0))
                pf, beta = cls.calculate_pf_and_beta(failures, num_simulations)
                state_results[int(elem_id)] = {
                    'failures': failures,
                    'Pf': pf,
                    'Beta': beta
                }
            element_reliability[state_name] = state_results

        return element_reliability

    def run_simulation(self, analysis_func: Callable,
                       random_vars: Dict,
                       performance_func: Callable = None) -> Dict:
        """
        Jalankan simulasi Monte Carlo.

        Parameters:
        - analysis_func: fungsi analisis struktural
        - random_vars: definisi variabel random
        - performance_func: fungsi evaluasi failure; boleh return bool atau dict detail
        """
        failures = 0
        analysis_failures = 0
        failure_indices: List[int] = []
        max_forces_history: List[Any] = []
        random_samples_history: List[Dict[str, float]] = []
        overall_failure_counts: Dict[int, int] = {}
        state_failure_counts: Dict[str, Dict[int, int]] = {}
        applicable_by_state: Dict[str, set] = {}

        print(f"Running Monte Carlo Simulation dengan {self.num_simulations} samples...")

        sample_arrays = self._generate_sample_arrays(random_vars)
        variable_names = list(sample_arrays.keys())

        for index in range(self.num_simulations):
            samples = {
                var_name: float(sample_arrays[var_name][index])
                for var_name in variable_names
            }
            random_samples_history.append(samples)

            analysis_result = None
            try:
                analysis_result = analysis_func(samples)
            except Exception as exc:
                analysis_failures += 1
                print(f"Error di simulasi {index}: {exc}")

            max_forces_history.append(analysis_result)

            if performance_func:
                performance_result = performance_func(analysis_result)
            else:
                performance_result = analysis_result is not None

            performance_details = self._normalize_performance_result(performance_result)
            self._merge_applicable_elements(
                applicable_by_state,
                performance_details['applicable_elements_by_state']
            )

            failed_elements = set()
            for state_name, elem_ids in performance_details['failed_elements_by_state'].items():
                state_counts = state_failure_counts.setdefault(state_name, {})
                for elem_id in elem_ids:
                    elem_id = int(elem_id)
                    state_counts[elem_id] = state_counts.get(elem_id, 0) + 1
                    failed_elements.add(elem_id)

            if not failed_elements:
                failed_elements.update(int(elem_id) for elem_id in performance_details['failed_elements'])

            for elem_id in failed_elements:
                overall_failure_counts[elem_id] = overall_failure_counts.get(elem_id, 0) + 1

            if not performance_details['is_safe']:
                failures += 1
                failure_indices.append(index)

            if (index + 1) % 1000 == 0:
                print(f"  ... {index + 1} simulasi selesai")

        pf, beta = self.calculate_pf_and_beta(failures, self.num_simulations)
        element_reliability = self._build_element_reliability_results(
            self.num_simulations,
            overall_failure_counts,
            state_failure_counts,
            applicable_by_state
        )

        results = {
            'num_simulations': self.num_simulations,
            'failures': failures,
            'analysis_failures': analysis_failures,
            'failure_indices': failure_indices,
            'Pf': pf,
            'Beta': beta,
            'max_forces_history': max_forces_history,
            'random_samples_history': random_samples_history,
            'element_reliability': element_reliability,
            'element_failure_counts': {
                'overall': overall_failure_counts,
                'by_limit_state': state_failure_counts
            }
        }

        self.results = results
        return results

    def get_statistics(self, data_values: List[float]) -> Dict:
        """Hitung statistik dari hasil simulasi."""
        data = np.asarray(data_values, dtype=float)

        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            '5th_percentile': float(np.percentile(data, 5)),
            '95th_percentile': float(np.percentile(data, 95)),
            'median': float(np.median(data)),
            'cov': float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else 0.0
        }


class ReliabilityAnalysis:
    """Analisis keandalan struktural menggunakan FORM dan SORM."""

    @staticmethod
    def calculate_pf_and_beta(failures: int, num_simulations: int) -> Tuple[float, float]:
        return MonteCarloAnalysis.calculate_pf_and_beta(failures, num_simulations)

    @staticmethod
    def get_safety_margin(capacity: float, demand: float) -> float:
        """Hitung safety margin: capacity - demand."""
        return float(capacity) - float(demand)

    @staticmethod
    def capacity_model_beton_beam(fc: float, fy: float,
                                  b: float, h: float, d: float,
                                  As: float, As_prime: float = 0) -> float:
        """
        Hitung kapasitas momen balok beton bertulang.

        Returns:
        - Mc: kapasitas momen (kN.m)
        """
        a_value = As * fy / (0.85 * fc * b)
        c_moment = As * fy * (d - a_value / 2.0) / 1e6

        if As_prime > 0:
            c_moment += As_prime * fy * (d - 50.0) / 1e6

        return float(0.9 * c_moment)

    @staticmethod
    def demand_from_analysis(max_moment: float) -> float:
        """Demand dari hasil analisis struktural."""
        return abs(float(max_moment))


class FailureRateEstimator:
    """Estimasi laju kegagalan dan analisis sensitivitas."""

    @staticmethod
    def sensitivity_analysis(results: Dict, variable_names: List[str]) -> Dict:
        """Analisis sensitivitas terhadap variabel random."""
        sensitivities = {}
        failure_indices = [
            int(index) for index in results.get('failure_indices', [])
        ]

        for var_name in variable_names:
            var_values = [
                sample[var_name]
                for sample in results['random_samples_history']
                if var_name in sample
            ]
            var_failure_values = [
                results['random_samples_history'][index][var_name]
                for index in failure_indices
                if 0 <= index < len(results['random_samples_history'])
                and var_name in results['random_samples_history'][index]
            ]

            if var_failure_values:
                sensitivities[var_name] = {
                    'mean_in_failure': float(np.mean(var_failure_values)),
                    'mean_overall': float(np.mean(var_values)),
                    'std_in_failure': float(np.std(var_failure_values))
                }
            else:
                sensitivities[var_name] = {
                    'mean': float(np.mean(var_values)) if var_values else 0.0,
                    'std': float(np.std(var_values)) if var_values else 0.0
                }

        return sensitivities
