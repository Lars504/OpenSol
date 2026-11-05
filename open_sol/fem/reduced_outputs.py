"""Cheap outputs of reduced frequency sweeps (experiment plan A4): the field policy of a sweep (fields published only for
stored snapshots and the last frequency) and ReducedRows, chosen rows of x = P V y + x0 and of A_s x precomputed as
small dense blocks, so a frequency's records cost O(rows x basis) instead of V y. Host numpy, scipy sparse inputs."""
# Contract: field_policy(options, frequencies) reads options['fields'] of the frequency_response study ('all' default |
# 'last': study_frequency_response passes 'last' unless every frequency keeps a snapshot; direct sweeps get it too, by
# plan_frequency_response with method 'direct', R6); wants_fields(policy, f) is True for every frequency under 'all',
# else for the last listed one only. A solver that skipped fields publishes them in publish_frequency_response (the
# study calls it after every sweep, so also after a cancel). Scalar harmonic systems (fem.harmonic) use
# HarmonicSystemBase.linear_output instead of ReducedRows (gathers of the full-numbering values).
import numpy as np

FIELD_POLICIES = ('all', 'last')
_FREQUENCY_MATCH_RTOL = 1e-12


def field_policy(options, frequencies) -> tuple:
    """(policy of FIELD_POLICIES, last frequency [Hz]) of a planned sweep (direct or reduced) over frequencies in solve
    order."""
    policy = (options or {}).get('fields', 'all')
    if policy not in FIELD_POLICIES:
        raise ValueError(f"frequency sweep fields must be one of {', '.join(FIELD_POLICIES)}, got {policy!r}")
    return policy, float(np.asarray(frequencies, dtype=np.float64).ravel()[-1])


def wants_fields(policy, frequency) -> bool:
    """True when a solve at frequency [Hz] publishes its fields under policy (None = every frequency)."""
    if policy is None:
        return True
    kind, last = policy
    return kind == 'all' or abs(float(frequency) - last) <= _FREQUENCY_MATCH_RTOL * abs(last)


class ReducedRows:
    """Rows of the full solution x = P V y + x0 and of A(omega) x = sum_s c_s(omega) A_s x of a reduced model, built
    once: prolongation P (n, n_free) sparse, basis V (n_free, k), offset x0 (n,), terms [(A_s (n, n) sparse,
    coefficient(omega) -> complex)]."""

    def __init__(self, rows, prolongation, basis, offset, terms):
        self.rows = np.unique(np.asarray(rows, dtype=np.int64).ravel())
        self.n = int(prolongation.shape[0])
        offset = np.asarray(offset, dtype=np.complex128).ravel()
        prolongation = prolongation.tocsr()
        self._x_block = np.asarray(prolongation[self.rows] @ basis, dtype=np.complex128)   # (r, k)
        self._x_offset = offset[self.rows]
        self._terms = []
        for matrix, coefficient in terms:
            selected = matrix.tocsr()[self.rows]
            block = np.asarray((selected @ prolongation) @ basis, dtype=np.complex128)        # (r, k)
            self._terms.append((coefficient, block, np.asarray(selected @ offset, dtype=np.complex128).ravel()))

    def evaluate(self, omega, coordinates) -> tuple:
        """(x, A(omega) x): complex (n,) vectors exact at the rows, 0 elsewhere, for reduced coordinates y (k,)."""
        coordinates = np.asarray(coordinates, dtype=np.complex128)
        solution = np.zeros(self.n, dtype=np.complex128)
        product = np.zeros(self.n, dtype=np.complex128)
        solution[self.rows] = self._x_block @ coordinates + self._x_offset
        values = np.zeros(self.rows.size, dtype=np.complex128)
        for coefficient, block, offset in self._terms:
            values += complex(coefficient(omega)) * (block @ coordinates + offset)
        product[self.rows] = values
        return solution, product
