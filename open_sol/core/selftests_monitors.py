"""Self-test check of the core.monitors ProbeRows host parking (CPU, no model run): rows recorded around parks, device
growth and a column added later read back exactly; the SelfTestCase lives in model.selftests_analytic, which passes its
SubChecks.add (core imports no model)."""
import numpy as np

_FIRST_ROWS = 1500    # > PROBE_INITIAL_ROWS: the device table doubles before the first park
_SECOND_ROWS = 700
_THIRD_ROWS = 3000    # after the second column joined: doubles again, below the min_rows of the refused park
_LAST_ROWS = 10
_DT = 0.5             # s per row (float64-exact times)


def _value(row, column):
    """float32-exact sample of a row in a column."""
    return np.float32(row + 0.25 * (column + 1))


def check_probe_rows_parking(add):
    """Fill the sub-checks through add(name, passed, **details) (model's SubChecks.add)."""
    from .monitors import PROBE_INITIAL_ROWS, ProbeRows

    class _Table(ProbeRows):
        def _build_launch(self):
            return ('probe launch',)

    table = _Table(np)

    def record(count):
        """count rows as a probe kernel writes them (device row = the counter), each stamped."""
        for _ in range(count):
            launch = table.next_launch()
            row = int(table._row_counter[0])
            for column in range(table.n_columns):
                table._samples[row, column] = _value(table.n_rows, column)
            table._row_counter[0] = row + 1
            table.stamp(table.n_rows * _DT, _DT)
        return launch

    table._add_column()
    record(_FIRST_ROWS)
    add('device_grew_before_park', table._samples.shape[0] > PROBE_INITIAL_ROWS, capacity=table._samples.shape[0])
    launch = table.next_launch()
    table.park(numpy_tables=True)
    add('park_empties_device', table.device_rows == 0 and int(table._row_counter[0]) == 0,
        device_rows=table.device_rows)
    add('park_keeps_launch', table.next_launch() is launch)   # same table and counter: graphs stay valid
    record(_SECOND_ROWS)
    second_start = table.n_rows
    second = table._add_column()
    capacity_before = table._samples.shape[0]
    record(_THIRD_ROWS)
    add('device_grew_after_park', table._samples.shape[0] > capacity_before, capacity=table._samples.shape[0])
    table.park(min_rows=_THIRD_ROWS + _SECOND_ROWS + 1, numpy_tables=True)
    add('park_below_min_rows_refused', table.device_rows == _SECOND_ROWS + _THIRD_ROWS, device_rows=table.device_rows)
    table.park(min_rows=_SECOND_ROWS + _THIRD_ROWS, numpy_tables=True)
    record(_LAST_ROWS)
    total = _FIRST_ROWS + _SECOND_ROWS + _THIRD_ROWS + _LAST_ROWS
    add('rows_counted', table.n_rows == total and table.device_rows == _LAST_ROWS, n_rows=table.n_rows)
    first_values = table.column_values(0, 0)
    add('first_column_exact', np.array_equal(first_values, np.array([_value(row, 0) for row in range(total)])),
        n_values=int(first_values.size))
    second_values = table.column_values(second, second_start)
    add('later_column_exact', np.array_equal(second_values, np.array([_value(row, second)
                                                                       for row in range(second_start, total)])),
        n_values=int(second_values.size))
    middle = _FIRST_ROWS - 3   # inside the first parked block
    add('partial_read_exact', np.array_equal(table.column_values(0, middle),
                                             np.array([_value(row, 0) for row in range(middle, total)])))
    add('device_only_read_exact', np.array_equal(table.column_values(0, total - 2),
                                                 np.array([_value(row, 0) for row in (total - 2, total - 1)])))
    add('times_kept', np.array_equal(table._times[:total], np.arange(total) * _DT))
    cpu_table = ProbeRows(np)
    cpu_table._add_column()
    cpu_table.stamp(0.0)
    cpu_table.park()   # numpy tables stay unparked by default (engines on the cpu)
    add('numpy_table_not_parked_by_default', cpu_table._parked is None)
