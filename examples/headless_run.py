"""Headless quickstart (no GUI): template -> check (CPU) -> run (GPU) -> Results, the path agents use. The CLI
equivalent: python -m open_sol check model.json, then python -m open_sol run model.json --figures."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # not needed after pip install -e .

from open_sol import api

spec = api.template('dielectric_cylinder')   # python -m open_sol describe templates lists every key
spec.study.settings['duration'] = 100e-15    # edit the spec like any dataclass (or api.load / api.save a JSON file)

report = api.check(spec, out_dir='headless_run_out')   # issues, mesh and step estimate -> check.json
if report['n_errors']:
    sys.exit('\n'.join(f"{issue['node']}: {issue['message']}" for issue in report['issues']
                       if issue['level'] == 'error'))
print(f"check: {report['status']}, {report['mesh']['n_cells']:,} cells, ~{report['study']['n_steps']} steps")

with api.run(spec, out_dir='headless_run_out', figures=True) as results:   # summary.json, log.txt, results.npz
    print(f'run: {results.status}; fields {results.field_names()}; results {results.result_names()}')
    flux = results.result('em.transmitted_flux')   # dict: wavelength [m], power ({flux['unit']})
    peak = int(np.argmax(flux['power']))
    print(f"transmitted flux peaks at {flux['wavelength'][peak] * 1e9:.0f} nm ({flux['unit']})")
    ez = results.field('em.ez')   # FieldData: values float32 (nx, ny), extent [m]
    print(f'final |E_z| max {np.abs(ez.values).max():.3g} V/m')
