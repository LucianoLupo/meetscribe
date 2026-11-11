#!/usr/bin/env python3
"""Mandatory model-free seven-finding gate with nonzero per-scenario accounting."""
import argparse
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    'f1_checkpoint': ('iterations', 5), 'f1_metadata': ('iterations', 5),
    'f2_retention': ('iterations', 5), 'f2_crash_budget': ('iterations', 5),
    'f2_retry_worker': ('iterations', 5), 'f2_child_retry_retention': ('iterations', 5),
    'f2_launch_failure_budget': ('iterations', 5), 'f2_launch_busy_free': ('iterations', 5),
    'f2_launch_reserved_once': ('iterations', 5), 'f2_launch_preserve_ineligible': ('iterations', 5),
    'f3_resume_identity': ('iterations', 5), 'f3_db_race': ('iterations', 5),
    'f3_child_cutpoints': ('iterations', 5), 'f4_merge': ('iterations_per_scenario', 5),
    'f5_self_source': ('iterations', 5), 'f5_interrupted_swap': ('iterations', 5),
    'f6_stale_control': ('iterations', 5), 'f7_callback': ('log_levels', 2),
    'f1_trusted_roll_inventory': ('iterations', 5),
    'f5_source_absence_reconcile': ('iterations', 5), 'f5_source_entry_identity': ('iterations', 5),
    'f6_inventory_failure': ('iterations', 5), 'f6_addressed_poll': ('iterations', 5),
}


def verify(events):
    by_case = {}
    for event in events:
        name = event.get('case')
        if name in by_case:
            raise ValueError('duplicate case receipt')
        by_case[name] = event
    for name, (field, minimum) in REQUIRED.items():
        value = by_case.get(name, {}).get(field)
        if type(value) is not int or value < minimum:
            raise ValueError(f'missing or insufficient execution: {name}')
    if by_case['f4_merge'].get('scenarios', 0) < 11:
        raise ValueError('speaker scenario accounting incomplete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    run = uuid.uuid4().hex
    logger = logging.getLogger('hardening-gate')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

    def event(name, level=logging.INFO, **fields):
        logger.log(level, json.dumps({'timestamp': time.time(), 'level': logging.getLevelName(level).lower(),
                                    'component': 'hardening-gate', 'run_id': run, 'event': name, **fields}))
    if args.self_test:
        valid = [{'case': name, field: minimum} for name, (field, minimum) in REQUIRED.items()]
        next(item for item in valid if item['case'] == 'f4_merge')['scenarios'] = 11
        verify(valid)
        for invalid in ([], valid[:-1], [dict(item, iterations=0) if item['case'] == 'f1_checkpoint' else item for item in valid]):
            try:
                verify(invalid)
            except ValueError:
                continue
            raise AssertionError('invalid execution accounting accepted')
        event('self_test_passed', rejected_cases=3)
        return 0
    if args.out is None:
        parser.error('--out is required unless --self-test is selected')
    try:
        args.out.mkdir(mode=0o700, exist_ok=False)
        event('command_started', target='cargo', required_cases=len(REQUIRED))
        started = time.monotonic()
        result = subprocess.run(['cargo', 'test', '--locked', '--bin', 'meetscribe', '--', '--nocapture', '--test-threads=1'],
                                cwd=ROOT, capture_output=True, timeout=600)
        for suffix, data in (('stdout', result.stdout), ('stderr', result.stderr)):
            with os.fdopen(os.open(args.out / f'tests.{suffix}', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as output:
                output.write(data)
        event('command_finished', exit=result.returncode, duration=time.monotonic() - started)
        if result.returncode:
            raise RuntimeError('Rust regression suite failed')
        receipts = [json.loads(match.group()) for match in re.finditer(rb'\{"event":"audit_case_complete"[^\n]*?\}', result.stdout)]
        verify(receipts)
        summary = {'run_id': run, 'ok': True, 'required_cases': len(REQUIRED), 'cases': receipts,
                   'real_model_release_gate': 'separate; not counted as CI coverage'}
        with os.fdopen(os.open(args.out / 'result.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as output:
            json.dump(summary, output)
        event('gate_passed', executed_cases=len(receipts))
        return 0
    except Exception as error:
        event('gate_failed', logging.ERROR, error_class=type(error).__name__)
        return 1


if __name__ == '__main__':
    sys.exit(main())
