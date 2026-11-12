#!/usr/bin/env python3
"""Opt-in macOS release gate: real synthetic-speech decode in four build/log cells.

Run with an existing public Whisper model; never downloads models or captures audio.
Every cell also exercises the native callback's safe warning/error probe in that build.
Missing prerequisites, decode failures and unexecuted cells are failures, never skips.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import struct
import sys
import time
import uuid
import wave

PHRASE = 'Violet lanterns orbit slowly'
ROOT = Path(__file__).resolve().parents[1]
RUN = uuid.uuid4().hex


class Formatter(logging.Formatter):
    def format(self, record):
        return json.dumps({'timestamp': time.time(), 'level': record.levelname.lower(),
                          'component': 'f7-native-privacy', 'event': record.msg,
                          'run_id': RUN, **getattr(record, 'fields', {})})


log = logging.getLogger('f7-native-privacy')
log.setLevel(logging.INFO)


def private_write(path, data):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as file:
        file.write(data)


def call(command, name, out, env=None, timeout=900):
    start = time.monotonic()
    log.info('external_started', extra={'fields': {'call': name, 'target': Path(command[0]).name}})
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        private_write(out / f'{name}.stdout', error.stdout or b'')
        private_write(out / f'{name}.stderr', error.stderr or b'')
        log.error('external_timeout', extra={'fields': {'call': name, 'duration': time.monotonic() - start}})
        raise RuntimeError('external call timed out') from None
    private_write(out / f'{name}.stdout', result.stdout)
    private_write(out / f'{name}.stderr', result.stderr)
    log.log(logging.INFO if result.returncode == 0 else logging.ERROR, 'external_finished',
            extra={'fields': {'call': name, 'exit': result.returncode, 'duration': time.monotonic() - start}})
    if result.returncode:
        raise RuntimeError(f'external call failed: {name}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='new evidence directory; never overwritten')
    parser.add_argument('--voice', default='Samantha', help='exact installed English say voice; never downloaded')
    args = parser.parse_args()
    model = args.model.resolve(strict=True)
    if not model.is_file() or model.stat().st_size < 1_000_000:
        parser.error('a real existing Whisper model is required')
    out = args.out.absolute()
    out.mkdir(mode=0o700, exist_ok=False)
    for handler in (logging.StreamHandler(), logging.FileHandler(out / 'run.jsonl')):
        handler.setFormatter(Formatter())
        log.addHandler(handler)
    os.chmod(out / 'run.jsonl', 0o600)
    log.info('gate_started', extra={'fields': {'model_bytes': model.stat().st_size, 'required_cells': 4}})
    try:
        voices = call(['/usr/bin/say', '-v', '?'], 'voices', out)
        installed = [match.group(1).strip() for match in re.finditer(r'^(.+?)\s+en_[A-Z]{2}\s+#', voices.stdout.decode(), re.M)]
        if args.voice not in installed:
            raise RuntimeError('selected installed English voice required; no voice download attempted')
        aiff, wav = out / 'synthetic.aiff', out / 'synthetic.wav'
        call(['/usr/bin/say', '-v', args.voice, '-r', '135', '-o', str(aiff),
              f'{PHRASE}. {PHRASE}.'], 'synthesize', out)
        call(['/usr/bin/afconvert', '-f', 'WAVE', '-d', 'LEI16@16000', '-c', '1', str(aiff), str(wav)], 'convert', out)
        os.chmod(aiff, 0o600)
        os.chmod(wav, 0o600)
        with wave.open(str(wav), 'rb') as source:
            frames, rate, channels = source.getnframes(), source.getframerate(), source.getnchannels()
        if frames < 16_000 or rate != 16_000 or channels != 1:
            raise RuntimeError('synthetic audio format/duration invalid')
        float_wav = out / 'synthetic-float.wav'
        call(['/usr/bin/afconvert', '-f', 'WAVE', '-d', 'LEF32@16000', '-c', '1', str(wav), str(float_wav)], 'convert-float', out)
        os.chmod(float_wav, 0o600)
        target = Path(os.environ.get('CARGO_TARGET_DIR', ROOT / 'target')).absolute()
        completed = []
        for build in ('debug', 'release'):
            flags = ['--release'] if build == 'release' else []
            call(['cargo', 'build', '--locked', '--bin', 'meetscribe', *flags], f'build-{build}', out)
            compiled = call(['cargo', 'test', '--locked', '--bin', 'meetscribe', '--no-run', '--message-format=json', *flags],
                            f'probe-build-{build}', out)
            artifacts = [json.loads(line) for line in compiled.stdout.splitlines() if line.startswith(b'{')]
            probes = [item['executable'] for item in artifacts if item.get('reason') == 'compiler-artifact'
                      and item.get('target', {}).get('name') == 'meetscribe' and item.get('profile', {}).get('test')
                      and item.get('executable')]
            if len(probes) != 1:
                raise RuntimeError('exactly one native callback probe executable required')
            for level in ('info', 'debug'):
                cell = f'{build}-{level}'
                capture = out / f'capture-{cell}'
                capture.mkdir(mode=0o700)
                shutil.copyfile(float_wav, capture / 'system.wav')
                os.chmod(capture / 'system.wav', 0o600)
                data_size = frames * 4
                header = struct.pack('<4sI4s4sIHHIIHH4sI', b'RIFF', 36 + data_size, b'WAVE', b'fmt ', 16,
                                     3, 1, rate, rate * 4, 4, 32, b'data', data_size)
                private_write(capture / 'mic.wav', header + b'\0' * data_size)
                sentinel = capture / 'database-must-not-open'
                sentinel.mkdir(mode=0o700)
                env = dict(os.environ, RUST_LOG=level)
                decoded = call([str(target / build / 'meetscribe'), 'transcribe', str(capture), '--model', str(model),
                                '--lang', 'en', '--no-split', '--speaker-model', str(capture / 'absent-speaker.onnx'),
                                '--no-store', '--db', str(sentinel)], f'decode-{cell}', out, env)
                normalized = ' '.join(re.findall(r'\w+', decoded.stdout.decode(errors='replace').casefold()))
                expected = PHRASE.casefold()
                if expected not in normalized:
                    raise RuntimeError(f'positive real decode assertion failed: {cell}')
                distinctive = set(expected.split())
                if distinctive.intersection(re.findall(r'\w+', decoded.stderr.decode(errors='replace').casefold())):
                    raise RuntimeError(f'decoded content reached diagnostic stderr: {cell}')
                if not sentinel.is_dir() or any(sentinel.iterdir()):
                    raise RuntimeError('no-store touched its database sentinel')
                probe_env = dict(env, MEETSCRIBE_TEST_NATIVE_LOG=level)
                probe = call([probes[0], '--exact', 'asr::tests::audit_f7_callback_privacy', '--nocapture'],
                             f'probe-{cell}', out, probe_env)
                if b'native_callback_ack' not in probe.stdout or b'SYNTHETIC_RELEASE_TEXT_MUST_NOT_LOG' in probe.stderr:
                    raise RuntimeError('native callback probe did not execute safely')
                events = [json.loads(line) for line in probe.stderr.splitlines()]
                for event in ('native_warning', 'native_error'):
                    if not any(item.get('event') == event and item.get('count', 0) > 0 for item in events):
                        raise RuntimeError(f'safe native diagnostic absent: {event}')
                completed.append(cell)
                log.info('cell_verified', extra={'fields': {'cell': cell, 'positive_decode': True,
                                                          'diagnostics_content_free': True, 'safe_native_counts': True}})
        if len(completed) != 4:
            raise RuntimeError('required cell accounting incomplete')
        private_write(out / 'result.json', json.dumps({'run_id': RUN, 'cells': completed, 'ok': True}).encode())
        log.info('gate_passed', extra={'fields': {'executed_cells': len(completed)}})
        return 0
    except Exception as error:
        log.error('gate_failed', extra={'fields': {'error_class': type(error).__name__,
                                                'reason': str(error) if isinstance(error, RuntimeError) else 'io_or_process_error'}})
        return 1


if __name__ == '__main__':
    sys.exit(main())
