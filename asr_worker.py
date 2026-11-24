"""Runs in a separate process so GPU allocations are released on exit."""
import argparse
import json
import os
import site
import sys
from pathlib import Path

from core import ROOT, atomic_json, clean_text, compact_segments

_dll_handles = []

STYLE_LIMITS = {
    'tiktok': (3.5, 42, 0.55, 1.1),
    'balanced': (5.2, 66, 0.70, 1.4),
    'full': (6.8, 86, 0.85, 1.6),
}


def register_cuda():
    if os.name != 'nt':
        return
    for base in site.getsitepackages():
        for name in ('cublas', 'cudnn', 'cuda_runtime'):
            p = Path(base) / 'nvidia' / name / 'bin'
            if p.exists():
                os.environ['PATH'] = str(p) + os.pathsep + os.environ.get('PATH', '')
                _dll_handles.append(os.add_dll_directory(str(p)))


def emit(**kwargs):
    print(json.dumps(kwargs, ensure_ascii=False), flush=True)


def split_segment(segment, segment_style='balanced'):
    max_dur, max_chars, pause_gap, min_punct_dur = STYLE_LIMITS.get(
        segment_style, STYLE_LIMITS['balanced']
    )
    words = list(segment.words or [])
    if not words:
        return [dict(start=segment.start, end=segment.end, source=clean_text(segment.text), vi='')]
    groups, current = [], []
    for word in words:
        if current and (
            word.end - current[0].start > max_dur
            or len(''.join(w.word for w in current)) + len(word.word) > max_chars
            or word.start - current[-1].end > pause_gap
        ):
            groups.append(current)
            current = []
        current.append(word)
        token = word.word.rstrip()
        if token.endswith(('.', '?', '!')) and word.end - current[0].start > min_punct_dur:
            groups.append(current)
            current = []
        elif (
            segment_style == 'tiktok'
            and token.endswith((',', ';', ':'))
            and len(''.join(w.word for w in current).strip()) >= 24
            and word.end - current[0].start >= 1.3
        ):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return [
        dict(
            start=g[0].start,
            end=max(g[-1].end, g[0].start + .05),
            source=clean_text(''.join(w.word for w in g)),
            vi='',
        )
        for g in groups
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='medium', choices=['small', 'medium', 'large-v3'])
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--language', default='en')
    parser.add_argument('--segment-style', default='balanced', choices=['tiktok', 'balanced', 'full'])
    args = parser.parse_args()
    register_cuda()
    from faster_whisper import WhisperModel
    emit(progress=0, message='Đang nạp mô hình nhận diện. Lần đầu cần tải mô hình từ Internet.')
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type='int8_float16' if args.device == 'cuda' else 'int8',
        download_root=str(ROOT / 'models' / 'whisper'),
    )
    iterator, info = model.transcribe(
        args.input,
        language=None if args.language == 'auto' else args.language,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    rows = []
    for segment in iterator:
        rows.extend(r for r in split_segment(segment, args.segment_style) if r['source'])
        emit(
            progress=min(segment.end / max(info.duration, 1), .99),
            message=f'Đã nhận diện đến {segment.end:.0f} / {info.duration:.0f} giây',
        )
    if not rows:
        raise RuntimeError('Không tìm thấy lời nói. Kiểm tra video có tiếng và chọn đúng ngôn ngữ.')
    max_chars = STYLE_LIMITS[args.segment_style][1]
    max_dur = STYLE_LIMITS[args.segment_style][0]
    rows = compact_segments(rows, max_chars=max_chars, max_duration=max_dur)
    atomic_json(args.output, dict(language=info.language, duration=info.duration, segments=rows))
    emit(progress=1, message='Đã nhận diện xong.')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'{type(e).__name__}: {e}', file=sys.stderr, flush=True)
        sys.exit(1)
