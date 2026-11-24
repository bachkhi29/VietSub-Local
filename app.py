from __future__ import annotations

import base64
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import gradio as gr
from core import (
    FONT_FILES,
    MODEL,
    OUTPUT,
    ROOT,
    SEGMENT_STYLES,
    SRT_IMPORT_MODES,
    TARGET_LANGUAGES,
    TRANSLATION_STYLES,
    compact_segments,
    export_project,
    folder,
    list_projects,
    load_project,
    new_project,
    parse_srt,
    parse_time,
    require_model,
    rows_to_segments,
    save_project,
    subtitle_delay,
    table_rows,
    timestamp,
    translate_project,
    unload_model,
    webvtt,
)
from render import render_video

OUTPUT.mkdir(exist_ok=True)
ASSETS = ROOT / 'assets'
LOGO_PATH = ASSETS / 'logo.png'
FAVICON_PATH = ASSETS / 'favicon.png'

BROWSER_VIDEO_EXTS = ('.mp4', '.webm', '.ogg')


def _logo_data_uri():
    if LOGO_PATH.is_file():
        encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode('ascii')
        return f'data:image/png;base64,{encoded}'
    return ''


STYLE_PRESETS = {
    'TikTok Gọn · Nền đen mờ': dict(
        color='#FFFFFF', secondary='#FDE68A', background='#120D1D',
        opacity=80, radius=14, backdrop='panel', scope='all',
        outline=1.5, font='Segoe UI', bold=True,
        outline_color='#09090B', shadow=0.0, box_padding=12, italic=False,
    ),
    'Hồng LocalBachSub · Nổi bật': dict(
        color='#FFFFFF', secondary='#FCE7F3', background='#DB2777',
        opacity=92, radius=18, backdrop='panel', scope='translation',
        outline=0.0, font='Segoe UI', bold=True,
        outline_color='#1F0916', shadow=0.0, box_padding=13, italic=False,
    ),
    'Đen trên nền vàng': dict(
        color='#111111', secondary='#FFFFFF', background='#FFD800',
        opacity=100, radius=22, backdrop='panel', scope='translation',
        outline=0.0, font='Arial Narrow', bold=True,
        outline_color='#111111', shadow=0.0, box_padding=13, italic=False,
    ),
    'Podcast · Chữ vàng viền nét': dict(
        color='#FACC15', secondary='#FFFFFF', background='#000000',
        opacity=0, radius=0, backdrop='none', scope='all',
        outline=3.5, font='Segoe UI', bold=True,
        outline_color='#09090B', shadow=2.0, box_padding=12, italic=False,
    ),
    'Trắng tối giản · Đổ bóng': dict(
        color='#FFFFFF', secondary='#F3E8FF', background='#000000',
        opacity=0, radius=0, backdrop='none', scope='all',
        outline=2.5, font='Segoe UI', bold=True,
        outline_color='#140C1C', shadow=2.0, box_padding=12, italic=False,
    ),
    'Trắng trên nền xanh đậm': dict(
        color='#FFFFFF', secondary='#BAE6FD', background='#163A5F',
        opacity=92, radius=16, backdrop='panel', scope='translation',
        outline=0.5, font='Segoe UI', bold=True,
        outline_color='#0B192C', shadow=0.0, box_padding=13, italic=False,
    ),
}

LAYOUT_PRESETS = {
    'tiktok_crop': dict(
        aspect='9:16', frame_fill='crop', position='Dưới', margin=215,
        text_width=76, max_chars=28, max_lines=2, size=44, min_size=22,
        box_padding=12, dual_gap=4, wrap_mode='balanced',
    ),
    'tiktok_fit': dict(
        aspect='9:16', frame_fill='fit', position='Dưới', margin=185,
        text_width=84, max_chars=34, max_lines=2, size=42, min_size=20,
        box_padding=12, dual_gap=4, wrap_mode='balanced',
    ),
    'youtube_wide': dict(
        aspect='16:9', frame_fill='fit', position='Dưới', margin=60,
        text_width=86, max_chars=50, max_lines=2, size=44, min_size=24,
        box_padding=14, dual_gap=6, wrap_mode='balanced',
    ),
    'original_clean': dict(
        aspect='original', frame_fill='fit', position='Dưới', margin=75,
        text_width=84, max_chars=42, max_lines=2, size=44, min_size=22,
        box_padding=13, dual_gap=6, wrap_mode='balanced',
    ),
}


def check_system():
    lines = [f'Python: {sys.version.split()[0]}']
    try:
        gpu = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        lines.append('GPU: ' + (gpu.stdout.strip() if gpu.returncode == 0 else 'chưa đọc được NVIDIA'))
    except (OSError, subprocess.TimeoutExpired):
        lines.append('GPU: chưa tìm thấy nvidia-smi; có thể chọn chế độ CPU.')
    try:
        require_model()
        lines.append(f'Ollama: sẵn sàng, đã có mô hình {MODEL}.')
    except RuntimeError as e:
        lines.append(str(e))
    if os.name == 'nt':
        import site
        for lib in ('cublas', 'cudnn'):
            found = any((Path(p) / 'nvidia' / lib / 'bin').exists() for p in site.getsitepackages())
            lines.append(f'Thư viện {lib}: ' + ('đã cài' if found else 'chưa có; chạy 1_CAI_DAT.bat'))
    return '\n'.join(lines)


def _auto_video_cc(project, media_input=None, preferred_mode='Song ngữ'):
    """Build a ready-to-play Video component with CC tracks if a browser-compatible video exists."""
    if not project:
        return None
    video = media_input or project.get('media')
    if not video or not Path(video).is_file():
        return None
    if Path(video).suffix.lower() not in BROWSER_VIDEO_EXTS:
        return None
    complete = all(s['vi'].strip() for s in project['segments'])
    if preferred_mode == 'Gốc' or not complete:
        preview_mode = 'source'
    elif preferred_mode == 'Bản dịch':
        preview_mode = 'vi'
    else:
        preview_mode = 'dual'
    chosen = folder(project) / f'preview-{preview_mode}.vtt'
    chosen.write_text(
        webvtt(project['segments'], preview_mode, project.get('subtitle_delay', 0)),
        encoding='utf-8',
    )
    return gr.Video(value=str(video), subtitles=str(chosen), interactive=False)


def preview_seek_update(project):
    segments = project['segments']
    delay = project.get('subtitle_delay', 0)
    first = segments[0]
    first_time = max(0, first['start'] + delay + min(.5, (first['end'] - first['start']) / 2))
    last_time = max(segment['end'] + delay for segment in segments)
    return gr.update(maximum=max(1, round(last_time + 1, 1)), value=round(first_time, 1))


def active_cue_at(project, moment, delay_value=0):
    if not project or not project.get('segments'):
        return None
    now = parse_time(moment or 0)
    delay = subtitle_delay(delay_value)
    for s in project['segments']:
        if s['start'] + delay <= now <= s['end'] + delay:
            return s
    return min(
        project['segments'],
        key=lambda s: min(abs(now - (s['start'] + delay)), abs(now - (s['end'] + delay))),
    )


def active_cue_fields(project, moment, delay_value=0):
    cue = active_cue_at(project, moment, delay_value)
    if not cue:
        return gr.update(value=None), '', ''
    return cue['id'], cue['source'], cue['vi']


def source_job(media, source_srt, model, device, language, target,
               srt_mode='auto', segment_style='balanced', progress=gr.Progress()):
    try:
        if target == language:
            raise ValueError('Ngôn ngữ gốc và ngôn ngữ đích đang giống nhau.')
        if source_srt:
            raw_text = Path(source_srt).read_text(encoding='utf-8-sig')
            segments = parse_srt(raw_text, mode=srt_mode, source_lang=language, target_lang=target)
            if segment_style == 'tiktok':
                segments = compact_segments(segments, max_chars=42, max_duration=3.5)
            project = new_project(
                Path(source_srt).name,
                segments,
                language if language != 'auto' else 'source',
                media,
                target,
            )
        else:
            if not media:
                raise ValueError('Hãy chọn video/âm thanh hoặc thả file SRT có sẵn vào trước.')
            progress(0, desc='Chuẩn bị nhận diện giọng nói…')
            unload_model()
            work = Path(tempfile.mkdtemp(prefix='asr_', dir=OUTPUT))
            result = work / 'result.json'
            log = work / 'asr.log'
            command = [
                sys.executable, '-u', str(ROOT / 'asr_worker.py'),
                '--input', str(media), '--output', str(result),
                '--model', model, '--device', device,
                '--language', language, '--segment-style', segment_style,
            ]
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            proc = None
            try:
                with log.open('w', encoding='utf-8') as err:
                    proc = subprocess.Popen(
                        command, stdout=subprocess.PIPE, stderr=err,
                        text=True, encoding='utf-8', errors='replace', env=env,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                    )
                    for line in proc.stdout:
                        try:
                            event = json.loads(line)
                            progress(event['progress'], desc=event['message'])
                        except (ValueError, KeyError):
                            continue
                    code = proc.wait()
                if code != 0:
                    detail = log.read_text(encoding='utf-8')[-1600:]
                    raise RuntimeError(
                        'Nhận diện chưa chạy được. Nếu lỗi CUDA/cuDNN, hãy thử chuyển Thiết bị sang CPU + small.\n'
                        + detail
                    )
                data = json.loads(result.read_text(encoding='utf-8'))
                project = new_project(Path(media).name, data['segments'], data['language'], media, target)
            finally:
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()

        files, complete = export_project(project)
        translated_count = sum(bool(s['vi'].strip()) for s in project['segments'])
        total_count = len(project['segments'])
        if complete:
            msg = f'Đã nạp {total_count} câu song ngữ (đủ cả gốc & dịch). Bạn có thể đối chiếu bên dưới hoặc sang tab 2 thiết kế video.'
        elif translated_count > 0:
            msg = f'Đã tạo {total_count} câu (có sẵn {translated_count} câu dịch). Bấm “Dịch các câu còn trống” ngay phía trên bảng.'
        else:
            msg = f'Đã tạo {total_count} câu gốc. Bấm “Dịch các câu còn trống” trên thanh công cụ để dịch sang {TARGET_LANGUAGES[target]}.'
        return (
            project,
            table_rows(project),
            files,
            msg,
            gr.update(choices=list_projects(), value=project['job']),
            _auto_video_cc(project, media),
            gr.update(value=target),
            gr.update(value=0),
            preview_seek_update(project),
        )
    except Exception as e:
        raise gr.Error(str(e)) from e


def edited(project, rows):
    if not project:
        raise ValueError('Hãy tạo hoặc mở một công việc trước.')
    result = copy.deepcopy(project)
    result['segments'] = rows_to_segments(rows, project['segments'])
    save_project(result)
    return result


def save_and_export(project, rows, media=None, cc_mode='Song ngữ'):
    try:
        project = edited(project, rows)
        files, complete = export_project(project)
        msg = (
            'Đã lưu phụ đề và cập nhật đủ bộ file SRT, VTT, ASS, ZIP.'
            if complete else
            'Đã lưu công việc và xuất bản gốc. Dịch nốt các câu trống để xuất đủ file song ngữ.'
        )
        return project, table_rows(project), files, msg, _auto_video_cc(project, media, cc_mode)
    except Exception as e:
        raise gr.Error(str(e)) from e


def compact_cues_job(project, rows, media=None, cc_mode='Song ngữ'):
    try:
        project = edited(project, rows)
        before = len(project['segments'])
        project['segments'] = compact_segments(project['segments'], max_chars=42, max_duration=3.5)
        save_project(project)
        files, _ = export_project(project)
        after = len(project['segments'])
        msg = f'Đã gom gọn câu chuẩn TikTok ({before} → {after} câu): tự nối từ lẻ bị cụt và tách câu quá dài.'
        return (
            project,
            table_rows(project),
            files,
            msg,
            _auto_video_cc(project, media, cc_mode),
            preview_seek_update(project),
        )
    except Exception as e:
        raise gr.Error(str(e)) from e


def translate_job(project, rows, glossary, target, style='natural',
                  media=None, cc_mode='Song ngữ', progress=gr.Progress()):
    try:
        project = edited(project, rows)
        if target != project.get('target_language', 'vi'):
            raise ValueError('Bạn vừa chọn ngôn ngữ đích khác. Hãy bấm “Đổi ngôn ngữ đích” trong mục Công cụ hệ thống trước khi dịch.')
        require_model()
    except Exception as e:
        raise gr.Error(str(e)) from e
    status = ''
    try:
        progress(0, desc='Đang kết nối mô hình dịch ngữ cảnh…')
        translate_project(
            project,
            glossary,
            lambda n, total: progress(n / total, desc=f'Đang dịch mạch ngữ cảnh: {n}/{total} câu'),
            style=style,
        )
        status = 'Đã dịch xong toàn bộ phụ đề! Đối chiếu trực tiếp với video bên trái hoặc sang tab 2 để dựng MP4.'
    except Exception as e:
        status = f'Tạm dừng: {e}\nCác câu dịch xong đã được lưu. Bấm Dịch lần nữa để dịch tiếp.'
    finally:
        unload_model()
    files, _ = export_project(project)
    return project, table_rows(project), files, status, _auto_video_cc(project, media, cc_mode)


def reopen(job, media=None):
    try:
        if not job:
            raise ValueError('Hãy chọn một công việc đã lưu trong danh sách.')
        project = load_project(job)
        files, _ = export_project(project)
        video_preview = _auto_video_cc(project, media)
        note = (
            'Đã mở lại công việc và nạp sẵn video đối chiếu.'
            if video_preview is not None else
            'Đã mở lại công việc. Để xem hình video, hãy thả lại file video gốc vào mục Nguồn dữ liệu.'
        )
        return (
            project,
            table_rows(project),
            files,
            note,
            video_preview,
            gr.update(value=project.get('target_language', 'vi')),
            gr.update(value=project.get('subtitle_delay', 0)),
            preview_seek_update(project),
        )
    except Exception as e:
        raise gr.Error(str(e)) from e


def reset_translation(project, rows, media=None):
    try:
        project = edited(project, rows)
        for s in project['segments']:
            s['vi'] = ''
        save_project(project)
        files, _ = export_project(project)
        return (
            project,
            table_rows(project),
            files,
            'Đã xóa bản dịch hiện tại; bấm “Dịch các câu còn trống” để tạo bản dịch mới.',
            _auto_video_cc(project, media, 'Gốc'),
        )
    except Exception as e:
        raise gr.Error(str(e)) from e


def set_target_job(project, rows, target, media=None):
    try:
        project = edited(project, rows)
        if target not in TARGET_LANGUAGES:
            raise ValueError('Ngôn ngữ đích không hợp lệ.')
        if target == project.get('language'):
            raise ValueError('Ngôn ngữ đích đang trùng với ngôn ngữ gốc.')
        if project.get('target_language', 'vi') == target:
            files, _ = export_project(project)
            return project, table_rows(project), files, 'Ngôn ngữ đích này đang được sử dụng.', _auto_video_cc(project, media)
        for row in project['segments']:
            row['vi'] = ''
        project['target_language'] = target
        save_project(project)
        files, _ = export_project(project)
        return (
            project,
            table_rows(project),
            files,
            f'Đã chuyển ngôn ngữ đích sang {TARGET_LANGUAGES[target]} và xóa bản dịch cũ.',
            _auto_video_cc(project, media, 'Gốc'),
        )
    except Exception as e:
        raise gr.Error(str(e)) from e


def preview_job(project, rows, media, mode):
    try:
        project = edited(project, rows)
        video = media or project.get('media')
        if not video or not Path(video).is_file():
            raise ValueError('Hãy tải video gốc ở mục Nguồn dữ liệu để xem kèm phụ đề.')
        if Path(video).suffix.lower() not in BROWSER_VIDEO_EXTS:
            raise ValueError('Trình duyệt hỗ trợ phát trực tiếp MP4/WebM/OGG. Các định dạng khác vẫn xuất MP4 bình thường.')
        files, complete = export_project(project)
        if mode != 'Gốc' and not complete:
            raise ValueError('Còn câu chưa có bản dịch. Hãy dịch đủ hoặc chọn chế độ “Gốc”.')
        preview_mode = 'source' if mode == 'Gốc' else 'vi' if mode == 'Bản dịch' else 'dual'
        chosen = folder(project) / f'preview-{preview_mode}.vtt'
        chosen.write_text(
            webvtt(project['segments'], preview_mode, project.get('subtitle_delay', 0)),
            encoding='utf-8',
        )
        return (
            project,
            table_rows(project),
            files,
            gr.Video(value=str(video), subtitles=str(chosen), interactive=False),
            f'Đang hiển thị phụ đề CC ({mode}) đồng bộ với bảng bên phải.',
        )
    except Exception as e:
        raise gr.Error(str(e)) from e


def apply_style_preset(name):
    style = STYLE_PRESETS[name]
    return (
        style['color'],
        style['secondary'],
        style['background'],
        style['opacity'],
        style['radius'],
        style['backdrop'],
        style['scope'],
        style['outline'],
        style['font'],
        style['bold'],
        style['outline_color'],
        style['shadow'],
        style['box_padding'],
        style['italic'],
    )


def jump_subtitle(project, current, delay_value, direction):
    if not project:
        return gr.update()
    delay = subtitle_delay(delay_value)
    times = [
        max(0, segment['start'] + delay + min(.5, (segment['end'] - segment['start']) / 2))
        for segment in project['segments']
    ]
    now = parse_time(current)
    if direction > 0:
        chosen = next((time for time in times if time > now + .05), times[-1])
    else:
        chosen = next((time for time in reversed(times) if time < now - .05), times[0])
    return round(chosen, 1)


def set_delay_job(project, rows, value):
    try:
        if not project:
            return None, gr.update(), 'Tạo hoặc mở công việc trước khi chỉnh độ trễ.'
        project = edited(project, rows)
        project['subtitle_delay'] = subtitle_delay(value)
        files, _ = export_project(project)
        direction = (
            'muộn hơn' if project['subtitle_delay'] > 0
            else 'sớm hơn' if project['subtitle_delay'] < 0
            else 'đúng mốc gốc'
        )
        return (
            project,
            files,
            f'Đã lưu độ trễ {project["subtitle_delay"]:+.1f}s ({direction}) cho toàn bộ file phụ đề và MP4.',
        )
    except Exception as e:
        raise gr.Error(str(e)) from e


def update_single_cue_job(project, rows, cue_id, new_source, new_vi):
    try:
        if not project:
            raise ValueError('Chưa có công việc đang mở.')
        project = edited(project, rows)
        cid = int(cue_id or 0)
        matched = False
        for seg in project['segments']:
            if seg['id'] == cid:
                if new_source and new_source.strip():
                    seg['source'] = ' '.join(str(new_source).split())
                seg['vi'] = ' '.join(str(new_vi or '').split())
                matched = True
                break
        if not matched:
            raise ValueError('Không tìm thấy câu cần sửa.')
        save_project(project)
        files, _ = export_project(project)
        return project, table_rows(project), files, f'Đã cập nhật trực tiếp câu #{cid}.'
    except Exception as e:
        raise gr.Error(str(e)) from e


POSITION_MAP = {'Dưới': 'bottom', 'Giữa': 'middle', 'Trên': 'top'}


def render_job(project, rows, media, mode, font, size, color, secondary_color,
               backdrop_color, backdrop_opacity, backdrop_radius, backdrop_scope,
               backdrop, position, margin,
               resolution, aspect, frame_fill, text_align, text_width, wrap_mode,
               auto_fit, max_lines, min_size, source_scale, start, delay_value,
               uppercase, bold, outline, max_chars,
               sample, progress=gr.Progress(), frame_only=False,
               outline_color='#171717', shadow=0.0, italic=False,
               letter_spacing=0.0, box_padding=16, dual_gap=6,
               dual_order='translation_first'):
    try:
        if frame_only:
            if not project:
                raise ValueError('Tạo hoặc mở công việc trước khi xem video.')
            project = copy.deepcopy(project)
            updated_segments = rows_to_segments(rows, project['segments'])
            unsaved_edits = updated_segments != project['segments']
            project['segments'] = updated_segments
        else:
            project = edited(project, rows)
        project['subtitle_delay'] = subtitle_delay(delay_value)
        if not any(segment['end'] + project['subtitle_delay'] > 0 for segment in project['segments']):
            raise ValueError('Độ trễ đưa toàn bộ phụ đề ra trước đầu video. Hãy giảm độ dịch sớm.')
        if not frame_only:
            save_project(project)
        video = media or project.get('media')
        chosen = {'Bản dịch': 'vi', 'Song ngữ': 'dual', 'Gốc': 'source'}[mode]
        fallback = (
            frame_only
            and chosen != 'source'
            and not all(segment['vi'].strip() for segment in project['segments'])
        )
        if fallback:
            chosen = 'source'
        if sample or frame_only:
            if start is not None and str(start).strip():
                sample_start = parse_time(start)
            else:
                first = project['segments'][0]
                sample_start = max(0, first['start'] + project['subtitle_delay'])
                if frame_only:
                    sample_start += min(.5, (first['end'] - first['start']) / 2)
        else:
            sample_start = 0
        if backdrop not in ('none', 'lines', 'panel'):
            raise ValueError('Kiểu nền phụ đề không hợp lệ.')
        pos_code = POSITION_MAP.get(position, position if position in ('bottom', 'middle', 'top') else 'bottom')
        result = render_video(
            project, video, chosen, font, size, color,
            backdrop != 'none', pos_code,
            margin, sample, sample_start, resolution, progress,
            secondary_color, uppercase, bold, outline, max_chars,
            aspect=aspect, frame_fill=frame_fill,
            text_align=text_align, text_width=text_width,
            wrap_mode=wrap_mode, auto_fit=auto_fit,
            max_lines=max_lines, min_size=min_size,
            source_scale=source_scale,
            backdrop_style='panel' if backdrop == 'panel' else 'lines',
            backdrop_color=backdrop_color,
            backdrop_opacity=backdrop_opacity,
            backdrop_radius=backdrop_radius,
            backdrop_scope=backdrop_scope,
            outline_color=outline_color,
            shadow=shadow,
            italic=italic,
            letter_spacing=letter_spacing,
            box_padding=box_padding,
            dual_gap=dual_gap,
            dual_order=dual_order,
            frame_only=frame_only,
        )
        if frame_only:
            active = any(
                segment['start'] + project['subtitle_delay'] <= sample_start < segment['end'] + project['subtitle_delay']
                for segment in project['segments']
            )
            status = f'Mốc xem: {timestamp(sample_start)}.'
            if not active:
                status += ' Mốc này chưa có câu phụ đề; bấm Câu sau ▶ để tới câu gần nhất.'
            if fallback:
                status += ' Chưa dịch đủ các câu nên đang hiển thị tạm chữ gốc.'
            if unsaved_edits:
                status += ' Bảng có chỉnh sửa chưa lưu.'
            return project, table_rows(project), result, gr.update(), status

        # Both 30s preview AND full MP4 render load straight into the inline video player first!
        msg = (
            'Đã dựng xong đoạn xem thử 30 giây — video đang phát trực tiếp ngay bên phải.'
            if sample else
            'Đã dựng xong toàn bộ video MP4! Bạn có thể xem trọn video ngay bên phải rồi bấm “Tải MP4 về máy”.'
        )
        return (
            project,
            table_rows(project),
            result,
            gr.Video(value=result, autoplay=True, interactive=False),
            msg,
        )
    except Exception as e:
        raise gr.Error(str(e)) from e


def preview_frame_job(*args):
    """Render a single live frame from all 38 render_inputs."""
    if not args[0]:
        return None, 'Hãy tạo hoặc mở một công việc ở tab 1 để đối chiếu trực tiếp trên khung hình.'
    base = args[:31]
    extra = args[31:]
    outline_color, shadow, italic, letter_spacing, box_padding, dual_gap, dual_order = (
        extra if len(extra) == 7 else ('#171717', 0.0, False, 0.0, 16, 6, 'translation_first')
    )
    try:
        result = render_job(
            *base, True, frame_only=True,
            outline_color=outline_color, shadow=shadow, italic=italic,
            letter_spacing=letter_spacing, box_padding=box_padding,
            dual_gap=dual_gap, dual_order=dual_order,
        )
        return result[2], result[4]
    except Exception as e:
        return None, f'Chưa dựng được khung xem trước: {e}'


def apply_preset_and_preview(preset_name, *current_render_inputs):
    """Atomically apply a color/style preset and render the updated preview frame in one step."""
    vals = apply_style_preset(preset_name)
    (
        c_color, c_sec, c_bg, c_op, c_rad, c_backdrop, c_scope,
        c_outline, c_font, c_bold, c_out_col, c_shadow, c_pad, c_italic,
    ) = vals
    updated = list(current_render_inputs)
    if len(updated) >= 38:
        updated[4] = c_font
        updated[6] = c_color
        updated[7] = c_sec
        updated[8] = c_bg
        updated[9] = c_op
        updated[10] = c_rad
        updated[11] = c_scope
        updated[12] = c_backdrop
        updated[28] = c_bold
        updated[29] = c_outline
        updated[31] = c_out_col
        updated[32] = c_shadow
        updated[33] = c_italic
        updated[35] = c_pad
    frame, msg = preview_frame_job(*updated)
    return (*vals, frame, msg)


def apply_layout_preset_and_preview(preset_key, *current_render_inputs):
    cfg = LAYOUT_PRESETS[preset_key]
    updated = list(current_render_inputs)
    if len(updated) >= 38:
        updated[5] = cfg['size']
        updated[13] = cfg['position']
        updated[14] = cfg['margin']
        updated[16] = cfg['aspect']
        updated[17] = cfg['frame_fill']
        updated[19] = cfg['text_width']
        updated[20] = cfg['wrap_mode']
        updated[22] = cfg['max_lines']
        updated[23] = cfg['min_size']
        updated[30] = cfg['max_chars']
        updated[35] = cfg['box_padding']
        updated[36] = cfg['dual_gap']
    frame, msg = preview_frame_job(*updated)
    return (
        cfg['aspect'], cfg['frame_fill'], cfg['position'], cfg['margin'],
        cfg['text_width'], cfg['max_chars'], cfg['max_lines'], cfg['size'],
        cfg['min_size'], cfg['box_padding'], cfg['dual_gap'], cfg['wrap_mode'],
        frame, msg,
    )


CSS = '''
:root {
  --lbs-bg: #f5f6f8;
  --lbs-surface: #ffffff;
  --lbs-surface-alt: #f8f9fb;
  --lbs-border: #e5e7eb;
  --lbs-border-strong: #d1d5db;
  --lbs-text: #111827;
  --lbs-muted: #6b7280;
  --lbs-accent: #cf3c74;
  --lbs-accent-hover: #b53063;
  --lbs-accent-soft: rgba(207, 60, 116, 0.08);
}

.dark, body.dark {
  --lbs-bg: #0f1115;
  --lbs-surface: #161920;
  --lbs-surface-alt: #1d212a;
  --lbs-border: #272c38;
  --lbs-border-strong: #363d4d;
  --lbs-text: #f3f4f6;
  --lbs-muted: #9ca3af;
  --lbs-accent: #de4d83;
  --lbs-accent-hover: #c93b71;
  --lbs-accent-soft: rgba(222, 77, 131, 0.14);
}

body, .gradio-container {
  background: var(--lbs-bg) !important;
  color: var(--lbs-text) !important;
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}

.gradio-container {
  max-width: 1540px !important;
  margin: 0 auto !important;
  padding: 10px 20px 28px !important;
}

/* Guarantee columns never wrap sideways into horizontal scrollbars */
.gradio-container .column,
.lbs-card,
#style-inspector-card,
#preview-sticky-col,
#video-review-panel,
#subtitle-table-panel {
  flex-wrap: nowrap !important;
  min-width: 0 !important;
  overflow-x: hidden !important;
}

/* Minimalist Brand Header */
#topbar-row {
  align-items: center !important;
  border-bottom: 1px solid var(--lbs-border);
  padding: 6px 2px 12px;
  margin-bottom: 12px;
}

.lbs-brand-wrap {
  display: flex;
  align-items: center;
  gap: 12px;
}
.lbs-logo-img {
  height: 34px;
  width: auto;
  object-fit: contain;
}
.lbs-brand-text h1 {
  margin: 0 !important;
  font-size: 20px !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em;
  color: var(--lbs-text) !important;
  line-height: 1.2;
}
.lbs-brand-sub {
  margin: 2px 0 0 !important;
  font-size: 12.5px !important;
  color: var(--lbs-muted) !important;
}

/* Clean Matte Studio Cards (no glow, no tinted borders) */
.lbs-card {
  background: var(--lbs-surface) !important;
  border: 1px solid var(--lbs-border) !important;
  border-radius: 10px !important;
  padding: 14px 16px !important;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
}

.lbs-toolbar {
  background: var(--lbs-surface) !important;
  border: 1px solid var(--lbs-border) !important;
  border-radius: 10px !important;
  padding: 10px 14px !important;
  margin-bottom: 12px !important;
  align-items: center !important;
}

/* Neutral Form Blocks & Inputs (remove Soft pink wash) */
.gradio-container .block,
.gradio-container .form {
  background: transparent !important;
  border-color: var(--lbs-border) !important;
  box-shadow: none !important;
}

.gradio-container input:not([type="radio"]):not([type="checkbox"]):not([type="color"]):not([type="range"]),
.gradio-container textarea,
.gradio-container .wrap-inner {
  background: var(--lbs-surface) !important;
  border-color: var(--lbs-border-strong) !important;
  color: var(--lbs-text) !important;
  box-shadow: none !important;
}

/* Clean crisp labels */
.gradio-container label > span,
.gradio-container .block > label > span,
.gradio-container span[data-testid="block-info"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  box-shadow: none !important;
  color: var(--lbs-text) !important;
  font-weight: 600 !important;
  font-size: 13px !important;
}

.gradio-container .prose,
.gradio-container p {
  color: var(--lbs-text) !important;
}
.gradio-container .prose h3,
.gradio-container .prose h4 {
  color: var(--lbs-text) !important;
  margin: 0 0 6px 0 !important;
  font-size: 14.5px !important;
  font-weight: 600 !important;
}
.muted-note p {
  color: var(--lbs-muted) !important;
  font-size: 12.5px !important;
  margin: 2px 0 0 !important;
}

/* Flat Matte Rose Primary Button & Pure Neutral Secondary Button */
.gradio-container button.primary {
  background: var(--lbs-accent) !important;
  border: 1px solid var(--lbs-accent) !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
  box-shadow: none !important;
  min-height: 40px !important;
  transition: background-color 0.15s ease, border-color 0.15s ease !important;
}
.gradio-container button.primary:hover {
  background: var(--lbs-accent-hover) !important;
  border-color: var(--lbs-accent-hover) !important;
  box-shadow: none !important;
}

.gradio-container button.secondary {
  background: var(--lbs-surface-alt) !important;
  border: 1px solid var(--lbs-border-strong) !important;
  color: var(--lbs-text) !important;
  font-weight: 500 !important;
  border-radius: 8px !important;
  box-shadow: none !important;
  min-height: 40px !important;
  transition: border-color 0.15s ease, background-color 0.15s ease !important;
}
.gradio-container button.secondary:hover {
  border-color: var(--lbs-muted) !important;
  background: var(--lbs-surface) !important;
}

/* Clean Tabs */
.gradio-container button[role="tab"] {
  font-size: 13.5px !important;
  padding: 8px 12px !important;
  color: var(--lbs-muted) !important;
}
.gradio-container button[role="tab"][aria-selected="true"] {
  color: var(--lbs-accent) !important;
  border-bottom-color: var(--lbs-accent) !important;
  font-weight: 600 !important;
}

/* Clean Radio Pills (neutral unselected, subtle rose border when selected) */
.gradio-container .wrap > label {
  background: var(--lbs-surface-alt) !important;
  border: 1px solid var(--lbs-border) !important;
  color: var(--lbs-text) !important;
  border-radius: 8px !important;
  box-shadow: none !important;
  transition: border-color 0.15s ease, background-color 0.15s ease !important;
}
.gradio-container .wrap > label.selected {
  border-color: var(--lbs-accent) !important;
  background: var(--lbs-accent-soft) !important;
  color: var(--lbs-text) !important;
}

/* Side-by-side Alignment */
#editor-workspace-row {
  align-items: stretch !important;
  gap: 16px !important;
}
.panel-header-row {
  align-items: center !important;
  justify-content: space-between !important;
  min-height: 38px;
  margin-bottom: 8px !important;
}
#cc-mode-radio .wrap {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  gap: 6px !important;
}
#cc-mode-radio label {
  padding: 5px 10px !important;
  white-space: nowrap !important;
}

#export-studio-row {
  align-items: flex-start !important;
  gap: 16px !important;
}
@media (min-width: 1024px) {
  #preview-sticky-col {
    position: sticky !important;
    top: 10px !important;
  }
}

#export-top-bar {
  align-items: center !important;
  gap: 8px !important;
  margin-bottom: 8px !important;
}
#export-top-bar button,
#export-top-bar a button {
  white-space: nowrap !important;
}

#live-preview {
  background: #0c0d10 !important;
  border: 1px solid var(--lbs-border) !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}
#live-preview img {
  object-fit: contain !important;
}

/* Scrubber bar */
#scrubber-row {
  align-items: center !important;
  gap: 10px !important;
  margin-top: 8px !important;
}

/* Color Pickers */
.color-grid input[type="color"] {
  width: 44px !important;
  height: 34px !important;
  border-radius: 6px !important;
  cursor: pointer !important;
}

/* Preset Radio Grid */
#preset-chips .wrap {
  display: grid !important;
  grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
  gap: 8px !important;
}
#preset-chips label {
  padding: 8px 11px !important;
  margin: 0 !important;
  cursor: pointer !important;
}
'''

STUDIO_THEME = gr.themes.Default(
    primary_hue='pink',
    secondary_hue='gray',
    neutral_hue='gray',
    radius_size=gr.themes.sizes.radius_md,
    font=['Segoe UI', 'Inter', 'system-ui', 'sans-serif'],
)

TOGGLE_DARK_JS = '''
() => {
  const isDark = document.body.classList.toggle('dark');
  document.documentElement.classList.toggle('dark', isDark);
  try { localStorage.setItem('localbachsub_dark_mode', isDark ? '1' : '0'); } catch (e) {}
}
'''

INIT_THEME_JS = '''
() => {
  try {
    const saved = localStorage.getItem('localbachsub_dark_mode');
    if (saved === '1') {
      document.body.classList.add('dark');
      document.documentElement.classList.add('dark');
    } else if (saved === '0') {
      document.body.classList.remove('dark');
      document.documentElement.classList.remove('dark');
    }
  } catch (e) {}
}
'''


def _brand_header_html():
    uri = _logo_data_uri()
    logo_tag = f'<img src="{uri}" class="lbs-logo-img" alt="LocalBachSub" />' if uri else ''
    return f'''
    <div class="lbs-brand-wrap">
      {logo_tag}
      <div class="lbs-brand-text">
        <h1>LocalBachSub</h1>
        <p class="lbs-brand-sub">Nhận diện giọng nói · Dịch ngữ cảnh · Biên tập đối chiếu &amp; Xuất MP4</p>
      </div>
    </div>
    '''


def build_app():
    with gr.Blocks(title='LocalBachSub') as demo:
        state = gr.State(None)

        # Top Minimal Header
        with gr.Row(elem_id='topbar-row'):
            with gr.Column(scale=8):
                gr.HTML(_brand_header_html())
            with gr.Column(scale=2, min_width=160):
                theme_btn = gr.Button('Sáng / Tối', variant='secondary', size='sm')

        with gr.Tabs():
            # =========================================================================
            # TAB 1: SIDE-BY-SIDE VIDEO & SUBTITLE EDITOR + CONTEXTUAL TRANSLATION
            # =========================================================================
            with gr.Tab('1 · Dịch & Biên tập đối chiếu'):
                with gr.Accordion(
                    'Nguồn video / SRT & Nhận diện giọng nói (tự thu gọn sau khi nạp)',
                    open=True,
                ) as setup_drawer:
                    with gr.Row():
                        with gr.Column(scale=7, elem_classes=['lbs-card']):
                            with gr.Row():
                                media = gr.File(
                                    label='Video hoặc âm thanh gốc',
                                    file_types=['.mp4', '.mkv', '.mov', '.webm', '.avi', '.mp3', '.wav', '.m4a', '.ogg'],
                                    type='filepath',
                                    height=105,
                                )
                                source_srt = gr.File(
                                    label='File SRT có sẵn (Gốc hoặc Song ngữ)',
                                    file_types=['.srt'],
                                    type='filepath',
                                    height=105,
                                )
                            with gr.Row():
                                language = gr.Dropdown(
                                    [(TARGET_LANGUAGES[code], code) for code in ('en', 'vi', 'zh', 'ja', 'ko', 'fr', 'es')]
                                    + [('Tự nhận diện', 'auto')],
                                    value='en',
                                    label='Ngôn ngữ gốc',
                                )
                                target = gr.Dropdown(
                                    [(label, code) for code, label in TARGET_LANGUAGES.items()],
                                    value='vi',
                                    label='Ngôn ngữ đích',
                                )
                                srt_mode = gr.Dropdown(
                                    [(label, code) for code, label in SRT_IMPORT_MODES.items()],
                                    value='auto',
                                    label='Cách đọc SRT',
                                )
                            with gr.Row():
                                model = gr.Dropdown(
                                    [('Nhanh (small)', 'small'), ('Cân bằng (medium)', 'medium'), ('Chính xác cao (large-v3)', 'large-v3')],
                                    value='medium',
                                    label='Mô hình Whisper',
                                )
                                device = gr.Dropdown(
                                    [('NVIDIA GPU (CUDA)', 'cuda'), ('CPU', 'cpu')],
                                    value='cuda',
                                    label='Thiết bị chạy',
                                )
                                segment_style = gr.Dropdown(
                                    [(label, code) for code, label in SEGMENT_STYLES.items()],
                                    value='tiktok',
                                    label='Độ gọn câu khi tách',
                                )
                            create = gr.Button('Tạo / Nhập phụ đề', variant='primary')

                        with gr.Column(scale=4, elem_classes=['lbs-card']):
                            gr.Markdown('### Công việc đã lưu & Hệ thống')
                            history = gr.Dropdown(choices=list_projects(), label='Mở lại công việc cũ (thư mục output)')
                            with gr.Row():
                                restore = gr.Button('Mở công việc', size='sm', variant='secondary')
                                refresh = gr.Button('Làm mới', size='sm', variant='secondary')
                            with gr.Row():
                                change_target = gr.Button('Đổi ngôn ngữ đích', size='sm', variant='secondary')
                                reset = gr.Button('Xóa bản dịch', size='sm', variant='secondary')
                                check = gr.Button('Kiểm tra máy', size='sm', variant='secondary')
                            health = gr.Textbox(label='Trạng thái GPU & Ollama', lines=3, interactive=False)

                # Compact Contextual AI Translation Bar
                with gr.Row(elem_classes=['lbs-toolbar']):
                    translation_style = gr.Dropdown(
                        [(label, code) for code, label in TRANSLATION_STYLES.items()],
                        value='natural',
                        label='Văn phong dịch AI',
                        scale=3,
                    )
                    glossary = gr.Textbox(
                        label='Tên riêng & thuật ngữ giữ nguyên (tùy chọn)',
                        lines=1,
                        placeholder='VD: Giữ nguyên Joe Dispenza, Alpha, Theta; subconscious mind = tiềm thức...',
                        scale=5,
                    )
                    translate = gr.Button('Dịch các câu còn trống', variant='primary', scale=2)

                # TRUE HORIZONTAL SIDE-BY-SIDE VIDEO + SUBTITLE TABLE
                with gr.Row(elem_id='editor-workspace-row'):
                    # LEFT: Video Player with Live CC
                    with gr.Column(scale=5, elem_id='video-review-panel', elem_classes=['lbs-card']):
                        with gr.Row(elem_classes=['panel-header-row']):
                            gr.Markdown('### Video đối chiếu')
                            mode = gr.Radio(
                                ['Song ngữ', 'Bản dịch', 'Gốc'],
                                value='Song ngữ',
                                show_label=False,
                                container=False,
                                elem_id='cc-mode-radio',
                            )
                            show = gr.Button('Cập nhật CC', size='sm', variant='secondary')
                        preview = gr.Video(
                            label='Phát video đối chiếu phụ đề',
                            show_label=False,
                            height=440,
                            interactive=False,
                        )
                        status = gr.Textbox(
                            label='Trạng thái',
                            lines=2,
                            interactive=False,
                            value='Tải video hoặc file SRT ở trên để bắt đầu đối chiếu ngang hàng.',
                        )

                    # RIGHT: Editable Subtitle Table at Exact Same Eye Level
                    with gr.Column(scale=7, elem_id='subtitle-table-panel', elem_classes=['lbs-card']):
                        with gr.Row(elem_classes=['panel-header-row']):
                            gr.Markdown('### Bảng chỉnh sửa phụ đề (nhấp đúp ô để sửa)')
                            compact_btn = gr.Button('Gom gọn câu TikTok', size='sm', variant='secondary')
                            save = gr.Button('Lưu & Xuất SRT / ZIP', size='sm', variant='primary')
                        table = gr.Dataframe(
                            headers=['#', 'Bắt đầu', 'Kết thúc', 'Câu gốc', 'Bản dịch'],
                            datatype=['number', 'str', 'str', 'str', 'str'],
                            type='array',
                            interactive=True,
                            column_count=(5, 'fixed'),
                            column_widths=['44px', '104px', '104px', '36%', '38%'],
                            max_height=440,
                            wrap=True,
                        )
                        with gr.Accordion('Tải file phụ đề rời (ZIP · SRT · VTT · ASS)', open=False):
                            downloads = gr.File(
                                label='Danh sách file phụ đề',
                                file_count='multiple',
                                interactive=False,
                            )

            # =========================================================================
            # TAB 2: SUBTITLE DESIGNER + DIRECT VIDEO PREVIEW (30S & FULL) BEFORE DOWNLOAD
            # =========================================================================
            with gr.Tab('2 · Thiết kế chữ & Xem / Xuất Video'):
                with gr.Row(elem_id='export-studio-row'):
                    # LEFT: Clean 5 Sub-tabs on ONE line
                    with gr.Column(scale=6, elem_id='style-inspector-card', elem_classes=['lbs-card']):
                        with gr.Tabs(overflow_behavior='wrap'):
                            # Sub-tab 1: Quick Presets & Aspect Ratio
                            with gr.Tab('Mẫu & Khung'):
                                palette = gr.Radio(
                                    list(STYLE_PRESETS.keys()),
                                    value='TikTok Gọn · Nền đen mờ',
                                    label='Bộ màu chọn nhanh',
                                    elem_id='preset-chips',
                                )
                                gr.Markdown('**Bố cục theo nền tảng**')
                                with gr.Row():
                                    preset_tiktok_crop = gr.Button('TikTok 9:16 · Phủ kín', size='sm', variant='secondary')
                                    preset_tiktok_fit = gr.Button('TikTok 9:16 · Giữ khung', size='sm', variant='secondary')
                                with gr.Row():
                                    preset_yt = gr.Button('YouTube 16:9', size='sm', variant='secondary')
                                    preset_orig = gr.Button('Giữ khung gốc', size='sm', variant='secondary')

                                render_mode = gr.Radio(
                                    ['Song ngữ', 'Bản dịch', 'Gốc'],
                                    value='Song ngữ',
                                    label='Nội dung chữ hiển thị trên video',
                                )
                                with gr.Row():
                                    aspect = gr.Dropdown(
                                        [
                                            ('Giữ khung gốc của video', 'original'),
                                            ('Dọc 9:16 (TikTok / Reels / Shorts)', '9:16'),
                                            ('Dọc 4:5 (Facebook / IG)', '4:5'),
                                            ('Vuông 1:1', '1:1'),
                                            ('Ngang 16:9 (YouTube)', '16:9'),
                                            ('Ngang 4:3', '4:3'),
                                        ],
                                        value='original',
                                        label='Tỷ lệ khung hình xuất',
                                    )
                                    frame_fill = gr.Dropdown(
                                        [
                                            ('Giữ trọn hình · Có viền đen', 'fit'),
                                            ('Phủ kín khung · Cắt rìa', 'crop'),
                                        ],
                                        value='fit',
                                        label='Cách đưa hình vào khung',
                                    )
                                resolution = gr.Dropdown(
                                    [('Độ phân giải gốc (Sắc nét nhất)', 'Gốc'), ('720p HD (Dựng nhanh, nhẹ máy)', '720p')],
                                    value='Gốc',
                                    label='Chất lượng video MP4',
                                )

                            # Sub-tab 2: Font & Compact TikTok Grouping
                            with gr.Tab('Chữ & Gọn'):
                                with gr.Row():
                                    font = gr.Dropdown(
                                        list(FONT_FILES),
                                        value='Segoe UI',
                                        label='Phông chữ',
                                        filterable=True,
                                    )
                                    size = gr.Slider(18, 88, value=44, step=1, label='Cỡ chữ chính')
                                dual_order = gr.Radio(
                                    [
                                        ('Bản dịch ở trên · Gốc ở dưới', 'translation_first'),
                                        ('Bản gốc ở trên · Dịch ở dưới', 'source_first'),
                                    ],
                                    value='translation_first',
                                    label='Thứ tự dòng khi song ngữ',
                                )
                                with gr.Row():
                                    source_scale = gr.Slider(45, 95, value=72, step=1, label='Tỷ lệ cỡ dòng gốc (%)')
                                    dual_gap = gr.Slider(0, 24, value=4, step=2, label='Khoảng cách 2 ngôn ngữ (px)')

                                gr.Markdown('**Gom gọn khối chữ (tối ưu đọc lướt TikTok)**')
                                with gr.Row():
                                    text_width = gr.Slider(40, 96, value=82, step=1, label='Bề rộng khối chữ (%)')
                                    max_chars = gr.Slider(14, 85, value=34, step=1, label='Ký tự tối đa mỗi dòng')
                                with gr.Row():
                                    max_lines = gr.Slider(1, 5, value=2, step=1, label='Số dòng tối đa mỗi ngôn ngữ')
                                    min_size = gr.Slider(12, 42, value=20, step=1, label='Cỡ chữ nhỏ nhất khi tự co')
                                with gr.Row():
                                    wrap_mode = gr.Radio(
                                        [('Cân đều độ dài dòng', 'balanced'), ('Đổ đầy dòng trên', 'compact')],
                                        value='balanced',
                                        label='Cách ngắt dòng',
                                    )
                                    letter_spacing = gr.Slider(-2.0, 5.0, value=0.0, step=0.5, label='Khoảng cách chữ cái (px)')
                                with gr.Row():
                                    bold = gr.Checkbox(label='Chữ đậm', value=True)
                                    italic = gr.Checkbox(label='Chữ nghiêng', value=False)
                                    uppercase = gr.Checkbox(label='IN HOA bản dịch', value=False)
                                    auto_fit = gr.Checkbox(label='Tự thu nhỏ câu dài', value=True)

                            # Sub-tab 3: Colors, Borders & Box
                            with gr.Tab('Màu & Nền'):
                                with gr.Row(elem_classes=['color-grid']):
                                    color = gr.ColorPicker(value='#FFFFFF', label='Màu chữ dịch')
                                    secondary_color = gr.ColorPicker(value='#FDE68A', label='Màu chữ gốc')
                                    outline_color = gr.ColorPicker(value='#09090B', label='Màu viền & bóng')
                                    backdrop_color = gr.ColorPicker(value='#120D1D', label='Màu hộp nền')
                                with gr.Row():
                                    backdrop = gr.Dropdown(
                                        [
                                            ('Hộp bo góc liền khối', 'panel'),
                                            ('Nền ôm rời từng dòng', 'lines'),
                                            ('Không hộp nền (chỉ viền & bóng)', 'none'),
                                        ],
                                        value='panel',
                                        label='Kiểu hộp nền',
                                    )
                                    backdrop_scope = gr.Radio(
                                        [('Ôm cả hai dòng', 'all'), ('Chỉ ôm bản dịch', 'translation')],
                                        value='all',
                                        label='Phạm vi hộp nền',
                                    )
                                with gr.Row():
                                    backdrop_opacity = gr.Slider(0, 100, value=80, step=2, label='Độ đậm nền (%)')
                                    backdrop_radius = gr.Slider(0, 40, value=14, step=2, label='Bo góc hộp nền')
                                    box_padding = gr.Slider(6, 32, value=12, step=1, label='Độ đệm hộp nền (ôm gọn)')
                                with gr.Row():
                                    outline = gr.Slider(0, 7, value=1.5, step=0.5, label='Độ dày viền chữ')
                                    shadow = gr.Slider(0, 6, value=0.0, step=0.5, label='Độ đổ bóng chữ')

                            # Sub-tab 4: Position & Sync
                            with gr.Tab('Vị trí'):
                                with gr.Row():
                                    position = gr.Radio(
                                        ['Dưới', 'Giữa', 'Trên'],
                                        value='Dưới',
                                        label='Vùng đặt phụ đề',
                                    )
                                    text_align = gr.Radio(
                                        [('Trái', 'left'), ('Giữa', 'center'), ('Phải', 'right')],
                                        value='center',
                                        label='Căn lề ngang',
                                    )
                                margin = gr.Slider(
                                    10, 480, value=80, step=5,
                                    label='Khoảng cách tới mép dọc (px trên khung 1080p)',
                                )
                                gr.Markdown(
                                    'Mẹo đăng TikTok / Reels: chọn **215** để phụ đề nằm gọn trên vùng tên kênh & mô tả.',
                                    elem_classes=['muted-note'],
                                )
                                with gr.Row():
                                    margin_low = gr.Button('Sát đáy (65)', size='sm', variant='secondary')
                                    margin_tiktok = gr.Button('Vùng an toàn TikTok (215)', size='sm', variant='secondary')
                                    margin_high = gr.Button('Cao thoáng (310)', size='sm', variant='secondary')
                                delay = gr.Slider(
                                    -10, 10, value=0, step=0.1,
                                    label='Độ trễ phụ đề (giây · Dương = trễ hơn, Âm = sớm hơn)',
                                )

                            # Sub-tab 5: Side-by-Side Direct Subtitle Editing
                            with gr.Tab('Sửa chữ'):
                                gr.Markdown(
                                    'Sửa trực tiếp nội dung từng câu ngay cạnh khung hình bên phải, rồi bấm **Áp dụng & Lưu phụ đề**.',
                                    elem_classes=['muted-note'],
                                )
                                apply_table_design = gr.Button('Áp dụng & Lưu phụ đề', variant='primary', size='sm')
                                table_design = gr.Dataframe(
                                    headers=['#', 'Bắt đầu', 'Kết thúc', 'Câu gốc', 'Bản dịch'],
                                    datatype=['number', 'str', 'str', 'str', 'str'],
                                    type='array',
                                    interactive=True,
                                    column_count=(5, 'fixed'),
                                    column_widths=['44px', '100px', '100px', '36%', '38%'],
                                    max_height=430,
                                    wrap=True,
                                )

                    # RIGHT: Sticky Preview Stage + Uniform Single-Row Action Bar
                    with gr.Column(scale=6, elem_id='preview-sticky-col', elem_classes=['lbs-card']):
                        with gr.Row(elem_id='export-top-bar'):
                            sample_button = gr.Button('Xem thử 30s', variant='secondary', scale=2)
                            render_button = gr.Button('Xuất & Xem toàn bộ MP4', variant='primary', scale=3)
                            download_mp4 = gr.DownloadButton(
                                'Tải MP4 về máy',
                                variant='secondary',
                                visible=False,
                                scale=2,
                            )

                        with gr.Tabs(selected='frame_tab', overflow_behavior='wrap') as preview_tabs:
                            with gr.Tab('Ảnh khung hình (tự cập nhật)', id='frame_tab'):
                                live_preview = gr.Image(
                                    label='Khung đối chiếu tức thì',
                                    show_label=False,
                                    height=470,
                                    format='png',
                                    interactive=False,
                                    buttons=['fullscreen'],
                                    elem_id='live-preview',
                                )
                            with gr.Tab('Phát video đã dựng', id='video_tab'):
                                rendered_preview = gr.Video(
                                    label='Xem trực tiếp video đã gắn phụ đề',
                                    show_label=False,
                                    height=470,
                                    interactive=False,
                                )

                        live_status = gr.Markdown(
                            'Chỉnh tùy chọn bên trái để xem khung hình ngay, hoặc bấm **Xem thử 30s** / **Xuất & Xem toàn bộ MP4** để phát thử trước khi tải.',
                            elem_classes=['muted-note'],
                        )

                        # Wide horizontal scrubber row (labels never wrap vertically!)
                        with gr.Row(elem_id='scrubber-row'):
                            previous_cue = gr.Button('◀ Câu trước', size='sm', variant='secondary', scale=1, min_width=105)
                            start = gr.Slider(0, 30, value=0, step=0.1, label='Mốc đang xem (giây)', scale=4, min_width=220)
                            next_cue = gr.Button('Câu sau ▶', size='sm', variant='secondary', scale=1, min_width=105)

                        with gr.Accordion('Sửa nhanh câu tại mốc đang xem', open=False):
                            with gr.Row():
                                quick_id = gr.Number(label='Câu #', precision=0, interactive=False, scale=1, min_width=70)
                                quick_source = gr.Textbox(label='Câu gốc', lines=1, scale=4)
                            with gr.Row():
                                quick_vi = gr.Textbox(label='Bản dịch', lines=1, scale=4)
                                quick_save = gr.Button('Cập nhật câu', variant='primary', size='sm', scale=1)

                        # Hidden file holder kept for programmatic compatibility if needed
                        rendered = gr.File(label='File MP4', interactive=False, visible=False)

        # =====================================================================
        # EVENT WIRING
        # =====================================================================
        demo.load(None, None, None, js=INIT_THEME_JS)
        theme_btn.click(None, None, None, js=TOGGLE_DARK_JS)
        check.click(check_system, outputs=health)

        common = dict(concurrency_id='local_gpu', concurrency_limit=1)

        def _create_and_focus(*args, progress=gr.Progress()):
            res = source_job(*args, progress=progress)
            # Collapse setup_drawer so Side-by-Side Video + Subtitle Table jump straight to eye level!
            return (*res, res[1], gr.update(open=False))

        def _reopen_and_focus(job, media_val):
            res = reopen(job, media_val)
            return (*res, res[1], gr.update(open=False))

        create.click(
            _create_and_focus,
            [media, source_srt, model, device, language, target, srt_mode, segment_style],
            [state, table, downloads, status, history, preview, target, delay, start, table_design, setup_drawer],
            **common,
        )
        restore.click(
            _reopen_and_focus,
            [history, media],
            [state, table, downloads, status, preview, target, delay, start, table_design, setup_drawer],
            **common,
        )

        def _sync_translate(*args, progress=gr.Progress()):
            res = translate_job(*args, progress=progress)
            return (*res, res[1])

        def _sync_save(proj, rows_data, media_val, cc_val):
            res = save_and_export(proj, rows_data, media_val, cc_val)
            return (*res, res[1])

        def _sync_compact(proj, rows_data, media_val, cc_val):
            res = compact_cues_job(proj, rows_data, media_val, cc_val)
            return (*res, res[1])

        translate.click(
            _sync_translate,
            [state, table, glossary, target, translation_style, media, mode],
            [state, table, downloads, status, preview, table_design],
            **common,
        )
        save.click(
            _sync_save,
            [state, table, media, mode],
            [state, table, downloads, status, preview, table_design],
            **common,
        )
        apply_table_design.click(
            _sync_save,
            [state, table_design, media, mode],
            [state, table, downloads, status, preview, table_design],
            **common,
        )
        compact_btn.click(
            _sync_compact,
            [state, table, media, mode],
            [state, table, downloads, status, preview, start, table_design],
            **common,
        )
        reset.click(
            lambda p, r, m: (*reset_translation(p, r, m), table_rows(p) if p else []),
            [state, table, media],
            [state, table, downloads, status, preview, table_design],
            **common,
        ).then(
            lambda p: table_rows(p) if p else [],
            inputs=state,
            outputs=table_design,
            queue=False,
        )
        change_target.click(
            set_target_job,
            [state, table, target, media],
            [state, table, downloads, status, preview],
            **common,
        ).then(
            lambda p: table_rows(p) if p else [],
            inputs=state,
            outputs=table_design,
            queue=False,
        )
        show.click(
            preview_job,
            [state, table, media, mode],
            [state, table, downloads, preview, status],
            **common,
        )
        refresh.click(lambda: gr.update(choices=list_projects()), outputs=history)

        delay.input(
            set_delay_job,
            [state, table, delay],
            [state, downloads, status],
            trigger_mode='always_last',
            **common,
        )

        previous_cue.click(
            lambda project, moment, offset: jump_subtitle(project, moment, offset, -1),
            [state, start, delay],
            start,
            queue=False,
        )
        next_cue.click(
            lambda project, moment, offset: jump_subtitle(project, moment, offset, 1),
            [state, start, delay],
            start,
            queue=False,
        )

        # Full 38-item render_inputs
        render_inputs = [
            state, table, media, render_mode, font, size, color,
            secondary_color, backdrop_color, backdrop_opacity,
            backdrop_radius, backdrop_scope, backdrop,
            position, margin, resolution, aspect, frame_fill,
            text_align, text_width, wrap_mode, auto_fit, max_lines,
            min_size, source_scale, start, delay,
            uppercase, bold, outline, max_chars,
            outline_color, shadow, italic, letter_spacing,
            box_padding, dual_gap, dual_order,
        ]

        def _preview_and_show_frame(*all_args):
            frame, msg = preview_frame_job(*all_args)
            return frame, msg, gr.update(selected='frame_tab')

        # Quick margin presets
        for m_btn, m_val in ((margin_low, 65), (margin_tiktok, 215), (margin_high, 310)):
            m_btn.click(lambda v=m_val: v, outputs=margin, queue=False).then(
                _preview_and_show_frame,
                inputs=render_inputs,
                outputs=[live_preview, live_status, preview_tabs],
                show_progress='hidden',
                concurrency_id='live_preview',
                concurrency_limit=1,
            )

        # Sync active cue quick-editor
        gr.on(
            triggers=[state.change, start.change, table.change],
            fn=active_cue_fields,
            inputs=[state, start, delay],
            outputs=[quick_id, quick_source, quick_vi],
            queue=False,
            show_progress='hidden',
        )
        quick_save.click(
            update_single_cue_job,
            [state, table, quick_id, quick_source, quick_vi],
            [state, table, downloads, status],
            **common,
        ).then(
            lambda p: table_rows(p) if p else [],
            inputs=state,
            outputs=table_design,
            queue=False,
        )

        # Export & Inline Video Preview (both 30s sample and full MP4 play right inside `video_tab` first!)
        def _run_export(*all_args, sample_flag=False, progress=gr.Progress()):
            base = all_args[:31]
            outline_c, shad, ital, l_space, b_pad, d_gap, d_ord = all_args[31:]
            proj, rows_out, mp4_path, video_comp, msg = render_job(
                *base, sample_flag, progress,
                outline_color=outline_c, shadow=shad, italic=ital,
                letter_spacing=l_space, box_padding=b_pad,
                dual_gap=d_gap, dual_order=d_ord,
            )
            dl_label = 'Tải bản thử 30s' if sample_flag else 'Tải về MP4'
            return (
                proj,
                rows_out,
                rows_out,
                mp4_path,
                video_comp,
                msg,
                msg,
                gr.update(selected='video_tab'),
                gr.update(value=mp4_path, label=dl_label, visible=True),
            )

        export_outputs = [
            state, table, table_design, rendered, rendered_preview,
            status, live_status, preview_tabs, download_mp4,
        ]
        sample_button.click(
            lambda *args, progress=gr.Progress(): _run_export(*args, sample_flag=True, progress=progress),
            render_inputs,
            export_outputs,
            **common,
        )
        render_button.click(
            lambda *args, progress=gr.Progress(): _run_export(*args, sample_flag=False, progress=progress),
            render_inputs,
            export_outputs,
            **common,
        )

        # Atomic Style Preset handler (switches back to frame_tab to display instant style change)
        palette.input(
            lambda name, *r_in: (*apply_preset_and_preview(name, *r_in), gr.update(selected='frame_tab')),
            inputs=[palette, *render_inputs],
            outputs=[
                color, secondary_color, backdrop_color, backdrop_opacity,
                backdrop_radius, backdrop, backdrop_scope, outline, font, bold,
                outline_color, shadow, box_padding, italic,
                live_preview, live_status, preview_tabs,
            ],
            trigger_mode='always_last',
            show_progress='hidden',
            concurrency_id='live_preview',
            concurrency_limit=1,
        )

        # Atomic Platform Layout Preset buttons
        layout_outputs = [
            aspect, frame_fill, position, margin, text_width, max_chars,
            max_lines, size, min_size, box_padding, dual_gap, wrap_mode,
            live_preview, live_status, preview_tabs,
        ]
        for btn, key in (
            (preset_tiktok_crop, 'tiktok_crop'),
            (preset_tiktok_fit, 'tiktok_fit'),
            (preset_yt, 'youtube_wide'),
            (preset_orig, 'original_clean'),
        ):
            btn.click(
                lambda *args, k=key: (*apply_layout_preset_and_preview(k, *args), gr.update(selected='frame_tab')),
                inputs=render_inputs,
                outputs=layout_outputs,
                show_progress='hidden',
                concurrency_id='live_preview',
                concurrency_limit=1,
            )

        # Live Frame Preview on direct user input or project/table/seek changes
        user_style_controls = (
            render_mode, font, size, color, secondary_color,
            backdrop_color, backdrop_opacity, backdrop_radius,
            backdrop_scope, backdrop, position, margin, resolution,
            aspect, frame_fill, text_align, text_width, wrap_mode,
            auto_fit, max_lines, min_size, source_scale,
            uppercase, bold, outline, max_chars,
            outline_color, shadow, italic, letter_spacing,
            box_padding, dual_gap, dual_order, delay,
        )
        live_triggers = [state.change, table.change, media.change, start.change]
        for comp in user_style_controls:
            live_triggers.append(comp.input if hasattr(comp, 'input') else comp.change)

        gr.on(
            triggers=live_triggers,
            fn=_preview_and_show_frame,
            inputs=render_inputs,
            outputs=[live_preview, live_status, preview_tabs],
            trigger_mode='always_last',
            show_progress='hidden',
            concurrency_id='live_preview',
            concurrency_limit=1,
        )

    return demo


if __name__ == '__main__':
    build_app().queue(max_size=12).launch(
        server_name='127.0.0.1',
        inbrowser=True,
        share=False,
        favicon_path=str(FAVICON_PATH) if FAVICON_PATH.is_file() else None,
        allowed_paths=[str(OUTPUT), str(ASSETS)],
        theme=STUDIO_THEME,
        css=CSS,
    )
