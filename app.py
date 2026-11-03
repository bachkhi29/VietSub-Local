from __future__ import annotations

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
from core import (FONT_FILES, MODEL, OUTPUT, ROOT, TARGET_LANGUAGES, api, export_project, folder, list_projects,
                  load_project, new_project, parse_srt, parse_time, require_model, rows_to_segments,
                  save_project, subtitle_delay, table_rows, timestamp, translate_project,
                  unload_model, webvtt)
from render import render_video

OUTPUT.mkdir(exist_ok=True)


def check_system():
    lines = [f'Python: {sys.version.split()[0]}']
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        lines.append('GPU: ' + (gpu.stdout.strip() if gpu.returncode == 0 else 'chưa đọc được NVIDIA'))
    except (OSError, subprocess.TimeoutExpired):
        lines.append('GPU: chưa tìm thấy nvidia-smi; có thể thử chế độ CPU.')
    try:
        require_model()
        lines.append(f'Ollama: sẵn sàng, đã có {MODEL}.')
    except RuntimeError as e:
        lines.append(str(e))
    if os.name == 'nt':
        import site
        for lib in ('cublas', 'cudnn'):
            found = any((Path(p) / 'nvidia' / lib / 'bin').exists() for p in site.getsitepackages())
            lines.append(f'Thư viện {lib}: ' + ('đã cài' if found else 'chưa có; chạy 1_CAI_DAT.bat'))
    lines.append('Đây là kiểm tra cài đặt; chỉ chạy video thật mới kiểm tra được toàn bộ GPU và mô hình.')
    return '\n'.join(lines)


def source_job(media, source_srt, model, device, language, target, progress=gr.Progress()):
    try:
        if target == language:
            raise ValueError('Ngôn ngữ gốc và ngôn ngữ đích đang giống nhau.')
        if source_srt:
            segments = parse_srt(Path(source_srt).read_text(encoding='utf-8-sig'))
            project = new_project(Path(source_srt).name, segments,
                                  language if language != 'auto' else 'source', media, target)
        else:
            if not media:
                raise ValueError('Hãy thả video hoặc file phụ đề SRT vào trước.')
            progress(0, desc='Chuẩn bị nhận diện…')
            unload_model()
            work = Path(tempfile.mkdtemp(prefix='asr_', dir=OUTPUT))
            result = work / 'result.json'
            log = work / 'asr.log'
            command = [sys.executable, '-u', str(ROOT / 'asr_worker.py'),
                       '--input', str(media), '--output', str(result), '--model', model,
                       '--device', device, '--language', language]
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            proc = None
            try:
                with log.open('w', encoding='utf-8') as err:
                    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err,
                                            text=True, encoding='utf-8', errors='replace', env=env)
                    for line in proc.stdout:
                        try:
                            event = json.loads(line)
                            progress(event['progress'], desc=event['message'])
                        except (ValueError, KeyError):
                            continue
                    code = proc.wait()
                if code != 0:
                    detail = log.read_text(encoding='utf-8')[-1600:]
                    raise RuntimeError('Nhận diện chưa chạy được. Nếu lỗi CUDA/cuDNN/DLL, '
                                       'chạy lại 1_CAI_DAT.bat hoặc thử CPU + small.\n' + detail)
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
        files, _ = export_project(project)
        return (project, table_rows(project), files,
                f'Đã tạo {len(project["segments"])} dòng. Rà bản gốc rồi bấm Dịch.',
                gr.update(choices=list_projects(), value=project['job']), None,
                gr.update(value=target), gr.update(value=0), preview_seek_update(project))
    except Exception as e:
        raise gr.Error(str(e)) from e


def edited(project, rows):
    if not project:
        raise ValueError('Hãy tạo hoặc mở một công việc trước.')
    result = copy.deepcopy(project)
    result['segments'] = rows_to_segments(rows, project['segments'])
    save_project(result)
    return result


def save_and_export(project, rows):
    try:
        project = edited(project, rows)
        files, complete = export_project(project)
        msg = 'Đã lưu và xuất đủ bản gốc, bản dịch, song ngữ và ASS.' if complete else (
            'Đã lưu công việc và xuất bản gốc. Còn câu chưa dịch nên chưa xuất bản dịch/song ngữ.')
        return project, table_rows(project), files, msg, None
    except Exception as e:
        raise gr.Error(str(e)) from e


def translate_job(project, rows, glossary, target, progress=gr.Progress()):
    try:
        project = edited(project, rows)
        if target != project.get('target_language', 'vi'):
            raise ValueError('Bạn vừa chọn ngôn ngữ đích khác. Hãy bấm “Đổi ngôn ngữ đích · xóa bản dịch cũ” trước khi dịch.')
        require_model()
    except Exception as e:
        raise gr.Error(str(e)) from e
    status = ''
    try:
        progress(0, desc='Đang nạp mô hình dịch…')
        translate_project(project, glossary,
                          lambda n, total: progress(n / total, desc=f'Đã dịch {n}/{total} câu còn thiếu'))
        status = 'Dịch xong. Hãy rà tên riêng, số liệu và cách xưng hô trước khi dùng.'
    except Exception as e:
        status = f'Đã dừng: {e}\nCác câu dịch xong vẫn được lưu. Bấm Dịch lần nữa để tiếp tục.'
    finally:
        unload_model()
    files, _ = export_project(project)
    return project, table_rows(project), files, status, None


def reopen(job):
    try:
        if not job:
            raise ValueError('Chọn một công việc đã lưu.')
        project = load_project(job)
        files, _ = export_project(project)
        return (project, table_rows(project), files, 'Đã mở lại. Nếu muốn xem video, chọn lại đúng video gốc.',
                None, gr.update(value=project.get('target_language', 'vi')),
                gr.update(value=project.get('subtitle_delay', 0)), preview_seek_update(project))
    except Exception as e:
        raise gr.Error(str(e)) from e


def reset_translation(project, rows):
    try:
        project = edited(project, rows)
        for s in project['segments']:
            s['vi'] = ''
        save_project(project)
        files, _ = export_project(project)
        return project, table_rows(project), files, 'Đã xóa bản dịch hiện tại; bấm Dịch để tạo lại.', None
    except Exception as e:
        raise gr.Error(str(e)) from e


def set_target_job(project, rows, target):
    try:
        project = edited(project, rows)
        if target not in TARGET_LANGUAGES:
            raise ValueError('Ngôn ngữ đích không hợp lệ.')
        if target == project.get('language'):
            raise ValueError('Ngôn ngữ đích đang trùng với ngôn ngữ gốc.')
        if project.get('target_language', 'vi') == target:
            files, _ = export_project(project)
            return project, table_rows(project), files, 'Ngôn ngữ đích đang được sử dụng.', None
        for row in project['segments']:
            row['vi'] = ''
        project['target_language'] = target
        save_project(project)
        files, _ = export_project(project)
        return (project, table_rows(project), files, f'Đã đổi sang {TARGET_LANGUAGES[target]}. '
                'Bản dịch trước đã được xóa khỏi công việc này; bấm Dịch để tạo bản mới.', None)
    except Exception as e:
        raise gr.Error(str(e)) from e


def preview_job(project, rows, media, mode):
    try:
        project = edited(project, rows)
        video = media or project.get('media')
        if not video or not Path(video).is_file():
            raise ValueError('Chọn lại đúng video gốc ở ô Video rồi bấm Xem trước.')
        if Path(video).suffix.lower() not in ('.mp4', '.webm', '.ogg'):
            raise ValueError('Xem trong trình duyệt hỗ trợ MP4/WebM/OGG. '
                             'Video khác vẫn nhận diện được; xem SRT bằng VLC.')
        files, complete = export_project(project)
        if mode != 'Gốc' and not complete:
            raise ValueError('Cần dịch đủ các câu trước khi xem bản dịch/song ngữ.')
        preview_mode = 'source' if mode == 'Gốc' else 'vi' if mode == 'Bản dịch' else 'dual'
        chosen = folder(project) / f'preview-{preview_mode}.vtt'
        chosen.write_text(webvtt(project['segments'], preview_mode,
                                 project.get('subtitle_delay', 0)), encoding='utf-8')
        return (project, table_rows(project), files,
                gr.Video(value=str(video), subtitles=str(chosen), interactive=False),
                'Đây là phụ đề CC để rà câu chữ. Kiểu chữ MP4 hiển thị ngay ở tab Xuất video.')
    except Exception as e:
        raise gr.Error(str(e)) from e


STYLE_PRESETS = {
    'Trắng trên nền đen': dict(color='#FFFFFF', secondary='#FFF2A8',
                              background='#000000', opacity=70, radius=12,
                              backdrop='panel', scope='all', outline=2.5,
                              font='Segoe UI', bold=True),
    'Đen trên nền vàng': dict(color='#111111', secondary='#FFFFFF',
                             background='#FFD800', opacity=100, radius=28,
                             backdrop='panel', scope='translation', outline=0,
                             font='Arial Narrow', bold=True),
    'Trắng trên nền xanh': dict(color='#FFFFFF', secondary='#D7F3FF',
                               background='#184A6B', opacity=92, radius=18,
                               backdrop='panel', scope='translation', outline=1,
                               font='Arial', bold=True),
    'Trắng tối giản': dict(color='#FFFFFF', secondary='#F5D4E0',
                          background='#000000', opacity=0, radius=0,
                          backdrop='none', scope='all', outline=3,
                          font='Segoe UI', bold=True),
}


def apply_style_preset(name):
    style = STYLE_PRESETS[name]
    return (style['color'], style['secondary'], style['background'],
            style['opacity'], style['radius'], style['backdrop'],
            style['scope'], style['outline'], style['font'], style['bold'])

LANGUAGE_FLAGS = {
    'vi': '🇻🇳', 'en': '🇬🇧', 'zh': '🇨🇳', 'ja': '🇯🇵',
    'ko': '🇰🇷', 'fr': '🇫🇷', 'es': '🇪🇸',
}


def preview_seek_update(project):
    segments = project['segments']
    delay = project.get('subtitle_delay', 0)
    first = segments[0]
    first_time = max(0, first['start'] + delay + min(.5, (first['end'] - first['start']) / 2))
    last_time = max(segment['end'] + delay for segment in segments)
    return gr.update(maximum=max(1, round(last_time + 1, 1)), value=round(first_time, 1))


def jump_subtitle(project, current, delay_value, direction):
    if not project:
        return gr.update()
    delay = subtitle_delay(delay_value)
    times = [max(0, segment['start'] + delay +
                 min(.5, (segment['end'] - segment['start']) / 2))
             for segment in project['segments']]
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
        direction = 'muộn hơn' if project['subtitle_delay'] > 0 else 'sớm hơn' if project['subtitle_delay'] < 0 else 'như mốc gốc'
        return (project, files,
                f'Đã lưu độ trễ {project["subtitle_delay"]:+.1f} giây: phụ đề hiện {direction}. '
                'File SRT/VTT/ASS và MP4 mới sẽ dùng mốc này.')
    except Exception as e:
        raise gr.Error(str(e)) from e


def render_job(project, rows, media, mode, font, size, color, secondary_color,
               backdrop_color, backdrop_opacity, backdrop_radius, backdrop_scope,
               backdrop, position, margin,
               resolution, aspect, frame_fill, text_align, text_width, wrap_mode,
               auto_fit, max_lines, min_size, source_scale, start, delay_value,
               uppercase, bold, outline, max_chars,
               sample, progress=gr.Progress(), frame_only=False):
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
        if not any(segment['end'] + project['subtitle_delay'] > 0
                   for segment in project['segments']):
            raise ValueError('Độ trễ đưa toàn bộ phụ đề ra trước đầu video. Hãy giảm độ dịch sớm.')
        if not frame_only:
            save_project(project)
        video = media or project.get('media')
        chosen = {'Bản dịch': 'vi', 'Song ngữ': 'dual', 'Gốc': 'source'}[mode]
        fallback = frame_only and chosen != 'source' and not all(
            segment['vi'].strip() for segment in project['segments'])
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
        result = render_video(project, video, chosen, font, size, color,
                              backdrop != 'none', 'bottom' if position == 'Dưới' else 'top',
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
                              frame_only=frame_only)
        if frame_only:
            active = any(segment['start'] + project['subtitle_delay'] <= sample_start <
                         segment['end'] + project['subtitle_delay']
                         for segment in project['segments'])
            status = f'Khung video tại {timestamp(sample_start)}.'
            if not active:
                status += ' Mốc này chưa có câu phụ đề; chọn mốc khác để rà chữ.'
            if fallback:
                status += ' Bản dịch chưa đủ nên đang hiện chữ gốc.'
            if unsaved_edits:
                status += ' Bảng có sửa chưa lưu; hãy bấm Lưu trước khi đóng công việc.'
            return project, table_rows(project), result, gr.update(), status
        msg = ('Đã dựng video 30 giây. Mở thẻ “MP4 thử 30 giây” bên phải để xem chuyển động; '
               'nếu ổn thì bấm Xuất toàn bộ MP4.' if sample else
               'Đã xuất toàn bộ MP4. Bấm vào file bên dưới để tải về.')
        return project, table_rows(project), result, (gr.update(value=result) if sample else gr.update()), msg
    except Exception as e:
        raise gr.Error(str(e)) from e


def preview_frame_job(*args):
    if not args[0]:
        return None, 'Tạo hoặc mở công việc để xem ngay trên khung video.'
    try:
        result = render_job(*args, True, frame_only=True)
        return result[2], result[4]
    except Exception as e:
        return None, f'Chưa xem được khung video: {e}'


CSS = '''
.gradio-container {max-width: 1450px !important; margin: auto; font-family: 'Segoe UI', Arial, sans-serif;}
body {background: #f6f7fb !important;}
#hero {padding: 24px 4px 14px; border-bottom: 1px solid #e8e5ee; margin-bottom: 16px;}
#hero h1 {font-size: 32px; letter-spacing: -.04em; color: #272430; margin-bottom: 5px;}
#hero p {color: #615e69; font-size: 15px;}
#upload-card, #review-card, #render-controls, #render-preview {
  background: #fff; border: 1px solid #e7e3ea; border-radius: 20px;
  padding: 20px; box-shadow: 0 10px 30px rgba(44, 28, 46, .045);
}
#render-controls {max-height: calc(100vh - 115px); overflow-y: auto;
  overscroll-behavior: contain; scrollbar-color: #d9a1b9 #f9f3f6;}
#render-controls .accordion {border-color: #eee4eb; border-radius: 14px;}
#render-controls .accordion > button {font-weight: 650; color: #42313c;}
#render-controls .accordion[open] > button {color: #b93b72;}
#render-preview {align-self: flex-start;}
#live-preview {background: #151419; border-radius: 14px; overflow: hidden;}
#live-preview img {object-fit: contain !important;}
.gradio-container button.primary {background: #cb4b82 !important; border-color: #cb4b82 !important; color: white !important;}
.gradio-container button.primary:hover {background: #ad386a !important;}
@media (max-width: 900px) {#render-controls {max-height: none; overflow: visible;}}
'''


def build_app():
    with gr.Blocks(title='VietSub Local') as demo:
        state = gr.State(None)
        gr.Markdown('# VietSub Local\nNhận diện giọng nói · Dịch và sửa phụ đề · Dán chữ vào video ngay trên máy.', elem_id='hero')
        with gr.Tabs():
            with gr.Tab('1 · Video & dịch'):
                with gr.Row():
                    with gr.Column(scale=4, elem_id='upload-card'):
                        gr.Markdown('### Chọn video và ngôn ngữ')
                        media = gr.File(label='Video hoặc âm thanh gốc', file_types=['.mp4', '.mkv', '.mov', '.webm', '.avi', '.mp3', '.wav', '.m4a', '.ogg'], type='filepath')
                        source_srt = gr.File(label='Có sẵn SRT gốc? Thả vào đây để khỏi nhận diện lại', file_types=['.srt'], type='filepath')
                        with gr.Row():
                            language = gr.Dropdown(
                                [(f'{LANGUAGE_FLAGS[code]} {TARGET_LANGUAGES[code]}', code)
                                 for code in ('en', 'vi', 'zh', 'ja', 'ko', 'fr', 'es')]
                                + [('🌐 Tự nhận diện', 'auto')],
                                value='en', label='Ngôn ngữ gốc')
                            target = gr.Dropdown([(f'{LANGUAGE_FLAGS[code]} {label}', code)
                                                  for code, label in TARGET_LANGUAGES.items()],
                                                 value='vi', label='Ngôn ngữ đích')
                        gr.Markdown('Nếu có cả video và SRT, app dùng SRT. SRT cần mã hóa UTF-8. '
                                    'Ngôn ngữ gốc của SRT là lựa chọn ở trên.')
                        with gr.Accordion('Tùy chọn nhận diện', open=False):
                            model = gr.Dropdown(['small', 'medium', 'large-v3'], value='medium', label='Mô hình')
                            device = gr.Dropdown([('NVIDIA GPU', 'cuda'), ('CPU', 'cpu')], value='cuda', label='Thiết bị')
                        create = gr.Button('Tạo phụ đề gốc', variant='primary')
                        glossary = gr.Textbox(label='Tên riêng và thuật ngữ (tùy chọn)', lines=2,
                                              placeholder='Ví dụ: giữ nguyên tên Mel Robbins')
                        translate = gr.Button('Dịch các câu còn trống', variant='primary')
                        with gr.Accordion('Đổi ngôn ngữ cho công việc đang mở', open=False):
                            gr.Markdown('Chọn ngôn ngữ đích khác ở trên, rồi bấm nút này. '
                                        '**Bản dịch của công việc hiện tại sẽ bị xóa** để tránh trộn hai ngôn ngữ. '
                                        'Nên tải ZIP trước khi đổi.')
                            change_target = gr.Button('Đổi ngôn ngữ đích · xóa bản dịch cũ')
                    with gr.Column(scale=6, elem_id='review-card'):
                        gr.Markdown('### Rà nội dung và thời gian')
                        preview = gr.Video(label='Video gốc + phụ đề CC', interactive=False)
                        with gr.Row():
                            mode = gr.Radio(['Gốc', 'Bản dịch', 'Song ngữ'], value='Song ngữ', label='Lớp phụ đề CC')
                            show = gr.Button('Cập nhật phụ đề CC')
                        gr.Markdown('Trình phát này dùng **CC để rà câu chữ**. Mở tab **3 · Xuất video** '
                                    'để xem kiểu chữ trên hình video và chỉnh trực tiếp.')
                        status = gr.Textbox(label='Trạng thái', lines=3, interactive=False,
                                            value='Chọn video và thử một đoạn ngắn trước.')
                with gr.Accordion('Kiểm tra máy & hướng dẫn', open=False):
                    check = gr.Button('Kiểm tra cài đặt')
                    health = gr.Textbox(label='Tình trạng', lines=5, interactive=False)
                    check.click(check_system, outputs=health)
            with gr.Tab('2 · Sửa & tải phụ đề'):
                gr.Markdown('### Biên tập từng câu\nNhấp đúp một ô để sửa. Thời gian dạng **00:01:02,500**. '
                            'Bấm **Lưu** trước khi xuất hoặc chuyển công việc.')
                table = gr.Dataframe(headers=['STT', 'Bắt đầu', 'Kết thúc', 'Bản gốc', 'Bản dịch'],
                                     datatype=['number', 'str', 'str', 'str', 'str'], type='array',
                                     interactive=True, column_count=(5, 'fixed'), row_count=(0, 'dynamic'), wrap=True)
                save = gr.Button('Lưu và xuất SRT / VTT / ASS', variant='primary')
                downloads = gr.File(label='Tải ZIP hoặc từng file phụ đề', file_count='multiple', interactive=False)
                with gr.Accordion('Mở lại công việc đã lưu', open=False):
                    history = gr.Dropdown(choices=list_projects(), label='Công việc')
                    with gr.Row():
                        refresh = gr.Button('Làm mới danh sách')
                        restore = gr.Button('Mở công việc')
                    gr.Markdown('Công việc ở thư mục **output**. Nếu mở lại, hãy chọn lại video gốc ở tab 1.')
                with gr.Accordion('Dịch lại từ đầu', open=False):
                    gr.Markdown('Nút này xóa bản dịch hiện tại. Tải ZIP để giữ bản cũ trước khi xóa.')
                    reset = gr.Button('Xóa bản dịch hiện tại')
            with gr.Tab('3 · Xuất video MP4'):
                gr.Markdown('### Chỉnh phụ đề trên hình video\n'
                            'Khung **Xem ngay** tự cập nhật khi đổi cài đặt. '
                            'Chỉ dựng MP4 30 giây khi muốn kiểm tra chuyển động trước khi xuất.')
                with gr.Row():
                    with gr.Column(scale=5, elem_id='render-controls'):
                        gr.Markdown('**🎨 Màu phụ đề** · Chọn nhanh một mẫu tại đây, hoặc mở **Màu & nền** để chỉnh từng màu.')
                        palette = gr.Dropdown(list(STYLE_PRESETS),
                                              value='Trắng trên nền đen',
                                              label='Bộ màu chọn nhanh',
                                              info='Ví dụ: Đen trên nền vàng. Khung bên phải tự cập nhật.')
                        with gr.Accordion('1 · Khung hình & đồng bộ', open=False):
                            with gr.Row():
                                aspect = gr.Dropdown([
                                    ('Giữ khung gốc', 'original'), ('Ngang 16:9', '16:9'),
                                    ('Ngang 4:3', '4:3'), ('Vuông 1:1', '1:1'),
                                    ('Dọc 4:5', '4:5'), ('Dọc 9:16', '9:16')],
                                    value='original', label='Tỷ lệ video')
                                frame_fill = gr.Dropdown([
                                    ('Giữ toàn bộ hình · có viền', 'fit'),
                                    ('Phủ kín khung · cắt rìa', 'crop')],
                                    value='fit', label='Cách đưa hình vào khung')
                            resolution = gr.Dropdown(['Gốc', '720p'], value='Gốc',
                                                     label='Độ phân giải MP4')
                            delay = gr.Slider(-10, 10, value=0, step=.1,
                                              label='Độ trễ phụ đề (giây)',
                                              info='Dương (+): hiện muộn. Âm (−): hiện sớm.')
                        with gr.Accordion('2 · Chữ & bố cục', open=False):
                            render_mode = gr.Radio(['Bản dịch', 'Song ngữ', 'Gốc'],
                                                   value='Song ngữ', label='Hiển thị trên video')
                            with gr.Row():
                                font = gr.Dropdown(list(FONT_FILES), value='Segoe UI',
                                                   label='Phông chữ', filterable=True)
                                size = gr.Slider(24, 80, value=46, step=1, label='Cỡ chữ')
                            with gr.Row():
                                text_align = gr.Radio([
                                    ('Trái', 'left'), ('Giữa', 'center'), ('Phải', 'right')],
                                    value='center', label='Căn chữ')
                                wrap_mode = gr.Radio([
                                    ('Cân độ dài các dòng', 'balanced'),
                                    ('Đổ đầy từng dòng', 'compact')],
                                    value='balanced', label='Cách xuống dòng')
                            text_width = gr.Slider(60, 95, value=90, step=1,
                                                   label='Bề rộng vùng chữ (%)')
                            with gr.Row():
                                position = gr.Radio(['Dưới', 'Trên'], value='Dưới', label='Vị trí chữ')
                                margin = gr.Slider(15, 240, value=65, step=5,
                                                   label='Cách mép video')
                            auto_fit = gr.Checkbox(value=True, label='Tự thu nhỏ chữ khi câu dài')
                            with gr.Accordion('Nâng cao · câu dài & song ngữ', open=False):
                                with gr.Row():
                                    max_lines = gr.Slider(2, 5, value=3, step=1,
                                                          label='Số dòng mục tiêu mỗi ngôn ngữ')
                                    min_size = gr.Slider(16, 36, value=22, step=1,
                                                         label='Cỡ chữ nhỏ nhất')
                                source_scale = gr.Slider(50, 90, value=74, step=1,
                                                         label='Cỡ dòng gốc so với bản dịch (%)')
                                max_chars = gr.Slider(30, 85, value=52, step=1,
                                                       label='Giới hạn ký tự mỗi dòng')
                                bold = gr.Checkbox(label='Chữ đậm', value=True)
                                uppercase = gr.Checkbox(label='IN HOA bản dịch', value=False)
                        with gr.Accordion('3 · Màu & nền · chỉnh chi tiết', open=True):
                            with gr.Row():
                                color = gr.ColorPicker(value='#FFFFFF', label='Màu bản dịch')
                                secondary_color = gr.ColorPicker(value='#FFF2A8', label='Màu dòng gốc')
                            backdrop_color = gr.ColorPicker(value='#000000', label='Màu nền phụ đề')
                            with gr.Row():
                                backdrop = gr.Dropdown([
                                    ('Nền liền khối', 'panel'), ('Nền từng dòng', 'lines'),
                                    ('Không nền', 'none')], value='panel', label='Kiểu nền')
                                backdrop_scope = gr.Radio([
                                    ('Chỉ bản dịch', 'translation'), ('Cả hai dòng', 'all')],
                                    value='all', label='Nền cho',
                                    info='Áp dụng khi dùng nền liền khối và song ngữ.')
                            with gr.Row():
                                backdrop_opacity = gr.Slider(0, 100, value=70, step=5,
                                                             label='Độ đậm nền (%)')
                                backdrop_radius = gr.Slider(0, 40, value=12, step=2,
                                                            label='Bo góc nền')
                            outline = gr.Slider(0, 6, value=2.5, step=.5,
                                                label='Viền chữ',
                                                info='Để 0 khi dùng chữ đen trên nền vàng.')
                        gr.Markdown('**Kiểm tra & xuất** · Màu, chữ và nền hiện ngay trong khung bên phải.')
                        with gr.Row():
                            sample_button = gr.Button('Dựng thử 30 giây')
                            render_button = gr.Button('Xuất toàn bộ MP4', variant='primary')
                    with gr.Column(scale=6, elem_id='render-preview'):
                        gr.Markdown('**Khung đối chiếu** · Thay đổi cài đặt bên trái để xem ngay tại mốc này.')
                        start = gr.Slider(0, 30, value=0, step=.1,
                                          label='Mốc đang xem (giây)')
                        with gr.Row():
                            previous_cue = gr.Button('◀ Câu trước', size='sm')
                            next_cue = gr.Button('Câu sau ▶', size='sm')
                        with gr.Tabs():
                            with gr.Tab('Xem ngay · tự cập nhật'):
                                live_preview = gr.Image(label='Khung video có phụ đề', height=570,
                                                        format='png', interactive=False,
                                                        buttons=['fullscreen'], elem_id='live-preview')
                                live_status = gr.Markdown('Tạo hoặc mở công việc để xem ngay trên khung video.')
                            with gr.Tab('MP4 thử 30 giây'):
                                rendered_preview = gr.Video(label='Video đã dán chữ · phát thử',
                                                            interactive=False)
                        rendered = gr.File(label='Tải MP4 vừa tạo', interactive=False)
                gr.Markdown('Ảnh **Xem ngay** là một khung hình tại mốc đã chọn và tự cập nhật khi chỉnh. '
                            'Dùng **Dựng thử 30 giây** để kiểm tra chuyển động và âm thanh trước khi xuất cả video.')
        common = dict(concurrency_id='local_gpu', concurrency_limit=1)
        create.click(source_job, [media, source_srt, model, device, language, target],
                     [state, table, downloads, status, history, preview, target, delay, start], **common)
        translate.click(translate_job, [state, table, glossary, target], [state, table, downloads, status, preview], **common)
        save.click(save_and_export, [state, table], [state, table, downloads, status, preview], **common)
        restore.click(reopen, [history], [state, table, downloads, status, preview, target, delay, start], **common)
        reset.click(reset_translation, [state, table], [state, table, downloads, status, preview], **common)
        change_target.click(set_target_job, [state, table, target],
                            [state, table, downloads, status, preview], **common)
        show.click(preview_job, [state, table, media, mode], [state, table, downloads, preview, status], **common)
        delay.change(set_delay_job, [state, table, delay], [state, downloads, status],
                     trigger_mode='always_last', **common)
        previous_cue.click(lambda project, moment, offset: jump_subtitle(project, moment, offset, -1),
                           [state, start, delay], start, queue=False)
        next_cue.click(lambda project, moment, offset: jump_subtitle(project, moment, offset, 1),
                       [state, start, delay], start, queue=False)
        palette.change(apply_style_preset, palette,
                       [color, secondary_color, backdrop_color, backdrop_opacity,
                        backdrop_radius, backdrop, backdrop_scope, outline, font, bold],
                       queue=False)
        render_inputs = [state, table, media, render_mode, font, size, color,
                         secondary_color, backdrop_color, backdrop_opacity,
                         backdrop_radius, backdrop_scope, backdrop,
                         position, margin, resolution, aspect, frame_fill,
                         text_align, text_width, wrap_mode, auto_fit, max_lines,
                         min_size, source_scale, start, delay,
                         uppercase, bold, outline, max_chars]
        render_outputs = [state, table, rendered, rendered_preview, status]
        sample_button.click(lambda *args, progress=gr.Progress(): render_job(*args, True, progress),
                            render_inputs, render_outputs, **common)
        render_button.click(lambda *args, progress=gr.Progress(): render_job(*args, False, progress),
                            render_inputs, render_outputs, **common)
        gr.on(
            triggers=[state.change, table.change, media.change, start.change]
                     + [component.change for component in (
                         render_mode, font, size, color, secondary_color,
                         backdrop_color, backdrop_opacity, backdrop_radius,
                         backdrop_scope, backdrop, position,
                         margin, resolution, aspect, frame_fill, text_align,
                         text_width, wrap_mode, auto_fit, max_lines, min_size,
                         source_scale, uppercase, bold, outline, max_chars)],
            fn=preview_frame_job, inputs=render_inputs,
            outputs=[live_preview, live_status],
            trigger_mode='always_last', show_progress='hidden',
            concurrency_id='live_preview', concurrency_limit=1,
        )
        refresh.click(lambda: gr.update(choices=list_projects()), outputs=history)
        gr.Markdown('Chạy cục bộ · Xem ngay trên khung video · Xuất MP4, SRT, VTT, ASS và ZIP. '
                    'AI có thể nghe hoặc dịch sai; hãy rà lại nội dung quan trọng.')
    return demo


if __name__ == '__main__':
    build_app().queue(max_size=8).launch(server_name='127.0.0.1', inbrowser=True, share=False,
                                       allowed_paths=[str(OUTPUT)],
                                       theme=gr.themes.Soft(primary_hue='pink', neutral_hue='slate'), css=CSS)
