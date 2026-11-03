"""Render a local video with subtitles using the FFmpeg shipped by imageio-ffmpeg."""
from __future__ import annotations

import re
import subprocess
import uuid
from pathlib import Path

import imageio_ffmpeg

from core import folder, parse_time, styled_ass

VIDEO_EXTS = {'.mp4', '.mkv', '.mov', '.webm', '.avi', '.m4v'}
FRAME_RATIOS = {'16:9': (16, 9), '4:3': (4, 3), '9:16': (9, 16),
                '1:1': (1, 1), '4:5': (4, 5)}


def video_size(source):
    frames = imageio_ffmpeg.read_frames(str(source))
    try:
        return frames.__next__()['size']
    finally:
        frames.close()


def frame_size(source_size, aspect, resolution):
    width_ratio, height_ratio = FRAME_RATIOS[aspect]
    short_side = 720 if resolution == '720p' else min(source_size)
    short_side = max(2, 2 * round(short_side / 2))
    if width_ratio >= height_ratio:
        return 2 * round(short_side * width_ratio / height_ratio / 2), short_side
    return short_side, 2 * round(short_side * height_ratio / width_ratio / 2)


def render_video(project, media, mode='dual', font='Arial', size=46,
                 color='#FFFFFF', backdrop=False, position='bottom', margin=65,
                 sample=False, sample_start=0, resolution='Gốc', progress=None,
                 secondary_color='#E7E7E7', uppercase=False, bold=True,
                 outline=2.5, max_chars=52, aspect='original', frame_fill='fit',
                 text_align='center', text_width=90, wrap_mode='balanced',
                 auto_fit=True, max_lines=3, min_size=22, source_scale=74,
                 backdrop_style='lines', frame_only=False,
                 backdrop_color='#000000', backdrop_opacity=70,
                 backdrop_radius=12, backdrop_scope='all'):
    source = Path(media or '').resolve()
    if not source.is_file():
        raise ValueError('Chọn lại video gốc ở ô Video trước khi xuất MP4.')
    if source.suffix.lower() not in VIDEO_EXTS:
        raise ValueError('Xuất MP4 cần video (không dùng file chỉ có âm thanh).')
    if resolution not in ('Gốc', '720p'):
        raise ValueError('Độ phân giải không hợp lệ.')
    if aspect != 'original' and aspect not in FRAME_RATIOS:
        raise ValueError('Tỷ lệ khung hình không hợp lệ.')
    if frame_fill not in ('fit', 'crop'):
        raise ValueError('Cách lấp đầy khung không hợp lệ.')
    if not frame_only and mode != 'source' and not all(s['vi'].strip() for s in project['segments']):
        raise ValueError('Còn câu chưa dịch. Dịch đủ các câu trước khi xuất bản Việt/song ngữ.')
    start = parse_time(sample_start) if sample or frame_only else 0
    duration = .2 if frame_only else 30 if sample else None
    if aspect == 'original':
        vf = ('scale=-2:720,' if resolution == '720p' else '') + 'ass=overlay.ass'
        script_width = 1920
    else:
        dimensions = frame_size(video_size(source) if resolution == 'Gốc' else (720, 720),
                                aspect, resolution)
        width, height = dimensions
        scale_mode = 'decrease' if frame_fill == 'fit' else 'increase'
        framing = (f'scale={width}:{height}:force_original_aspect_ratio={scale_mode}:'
                   'force_divisible_by=2,')
        if frame_fill == 'fit':
            framing += f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,'
        else:
            framing += f'crop={width}:{height},'
        vf = framing + 'setsar=1,ass=overlay.ass'
        script_width = round(1080 * width / height)
    ass = styled_ass(project['segments'], mode, font, size, color, backdrop,
                     position, margin, clip_start=start, clip_duration=duration,
                     secondary_color=secondary_color, uppercase=uppercase,
                     bold=bold, outline=outline, max_chars=max_chars,
                     script_width=script_width, text_align=text_align,
                     text_width=text_width, wrap_mode=wrap_mode,
                     auto_fit=auto_fit, max_lines=max_lines,
                     min_size=min_size, source_scale=source_scale,
                     backdrop_style=backdrop_style,
                     backdrop_color=backdrop_color,
                     backdrop_opacity=backdrop_opacity,
                     backdrop_radius=backdrop_radius,
                     backdrop_scope=backdrop_scope,
                     delay=project.get('subtitle_delay', 0))
    if not frame_only and 'Dialogue: ' not in ass:
        raise ValueError('Không có phụ đề ở mốc thử này. Chọn mốc khác.')
    if frame_only:
        directory = folder(project) / 'previews'
        directory.mkdir(parents=True, exist_ok=True)
        stamp = uuid.uuid4().hex[:12]
        overlay = directory / f'{stamp}.ass'
        frame = directory / f'{stamp}.png'
        overlay.write_text(ass, encoding='utf-8-sig')
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-nostdin', '-y',
                   '-loglevel', 'error', '-ss', str(start), '-i', str(source),
                   '-frames:v', '1', '-an', '-vf', vf.replace('overlay.ass', overlay.name),
                   '-update', '1', str(frame)]
        try:
            result = subprocess.run(command, cwd=directory, capture_output=True,
                                    text=True, encoding='utf-8', errors='replace',
                                    timeout=45,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if result.returncode != 0 or not frame.is_file():
                raise RuntimeError('Không lấy được khung hình tại mốc đã chọn.\n'
                                   + result.stderr[-1000:])
            old = sorted(directory.glob('*.png'), key=lambda path: path.stat().st_mtime)
            for path in old[:-8]:
                path.unlink(missing_ok=True)
            return str(frame)
        finally:
            overlay.unlink(missing_ok=True)
    estimated_duration = duration or max(s['end'] for s in project['segments'])
    directory = folder(project) / 'renders' / uuid.uuid4().hex[:10]
    directory.mkdir(parents=True)
    (directory / 'overlay.ass').write_text(ass, encoding='utf-8-sig')
    label = {'source': 'goc', 'vi': 'viet', 'dual': 'song-ngu'}[mode]
    stem = f"{project['name']}-{label}" + ('-thu-30s' if sample else '')
    if aspect != 'original':
        stem += f'-{aspect.replace(":", "x")}-{frame_fill}'
    output = directory / f'{stem}.mp4'
    partial = directory / 'rendering.part.mp4'
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-nostdin', '-y',
           '-loglevel', 'error', '-ss', str(start), '-i', str(source)]
    if sample:
        cmd += ['-t', str(duration)]
    cmd += ['-map', '0:v:0', '-map', '0:a?', '-vf', vf,
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k',
            '-movflags', '+faststart', '-progress', 'pipe:1', '-nostats', str(partial)]
    # The temporary name needs an explicit container; its suffix is .mp4.
    errors = []
    try:
        proc = subprocess.Popen(cmd, cwd=directory, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                errors='replace', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            for line in proc.stdout:
                line = line.strip()
                errors.append(line)
                if len(errors) > 80:
                    errors.pop(0)
                if progress and line.startswith('out_time='):
                    value = line.partition('=')[2]
                    if re.fullmatch(r'\d+:\d+:\d+\.\d+', value):
                        h, m, s = value.split(':')
                        progress(min(.95, (int(h)*3600 + int(m)*60 + float(s))/estimated_duration),
                                 desc=('Đang tạo video mẫu…' if sample else
                                       'Đang xuất toàn bộ MP4 · tiến độ ước lượng…'))
            code = proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        finally:
            proc.stdout.close()
        if code != 0 or not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError('Không xuất được MP4. Kiểm tra video gốc và chỗ trống ổ đĩa.\n'
                               + '\n'.join(errors[-12:])[-1400:])
        partial.rename(output)
        if progress:
            progress(1, desc='Đã xuất video.')
        return str(output)
    finally:
        partial.unlink(missing_ok=True)
