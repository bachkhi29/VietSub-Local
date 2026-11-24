import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import core
from render import frame_size, render_video


class RenderTests(unittest.TestCase):
    def test_frame_sizes_for_export_formats(self):
        expected = {'16:9': (1280, 720), '4:3': (960, 720),
                    '1:1': (720, 720), '4:5': (720, 900),
                    '9:16': (720, 1280)}
        for aspect, size in expected.items():
            with self.subTest(aspect=aspect):
                self.assertEqual(frame_size((1920, 1080), aspect, '720p'), size)
        self.assertEqual(frame_size((640, 360), '9:16', 'Gốc'), (360, 640))

    def test_reframed_sample_and_full_video(self):
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as temp, patch.object(core, 'OUTPUT', Path(temp)):
            source = Path(temp) / 'sample.mp4'
            subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-f', 'lavfi',
                            '-i', 'color=black:s=640x360:r=12:d=1', '-c:v', 'libx264',
                            str(source)], check=True, capture_output=True)
            project = core.new_project('sample.mp4', [dict(
                start=0, end=.9, source='Hello', vi='Xin chào')])
            for aspect, fill, resolution, expected, script_width, sample in [
                ('9:16', 'fit', '720p', '720x1280', 608, True),
                ('1:1', 'crop', 'Gốc', '360x360', 1080, False),
            ]:
                with self.subTest(aspect=aspect, fill=fill):
                    output = Path(render_video(project, source, mode='vi', sample=sample,
                                               resolution=resolution, aspect=aspect,
                                               frame_fill=fill))
                    info = subprocess.run([ffmpeg, '-hide_banner', '-i', str(output)],
                                          capture_output=True, text=True).stderr
                    self.assertIn(expected, info)
                    self.assertIn(f'PlayResX: {script_width}',
                                  (output.parent / 'overlay.ass').read_text(encoding='utf-8-sig'))

    def test_exported_position_matches_selected_side(self):
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as temp, patch.object(core, 'OUTPUT', Path(temp)):
            source = Path(temp) / 'black.mp4'
            subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-f', 'lavfi',
                            '-i', 'color=black:s=640x360:r=24:d=2', '-c:v', 'libx264',
                            str(source)], check=True, capture_output=True)
            project = core.new_project('black.mp4', [dict(
                start=0, end=2, source='HELLO FROM THE VIDEO', vi='XIN CHÀO MỌI NGƯỜI')])
            for position, expected in [('bottom', 1), ('top', 0)]:
                out = render_video(project, source, mode='dual', backdrop=True, position=position)
                frame = subprocess.run([ffmpeg, '-loglevel', 'error', '-ss', '1',
                                        '-i', out, '-frames:v', '1', '-vf',
                                        'scale=320:180,format=gray', '-f', 'rawvideo', '-'],
                                       check=True, capture_output=True).stdout
                halves = [sum(pixel > 130 for pixel in frame[i*28800:(i+1)*28800])
                          for i in range(2)]
                self.assertGreater(halves[expected], 50)
                self.assertLess(halves[1-expected], 10)

    def test_sample_mp4_has_audio_and_video(self):
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as temp, patch.object(core, 'OUTPUT', Path(temp)):
            source = Path(temp) / 'sample.mp4'
            subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                            '-f', 'lavfi', '-i', 'color=c=black:s=640x360:r=24:d=2',
                            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                            str(source)], check=True, capture_output=True)
            project = core.new_project('sample.mp4', [dict(
                start=.1, end=1.8, source='Hello', vi='Xin chào')], media=source)
            result = Path(render_video(project, source, sample=True, resolution='720p'))
            self.assertGreater(result.stat().st_size, 1000)
            info = subprocess.run([ffmpeg, '-hide_banner', '-i', str(result)],
                                  capture_output=True, text=True).stderr
            self.assertIn('Video: h264', info)
            self.assertIn('Audio: aac', info)
            self.assertIn('1280x720', info)

    def test_single_frame_preview_uses_tiktok_compact_middle_and_shadow(self):
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as temp, patch.object(core, 'OUTPUT', Path(temp)):
            source = Path(temp) / 'sample.mp4'
            subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-f', 'lavfi',
                            '-i', 'color=gray:s=640x360:r=12:d=1', '-c:v', 'libx264',
                            str(source)], check=True, capture_output=True)
            project = core.new_project('sample.mp4', [dict(
                start=.1, end=.9, source='When you wake up in the morning', vi='Khi bạn thức dậy vào buổi sáng')])
            project['subtitle_delay'] = .2
            frame = Path(render_video(
                project, source, mode='dual', aspect='9:16',
                resolution='720p', frame_only=True,
                sample_start=.5, backdrop=True,
                backdrop_style='panel', position='middle',
                text_width=70, max_chars=24, max_lines=2,
                box_padding=10, outline_color='#09090B', shadow=1.5,
            ))
            self.assertEqual(frame.suffix, '.png')
            self.assertGreater(frame.stat().st_size, 1000)
            info = subprocess.run([ffmpeg, '-hide_banner', '-i', str(frame)],
                                  capture_output=True, text=True).stderr
            self.assertIn('720x1280', info)

    def test_portrait_original_aspect_panel_and_lines_backdrop_colors(self):
        import imageio_ffmpeg
        from PIL import Image

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as temp, patch.object(core, 'OUTPUT', Path(temp)):
            # Portrait 720x1280 source with aspect='original' (the exact case that broke previously)
            source = Path(temp) / 'portrait.mp4'
            subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-f', 'lavfi',
                            '-i', 'color=black:s=720x1280:r=12:d=1', '-c:v', 'libx264',
                            str(source)], check=True, capture_output=True)
            project = core.new_project('portrait.mp4', [dict(
                start=0, end=1, source="There's two times", vi='Có hai thời điểm cánh cửa')])
            for style in ('panel', 'lines'):
                with self.subTest(backdrop_style=style):
                    frame = Path(render_video(
                        project, source, mode='dual', font='Arial Narrow', size=44,
                        color='#111111', secondary_color='#FFFFFF',
                        backdrop=True, backdrop_style=style,
                        backdrop_color='#FFD800', backdrop_opacity=100,
                        backdrop_radius=22, backdrop_scope='translation',
                        outline=0, aspect='original', frame_only=True, sample_start=0.4,
                    ))
                    img = Image.open(frame).convert('RGB')
                    self.assertEqual(img.size, (720, 1280))
                    # Verify wide yellow box spans across the portrait frame (not 31% width)
                    xs = [
                        x for y in range(900, 1270) for x in range(720)
                        if img.getpixel((x, y))[0] > 200 and img.getpixel((x, y))[1] > 180 and img.getpixel((x, y))[2] < 50
                    ]
                    self.assertGreater(len(xs), 1500)
                    self.assertGreater(max(xs) - min(xs), 340)


if __name__ == '__main__':
    unittest.main()
