import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import app
import core


class AppTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output_patch = patch.object(core, 'OUTPUT', Path(self.tmp.name) / 'output')
        self.output_patch.start()
        self.addCleanup(self.output_patch.stop)
        self.srt = Path(self.tmp.name) / 'sample.srt'
        self.srt.write_text('1\n00:00:00,000 --> 00:00:01,000\nXin chào.\n', encoding='utf-8')

    def test_selected_srt_source_language_is_kept_in_project_and_export(self):
        for language in ('vi', 'zh'):
            with self.subTest(language=language):
                project, _, files, _, _, _, _, _, seek = app.source_job(
                    None, str(self.srt), 'small', 'cpu', language, 'en')
                self.assertEqual(project['language'], language)
                self.assertGreater(seek['value'], 0)
                self.assertEqual(core.load_project(project['job'])['language'], language)
                with zipfile.ZipFile(files[0]) as archive:
                    self.assertIn(f'sample.{language}.srt', archive.namelist())

    def test_export_controls_reach_video_renderer(self):
        project = core.new_project('sample.mp4', [dict(
            start=0, end=1, source='Hello', vi='Xin chào')])
        with patch.object(app, 'render_video', return_value='sample.mp4') as render:
            app.render_job(project, core.table_rows(project), None, 'Bản dịch',
                           'Arial', 46, '#FFFFFF', '#FFF2A8', '#FFD800', 100,
                           28, 'translation', 'panel',
                           'Dưới', 65, '720p', '9:16', 'crop',
                           'left', 86, 'balanced', True, 3, 22, 74, '00:00:00', 0,
                           False, True, 2.5, 52, True)
        self.assertEqual(render.call_args.kwargs['aspect'], '9:16')
        self.assertEqual(render.call_args.kwargs['frame_fill'], 'crop')
        self.assertEqual(render.call_args.kwargs['text_align'], 'left')
        self.assertEqual(render.call_args.kwargs['text_width'], 86)
        self.assertEqual(render.call_args.kwargs['backdrop_style'], 'panel')
        self.assertEqual(render.call_args.kwargs['backdrop_color'], '#FFD800')
        self.assertEqual(render.call_args.kwargs['backdrop_scope'], 'translation')

    def test_delay_setting_is_saved_and_updates_downloads(self):
        project = core.new_project('sample.mp4', [dict(
            start=.2, end=1.2, source='Hello', vi='Xin chào')])
        updated, files, _ = app.set_delay_job(project, core.table_rows(project), 1.2)
        self.assertEqual(core.load_project(project['job'])['subtitle_delay'], 1.2)
        with zipfile.ZipFile(files[0]) as archive:
            content = archive.read('sample.vi.srt').decode('utf-8-sig')
            original = archive.read('sample.source-original.srt').decode('utf-8-sig')
        self.assertIn('00:00:01,400 --> 00:00:02,400', content)
        self.assertIn('00:00:00,200 --> 00:00:01,200', original)
        self.assertEqual(updated['subtitle_delay'], 1.2)

    def test_jump_between_subtitle_cues(self):
        project = core.new_project('sample.mp4', [
            dict(start=1, end=2, source='One', vi='Một'),
            dict(start=5, end=6, source='Two', vi='Hai')])
        self.assertEqual(app.jump_subtitle(project, 2, .5, 1), 6)
        self.assertEqual(app.jump_subtitle(project, 6, .5, -1), 2)

    def test_live_frame_preview_uses_selected_zero_time_without_saving(self):
        project = core.new_project('sample.mp4', [dict(
            start=1, end=2, source='Hello', vi='Xin chào')])
        rows = core.table_rows(project)
        rows[0][4] = 'Chào bạn'
        controls = (project, rows, None, 'Bản dịch',
                    'Arial', 46, '#FFFFFF', '#FFF2A8', '#000000', 70,
                    12, 'all', 'panel',
                    'Dưới', 65, '720p', '9:16', 'crop',
                    'center', 90, 'balanced', True, 3, 22, 74, 0, .5,
                    False, True, 2.5, 52)
        with patch.object(app, 'render_video', return_value='frame.png') as render:
            frame, message = app.preview_frame_job(*controls)
        self.assertEqual(frame, 'frame.png')
        self.assertIn('Mốc này chưa có câu phụ đề', message)
        self.assertIn('Bảng có sửa chưa lưu', message)
        self.assertTrue(render.call_args.kwargs['frame_only'])
        self.assertEqual(render.call_args.args[10], 0)
        self.assertEqual(core.load_project(project['job'])['subtitle_delay'], 0)

    def test_yellow_preset_sets_readable_contrast(self):
        preset = app.apply_style_preset('Đen trên nền vàng')
        self.assertEqual(preset[:3], ('#111111', '#FFFFFF', '#FFD800'))
        self.assertEqual(preset[5:7], ('panel', 'translation'))
        self.assertEqual(preset[8], 'Arial Narrow')


if __name__ == '__main__':
    unittest.main()
