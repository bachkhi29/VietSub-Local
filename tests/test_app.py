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
                    None, str(self.srt), 'small', 'cpu', language, 'en', 'auto', 'balanced'
                )
                self.assertEqual(project['language'], language)
                self.assertGreater(seek['value'], 0)
                self.assertEqual(core.load_project(project['job'])['language'], language)
                with zipfile.ZipFile(files[0]) as archive:
                    self.assertIn(f'sample.{language}.srt', archive.namelist())

    def test_bilingual_srt_import_populates_both_columns_and_exports_dual(self):
        dual_srt = Path(self.tmp.name) / 'bilingual.srt'
        dual_srt.write_text(
            '1\n00:00:00,500 --> 00:00:02,500\n'
            'When you wake up in the morning,\n'
            'Khi bạn thức dậy vào buổi sáng,\n\n'
            '2\n00:00:02,600 --> 00:00:04,500\n'
            'your brain waves start to change.\n'
            'sóng não của bạn bắt đầu thay đổi.\n',
            encoding='utf-8',
        )
        project, rows, files, msg, _, _, _, _, _ = app.source_job(
            None, str(dual_srt), 'small', 'cpu', 'en', 'vi', 'auto', 'balanced'
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][3], 'When you wake up in the morning,')
        self.assertEqual(rows[0][4], 'Khi bạn thức dậy vào buổi sáng,')
        self.assertIn('song ngữ', msg)
        with zipfile.ZipFile(files[0]) as archive:
            self.assertIn('bilingual.vi.srt', archive.namelist())
            self.assertIn('bilingual.vi-en.srt', archive.namelist())

    def test_export_controls_reach_video_renderer(self):
        project = core.new_project('sample.mp4', [dict(
            start=0, end=1, source='Hello', vi='Xin chào')])
        with patch.object(app, 'render_video', return_value='sample.mp4') as render:
            _, _, out_path, video_preview, _ = app.render_job(
                project, core.table_rows(project), None, 'Bản dịch',
                'Arial', 46, '#FFFFFF', '#FFF2A8', '#FFD800', 100,
                28, 'translation', 'panel',
                'Giữa', 215, '720p', '9:16', 'crop',
                'left', 76, 'balanced', True, 2, 22, 74, '00:00:00', 0,
                False, True, 2.5, 28, False,
                outline_color='#112233', shadow=1.5, italic=True,
                letter_spacing=-0.5, box_padding=11, dual_gap=4,
                dual_order='source_first',
            )
        self.assertEqual(out_path, 'sample.mp4')
        self.assertIsNotNone(video_preview)
        self.assertEqual(render.call_args.args[7], 'middle')
        self.assertEqual(render.call_args.kwargs['aspect'], '9:16')
        self.assertEqual(render.call_args.kwargs['frame_fill'], 'crop')
        self.assertEqual(render.call_args.kwargs['text_align'], 'left')
        self.assertEqual(render.call_args.kwargs['text_width'], 76)
        self.assertEqual(render.call_args.kwargs['backdrop_style'], 'panel')
        self.assertEqual(render.call_args.kwargs['backdrop_color'], '#FFD800')
        self.assertEqual(render.call_args.kwargs['backdrop_scope'], 'translation')
        self.assertEqual(render.call_args.kwargs['outline_color'], '#112233')
        self.assertEqual(render.call_args.kwargs['box_padding'], 11)
        self.assertEqual(render.call_args.kwargs['dual_order'], 'source_first')

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

    def test_live_frame_preview_and_atomic_style_preset(self):
        project = core.new_project('sample.mp4', [dict(
            start=1, end=2, source='Hello', vi='Xin chào')])
        rows = core.table_rows(project)
        rows[0][4] = 'Chào bạn'
        controls = (
            project, rows, None, 'Song ngữ',
            'Segoe UI', 44, '#FFFFFF', '#FDE68A', '#0F172A', 78,
            14, 'all', 'panel',
            'Dưới', 80, '720p', '9:16', 'crop',
            'center', 82, 'balanced', True, 2, 20, 72, 0, .5,
            False, True, 1.5, 36,
            '#09090B', 0.0, False, 0.0, 12, 4, 'translation_first',
        )
        with patch.object(app, 'render_video', return_value='frame.png') as render:
            result = app.apply_preset_and_preview('Đen trên nền vàng', *controls)
        self.assertEqual(result[:3], ('#111111', '#FFFFFF', '#FFD800'))
        self.assertEqual(result[5:7], ('panel', 'translation'))
        self.assertEqual(result[8], 'Arial Narrow')
        self.assertEqual(result[-2], 'frame.png')
        self.assertEqual(render.call_args.args[3], 'Arial Narrow')
        self.assertEqual(render.call_args.args[4], 44)
        self.assertEqual(render.call_args.args[5], '#111111')
        self.assertEqual(render.call_args.kwargs['backdrop_color'], '#FFD800')
        self.assertEqual(render.call_args.kwargs['backdrop_scope'], 'translation')
        self.assertTrue(render.call_args.kwargs['frame_only'])

    def test_compact_and_quick_cue_edit_jobs(self):
        project = core.new_project('sample.mp4', [
            dict(start=0, end=3.5, source='This is the first long part of the analytical', vi=''),
            dict(start=3.5, end=3.9, source='mind.', vi=''),
        ])
        project, rows, _, _, _, _ = app.compact_cues_job(project, core.table_rows(project))
        self.assertNotEqual(rows[-1][3], 'mind.')
        project, rows, _, _ = app.update_single_cue_job(project, rows, 1, rows[0][3], 'Câu dịch mới')
        self.assertEqual(rows[0][4], 'Câu dịch mới')

    def test_gradio_app_builds_cleanly(self):
        import asyncio

        class DummyLoop(asyncio.AbstractEventLoop):
            pass

        with patch('asyncio.new_event_loop', side_effect=DummyLoop):
            demo = app.build_app()
        self.assertIsNotNone(demo)
        self.assertGreater(len(demo.fns), 15)


if __name__ == '__main__':
    unittest.main()
