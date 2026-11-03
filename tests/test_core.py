import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import core


class SubtitleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = patch.object(core, 'OUTPUT', Path(self.tmp.name))
        self.patch.start()
        self.rows = [dict(start=0.2, end=2.1, source="Don't give up.", vi='Đừng bỏ cuộc.'),
                     dict(start=2.2, end=4.9, source='Keep going.', vi='Hãy tiếp tục.')]

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_timing_roundtrip(self):
        for t in [0, .001, 59.9996, 3599.9996, 360000.234]:
            self.assertAlmostEqual(core.parse_time(core.timestamp(t)), t, places=3)
        self.assertEqual(core.timestamp(59.9996), '00:01:00,000')

    def test_bom_crlf_and_multiline(self):
        data = '\ufeff1\r\n00:00:00,200 --> 00:00:02,100\r\nHello\r\nworld.\r\n'
        rows = core.parse_srt(data)
        self.assertEqual(rows[0]['source'], 'Hello world.')

    def test_bad_timing_rejected(self):
        for start, end in [(-1, 1), (1, 1), (float('nan'), 2), (0, float('inf'))]:
            with self.assertRaises(ValueError):
                core.validate_segments([dict(start=start, end=end, source='x')])

    def test_preview_vtt(self):
        result = core.webvtt(self.rows, 'dual')
        self.assertTrue(result.startswith('WEBVTT\n\n'))
        lines = [line for line in result.splitlines() if '-->' in line]
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(line.count('-->') == 1 for line in lines))
        self.assertIn('00:00:00.200 --> 00:00:02.100', lines)
        self.assertIn("Đừng bỏ cuộc.\nDon't give up.", result)

    def test_subtitle_delay_shifts_and_clips_export_times(self):
        delayed = core.srt(self.rows, 'vi', delay=1.25)
        self.assertIn('00:00:01,450 --> 00:00:03,350', delayed)
        early = core.webvtt(self.rows, 'vi', delay=-1)
        self.assertIn('00:00:00.000 --> 00:00:01.100', early)
        self.assertIn('00:00:01.200 --> 00:00:03.900', early)
        skipped = core.srt([dict(start=0, end=.5, source='Gone', vi='Mất'),
                            self.rows[1]], 'vi', delay=-1)
        self.assertNotIn('Mất', skipped)
        self.assertTrue(skipped.startswith('1\n00:00:01,200'))
        ass = core.styled_ass(self.rows, 'vi', delay=1.25)
        self.assertIn('Dialogue: 0,0:00:01.45', ass)

    def test_bad_subtitle_delay_rejected(self):
        for value in (float('nan'), 31, -31, 'not a number'):
            with self.assertRaises(ValueError):
                core.subtitle_delay(value)

    def test_ass_style_and_clip_shift(self):
        data = core.styled_ass(self.rows, 'vi', font='Tahoma', size=50,
                               color='#FFFF00', backdrop=True, position='top',
                               margin=80, clip_start=1, clip_duration=3)
        self.assertIn('Style: Default,Tahoma,50,&H0000FFFF&', data)
        self.assertIn(',3,2.5,0,8,96,96,80,1', data)
        self.assertIn('Dialogue: 0,0:00:00.00,0:00:01.10', data)
        self.assertNotIn("Don't give up.", data)
        self.assertIn('Hãy tiếp tục.', data)

    def test_ass_escapes_model_sequences(self):
        row = dict(start=0, end=2, source='Hey', vi=r'{\pos(0,0)} Xin')
        data = core.styled_ass([row], mode='vi')
        self.assertNotIn(r'{\pos(0,0)}', data)

    def test_ass_uppercase_only_translated_line(self):
        data = core.styled_ass(self.rows[:1], 'dual', secondary_color='#FFF2A8', uppercase=True)
        self.assertIn('ĐỪNG BỎ CUỘC.', data)
        self.assertIn("Don't give up.", data)
        self.assertIn(r'\c&H00A8F2FF&}', data)

    def test_long_portrait_caption_shrinks_and_balances_lines(self):
        value = ('Đây là một câu khá dài với nhiều phần khác nhau và cần được '
                 'sắp xếp cân đối giữa các dòng phụ đề.')
        size, lines = core.fitted_lines(value, 'Segoe UI', 46, True, 505, 52,
                                        True, True, 3, 22)
        self.assertLess(size, 46)
        self.assertLessEqual(len(lines), 3)
        self.assertEqual(' '.join(lines), value)
        compact = core.subtitle_lines(value, 'Segoe UI', 37, True, 505, 52, False)
        balanced = core.subtitle_lines(value, 'Segoe UI', 37, True, 505, 52, True)
        face = core.measure_font('Segoe UI', 37, True)
        spread = lambda rows: max(face.getlength(row) for row in rows) - min(
            face.getlength(row) for row in rows)
        self.assertLess(spread(balanced), spread(compact))

    def test_portrait_layout_controls_ass_alignment_and_margins(self):
        data = core.styled_ass(self.rows[:1], 'vi', script_width=608,
                               text_align='left', text_width=90)
        self.assertIn('PlayResX: 608', data)
        self.assertIn(',1,30,30,65,1', data)

    def test_panel_backdrop_is_one_layer_behind_caption(self):
        data = core.styled_ass(self.rows[:1], 'dual', script_width=608,
                               backdrop=True, backdrop_style='panel')
        self.assertEqual(data.count('Dialogue: '), 2)
        self.assertIn(r'\p1', data)
        self.assertIn('Dialogue: 1,', data)

    def test_yellow_rounded_panel_and_added_fonts(self):
        for font in ('Arial Narrow', 'Bahnschrift',
                     'Times New Roman', 'Comic Sans MS'):
            with self.subTest(font=font):
                self.assertGreater(core.measure_font(font, 32, True).getlength('Tiếng Việt'), 0)
        data = core.styled_ass(self.rows[:1], 'dual', font='Arial Narrow',
                               color='#111111', secondary_color='#FFFFFF',
                               backdrop=True, backdrop_style='panel',
                               backdrop_color='#FFD800', backdrop_opacity=100,
                               backdrop_radius=28, backdrop_scope='translation')
        self.assertIn(r'\1c&H00D8FF&\1a&H00&', data)
        self.assertIn(' b ', data)
        self.assertIn('Style: Default,Arial Narrow', data)

    def test_target_language_preserved_in_project_and_export(self):
        p = core.new_project('sample.mp4', self.rows, target_language='ja')
        self.assertEqual(core.load_project(p['job'])['target_language'], 'ja')
        files, _ = core.export_project(p)
        with zipfile.ZipFile(files[0]) as z:
            self.assertIn('sample.ja.srt', z.namelist())
            self.assertIn('sample.ja-en.srt', z.namelist())

    def test_export_unicode_and_same_timestamps(self):
        p = core.new_project('demo.mp4', self.rows)
        files, complete = core.export_project(p)
        self.assertTrue(complete)
        with zipfile.ZipFile(files[0]) as z:
            self.assertIn('demo.en.srt', z.namelist())
            self.assertIn('demo.vi.srt', z.namelist())
            en = core.parse_srt(z.read('demo.en.srt').decode('utf-8-sig'))
            vi = core.parse_srt(z.read('demo.vi.srt').decode('utf-8-sig'))
            self.assertEqual([x['start'] for x in en], [x['start'] for x in vi])
            self.assertEqual(vi[0]['source'], 'Đừng bỏ cuộc.')

    def test_incomplete_does_not_export_stale_translation(self):
        p = core.new_project('test.mp4', self.rows)
        core.export_project(p)
        p['segments'][0]['vi'] = ''
        files, complete = core.export_project(p)
        self.assertFalse(complete)
        with zipfile.ZipFile(files[0]) as z:
            self.assertNotIn('test.vi.srt', z.namelist())

    def test_source_edit_invalidates_unchanged_translation(self):
        p = core.new_project('demo.mp4', self.rows)
        table = core.table_rows(p)
        table[0][3] = 'Give up.'
        updated = core.rows_to_segments(table, p['segments'])
        self.assertEqual(updated[0]['vi'], '')
        self.assertEqual(updated[1]['vi'], 'Hãy tiếp tục.')

    def test_translation_resume_and_preserve_ids(self):
        p = core.new_project('demo.mp4', self.rows)
        p['segments'][1]['vi'] = ''
        seen = []
        def fake(endpoint, payload):
            items = json.loads(payload['messages'][-1]['content'])['items']
            seen.extend(x['id'] for x in items)
            return {'message': {'content': json.dumps({'translations': [{'id': 2, 'vi': 'Tiếp tục.'}]})}}
        core.translate_project(p, request=fake)
        self.assertEqual(seen, [2])
        self.assertEqual(core.load_project(p['job'])['segments'][1]['vi'], 'Tiếp tục.')

    def test_wrong_ids_rejected(self):
        p = core.new_project('demo', self.rows)
        fake = lambda *_: {'message': {'content': '{"translations":[{"id":99,"vi":"Sai"}]}'}}
        with self.assertRaises(RuntimeError):
            core.translate_batch(p['segments'][:1], [], request=fake)

    def test_partial_progress_saved_on_api_failure(self):
        rows = [dict(start=i, end=i+.9, source=f'Line {i}', vi='') for i in range(8)]
        p = core.new_project('long', rows)
        def fake(endpoint, payload):
            items = json.loads(payload['messages'][-1]['content'])['items']
            if items[0]['id'] > 6:
                raise RuntimeError('Disconnected')
            return {'message': {'content': json.dumps({'translations': [
                {'id': x['id'], 'vi': 'Dòng dịch'} for x in items]})}}
        with self.assertRaises(RuntimeError):
            core.translate_project(p, request=fake)
        saved = core.load_project(p['job'])
        self.assertEqual(sum(bool(s['vi']) for s in saved['segments']), 6)


if __name__ == '__main__':
    unittest.main()
