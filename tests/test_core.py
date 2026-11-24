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
        self.rows = [
            dict(start=0.2, end=2.1, source="Don't give up.", vi='Đừng bỏ cuộc.'),
            dict(start=2.2, end=4.9, source='Keep going.', vi='Hãy tiếp tục.'),
        ]

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
        self.assertEqual(rows[0]['vi'], '')

    def test_bilingual_srt_auto_detects_both_line_orders(self):
        vi_top = (
            '1\n00:00:00,200 --> 00:00:02,100\n'
            'Có hai thời điểm cánh cửa tiềm thức mở ra.\n'
            "There's two times when the door to the subconscious mind opens up.\n\n"
            '2\n00:00:02,200 --> 00:00:04,000\n'
            'Khi bạn thức dậy vào buổi sáng.\n'
            'When you wake up in the morning.\n'
        )
        rows_vi_top = core.parse_srt(vi_top, mode='auto', source_lang='en', target_lang='vi')
        self.assertEqual(rows_vi_top[0]['source'], "There's two times when the door to the subconscious mind opens up.")
        self.assertEqual(rows_vi_top[0]['vi'], 'Có hai thời điểm cánh cửa tiềm thức mở ra.')
        self.assertEqual(rows_vi_top[1]['vi'], 'Khi bạn thức dậy vào buổi sáng.')

        en_top = (
            '1\n00:00:00,200 --> 00:00:02,100\n'
            '{\\an8}When you wake up in the morning,\n'
            'Khi bạn thức dậy vào buổi sáng,\n\n'
            '2\n00:00:02,200 --> 00:00:03,000\n'
            'Okay.\n'
            'Okay.\n'
        )
        rows_en_top = core.parse_srt(en_top, mode='auto', source_lang='en', target_lang='vi')
        self.assertEqual(rows_en_top[0]['source'], 'When you wake up in the morning,')
        self.assertEqual(rows_en_top[0]['vi'], 'Khi bạn thức dậy vào buổi sáng,')
        self.assertEqual(rows_en_top[1]['source'], 'Okay.')
        self.assertEqual(rows_en_top[1]['vi'], 'Okay.')

    def test_compact_segments_merges_orphans_and_splits_long_cues(self):
        messy = [
            dict(start=0.0, end=3.8,
                 source='And what separates the conscious mind from the subconscious mind is the analytical',
                 vi=''),
            dict(start=3.8, end=4.2, source='mind.', vi=''),
        ]
        cleaned = core.compact_segments(messy, max_chars=44, max_duration=3.2)
        self.assertGreaterEqual(len(cleaned), 2)
        self.assertNotEqual(cleaned[-1]['source'], 'mind.')
        self.assertTrue(cleaned[-1]['source'].endswith('analytical mind.'))

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

    def test_ass_tiktok_compact_middle_shadow_and_dual_order(self):
        data = core.styled_ass(
            self.rows[:1], 'dual', font='Segoe UI', size=44,
            color='#FFFFFF', secondary_color='#FDE68A',
            backdrop=True, backdrop_style='panel',
            position='middle', margin=220,
            text_width=55, max_chars=18, max_lines=1,
            outline=2, outline_color='#112233', shadow=2.5,
            italic=True, letter_spacing=-0.5, box_padding=10,
            dual_gap=14, dual_order='source_first',
        )
        self.assertIn('&H00332211&', data)
        self.assertIn(',-1,-1,0,0,100,100,-0.5,0,1,2,2.5,5,', data)
        self.assertLess(data.index("Don't give up."), data.index('Đừng bỏ cuộc.'))

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

    def test_yellow_rounded_panel_and_unboxed_secondary_legibility(self):
        for font in ('Arial Narrow', 'Bahnschrift', 'Times New Roman', 'Consolas'):
            with self.subTest(font=font):
                self.assertGreater(core.glyph_width(font, 32, True, 'Tiếng Việt'), 0)
        data = core.styled_ass(self.rows[:1], 'dual', font='Arial Narrow',
                               color='#111111', secondary_color='#FFFFFF',
                               backdrop=True, backdrop_style='panel',
                               backdrop_color='#FFD800', backdrop_opacity=100,
                               backdrop_radius=28, backdrop_scope='translation',
                               outline=0)
        self.assertIn(r'\1c&H00D8FF&\1a&H00&', data)
        self.assertIn(r'\bord2\3c&H141414&\3a&H00&\4c&H0D0D0D&\4a&H66&\shad1', data)
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

    def test_translation_resume_preserve_ids_and_upcoming_context(self):
        rows = [
            dict(start=0, end=1, source='First part of', vi='Phần đầu của'),
            dict(start=1, end=2, source='a connected sentence', vi=''),
            dict(start=2, end=3, source='that finishes smoothly.', vi='kết thúc mượt mà.'),
        ]
        p = core.new_project('demo.mp4', rows)
        captured_payloads = []

        def fake(endpoint, payload):
            user_data = json.loads(payload['messages'][-1]['content'])
            captured_payloads.append(user_data)
            return {'message': {'content': json.dumps({'translations': [{'id': 2, 'vi': 'một câu liên kết'}]})}}

        core.translate_project(p, request=fake, style='inspirational')
        self.assertEqual([x['id'] for x in captured_payloads[0]['items']], [2])
        self.assertEqual(captured_payloads[0]['previous_context'][0]['vi'], 'Phần đầu của')
        self.assertEqual(captured_payloads[0]['upcoming_context'][0]['text'], 'that finishes smoothly.')
        self.assertEqual(core.load_project(p['job'])['segments'][1]['vi'], 'một câu liên kết')

    def test_wrong_ids_rejected(self):
        p = core.new_project('demo', self.rows)
        fake = lambda *_: {'message': {'content': '{"translations":[{"id":99,"vi":"Sai"}]}'}}
        with self.assertRaises(RuntimeError):
            core.translate_batch(p['segments'][:1], [], request=fake)

    def test_partial_progress_saved_on_api_failure(self):
        rows = [dict(start=i, end=i+.9, source=f'Line {i}', vi='') for i in range(14)]
        p = core.new_project('long', rows)

        def fake(endpoint, payload):
            items = json.loads(payload['messages'][-1]['content'])['items']
            if items[0]['id'] > 10:
                raise RuntimeError('Disconnected')
            return {'message': {'content': json.dumps({'translations': [
                {'id': x['id'], 'vi': 'Dòng dịch'} for x in items]})}}

        with self.assertRaises(RuntimeError):
            core.translate_project(p, request=fake)
        saved = core.load_project(p['job'])
        self.assertEqual(sum(bool(s['vi']) for s in saved['segments']), 10)


if __name__ == '__main__':
    unittest.main()
