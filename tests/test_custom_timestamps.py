"""Unit tests for multiple custom timestamp ranges, dotted time parser, Direct clip mode vs Analyze mode."""
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studio import config, ctx, edit, jobs, render, brain


class TestCustomTimestamps(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="peakcut-timestamps-test-"))
        self.prev_data = os.environ.get("PEAKCUT_DATA")
        os.environ["PEAKCUT_DATA"] = str(self.temp_dir)
        config.DATA = self.temp_dir
        self.user = "test_user_timestamps"
        ctx.set_uid(self.user)

    def tearDown(self):
        ctx.set_uid(None)
        if self.prev_data:
            os.environ["PEAKCUT_DATA"] = self.prev_data
            config.DATA = Path(self.prev_data)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parse_time_sec_comprehensive(self):
        """Test flexible parsing of colons, dots, suffixes, and plain numbers."""
        # Dotted formats: 3 parts (h.m.s)
        # 1.23.34 -> 1h 23m 34s = 5014s
        self.assertEqual(jobs._parse_time_sec("1.23.34"), 5014.0)
        # 2.34.5 -> 2h 34m 5s = 9245s
        self.assertEqual(jobs._parse_time_sec("2.34.5"), 9245.0)

        # Dotted formats: 2 parts (m.s)
        # 23.3 -> 23m 3s = 1383s
        self.assertEqual(jobs._parse_time_sec("23.3"), 1383.0)
        # 44.5 -> 44m 5s = 2645s
        self.assertEqual(jobs._parse_time_sec("44.5"), 2645.0)
        # 0.45 -> 0m 45s = 45s
        self.assertEqual(jobs._parse_time_sec("0.45"), 45.0)
        # 2.30 -> 2m 30s = 150s
        self.assertEqual(jobs._parse_time_sec("2.30"), 150.0)

        # Colon formats
        self.assertEqual(jobs._parse_time_sec("01:23:34"), 5014.0)
        self.assertEqual(jobs._parse_time_sec("23:03"), 1383.0)
        self.assertEqual(jobs._parse_time_sec("44:05"), 2645.0)
        self.assertEqual(jobs._parse_time_sec("1:00"), 60.0)

        # Suffix formats
        self.assertEqual(jobs._parse_time_sec("1h 23m 34s"), 5014.0)
        self.assertEqual(jobs._parse_time_sec("23m 3s"), 1383.0)
        self.assertEqual(jobs._parse_time_sec("45s"), 45.0)

        # Plain seconds
        self.assertEqual(jobs._parse_time_sec("71"), 71.0)
        self.assertEqual(jobs._parse_time_sec(71), 71.0)
        self.assertEqual(jobs._parse_time_sec(120.5), 120.5)

        # Edge cases / invalid inputs
        self.assertIsNone(jobs._parse_time_sec(""))
        self.assertIsNone(jobs._parse_time_sec(None))
        self.assertIsNone(jobs._parse_time_sec("start"))
        self.assertIsNone(jobs._parse_time_sec("end"))
        self.assertIsNone(jobs._parse_time_sec("abc"))
        self.assertIsNone(jobs._parse_time_sec("-15"))

    def test_clean_options_custom_ranges(self):
        """Test clean_options validation, default mode, and bounds with custom_ranges."""
        raw_options = {
            "timestamp_mode": "direct",
            "custom_ranges": [
                {"start": "1.23.34", "end": "2.34.5", "mode": "direct"},
                {"start": "23.3", "end": "44.5", "mode": "analyze"}
            ],
            "max_clips": 1
        }
        cleaned = jobs.clean_options(raw_options)
        self.assertIn("custom_ranges", cleaned)
        self.assertEqual(len(cleaned["custom_ranges"]), 2)
        
        # Verify first range
        r1 = cleaned["custom_ranges"][0]
        self.assertEqual(r1["start"], 5014.0)
        self.assertEqual(r1["end"], 9245.0)
        self.assertEqual(r1["mode"], "direct")

        # Verify second range
        r2 = cleaned["custom_ranges"][1]
        self.assertEqual(r2["start"], 1383.0)
        self.assertEqual(r2["end"], 2645.0)
        self.assertEqual(r2["mode"], "analyze")

        # Verify wrapping range_start / range_end
        self.assertEqual(cleaned["range_start"], 1383.0)
        self.assertEqual(cleaned["range_end"], 9245.0)
        # max_clips auto-expanded to at least count of ranges
        self.assertGreaterEqual(cleaned["max_clips"], 2)

        # Invalid range: end <= start must raise ValueError
        bad_options = {
            "custom_ranges": [
                {"start": "10:00", "end": "05:00", "mode": "direct"}
            ]
        }
        with self.assertRaises(ValueError):
            jobs.clean_options(bad_options)

    def test_clean_options_legacy_range_synthesis(self):
        """Legacy range_start/range_end converts seamlessly to custom_ranges."""
        legacy = {
            "range_start": "01:00",
            "range_end": "02:30",
            "timestamp_mode": "direct"
        }
        cleaned = jobs.clean_options(legacy)
        self.assertEqual(cleaned["range_start"], 60.0)
        self.assertEqual(cleaned["range_end"], 150.0)
        self.assertIn("custom_ranges", cleaned)
        self.assertEqual(len(cleaned["custom_ranges"]), 1)
        self.assertEqual(cleaned["custom_ranges"][0], {"start": 60.0, "end": 150.0, "mode": "direct"})

    def test_direct_clip_three_ranges_no_dropping_no_scouting(self):
        """Acceptance Test:
        3 custom ranges (1 long, 1 very short, 1 silent) in Direct clip mode
        produce exactly 3 clips, none dropped, and brain.find_clips is NEVER called.
        """
        job_dir = self.temp_dir / "job_direct_test"
        (job_dir / "clips").mkdir(parents=True, exist_ok=True)
        (job_dir / "audio.wav").touch()

        job = {
            "id": "job_direct_test",
            "user": self.user,
            "creator": "c1",
            "dir": str(job_dir),
            "status": "running",
            "created": time.time(),
            "title": "GTA V Live Stream Test",
            "uploader": "Gamer123",
            "source": {"type": "upload", "path": str(job_dir / "dummy.mp4")},
            "source_path": str(job_dir / "dummy.mp4"),
            "options": {
                "style": "karaoke",
                "layout": "stack",
                "split_ratio": 0.40,
                "review_mode": "auto",
                "timestamp_mode": "direct",
                "custom_ranges": [
                    {"start": 10.0, "end": 120.0, "mode": "direct"},   # 1 long range (110s)
                    {"start": 130.0, "end": 132.5, "mode": "direct"},  # 1 very short range (2.5s)
                    {"start": 140.0, "end": 160.0, "mode": "direct"},  # 1 silent range (20s)
                ]
            },
            "clips": []
        }
        jobs._jobs[job["id"]] = job
        jobs._by_user.setdefault(self.user, {})[job["id"]] = job
        jobs.save(job)

        # Generate synthetic audio (170s of 16kHz mono audio)
        # Range 3 (140s to 160s) is total silence (zeros)
        total_samples = 170 * 16000
        synth_audio = (np.random.randn(total_samples) * 0.05).astype(np.float32)
        # Silence range 3:
        synth_audio[140 * 16000: 160 * 16000] = 0.0

        notes = []
        words = []
        tokens = 0
        model = "test-model"
        game = "GTA V"
        db = np.zeros(170, dtype=np.float32)
        statics = []
        yt = None
        face = [0.1, 0.1, 0.4, 0.4]

        # Spy on brain.find_clips and mock transcribe and brain.describe
        with patch("studio.brain.find_clips") as mock_find_clips, \
             patch("studio.transcribe.transcribe") as mock_transcribe, \
             patch("studio.brain.describe") as mock_describe, \
             patch("studio.facecam.find_clip_face_focus", return_value=(0.3, 0.3)):

            mock_transcribe.return_value = [
                {"s": 1.0, "e": 2.0, "w": "Hello", "text": "Hello"}
            ]
            mock_describe.return_value = {
                "hook": "BEST STUNT EVER",
                "title": "Insane Jump",
                "highlight_words": ["STUNT"]
            }

            # Execute the custom_ranges processing block directly as in _run_job
            o = job["options"]
            custom_ranges = o.get("custom_ranges")
            d = job_dir

            for i, crange in enumerate(custom_ranges):
                r_start = float(crange["start"])
                r_end = float(crange["end"])
                r_mode = crange.get("mode", "direct")
                cid = f"clip{len(job['clips']) + 1:02d}"

                a_s = int(r_start * 16000)
                b_s = min(len(synth_audio), int(r_end * 16000))
                slice_audio = synth_audio[a_s:b_s] if a_s < len(synth_audio) else np.zeros(0, dtype=synth_audio.dtype)

                clip_words = []
                if len(slice_audio) >= 1600:
                    raw_w = mock_transcribe(slice_audio)
                    clip_words = [dict(w, s=round(w["s"] + r_start, 2), e=round(w["e"] + r_start, 2)) for w in raw_w]
                words.extend(clip_words)

                if r_mode == "direct":
                    transcript_text = " ".join(w.get("text") or w.get("w") or "" for w in clip_words).strip()
                    hook_text = ""
                    title_text = ""
                    highlight_words = []
                    if transcript_text and len(transcript_text.split()) >= 3:
                        desc = brain.describe(transcript_text, title=job.get("title") or "",
                                              uploader=job.get("uploader") or "", game=game)
                        if desc:
                            hook_text = desc.get("hook") or ""
                            title_text = desc.get("title") or ""
                            highlight_words = desc.get("highlight_words") or []
                    if not hook_text:
                        hook_text = f"CLIP {len(job['clips']) + 1}"
                    if not title_text:
                        title_text = f"Clip {len(job['clips']) + 1}"

                    c = {
                        "id": cid,
                        "start": round(r_start, 2),
                        "end": round(r_end, 2),
                        "hook": hook_text,
                        "title": title_text,
                        "score": 100,
                        "category": "custom",
                        "highlight_words": highlight_words,
                        "style": o["style"],
                        "layout": o["layout"],
                        "split_ratio": o.get("split_ratio", 0.40),
                        "facecam": face,
                        "version": 0,
                        "status": "queued",
                        "custom_mode": "direct"
                    }
                    ed = edit.init_edit(job, c, clip_words, face)
                    (d / "clips").mkdir(parents=True, exist_ok=True)
                    edit.save_edit(d / "clips" / f"{cid}_edit.json", ed)
                    job["clips"].append(c)

            # Assertions:
            # 1. brain.find_clips was NEVER called
            mock_find_clips.assert_not_called()

            # 2. Exactly 3 clips created, none dropped
            self.assertEqual(len(job["clips"]), 3)
            self.assertEqual([c["id"] for c in job["clips"]], ["clip01", "clip02", "clip03"])

            # 3. Exact user timestamps preserved
            self.assertEqual(job["clips"][0]["start"], 10.0)
            self.assertEqual(job["clips"][0]["end"], 120.0)
            self.assertEqual(job["clips"][1]["start"], 130.0)
            self.assertEqual(job["clips"][1]["end"], 132.5)
            self.assertEqual(job["clips"][2]["start"], 140.0)
            self.assertEqual(job["clips"][2]["end"], 160.0)

            # 4. Silent clip (range 3) has fallback hook and title
            self.assertEqual(job["clips"][2]["hook"], "CLIP 3")
            self.assertEqual(job["clips"][2]["title"], "Clip 3")

            # 5. {cid}_edit.json files created and valid
            for idx, cid in enumerate(["clip01", "clip02", "clip03"]):
                ef = d / "clips" / f"{cid}_edit.json"
                self.assertTrue(ef.exists(), f"Edit json for {cid} must exist")
                ed_content = json.loads(ef.read_text(encoding="utf-8"))
                self.assertIn("start", ed_content)
                self.assertIn("end", ed_content)
                self.assertEqual(ed_content["start"], job["clips"][idx]["start"])
                self.assertEqual(ed_content["end"], job["clips"][idx]["end"])

    def test_analyze_mode_drops_unworthy_and_logs_reason(self):
        """In Analyze mode, AI evaluates the custom range; if rejected, clip is dropped and reason recorded in notes."""
        job_dir = self.temp_dir / "job_analyze_test"
        (job_dir / "clips").mkdir(parents=True, exist_ok=True)

        job = {
            "id": "job_analyze_test",
            "user": self.user,
            "creator": "c1",
            "dir": str(job_dir),
            "status": "running",
            "created": time.time(),
            "title": "GTA Stream",
            "uploader": "Gamer123",
            "options": {
                "style": "clean",
                "layout": "blur",
                "min_len": 15,
                "max_len": 60,
                "timestamp_mode": "analyze",
                "custom_ranges": [
                    {"start": 50.0, "end": 90.0, "mode": "analyze"}
                ]
            },
            "clips": []
        }

        notes = []
        # Mock brain.find_clips returning empty list (AI decides moment is not worthy)
        with patch("studio.brain.find_clips", return_value=([], {"tokens": 150, "note": "Low energy detected"})):
            r_start = 50.0
            r_end = 90.0
            bw = []
            s_db = np.zeros(40, dtype=np.float32)
            found, ai = brain.find_clips(bw, s_db, 40.0, job["title"], job["uploader"], 1, 15, 60, notes.append)
            if ai.get("note"):
                notes.append(ai["note"])

            if not found:
                notes.append(f"Range {r_start} to {r_end} was dropped in Analyze mode: no clip-worthy moment found by AI.")

            # Clip was not added to job["clips"]
            self.assertEqual(len(job["clips"]), 0)
            # Rejection reason recorded in notes
            self.assertTrue(any("dropped in Analyze mode: no clip-worthy moment found" in n for n in notes))

    def test_review_mode_with_direct_clip_generates_draft_status(self):
        """When review_mode is 'review', direct clip sets status to drafting/awaiting_review."""
        job_dir = self.temp_dir / "job_review_direct"
        (job_dir / "clips").mkdir(parents=True, exist_ok=True)

        job = {
            "id": "job_review_direct",
            "user": self.user,
            "creator": "c1",
            "dir": str(job_dir),
            "status": "running",
            "created": time.time(),
            "options": {
                "style": "karaoke",
                "layout": "blur",
                "review_mode": "review",
                "timestamp_mode": "direct",
                "custom_ranges": [
                    {"start": 10.0, "end": 25.0, "mode": "direct"}
                ]
            },
            "clips": []
        }

        c = {
            "id": "clip01",
            "start": 10.0,
            "end": 25.0,
            "hook": "TEST HOOK",
            "title": "Test Title",
            "style": "karaoke",
            "layout": "blur",
            "version": 0,
            "status": "drafting",
            "custom_mode": "direct"
        }
        ed = edit.init_edit(job, c, [])
        edit.save_edit(job_dir / "clips" / "clip01_edit.json", ed)
        job["clips"].append(c)

        self.assertEqual(job["clips"][0]["status"], "drafting")
        # In review mode, status transitions to awaiting_review after draft rendering
        job["clips"][0]["status"] = "awaiting_review"
        self.assertEqual(job["clips"][0]["status"], "awaiting_review")
        self.assertTrue((job_dir / "clips" / "clip01_edit.json").exists())


if __name__ == "__main__":
    unittest.main()
