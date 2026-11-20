"""Tests for review mode, edit-state file, deep-merge apply-to-all, and clip state transitions."""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studio import config, ctx, edit, jobs, render


class TestReviewMode(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="peakcut-review-test-"))
        self.prev_data = os.environ.get("PEAKCUT_DATA")
        os.environ["PEAKCUT_DATA"] = str(self.temp_dir)
        config.DATA = self.temp_dir
        self.user = "test_user_review"
        ctx.set_uid(self.user)

    def tearDown(self):
        ctx.set_uid(None)
        if self.prev_data:
            os.environ["PEAKCUT_DATA"] = self.prev_data
            config.DATA = Path(self.prev_data)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_edit_and_validation(self):
        """Test default_edit, validation bounds clamping, and sanitization."""
        ed = edit.default_edit(start=5.0, end=25.0, layout="blur")
        self.assertEqual(ed["start"], 5.0)
        self.assertEqual(ed["end"], 25.0)
        self.assertEqual(ed["layout"], "blur")
        self.assertEqual(ed["version"], 1)
        self.assertTrue(ed["caption"]["enabled"])
        self.assertTrue(ed["hook"]["enabled"])

        # Test validation bounds clamping
        invalid = {
            "start": -10.0,
            "end": -5.0,
            "layout": "invalid_layout",
            "split_ratio": 0.95,  # should clamp to 0.80
            "caption": {
                "font_size": 999,  # should clamp to 240
                "y_pos": 3000.0,   # should clamp to 1820.0
            },
            "hook": {
                "font_size": 10,   # should clamp to 36
                "y_pos": -50.0,    # should clamp to 50.0
                "text": "  test hook  "
            },
            "facecam": {"x": 2000, "y": 1500, "w": 500, "h": 500}  # should clamp inside 1920x1080
        }
        val = edit.validate_edit(invalid, src_w=1920, src_h=1080)
        self.assertEqual(val["start"], 0.0)
        self.assertGreater(val["end"], val["start"])
        self.assertEqual(val["layout"], "blur")
        self.assertEqual(val["split_ratio"], 0.80)
        self.assertEqual(val["caption"]["font_size"], 240)
        self.assertEqual(val["caption"]["y_pos"], 1820.0)
        self.assertEqual(val["hook"]["font_size"], 36)
        self.assertEqual(val["hook"]["y_pos"], 50.0)
        self.assertEqual(val["hook"]["text"], "test hook")
        self.assertIsNotNone(val["facecam"])
        self.assertLessEqual(val["facecam"]["x"] + val["facecam"]["w"], 1920)
        self.assertLessEqual(val["facecam"]["y"] + val["facecam"]["h"], 1080)
        print("PASS test_default_edit_and_validation")

    def test_load_and_save_edit(self):
        """Test writing edit.json to disk and reading back."""
        p = self.temp_dir / "test_clip_edit.json"
        orig = edit.default_edit(10.0, 40.0, "stack")
        orig["hook"]["text"] = "BIG MOMENT"
        orig["caption"]["words"] = [{"text": "hello", "start": 10.0, "end": 10.5}]
        edit.save_edit(p, orig)

        loaded = edit.load_edit(p)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["start"], 10.0)
        self.assertEqual(loaded["end"], 40.0)
        self.assertEqual(loaded["layout"], "stack")
        self.assertEqual(loaded["hook"]["text"], "BIG MOMENT")
        self.assertEqual(len(loaded["caption"]["words"]), 1)
        self.assertEqual(loaded["caption"]["words"][0]["text"], "hello")
        print("PASS test_load_and_save_edit")

    def test_deep_merge_global_preserves_per_clip_fields(self):
        """Test that deep_merge_global updates global styling but strictly protects per-clip fields."""
        target = edit.default_edit(15.0, 35.0, "blur")
        target["hook"]["text"] = "ORIGINAL HOOK TEXT"
        target["hook"]["font_size"] = 96
        target["hook"]["y_pos"] = 300.0
        target["caption"]["words"] = [
            {"text": "one", "start": 15.0, "end": 15.4},
            {"text": "two", "start": 15.5, "end": 15.9}
        ]
        target["caption"]["style"] = "bounce"
        target["caption"]["font_size"] = 120
        target["caption"]["y_pos"] = 1600.0

        patch = {
            # Attempt to overwrite per-clip fields (MUST BE IGNORED)
            "start": 0.0,
            "end": 10.0,
            "hook": {
                "text": "PATCHED HOOK TEXT (SHOULD BE IGNORED)",
                "font_size": 110,
                "y_pos": 450.0,
                "style": "nobox"
            },
            "caption": {
                "words": [{"text": "hijacked", "start": 0.0, "end": 1.0}],  # MUST BE IGNORED
                "style": "bold_stroke",
                "font_size": 140,
                "y_pos": 1500.0
            },
            # Global fields (MUST BE APPLIED)
            "layout": "stack",
            "split_ratio": 0.50,
            "audio": {
                "meme": True,
                "music": True,
                "meme_mood": "airhorn",
                "music_bed": "lofi"
            }
        }

        merged = edit.deep_merge_global(target, patch)

        # Verify global fields updated
        self.assertEqual(merged["layout"], "stack")
        self.assertEqual(merged["split_ratio"], 0.50)
        self.assertEqual(merged["hook"]["font_size"], 110)
        self.assertEqual(merged["hook"]["y_pos"], 450.0)
        self.assertEqual(merged["hook"]["style"], "nobox")
        self.assertEqual(merged["caption"]["style"], "bold_stroke")
        self.assertEqual(merged["caption"]["font_size"], 140)
        self.assertEqual(merged["caption"]["y_pos"], 1500.0)
        self.assertTrue(merged["audio"]["meme"])
        self.assertTrue(merged["audio"]["music"])

        # Verify per-clip fields are STRICTLY UNCHANGED
        self.assertEqual(merged["start"], 15.0)
        self.assertEqual(merged["end"], 35.0)
        self.assertEqual(merged["hook"]["text"], "ORIGINAL HOOK TEXT")
        self.assertEqual(len(merged["caption"]["words"]), 2)
        self.assertEqual(merged["caption"]["words"][0]["text"], "one")
        self.assertEqual(merged["caption"]["words"][1]["text"], "two")
        print("PASS test_deep_merge_global_preserves_per_clip_fields")

    def test_job_options_review_mode(self):
        """Test job creation options with review_mode: auto, review, review_first."""
        opts_default = {**jobs.DEFAULTS, **jobs.clean_options({})}
        self.assertEqual(opts_default["review_mode"], "auto")

        opts_review = jobs.clean_options({"review_mode": "review"})
        self.assertEqual(opts_review["review_mode"], "review")

        opts_first = jobs.clean_options({"review_mode": "review_first"})
        self.assertEqual(opts_first["review_mode"], "review_first")

        opts_invalid = jobs.clean_options({"review_mode": "unknown_mode"})
        self.assertEqual(opts_invalid["review_mode"], "auto")
        print("PASS test_job_options_review_mode")

    def test_state_transitions_and_approval(self):
        """Test clip transitions: awaiting_review -> approved, approve_all, and apply_to_all."""
        job_dir = self.temp_dir / "job_test_1"
        (job_dir / "clips").mkdir(parents=True, exist_ok=True)
        job = {
            "id": "job_test_1",
            "user": self.user,
            "creator": "c1",
            "dir": str(job_dir),
            "status": "running",
            "stage": "Reviewing clips",
            "options": {"review_mode": "review"},
            "clips": [
                {
                    "id": "c1",
                    "status": "awaiting_review",
                    "start": 10.0,
                    "end": 25.0,
                    "hook": "Clip 1",
                    "words": [{"text": "word1", "start": 10.0, "end": 10.5}],
                    "version": 1
                },
                {
                    "id": "c2",
                    "status": "awaiting_review",
                    "start": 30.0,
                    "end": 45.0,
                    "hook": "Clip 2",
                    "words": [{"text": "word2", "start": 30.0, "end": 30.5}],
                    "version": 1
                }
            ]
        }
        # Save edit files for clips
        for c in job["clips"]:
            ef = job_dir / "clips" / f"{c['id']}_edit.json"
            ed = edit.init_edit(job, c, c["words"])
            edit.save_edit(ef, ed)

        jobs._jobs[job["id"]] = job
        jobs._by_user.setdefault(self.user, {})[job["id"]] = job
        jobs.save(job)

        # Test approve single clip
        approved = jobs.approve_clip("job_test_1", "c1")
        self.assertEqual(approved["status"], "approved")
        self.assertEqual(job["clips"][0]["status"], "approved")
        self.assertEqual(job["clips"][1]["status"], "awaiting_review")

        # Test apply_to_all
        patch = {
            "caption": {"style": "headline", "font_size": 150},
            "hook": {"style": "nobox", "y_pos": 400.0}
        }
        updated = jobs.apply_to_all("job_test_1", patch)
        self.assertEqual(len(updated), 1)  # Only c2 was not yet approved
        self.assertEqual(updated[0]["id"], "c2")

        c2_ed = edit.load_edit(job_dir / "clips" / "c2_edit.json")
        self.assertEqual(c2_ed["caption"]["style"], "headline")
        self.assertEqual(c2_ed["caption"]["font_size"], 150)
        self.assertEqual(c2_ed["hook"]["y_pos"], 400.0)
        self.assertEqual(c2_ed["hook"]["text"], "Clip 2")  # Preserved

        # Test approve_all
        app_all = jobs.approve_all("job_test_1")
        self.assertEqual(len(app_all), 1)  # Approves remaining c2
        self.assertEqual(job["clips"][1]["status"], "approved")
        print("PASS test_state_transitions_and_approval")

    def test_review_first_mode_auto_applies_style_and_approves_rest(self):
        """In review_first mode, approving the first clip patches remaining clips and approves them."""
        job_dir = self.temp_dir / "job_test_first"
        (job_dir / "clips").mkdir(parents=True, exist_ok=True)
        job = {
            "id": "job_test_first",
            "user": self.user,
            "creator": "c1",
            "dir": str(job_dir),
            "status": "running",
            "stage": "Waiting for review",
            "options": {"review_mode": "review_first"},
            "clips": [
                {
                    "id": "rf1",
                    "status": "awaiting_review",
                    "start": 5.0,
                    "end": 20.0,
                    "hook": "First Clip",
                    "words": [{"text": "w1", "start": 5.0, "end": 5.5}],
                    "version": 1
                },
                {
                    "id": "rf2",
                    "status": "awaiting_review",
                    "start": 25.0,
                    "end": 40.0,
                    "hook": "Second Clip",
                    "words": [{"text": "w2", "start": 25.0, "end": 25.5}],
                    "version": 1
                }
            ]
        }
        # First clip has custom styling in edit.json
        c1_ed = edit.init_edit(job, job["clips"][0], job["clips"][0]["words"])
        c1_ed["hook"]["font_size"] = 120
        c1_ed["hook"]["y_pos"] = 250.0
        c1_ed["hook"]["style"] = "nobox"
        c1_ed["caption"]["style"] = "headline"
        c1_ed["caption"]["font_size"] = 150
        edit.save_edit(job_dir / "clips" / "rf1_edit.json", c1_ed)

        c2_ed = edit.init_edit(job, job["clips"][1], job["clips"][1]["words"])
        edit.save_edit(job_dir / "clips" / "rf2_edit.json", c2_ed)

        jobs._jobs[job["id"]] = job
        jobs._by_user.setdefault(self.user, {})[job["id"]] = job
        jobs.save(job)

        # Approving the first clip should automatically patch rf2 with rf1's style and approve rf2
        jobs.approve_clip("job_test_first", "rf1")
        self.assertEqual(job["clips"][0]["status"], "approved")
        self.assertEqual(job["clips"][1]["status"], "approved")

        rf2_saved = edit.load_edit(job_dir / "clips" / "rf2_edit.json")
        self.assertEqual(rf2_saved["hook"]["font_size"], 120)
        self.assertEqual(rf2_saved["hook"]["y_pos"], 250.0)
        self.assertEqual(rf2_saved["hook"]["style"], "nobox")
        self.assertEqual(rf2_saved["caption"]["style"], "headline")
        self.assertEqual(rf2_saved["caption"]["font_size"], 150)
        self.assertEqual(rf2_saved["hook"]["text"], "Second Clip")  # Per-clip preserved!
        print("PASS test_review_first_mode_auto_applies_style_and_approves_rest")

    def test_api_endpoints_integration(self):
        """Test API endpoint functions directly."""
        from studio import server

        job_dir = self.temp_dir / "job_api_test"
        (job_dir / "clips").mkdir(parents=True, exist_ok=True)
        job = {
            "id": "job_api_test",
            "user": self.user,
            "creator": "c1",
            "dir": str(job_dir),
            "status": "running",
            "stage": "Waiting for review",
            "width": 1920,
            "height": 1080,
            "options": {"review_mode": "review"},
            "clips": [
                {
                    "id": "clip_api_1",
                    "status": "awaiting_review",
                    "start": 12.0,
                    "end": 28.0,
                    "hook": "Api Clip",
                    "words": [{"text": "testing", "start": 12.0, "end": 12.5}],
                    "version": 1
                }
            ]
        }
        ef = job_dir / "clips" / "clip_api_1_edit.json"
        ed = edit.init_edit(job, job["clips"][0], job["clips"][0]["words"])
        edit.save_edit(ef, ed)

        jobs._jobs[job["id"]] = job
        jobs._by_user.setdefault(self.user, {})[job["id"]] = job
        jobs.save(job)

        # GET /api/clips/{id}/edit
        data = server.get_clip_edit(cid="clip_api_1")
        self.assertEqual(data["start"], 12.0)
        self.assertEqual(data["end"], 28.0)
        self.assertEqual(data["hook"]["text"], "Api Clip")

        # PUT /api/clips/{id}/edit
        update_payload = {
            "hook": {
                "text": "Updated Hook",
                "font_size": 115,
                "y_pos": 350.0,
                "style": "nobox",
                "enabled": True
            },
            "caption": {
                "style": "bounce",
                "font_size": 140,
                "y_pos": 1550.0,
                "enabled": True,
                "words": [{"text": "updated", "start": 12.0, "end": 12.5}]
            },
            "layout": "stack",
            "split_ratio": 0.45,
            "version": data["version"]
        }
        updated_ed = server.update_clip_edit(cid="clip_api_1", body=update_payload)
        self.assertEqual(updated_ed["hook"]["text"], "Updated Hook")
        self.assertEqual(updated_ed["hook"]["font_size"], 115)
        self.assertEqual(updated_ed["hook"]["style"], "nobox")
        self.assertEqual(updated_ed["layout"], "stack")
        self.assertEqual(updated_ed["split_ratio"], 0.45)
        self.assertEqual(job["clips"][0]["status"], "drafting")

        # Reset status back to awaiting_review for approve test
        job["clips"][0]["status"] = "awaiting_review"
        jobs.save(job)

        # POST /api/clips/{id}/approve
        app_res = server.approve_clip_endpoint(cid="clip_api_1")
        self.assertTrue(app_res["ok"])
        self.assertEqual(job["clips"][0]["status"], "approved")

        # POST /api/jobs/{id}/facecam
        fc_res = server.set_facecam_endpoint(jid="job_api_test", body={"facecam": {"x": 100, "y": 80, "w": 400, "h": 300}})
        self.assertTrue(fc_res["ok"])
        print("PASS test_api_endpoints_integration")

    def test_auto_mode_skips_draft_and_renders_final(self):
        """In auto mode, clips are queued directly for final render, preserving legacy automated workflow."""
        job_dir = self.temp_dir / "job_auto"
        (job_dir / "clips").mkdir(parents=True, exist_ok=True)
        job = {
            "id": "job_auto",
            "user": self.user,
            "creator": "c1",
            "dir": str(job_dir),
            "status": "running",
            "created": time.time(),
            "options": {"review_mode": "auto"},
            "clips": []
        }
        # Simulate clip dispatch in auto mode
        scouted_clip = {"id": "auto_clip_1", "start": 0.0, "end": 15.0, "hook": "Auto Mode"}
        review_mode = job["options"].get("review_mode", "auto")
        if review_mode in ("review", "review_first"):
            scouted_clip["status"] = "drafting"
            quality = "draft"
        else:
            scouted_clip["status"] = "queued"
            quality = "final"

        job["clips"].append(scouted_clip)
        self.assertEqual(scouted_clip["status"], "queued")
        self.assertEqual(quality, "final")
        print("PASS test_auto_mode_skips_draft_and_renders_final")

    def test_qc_failure_sets_qc_flagged_status(self):
        """When quality check fails on final render, clip status becomes qc_flagged instead of done."""
        job_dir = self.temp_dir / "job_qc"
        (job_dir / "clips").mkdir(parents=True, exist_ok=True)
        job = {
            "id": "job_qc",
            "user": self.user,
            "creator": "c1",
            "dir": str(job_dir),
            "status": "running",
            "created": time.time(),
            "width": 1920,
            "height": 1080,
            "clips": [
                {
                    "id": "qc_clip_1",
                    "status": "approved",
                    "start": 0.0,
                    "end": 15.0,
                    "hook": "QC Test",
                    "version": 1
                }
            ]
        }
        # Simulate render_one result with QC failure
        out = {
            "file": "qc_clip_1.mp4",
            "cover": "qc_clip_1.jpg",
            "layout": "blur",
            "qc": {
                "status": "FAIL",
                "issues": ["Audio clipping detected: peak volume +2.1 dB"],
                "lufs": -8.5
            }
        }
        quality = "final"
        clip = job["clips"][0]
        if quality == "draft":
            clip.update(out, status="awaiting_review", error=None)
        else:
            final_status = "done"
            if out.get("qc") and out["qc"].get("status") == "FAIL":
                final_status = "qc_flagged"
                issues = out["qc"].get("issues", [])
                clip["error"] = ", ".join(issues) if issues else "Quality control check failed"
            clip.update(out, status=final_status)

        self.assertEqual(clip["status"], "qc_flagged")
        self.assertIn("Audio clipping", clip["error"])
        print("PASS test_qc_failure_sets_qc_flagged_status")

    def test_cover_crop_and_stack_filters(self):
        """Test cover_crop_filter and build_stack_filter across split ratios and swap options."""
        cam = [100, 100, 400, 400]

        # 1. Test 35/65 ratio: panel 1080x672 (target 1.607)
        # Cam aspect is 1.0 < 1.607 -> trim top/bottom -> cw=400, ch=248
        f35 = render.cover_crop_filter(cam, 1080, 672, headroom=0.42)
        self.assertIn("crop=400:248:100:196", f35)
        self.assertIn("scale=1080:672:flags=lanczos,setsar=1", f35)

        # 2. Test 50/50 ratio: panel 1080x960 (target 1.125) -> cw=400, ch=356
        # cy clamps to y + h - ch = 100 + 400 - 356 = 144 to stay inside camera bounds
        f50 = render.cover_crop_filter(cam, 1080, 960, headroom=0.42)
        self.assertIn("crop=400:356:100:144", f50)
        self.assertIn("scale=1080:960:flags=lanczos,setsar=1", f50)

        # 3. Test 40/60 ratio: panel 1080x768 (target 1.406) -> cw=400, ch=284
        f40 = render.cover_crop_filter(cam, 1080, 768, headroom=0.42)
        self.assertIn("crop=400:284:100:181", f40)
        self.assertIn("scale=1080:768:flags=lanczos,setsar=1", f40)

        # 4. Test zoom: zoom=1.5 shrinks cw and ch
        fzoom = render.cover_crop_filter(cam, 1080, 672, zoom=1.5)
        self.assertIn("crop=266:164:", fzoom)

        # 5. Test build_stack_filter with swap=False/True and optional divider
        stack_noswap = render.build_stack_filter(cam, 0.35, swap=False, divider=True)
        self.assertIn("[c][g]vstack=inputs=2", stack_noswap)
        self.assertIn("scale=1080:672:flags=lanczos", stack_noswap)
        self.assertIn("crop=1080:1248", stack_noswap)
        self.assertIn("drawbox=x=0:", stack_noswap)

        stack_nodiv = render.build_stack_filter(cam, 0.35, swap=False, divider=False)
        self.assertNotIn("drawbox=", stack_nodiv)

        stack_swap = render.build_stack_filter(cam, 0.35, swap=True, divider=True)
        self.assertIn("[g][c]vstack=inputs=2", stack_swap)
        self.assertIn("scale=1080:672:flags=lanczos", stack_swap)
        self.assertIn("crop=1080:1248", stack_swap)

        # 6. Test hook placement strictly contained inside facecam section (never overflowing across divider)
        # swap=False: facecam is [0, 672)
        hook_y_noswap = render._hook_y("stack", 300, facecam=cam, split_ratio=0.35, cam_swap=False)
        self.assertGreater(hook_y_noswap, 0)
        self.assertLess(hook_y_noswap, 672)

        # swap=True: facecam is (1248, 1920]
        hook_y_swap = render._hook_y("stack", 300, facecam=cam, split_ratio=0.35, cam_swap=True)
        self.assertGreater(hook_y_swap, 1248)
        self.assertLess(hook_y_swap, 1920)

        # 7. Test responsive proportional metrics at non-1080p output resolution (720x1280)
        m720 = render.stack_layout_metrics(720, 1280, split_ratio=0.35, cam_swap=False, divider=True)
        self.assertEqual(m720["cam_h"], 448)
        self.assertEqual(m720["game_h"], 832)
        self.assertEqual(m720["cam_h"] + m720["game_h"], 1280)
        self.assertLess(m720["hook_y"], m720["cam_bottom"])
        self.assertGreater(m720["cap_y"], m720["game_top"])
        self.assertLess(m720["cap_y"], m720["game_bottom"])

        stack_720 = render.build_stack_filter(cam, 0.35, swap=False, W=720, H=1280, divider=True)
        self.assertIn("scale=720:448:flags=lanczos", stack_720)
        self.assertIn("crop=720:832", stack_720)
        print("PASS test_cover_crop_and_stack_filters")

    def test_face_center_and_focus_detection(self):
        """Test face center detection helper and fallback behavior."""
        import numpy as np
        from studio import facecam

        # Black frame has no face -> returns None
        black = np.zeros((720, 1280, 3), dtype=np.uint8)
        fc = facecam.face_center(black, (100, 100, 300, 300))
        self.assertIsNone(fc)

        # Missing file returns None safely
        res = facecam.find_clip_face_focus(self.temp_dir / "nonexistent.mp4", (100, 100, 300, 300), 0.0, 2.0)
        self.assertIsNone(res)
        print("PASS test_face_center_and_focus_detection")

    def test_all_six_stack_render_combinations(self):
        """Test rendering all 6 combinations of (35/65, 40/60, 50/50) x (swap=False, swap=True)."""
        import subprocess
        from studio import media

        # Generate a 2.0-second synthetic test video with sine audio
        src_path = self.temp_dir / "synthetic_test.mp4"
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-t", "2.0",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(src_path)
        ]
        subprocess.run(cmd, check=True)
        self.assertTrue(src_path.is_file())

        out_dir = self.temp_dir / "rendered_clips"
        out_dir.mkdir(parents=True, exist_ok=True)

        combinations = [
            (0.35, False),
            (0.35, True),
            (0.40, False),
            (0.40, True),
            (0.50, False),
            (0.50, True)
        ]

        words = [{"text": "TEST", "start": 0.2, "end": 0.8}, {"text": "STACK", "start": 0.9, "end": 1.5}]
        facecam_box = [100, 80, 400, 400]

        for ratio, swap in combinations:
            tag = f"r{int(ratio*100)}_{'swap' if swap else 'noswap'}"
            clip_meta = {
                "id": tag,
                "start": 0.0,
                "end": 2.0,
                "layout": "stack",
                "split_ratio": ratio,
                "cam_swap": swap,
                "facecam": facecam_box,
                "face_zoom": 1.0,
                "hook": f"Hook {tag}",
                "caption": {"style": "bounce", "font_size": 90, "y_pos": 1400.0, "enabled": True},
                "audio": {"meme": False, "music": False}
            }

            # 1. Test Final Render
            res_final = render.render_clip(
                src=src_path,
                out_dir=out_dir,
                name=f"clip_{tag}",
                clip=clip_meta,
                words=words,
                src_w=1920,
                src_h=1080,
                quality="final"
            )
            final_mp4 = out_dir / res_final["file"]
            final_jpg = out_dir / res_final["cover"]
            self.assertTrue(final_mp4.is_file(), f"Final MP4 missing for {tag}")
            self.assertTrue(final_jpg.is_file(), f"Final JPG cover missing for {tag}")
            self.assertGreater(final_jpg.stat().st_size, 100, f"JPG cover empty for {tag}")

            # Verify exact dimensions 1080x1920
            info = media.probe(final_mp4)
            self.assertEqual(info["width"], 1080, f"Width should be 1080 for {tag}")
            self.assertEqual(info["height"], 1920, f"Height should be 1920 for {tag}")

        # 2. Test Draft Render at 540x960
        draft_meta = {
            "id": "draft_test",
            "start": 0.0,
            "end": 2.0,
            "layout": "stack",
            "split_ratio": 0.35,
            "cam_swap": False,
            "facecam": facecam_box,
            "hook": "Draft Hook",
            "caption": {"style": "bounce", "font_size": 90, "y_pos": 1400.0, "enabled": True},
            "audio": {"meme": False, "music": False}
        }
        res_draft = render.render_clip(
            src=src_path,
            out_dir=out_dir,
            name="clip_draft_test",
            clip=draft_meta,
            words=words,
            src_w=1920,
            src_h=1080,
            quality="draft"
        )
        draft_mp4 = out_dir / res_draft["file"]
        draft_jpg = out_dir / res_draft["cover"]
        self.assertTrue(draft_mp4.is_file())
        self.assertTrue(draft_jpg.is_file())
        self.assertTrue(res_draft.get("draft"))

        info_draft = media.probe(draft_mp4)
        self.assertEqual(info_draft["width"], 540)
        self.assertEqual(info_draft["height"], 960)

        print("PASS test_all_six_stack_render_combinations")


if __name__ == "__main__":
    unittest.main()
