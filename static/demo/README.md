# static/demo/

Place two video files here for the demo page to work:

| File | What it is |
|---|---|
| `demo_fighting.mp4` | The full original video analyzed |
| `demo_clip_Fighting.mp4` | A short crime clip extracted from it |

These are served at `/static/demo/<filename>` by both `app.py` and `app_demo.py`.

**These files are gitignored** (via `static/uploads/` rule — add `static/demo/*.mp4` to .gitignore if needed).
On Render, upload them as part of your repo or serve them from a CDN/cloud storage URL.
