# Signature Frontend

Minimal vanilla JS frontend for the FastAPI signature verification service.

## Features
- Register signatures from drawing pad (mouse / touch) or image upload.
- View list/thumbnail of registered samples.
- Verify new signature (draw or upload) and view similarity score.
- Logs panel for debugging responses.

## Run
From project root (where `signature_verification.py` lives):

```
uvicorn signature_verification:app --reload --port 8000
```

Then open: http://localhost:8000/ (auto-redirects to `/frontend`).

If CUDA is available, the model will use GPU automatically, else CPU.

## Files
- `index.html` main UI
- (No build step; plain static HTML/JS)

## Notes
- Adjust CORS origins in `signature_verification.py` if serving UI elsewhere.
- Tune `VERIFY_THRESHOLD` for better FAR/FRR based on validation data.
- Add authentication before production use.
