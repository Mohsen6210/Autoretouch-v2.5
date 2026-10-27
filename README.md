# Auto Retouch

Automatic portrait skin retouching (dodge & burn) for Windows. Everything runs locally; no photo is uploaded anywhere.

**Features**
- Automatic face detection for close-ups, half-body and full-body photos (Auto / Face / Half-body / Full-body)
- Skin mask that excludes eyes, eyebrows and lips; fix it by hand with Brush / Eraser / Invert / Undo. Soft brush (Hardness, Opacity) and adjustable Mask feather
- Views: Original, Preview (retouched), Mask
- Manual Dodge / Burn brushes (like Photoshop: size, hardness, flow, own Undo, optional "skin mask only")
- Global Sharpen and Contrast sliders (optional "skin mask only")
- Sliders: Warmth, Blend, Strength, Texture (100 = keep skin texture exactly; above boosts, below softens), Reset (R)
- Zoom (+ / -, Fit, 100%, mouse wheel), pan (hold Space and drag, Move tool, or right-drag)
- Saves at full resolution, keeps colour profile and EXIF. "Save" never overwrites the original (adds `_retouched`).

**Shortcuts:** V move, B brush, E eraser, I invert, 1 dodge, 2 burn, [ ] brush size, R reset sliders, Space+drag pan, Ctrl+Z undo, 0 fit, Ctrl+1 100%. Ctrl+click with Brush fills the mask, with Eraser clears it.

**Download:** see the Releases page. If Windows SmartScreen warns, click More info -> Run anyway (the app is unsigned).

**Build:** push to GitHub; the workflow in `.github/workflows/build.yml` builds the exe and installer (Actions -> Artifacts).

Errors are written to `%TEMP%\AutoRetouch_error.log`.
