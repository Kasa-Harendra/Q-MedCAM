# Q-MedCAM: Frontend Application

This directory is designated for the fullstack application's clinical web interface.

## Planned Stack
- **Framework:** Next.js / React (TypeScript)
- **Styling:** Tailwind CSS + shadcn/ui
- **Icons & Visuals:** Lucide React, Recharts (for risk distribution charts)
- **Image Inspection:** Canvas-based interactive image comparator / slider (Raw vs Grad-CAM overlay)

## Planned Clinical Features
1. **Dermoscopic Image Uploader:** Drag-and-drop support with real-time format validation and EXIF metadata extraction.
2. **Diagnostic Risk Gauge:** Visual risk meter displaying malignancy probability with clinical thresholds:
   - `< 30%`: Low Risk / Benign (Routine Follow-up)
   - `30% - 60%`: Indeterminate / Borderline (Secondary Review)
   - `> 60%`: High Risk / Malignant (Immediate Biopsy Recommendation)
3. **Tripartite XAI Viewer:**
   - Raw Lesion
   - Quantum Grad-CAM Activation Heatmap
   - Interactive Alpha-Blend Overlay with adjustable opacity slider
4. **Export Report:** Downloadable clinical PDF summary artifact at 300 DPI for patient charts and electronic medical records (EMR).
