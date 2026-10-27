# Milestones

Scope: about 2–3 weeks part-time for the minimum version (one task, two ports, two-date comparison,
clean report). Each milestone ends with something worth posting. Draft posts live in `docs/linkedin/`.

| # | Milestone | Ships | LinkedIn post | Est. |
|---|---|---|---|---|
| M0 | Plan & scaffold | Repo structure, architecture doc + diagram, tests running | **Post 1: The idea** | 1–2 days |
| M1 | Imagery pipeline + classical baseline | STAC search, AOI crop, cloud/water masks, classical vessel detector, annotated PNG, `ask-a-satellite detect` CLI | (feeds Post 2) | 3–4 days |
| M2 | Perception works (measured) | Hand-labelled test set for both ports; precision/recall for classical; YOLO fine-tuned on Kaggle and compared; `method=` switch | **Post 2: Perception works** | 4–5 days |
| M2.5 | Explainer | Diagram + tiny code snippet of one tool-use round trip (can use the M3 code skeleton) | **Post 2.5: How agents choose tools** | 1 day |
| M3 | The agent reasons | MCP server exposing the tools; Claude agent loop; `ask-a-satellite ask "..."`; markdown/HTML report; count validator | **Post 3: Demo** | 4–5 days |
| M4 | Polish + lessons | README results table, optional minimal web UI, limitations write-up, public repo | **Post 4: Lessons & what's next** | 2–3 days |

---

## M0: Plan & scaffold → Post 1

**Done when:** repo builds with `uv sync`, `uv run pytest` passes, `docs/architecture.md` has the diagram.

**Visual assets**
- Architecture diagram (from the Mermaid source in `docs/architecture.md`).
  - Tips: redraw it in Excalidraw or export from mermaid.live at 2× scale. Use one accent colour for the
    agent and a neutral colour for tools. Keep it to 6–8 boxes with short verb labels, size 1200×1200
    (square) or 1200×627, and include a small Sentinel-2 thumbnail of the port as the "input".
- Optional second image: a true-colour Sentinel-2 crop of Port Everglades (first output of M1) with the
  credit line.

Draft: `docs/linkedin/post-1-the-idea.md`.

## M1: Imagery pipeline + classical baseline

**Done when:** `uv run ask-a-satellite detect --aoi port-everglades --start ... --end ...` finds a clear
scene, writes an annotated PNG plus JSON with real counts, and tests cover search parsing, cropping,
masks, detection, annotation, and comparison.

Tasks
- [x] AOI registry (Port of Miami, Port Everglades + anchorage)
- [x] STAC search with scene-level cloud filter → `SceneInfo`
- [x] Windowed COG reads with reflectance scaling; SCL resampled to 10 m
- [x] AOI cloud fraction (SCL) and NDWI water mask
- [x] Classical vessel detector + `DetectionResult`
- [x] Annotated PNG with credit line
- [x] Two-date count comparison
- [x] `detect` / `search` / `aois` CLI
- [x] First real-imagery runs on both ports (see `docs/dev-log.md`: offset bug, white-hull NDWI, land leakage)
- [ ] Simple on-disk cache of AOI crops (avoid re-downloading during tuning)
- [ ] Reduce shoreline clutter: shape filter for surf strips, jetties, and marinas (currently dominant false positives)
- [ ] Tune thresholds on 3–5 real scenes per port against hand labels and record the chosen parameters

**Visual assets (captured now, used in Post 2)**
- Side-by-side pairs: raw true colour | NDWI water mask | detections. Same crop and scale, numbered boxes.
- Tips: pick a cloud-free scene with anchored ships offshore (they look clean). Upscale 3–4× with nearest
  neighbour so pixels are honest and visible, and add a 100 m scale bar.

## M2: Perception works (measured) → Post 2

**Done when:** a results table with precision, recall, and F1 per method, per port, and per date, measured
against hand labels, with the conditions (cloud %, date, sun-glint) stated.

Tasks
- Label 8–12 AOI crops (two ports × several dates) in Roboflow, using classical detections as
  pre-annotations. Keep them as a **test set only**.
- Check the licence for the S2-SHIPS dataset or alternatives, and write it into `docs/data.md`.
- Kaggle notebook: tile Sentinel-2 chips (upsampled 2–3×), fine-tune YOLO (small model), export weights.
  Weights are not committed. Download instructions go in the README.
- `detect_vessels(method="yolo")` with CPU inference, then evaluate both methods on the same test set.

**Visual assets**
- 2×2 grid: classical vs. YOLO on the same scene, with true positives green, false positives red, misses
  yellow.
- Small metrics table as an image (or in the post text).
- One "failure gallery" image (pier, wake, merged ships). Honest failures earn more trust than a clean
  demo.

**Post 2 outline:** what 10 m really means (pixel math) → the classical pipeline in 4 steps → how I
measured (hand labels, not vibes) → results table → 3 failure types → what YOLO changed (or didn't) →
question: "What's your go-to method for small-object detection in medium-resolution imagery?"

## M2.5: How agents choose tools → Post 2.5

**Visual assets**
- One diagram with three panels: (1) the "menu" (tool names + one-line descriptions), (2) Claude's
  structured request (`{"tool": "search_imagery", "input": {...}}`), (3) the code runs it and returns
  JSON. A loop arrow goes back to Claude.
- Optional: a 15-line code snippet screenshot (carbon.now.sh) of the loop.

**Post 2.5 outline:** a restaurant-menu analogy → the model never touches the satellite: it writes a
request, and my code does the work → why this matters for honesty (numbers come from code) → MCP as a
standard menu format → question: "What part of agents was hardest to understand when you started?"

## M3: The agent reasons → Post 3

**Done when:** `uv run ask-a-satellite ask "How many large ships are in Port Everglades this week vs last
month?"` produces a report with an answer, counts, dates, annotated images, and limitations. The same
tools also work from Claude Desktop through the MCP server.

Tasks
- MCP server (`mcp` Python SDK, stdio transport) wrapping the tools. Large rasters stay in a server-side
  cache keyed by ID.
- Agent loop with the Anthropic SDK and tool use. The system prompt states the honesty rules. Streaming
  output shows the plan and tool calls live.
- Report renderer (Markdown + optional HTML) plus a validator that compares narrative numbers with tool
  outputs.

**Visual assets**
- 30–60 s screen recording: type the question, then show tool calls streaming (search → fetch →
  detect → compare), then the report with images.
  - Tips: use a large terminal font (18–20 pt) and a dark theme, and trim the waiting time (cache the
    imagery first and say so in the post). Record at 1080p with OBS or ScreenToGif, and keep the GIF
    under 8 MB or upload as MP4. Add captions, since most people watch muted.

## M4: Polish + lessons → Post 4 (optional)

- README results table (filled only from real runs), hero GIF, limitations section.
- Optional minimal web UI (Streamlit or FastAPI + HTMX) with a question box and the report.
- Make the repo public **after Bernard approves**.

**Post 4 outline:** what worked → what didn't (honest numbers) → the limits of 10 m (and what 3 m or SAR
would change) → where agentic geospatial AI is going → repo link → question.
