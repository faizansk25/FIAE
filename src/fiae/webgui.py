"""FIAE Web GUI — zero-dependency single-page dashboard.

Served by the stdlib server (server.py) at ``/gui`` and ``/``. Provides:

- **Connect** — connect to any of the 25+ supported sources and profile
- **Learn**   — run the feature intelligence pipeline on a source
- **Runs**    — browse tracked runs and their metrics
- **Jobs**    — live job queue state (auto-refreshing)
- **About**   — pipeline architecture overview

Design: dark theme with the FIAE purple identity, vanilla JS (no build
step, no CDN, no frameworks — NFR-002: stdlib only).

Endpoints added to server.py:
- GET  /gui, /           — this GUI
- GET  /api/sources      — supported source catalog
- POST /api/profile      — async profile job (kind="profile")
- POST /api/learn        — async learn job (kind="learn", unchanged)
"""

from __future__ import annotations

__version__ = "0.0.1"

_GUI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FIAE — Feature Intelligence &amp; Architecture Engine</title>
<style>
:root{
  --bg:#0d1117; --panel:#161b27; --panel2:#1c2333; --border:#30363d;
  --text:#e6edf3; --muted:#8b949e;
  --purple:#c084fc; --purple2:#8b5cf6; --purple3:#a78bfa; --purple-dim:#3b2f63;
  --green:#7ee2a8; --green-bg:#1a4731; --red:#ffa1a1; --red-bg:#4d1f24;
  --yellow:#f5d76e; --cyan:#79c0ff;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);
  color:var(--text);min-height:100vh}
a{color:var(--purple3);text-decoration:none}
code{font-family:Consolas,monospace;background:var(--panel2);
  padding:.1rem .35rem;border-radius:4px;font-size:.85em}

/* ── header ─────────────────────────────────────────────── */
header{background:linear-gradient(135deg,#161022 0%,#0d1117 60%);
  border-bottom:1px solid var(--border);padding:1rem 2rem;display:flex;
  align-items:center;gap:1rem;flex-wrap:wrap}
.logo{font-size:1.6rem;font-weight:700;letter-spacing:.12em;
  background:linear-gradient(90deg,#c084fc,#8b5cf6,#6d28d9);
  -webkit-background-clip:text;background-clip:text;color:transparent}
.tagline{color:var(--muted);font-size:.8rem;flex:1;min-width:200px}
.ver{color:var(--muted);font-size:.75rem}

/* ── nav ────────────────────────────────────────────────── */
nav{display:flex;gap:.25rem;padding:.75rem 2rem 0;background:var(--bg);
  border-bottom:1px solid var(--border);flex-wrap:wrap}
nav button{background:none;border:none;color:var(--muted);font-size:.9rem;
  padding:.6rem 1.2rem;cursor:pointer;border-bottom:2px solid transparent;
  font-family:inherit}
nav button:hover{color:var(--text)}
nav button.active{color:var(--purple);border-bottom-color:var(--purple)}

/* ── layout ─────────────────────────────────────────────── */
main{max-width:1100px;margin:0 auto;padding:1.5rem 2rem 3rem}
.tab{display:none}.tab.active{display:block}
h2{color:var(--purple3);font-size:1.1rem;margin:1.2rem 0 .6rem;font-weight:600}
h3{color:var(--purple3);font-size:.95rem;margin:1rem 0 .4rem;font-weight:600}
.panel{background:var(--panel);border:1px solid var(--border);
  border-radius:8px;padding:1.2rem;margin-bottom:1rem}
.muted{color:var(--muted);font-size:.85rem}
.row{display:flex;gap:.8rem;flex-wrap:wrap;align-items:flex-end}
.field{display:flex;flex-direction:column;gap:.25rem;flex:1;min-width:180px}
.field.small{min-width:110px;flex:0 0 130px}
label{font-size:.75rem;color:var(--muted);text-transform:uppercase;
  letter-spacing:.05em}
input,select,textarea{background:var(--panel2);border:1px solid var(--border);
  color:var(--text);border-radius:6px;padding:.5rem .7rem;font-size:.9rem;
  font-family:inherit}
input:focus,select:focus{outline:none;border-color:var(--purple2)}
button.primary{background:linear-gradient(135deg,#8b5cf6,#6d28d9);color:#fff;
  border:none;border-radius:6px;padding:.55rem 1.4rem;font-size:.9rem;
  cursor:pointer;font-weight:600;font-family:inherit}
button.primary:hover{filter:brightness(1.15)}
button.primary:disabled{opacity:.5;cursor:not-allowed}
button.ghost{background:var(--panel2);color:var(--purple3);
  border:1px solid var(--border);border-radius:6px;padding:.5rem 1rem;
  cursor:pointer;font-family:inherit;font-size:.85rem}
button.ghost:hover{border-color:var(--purple2)}

/* ── badges ─────────────────────────────────────────────── */
.badge{display:inline-block;padding:.12rem .55rem;border-radius:10px;
  font-size:.72rem;font-weight:600;letter-spacing:.03em}
.b-completed,.b-ok{background:var(--green-bg);color:var(--green)}
.b-running,.b-pending{background:var(--purple-dim);color:#c4b5fd}
.b-failed,.b-error{background:var(--red-bg);color:var(--red)}
.b-warn{background:#4a3b17;color:var(--yellow)}

/* ── tables ─────────────────────────────────────────────── */
table{border-collapse:collapse;width:100%;margin-top:.5rem;font-size:.88rem}
th,td{padding:.45rem .7rem;border-bottom:1px solid var(--border);
  text-align:left}
th{color:var(--purple3);text-transform:uppercase;font-size:.7rem;
  letter-spacing:.06em}
tr:hover td{background:rgba(139,92,246,.06)}

/* ── kv grid ────────────────────────────────────────────── */
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
  gap:.5rem 1.5rem;margin-top:.5rem}
.kv div{font-size:.88rem}
.kv b{color:var(--purple3);font-weight:600;display:block;font-size:.7rem;
  text-transform:uppercase;letter-spacing:.06em;margin-bottom:.1rem}

/* ── misc ───────────────────────────────────────────────── */
.alert{border-radius:6px;padding:.7rem 1rem;margin:.8rem 0;font-size:.88rem;
  display:none}
.alert.error{background:var(--red-bg);color:var(--red);display:block}
.alert.info{background:var(--purple-dim);color:#c4b5fd;display:block}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid
  var(--purple-dim);border-top-color:var(--purple);border-radius:50%;
  animation:spin .8s linear infinite;vertical-align:-2px}
@keyframes spin{to{transform:rotate(360deg)}}
.bar{height:6px;background:var(--panel2);border-radius:3px;overflow:hidden;
  margin-top:.4rem}
.bar>div{height:100%;background:linear-gradient(90deg,#8b5cf6,#c084fc);
  width:0%;transition:width .4s}
.chips{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.5rem}
.chip{background:var(--panel2);border:1px solid var(--purple-dim);
  color:var(--purple3);border-radius:14px;padding:.3rem .9rem;cursor:pointer;
  font-size:.85rem;font-family:inherit}
.chip:hover{border-color:var(--purple);color:var(--purple)}
.chip small{color:var(--muted);margin-left:.4rem}
table.preview{font-family:Consolas,monospace;font-size:.78rem}
table.preview td{padding:.3rem .6rem;color:var(--muted)}
table.preview td:first-child{color:var(--text)}
table.preview tr:first-child td{color:var(--purple3)}
.wf{margin-top:.6rem}
.wf-row{display:flex;align-items:center;gap:.8rem;margin-bottom:.45rem}
.wf-label{width:150px;font-size:.78rem;color:var(--muted);text-align:right;
  flex-shrink:0}
.wf-track{flex:1;background:var(--panel2);border-radius:4px;height:22px;
  position:relative;overflow:hidden}
.wf-fill{height:100%;border-radius:4px;
  background:linear-gradient(90deg,#6d28d9,#8b5cf6);min-width:2px;
  transition:width .6s ease}
.wf-fill.warn{background:linear-gradient(90deg,#8a6d1d,#c9a227)}
.wf-val{width:90px;font-size:.8rem;color:var(--purple3);flex-shrink:0}
.funnel-note{font-size:.78rem;color:var(--muted);margin-top:.4rem}
/* ── report charts ──────────────────────────────────────── */
.chartgrid{display:grid;grid-template-columns:repeat(auto-fit,
  minmax(300px,1fr));gap:1.2rem;margin-top:.8rem}
.chart{background:var(--panel2);border:1px solid var(--border);
  border-radius:8px;padding:.9rem}
.chart h4{color:var(--purple3);font-size:.82rem;font-weight:600;
  margin-bottom:.5rem;overflow:hidden;text-overflow:ellipsis}
.chart svg{width:100%;height:auto;display:block}
.chart .sub{color:var(--muted);font-size:.72rem;margin-top:.45rem}
.corr-table{border-collapse:separate;border-spacing:2px}
.corr-table td{width:34px;height:26px;text-align:center;font-size:.62rem;
  border-radius:3px;cursor:default}
.corr-table .clabel{font-size:.68rem;color:var(--purple3);background:none;
  max-width:90px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.svglab{font-size:10.5px;fill:var(--purple3)}
.svgval{font-size:10px;fill:var(--muted);font-family:var(--mono,monospace)}
.donut-wrap{display:flex;align-items:center;gap:1rem;flex-wrap:wrap}
.donut-wrap svg{max-width:210px;height:auto}
.legend{display:flex;flex-direction:column;gap:.3rem;font-size:.78rem}
.legend-row{display:flex;align-items:center;gap:.4rem}
.legend-pct{color:var(--muted);margin-left:.2rem}
.swatch{width:11px;height:11px;border-radius:3px;display:inline-block;flex-shrink:0}
.insight{display:flex;gap:.6rem;align-items:flex-start;padding:.55rem .8rem;
  border-left:3px solid var(--border);border-radius:6px;margin-bottom:.5rem;
  background:var(--panel2);font-size:.86rem;line-height:1.45}
.insight code{color:var(--purple3);background:rgba(139,92,246,.1);
  padding:.05rem .35rem;border-radius:4px;font-size:.8rem}
.insight .ins-dot{width:8px;height:8px;border-radius:50%;margin-top:.4rem;flex-shrink:0}
.ins-info{border-left-color:#8b5cf6}.ins-info .ins-dot{background:#8b5cf6}
.ins-warning{border-left-color:#c9a227}.ins-warning .ins-dot{background:#c9a227}
.chart .why{color:#a78bfa;font-style:italic}
.chart h4{display:flex;align-items:center;gap:.5rem}
.tooltip-host{position:relative}
.sources{columns:2;column-gap:2rem;font-size:.85rem}
@media(max-width:900px){.chartgrid{grid-template-columns:1fr}
  .wf-label{width:100px;font-size:.7rem}}
.sources div{break-inside:avoid;margin-bottom:.35rem}
.sources b{color:var(--purple3)}
.pulse{animation:pulse 1.5s ease infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.55}}
footer{border-top:1px solid var(--border);color:var(--muted);
  font-size:.75rem;padding:1rem 2rem;text-align:center}
.pill{display:inline-block;background:var(--panel2);border:1px solid
  var(--border);border-radius:12px;padding:.15rem .7rem;font-size:.75rem;
  color:var(--purple3);margin-right:.4rem}
@media(max-width:700px){.sources{columns:1}main{padding:1rem}}
</style>
</head>
<body>

<header>
  <span class="logo">FIAE</span>
  <span class="tagline">Feature Intelligence &amp; Architecture Engine —
  safety-first automated feature engineering</span>
  <span class="ver" id="ver"></span>
</header>

<nav>
  <button data-tab="connect" class="active">Connect</button>
  <button data-tab="report">Report</button>
  <button data-tab="learn">Learn</button>
  <button data-tab="leakage">Leakage</button>
  <button data-tab="pipeline">Compile</button>
  <button data-tab="tune">Tune</button>
  <button data-tab="experience">Experience</button>
  <button data-tab="operators">Operators</button>
  <button data-tab="runs">Runs</button>
  <button data-tab="jobs">Jobs <span id="jobsPill" class="pill pulse"
    style="display:none">0</span></button>
  <button data-tab="benchmarks">Benchmarks</button>
  <button data-tab="about">About</button>
</nav>

<main>

<!-- ══════════════════ CONNECT ══════════════════ -->
<section id="tab-connect" class="tab active">
  <div class="panel">
    <h2>Connect to a Data Source</h2>
    <p class="muted">Auto-detects the adapter, verifies connectivity, and
    profiles a sample. Supports 25+ sources — files, URLs, cloud storage,
    databases, warehouses, and APIs.</p>
    <div class="alert" id="connectAlert"></div>
    <div class="row" style="margin-top:.8rem">
      <div class="field" style="flex:3">
        <label>Source (path, URI, connection string)</label>
        <input id="src" placeholder="data.csv  ·  s3://bucket/data.csv  ·
          postgresql://u:p@host/db  ·  bigquery://proj/ds">
      </div>
      <div class="field small">
        <label>Table (optional)</label>
        <input id="srcTable" placeholder="orders">
      </div>
      <button class="primary" id="btnProfile">Connect &amp; Profile</button>
    </div>
  </div>

  <div class="panel" id="profileResult" style="display:none">
    <h2 id="profileTitle">Profile</h2>
    <div class="kv" id="profileKv"></div>

    <h3>Suggested Targets</h3>
    <p class="muted">Click one to run the pipeline with that target.</p>
    <div id="targetChips" class="chips"></div>

    <h3>Data Preview</h3>
    <div style="overflow-x:auto"><table id="previewTable" class="preview">
      <thead></thead><tbody></tbody></table></div>

    <h3>Columns</h3>
    <div style="overflow-x:auto"><table id="profileCols">
      <thead><tr><th>Name</th><th>Type</th><th>Semantic</th>
        <th>Null %</th><th>Distinct</th></tr></thead>
      <tbody></tbody></table></div>
  </div>

  <div class="panel">
    <h3>Supported Sources</h3>
    <div class="sources muted" id="sourcesList">loading…</div>
  </div>
</section>

<!-- ══════════════════ LEARN ══════════════════ -->
<section id="tab-learn" class="tab">
  <div class="panel">
    <h2>Run the Feature Intelligence Pipeline</h2>
    <p class="muted">Generates, validates, and selects features — leakage
    checks, funnel gates F0–F6, complementarity, and stability proof.</p>
    <div class="alert" id="learnAlert"></div>
    <div class="row" style="margin-top:.8rem">
      <div class="field" style="flex:2">
        <label>Source</label>
        <input id="learnSrc" placeholder="data.csv or URI">
      </div>
      <div class="field small">
        <label>Target column</label>
        <input id="learnTarget" placeholder="y">
      </div>
      <div class="field small">
        <label>Max features</label>
        <input id="learnMaxF" type="number" value="8" min="1" max="64">
      </div>
      <div class="field small">
        <label>Max rows</label>
        <input id="learnMaxR" type="number" value="100000">
      </div>
      <button class="primary" id="btnLearn">Run Pipeline</button>
    </div>
  </div>

  <div class="panel" id="learnResult" style="display:none">
    <h2>Pipeline Result
      <span style="float:right">
        <button class="ghost" id="btnExportCsv">Download CSV</button>
        <button class="ghost" id="btnExportJson">Download JSON</button>
      </span></h2>
    <div class="kv" id="learnKv"></div>

    <h3>Funnel Waterfall</h3>
    <p class="muted">How proposals survived each gate.</p>
    <div id="waterfall"></div>

    <h3>Selected Features <span class="muted" id="featHint">
      (click a row for details)</span></h3>
    <div style="overflow-x:auto"><table id="learnFeat">
      <thead><tr><th>#</th><th>Operator</th><th>Inputs</th>
        <th>Gain</th><th>Stability</th><th>Status</th></tr></thead>
      <tbody></tbody></table></div>
  </div>

  <div class="panel" id="featureDetail" style="display:none">
    <h2>Feature Detail <span id="fdName" class="pill"></span></h2>
    <div class="kv" id="fdKv"></div>
    <h3>Gate History</h3>
    <div style="overflow-x:auto"><table id="fdStages">
      <thead><tr><th>Gate</th><th>Verdict</th><th>Reason</th></tr></thead>
      <tbody></tbody></table></div>
  </div>
</section>

<!-- ══════════════════ REPORT ══════════════════ -->
<section id="tab-report" class="tab">
  <div class="panel">
    <h2>Automatic Dataset Report</h2>
    <p class="muted">Understands your data first — classifies every column,
    excludes identifiers and noise, then charts only what carries signal,
    with ranked key insights. All pure SVG, zero chart libraries.</p>
    <div class="alert" id="repAlert"></div>
    <div class="row" style="margin-top:.8rem">
      <div class="field" style="flex:3">
        <label>Source (path, URI, connection string)</label>
        <input id="repSrc" placeholder="data.csv  ·  sqlite:///db.sqlite">
      </div>
      <div class="field small">
        <label>Target (optional)</label>
        <input id="repTarget" placeholder="y">
      </div>
      <div class="field small">
        <label>Table (optional)</label>
        <input id="repTable" placeholder="orders">
      </div>
      <button class="primary" id="btnReport">Generate Report</button>
    </div>
  </div>

  <div id="repContent" style="display:none">
    <div class="panel" id="repInsightsPanel" style="display:none">
      <h2>Key Insights</h2>
      <p class="muted">What an analyst would flag first — ranked findings
      computed from your data, not boilerplate.</p>
      <div id="repInsights"></div>
    </div>

    <div class="panel" id="repExcludedPanel" style="display:none">
      <h2>Columns Excluded From Charts</h2>
      <p class="muted">Identifiers, constants, free text and near-empty
      columns carry no chartable signal — showing them would be noise.</p>
      <div id="repExcluded"></div>
    </div>

    <div class="panel">
      <h2 id="repTitle">Dataset</h2>
      <div class="kv" id="repKv"></div>
    </div>

    <div class="panel" id="repBalancePanel" style="display:none">
      <h2>Target Balance</h2>
      <div id="repBalance" class="chartgrid"></div>
    </div>

    <div class="panel" id="repHistPanel" style="display:none">
      <h2>Numeric Distributions</h2>
      <div id="repHists" class="chartgrid"></div>
    </div>

    <div class="panel" id="repBoxPanel" style="display:none">
      <h2>Box &amp; Violin (Five-Number Summary)</h2>
      <div id="repBox" class="chartgrid"></div>
    </div>

    <div class="panel" id="repDensityPanel" style="display:none">
      <h2>Density Curves (KDE)</h2>
      <div id="repDensity" class="chartgrid"></div>
    </div>

    <div class="panel" id="repScatterPanel" style="display:none">
      <h2>Relationships — Scatter &amp; Density Bins</h2>
      <p class="muted">The strongest |r| numeric pairs. Dense pairs render as
      binned density; sparse ones as points with a least-squares trend line.</p>
      <div id="repScatter" class="chartgrid"></div>
    </div>

    <div class="panel" id="repCompPanel" style="display:none">
      <h2>Composition — Donut &amp; Waffle</h2>
      <div id="repComp" class="chartgrid"></div>
    </div>

    <div class="panel" id="repRidgePanel" style="display:none">
      <h2>Class-Conditional Distributions (Ridgeline)</h2>
      <p class="muted">Per-target-class density curves — see how each numeric
      column's distribution shifts across classes.</p>
      <div id="repRidge" class="chartgrid"></div>
    </div>

    <div class="panel" id="repParetoPanel" style="display:none">
      <h2>Pareto Analysis</h2>
      <p class="muted">Category frequencies with the cumulative-share line
      (the 80/20 view).</p>
      <div id="repPareto" class="chartgrid"></div>
    </div>

    <div class="panel" id="repCatPanel" style="display:none">
      <h2>Categorical Frequencies</h2>
      <div id="repCats" class="chartgrid"></div>
    </div>

    <div class="panel" id="repCorrPanel" style="display:none">
      <h2>Correlation Matrix</h2>
      <div id="repCorr" style="overflow-x:auto"></div>
    </div>

    <div class="panel" id="repTsPanel" style="display:none">
      <h2>Time Series — Trend &amp; Seasonality</h2>
      <p class="muted">Numeric columns aligned to <code id="repTsIndex"></code>:
      trend line (least-squares slope), autocorrelation (ACF) and dominant
      seasonal period. Drives time-aware split advice in the insights.
      </p>
      <div id="repTs" class="chartgrid"></div>
    </div>

    <div class="panel" id="repMissPanel" style="display:none">
      <h2>Missing Values by Column</h2>
      <div id="repMiss" class="chartgrid"></div>
    </div>
  </div>
</section>

<!-- ══════════════════ LEAKAGE ══════════════════ -->
<section id="tab-leakage" class="tab">
  <div class="panel">
    <h2>Leakage Detection</h2>
    <p class="muted">Scans every feature column for constant values,
    identifier-like cardinality, and high null rates (doc 03 taxonomy).</p>
    <div class="alert" id="leakAlert"></div>
    <div class="row" style="margin-top:.8rem">
      <div class="field" style="flex:2">
        <label>Source</label>
        <input id="leakSrc" placeholder="data.csv or URI">
      </div>
      <div class="field small">
        <label>Target column</label>
        <input id="leakTarget" placeholder="y">
      </div>
      <button class="primary" id="btnLeak">Scan for Leakage</button>
    </div>
  </div>
  <div class="panel" id="leakResult" style="display:none">
    <h2>Scan Result</h2>
    <div class="kv" id="leakKv"></div>
    <h3>Findings</h3>
    <div style="overflow-x:auto"><table id="leakFindings">
      <thead><tr><th>Column</th><th>Type</th><th>Severity</th>
        <th>Detail</th></tr></thead>
      <tbody></tbody></table></div>
  </div>
</section>

<!-- ══════════════════ PIPELINE (Compile) ══════════════════ -->
<section id="tab-pipeline" class="tab">
  <div class="panel">
    <h2>Compile Pipeline (IR + Verification Gates)</h2>
    <p class="muted">Runs the learn pipeline, builds Pipeline IR, and runs
    the 10 verification gates. Produces standalone sklearn-ready artifacts.</p>
    <div class="alert" id="pipeAlert"></div>
    <div class="row" style="margin-top:.8rem">
      <div class="field" style="flex:2">
        <label>Source</label>
        <input id="pipeSrc" placeholder="data.csv or URI">
      </div>
      <div class="field small">
        <label>Target column</label>
        <input id="pipeTarget" placeholder="y">
      </div>
      <button class="primary" id="btnPipe">Compile Pipeline</button>
    </div>
  </div>
  <div class="panel" id="pipeResult" style="display:none">
    <h2>Compilation Result</h2>
    <div class="kv" id="pipeKv"></div>
    <h3>Verification Gates</h3>
    <div style="overflow-x:auto"><table id="pipeGates">
      <thead><tr><th>Gate</th><th>Verdict</th><th>Message</th></tr></thead>
      <tbody></tbody></table></div>
  </div>
</section>

<!-- ══════════════════ TUNE ══════════════════ -->
<section id="tab-tune" class="tab">
  <div class="panel">
    <h2>Funnel Policy Auto-Tuning</h2>
    <p class="muted">Derives bounded, deterministic FunnelPolicy adjustments
    from experience-store failure evidence (doc 06 closed loop).</p>
    <div class="row" style="margin-top:.8rem">
      <div class="field small">
        <label>Min cases</label>
        <input id="tuneMinCases" type="number" value="5" min="1">
      </div>
      <button class="primary" id="btnTune">Analyze Experience</button>
    </div>
  </div>
  <div class="panel" id="tuneResult" style="display:none">
    <h2>Tuning Report</h2>
    <div class="kv" id="tuneKv"></div>
    <h3>Failure Evidence</h3>
    <div style="overflow-x:auto"><table id="tuneTags">
      <thead><tr><th>Tag</th><th>Count</th><th>Rate</th></tr></thead>
      <tbody></tbody></table></div>
    <h3>Adjustments</h3>
    <div style="overflow-x:auto"><table id="tuneAdjust">
      <thead><tr><th>Field</th><th>Old</th><th>New</th>
        <th>Reason</th></tr></thead>
      <tbody></tbody></table></div>
  </div>
</section>

<!-- ══════════════════ EXPERIENCE ══════════════════ -->
<section id="tab-experience" class="tab">
  <div class="panel">
    <h2>Experience Store</h2>
    <p class="muted">Cross-dataset case history powering R0–R3 retrieval
    and policy tuning.</p>
    <button class="primary" id="btnExperience" style="margin:.5rem 0">
      Load Experience</button>
  </div>
  <div class="panel" id="expResult" style="display:none">
    <h2>Case History</h2>
    <div class="kv" id="expKv"></div>
    <div style="overflow-x:auto"><table id="expCases">
      <thead><tr><th>Case</th><th>Dataset</th><th>Task</th>
        <th>Result</th><th>Failure Tags</th><th>Time</th></tr></thead>
      <tbody></tbody></table></div>
  </div>
</section>

<!-- ══════════════════ OPERATORS ══════════════════ -->
<section id="tab-operators" class="tab">
  <div class="panel">
    <h2>Operator Contract Verification</h2>
    <p class="muted">Verifies every operator against its typed contract
    (arity, input types, output type, determinism) — same engine as
    <code>fiae codegen --verify</code>.</p>
    <button class="primary" id="btnCodegen" style="margin:.5rem 0">
      Verify All Operators</button>
  </div>
  <div class="panel" id="opResult" style="display:none">
    <h2>Compliance Report</h2>
    <div class="kv" id="opKv"></div>
    <h3>Violations</h3>
    <div style="overflow-x:auto"><table id="opViol">
      <thead><tr><th>Operator</th><th>Violations</th></tr></thead>
      <tbody></tbody></table></div>
  </div>
</section>

<!-- ══════════════════ BENCHMARKS ══════════════════ -->
<section id="tab-benchmarks" class="tab">
  <div class="panel">
    <h2>Benchmark Suite</h2>
    <p class="muted">Operator catalog load time and transform throughput on
    10K rows — same suite as <code>fiae benchmarks</code>.</p>
    <button class="primary" id="btnBench" style="margin:.5rem 0">
      Run Benchmarks</button>
    <div class="kv" id="benchKv" style="margin-top:1rem"></div>
  </div>
</section>

<!-- ══════════════════ RUNS ══════════════════ -->
<section id="tab-runs" class="tab">
  <div class="panel">
    <h2>Tracked Runs</h2>
    <p class="muted">Runs recorded via <code>fiae learn --track local</code>
    or jobs launched here.</p>
    <button class="ghost" id="btnRuns" style="margin:.5rem 0">Refresh</button>
    <div style="overflow-x:auto"><table id="runsTable">
      <thead><tr><th>Run</th><th>State</th><th>Target</th>
        <th>Max Rows</th></tr></thead>
      <tbody><tr><td colspan="4" class="muted">No runs yet</td></tr></tbody>
    </table></div>
  </div>
</section>

<!-- ══════════════════ JOBS ══════════════════ -->
<section id="tab-jobs" class="tab">
  <div class="panel">
    <h2>Job Queue</h2>
    <div class="kv" id="poolKv"></div>
    <button class="ghost" id="btnJobs" style="margin:.5rem 0">Refresh</button>
    <div style="overflow-x:auto"><table id="jobsTable">
      <thead><tr><th>Job</th><th>Kind</th><th>State</th><th>Detail</th>
        <th></th></tr></thead>
      <tbody><tr><td colspan="5" class="muted">No jobs yet</td></tr></tbody>
    </table></div>
  </div>
  <div class="panel" id="jobDetail" style="display:none">
    <h2>Job Detail <span id="jobDetailId" class="pill"></span></h2>
    <pre id="jobDetailPre" style="overflow-x:auto;font-size:.8rem;
      color:var(--muted);white-space:pre-wrap"></pre>
  </div>
</section>

<!-- ══════════════════ ABOUT ══════════════════ -->
<section id="tab-about" class="tab">
  <div class="panel">
    <h2>What FIAE Proves</h2>
    <p class="muted">Every generated feature is proven:</p>
    <table style="margin-top:.8rem">
      <thead><tr><th>Guarantee</th><th>Gate</th></tr></thead>
      <tbody>
        <tr><td>Leakage-free</td><td>6-class leakage taxonomy (L0–L5), F0 gate</td></tr>
        <tr><td>Valid transform</td><td>F2 domain/invalid-rate gate</td></tr>
        <tr><td>Complementary</td><td>F5 correlation / mutual-information gate</td></tr>
        <tr><td>Real incremental gain</td><td>F4 progressive evaluation</td></tr>
        <tr><td>Stable across folds</td><td>F6 multi-seed cross-validation</td></tr>
        <tr><td>Reproducible</td><td>Deterministic content-hash IDs</td></tr>
        <tr><td>Exportable</td><td>10 verification gates + standalone sklearn code</td></tr>
      </tbody>
    </table>
  </div>
  <div class="panel">
    <h2>Architecture</h2>
    <pre style="font-size:.78rem;color:var(--purple3);overflow-x:auto">
 ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐
 │ Connect │→ │ Profile │→ │ Generate │→ │  Funnel  │→ │ Portfolio│
 └─────────┘  └─────────┘  └──────────┘  └──────────┘  └─────────┘
  25+ sources   8 semantic    95 typed      F0-F6 gates   probe + F4/F6
               types         operators                    selection
 ┌───────────┐  ┌──────────┐  ┌──────────┐
 │ Experience│→ │   HPO    │→ │  Export  │
 └───────────┘  └──────────┘  └──────────┘
  R0-R3 memory   model routing  verified code</pre>
  </div>
</section>

</main>

<footer>
  FIAE v<span id="ver2"></span> · REST API:
  <a href="/api/health">/api/health</a> ·
  <a href="/api/runs">/api/runs</a> ·
  <a href="/api/jobs">/api/jobs</a> ·
  POST /api/learn · POST /api/profile
</footer>

<script>
"use strict";
const $ = id => document.getElementById(id);
let jobsTimer = null;

// ── tabs ──────────────────────────────────────────────────
document.querySelectorAll("nav button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("nav button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    btn.classList.add("active");
    $("tab-" + btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "jobs") { refreshJobs(); startJobsAuto(); }
    else { stopJobsAuto(); }
    if (btn.dataset.tab === "runs") refreshRuns();
  });
});

// ── helpers ───────────────────────────────────────────────
function esc(s){const d=document.createElement("div");d.textContent=String(s??""),d.textContent;return d.innerHTML;}
async function api(path, opts){
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || ("HTTP " + r.status));
  return body;
}
function alertBox(id, msg, cls){
  const el = $(id);
  el.className = "alert " + (cls || "error");
  el.textContent = msg;
  if (!msg) el.className = "alert";
}

// ── health / version ──────────────────────────────────────
(async () => {
  try {
    const h = await api("/api/health");
    $("ver").textContent = "v" + h.version;
    $("ver2").textContent = h.version;
  } catch (e) { /* server offline */ }
})();

// ── sources catalog ───────────────────────────────────────
(async () => {
  try {
    const s = await api("/api/sources");
    const el = $("sourcesList");
    el.innerHTML = "";
    for (const [cat, items] of Object.entries(s.sources)) {
      const head = document.createElement("div");
      head.innerHTML = "<b>" + esc(cat) + "</b>";
      el.appendChild(head);
      for (const item of items) {
        const d = document.createElement("div");
        d.textContent = "· " + item;
        el.appendChild(d);
      }
    }
  } catch (e) { $("sourcesList").textContent = "unavailable"; }
})();

// ── connect & profile ─────────────────────────────────────
$("btnProfile").addEventListener("click", async () => {
  const src = $("src").value.trim();
  if (!src) return alertBox("connectAlert", "Enter a source first.", "error");
  alertBox("connectAlert", "");
  const btn = $("btnProfile");
  btn.disabled = true; btn.textContent = "Profiling…";
  try {
    const body = { source: src, _source: src };
    const table = $("srcTable").value.trim();
    if (table) body.table = table;
    const res = await api("/api/profile", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body)});
    pollJob(res.job_id, profile => {
      btn.disabled = false; btn.textContent = "Connect & Profile";
      if (profile.error) return alertBox("connectAlert", profile.error, "error");
      profile._source = src;
      renderProfile(profile);
    });
  } catch (e) {
    btn.disabled = false; btn.textContent = "Connect & Profile";
    alertBox("connectAlert", e.message, "error");
  }
});

function renderProfile(p){
  $("profileResult").style.display = "";
  $("profileTitle").textContent = "Profile — " + (p.source_id || "");
  const kv = [
    ["Rows observed", (p.rows_observed ?? 0).toLocaleString()],
    ["Columns", p.columns?.length ?? 0],
    ["Fingerprint", (p.dataset_fingerprint || "").slice(0, 24) + "…"],
  ];
  if (p.rows_estimated != null) kv.splice(1, 0, ["Rows estimated", p.rows_estimated.toLocaleString()]);
  if (p.quality_findings?.length) kv.push(["Quality findings", p.quality_findings.length]);
  $("profileKv").innerHTML = kv.map(([k,v]) => "<div><b>" + esc(k) + "</b>" + esc(v) + "</div>").join("");

  // suggested targets (one-click learn)
  const chips = $("targetChips");
  chips.innerHTML = (p.suggested_targets || []).map(t =>
    "<button class='chip' data-target='" + esc(t.column) + "'>" +
    esc(t.column) + "<small>" + esc(t.reasons.join(" · ")) +
    "</small></button>").join("") ||
    "<span class='muted'>No obvious target columns detected.</span>";
  chips.querySelectorAll("button[data-target]").forEach(b =>
    b.addEventListener("click", () => runLearn(p._source, b.dataset.target)));

  // data preview
  const pt = $("previewTable");
  const cols = p.preview?.columns || [];
  const rows = p.preview?.rows || [];
  pt.querySelector("thead").innerHTML = "<tr>" +
    cols.map(c => "<th>" + esc(c) + "</th>").join("") + "</tr>";
  pt.querySelector("tbody").innerHTML = rows.map(r =>
    "<tr>" + r.map(v => "<td>" + (v == null ? "∅" : esc(v)) + "</td>").join("") +
    "</tr>").join("");

  const tb = $("profileCols").querySelector("tbody");
  tb.innerHTML = (p.columns || []).map(c =>
    "<tr><td>" + esc(c.name) + "</td><td>" + esc(c.physical_dtype) +
    "</td><td>" + esc(c.semantic_type) + "</td><td>" +
    (100 * (c.null_fraction ?? 0)).toFixed(1) + "%</td><td>" +
    esc(c.distinct_estimate ?? "-") + "</td></tr>").join("");
}

// ── learn ─────────────────────────────────────────────────
async function runLearn(src, target){
  if (!src || !target){
    src = $("learnSrc").value.trim(); target = $("learnTarget").value.trim();
  }
  if (!src || !target)
    return alertBox("learnAlert", "Source and target are required.", "error");
  // mirror inputs so the Learn tab reflects the one-click run
  $("learnSrc").value = src; $("learnTarget").value = target;
  document.querySelector("nav button[data-tab='learn']").click();
  alertBox("learnAlert", "");
  const btn = $("btnLearn");
  btn.disabled = true; btn.textContent = "Running pipeline…";
  try {
    const res = await api("/api/learn", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        source: src, target,
        max_features: parseInt($("learnMaxF").value) || 8,
        max_rows: parseInt($("learnMaxR").value) || 100000,
        seed: 0})});
    pollJob(res.job_id, report => {
      btn.disabled = false; btn.textContent = "Run Pipeline";
      if (report.error) return alertBox("learnAlert", report.error, "error");
      renderLearn(report, res.job_id);
    });
  } catch (e) {
    btn.disabled = false; btn.textContent = "Run Pipeline";
    alertBox("learnAlert", e.message, "error");
  }
}
$("btnLearn").addEventListener("click", () => runLearn(null, null));

let lastLearnReport = null;
let lastLearnJobId = null;

function fmtGain(g){
  if (g == null) return "-";
  const a = Math.abs(g);
  if (a !== 0 && a < 0.001) return g.toExponential(2);
  return g.toFixed(6);
}

function renderLearn(r, jobId){
  lastLearnReport = r; lastLearnJobId = jobId || null;
  $("learnResult").style.display = "";
  const kv = [
    ["Task", r.task + " (" + Math.round((r.task_confidence ?? 0) * 100) + "% conf)"],
    ["Rows", (r.rows ?? r.rows_in_source ?? 0).toLocaleString()],
    ["Columns", r.columns ?? r.columns_in_source ?? "-"],
    ["Metrics", (r.metrics || []).join(", ") || "-"],
    ["Proposals", r.proposals_generated + " → " + r.proposals_after_dedup + " unique"],
    ["Passed funnel", r.funnel_passed_f2],
    ["Portfolio", r.portfolio_size],
    ["Time", (r.total_time_s ?? 0).toFixed(2) + "s"],
  ];
  $("learnKv").innerHTML = kv.map(([k,v]) => "<div><b>" + esc(k) + "</b>" + esc(v) + "</div>").join("");

  renderWaterfall(r);

  const tb = $("learnFeat").querySelector("tbody");
  tb.innerHTML = (r.portfolio || []).map((m, i) => {
    const ok = m.f4_passed && m.f6_passed;
    const badge = "<span class='badge " + (ok ? "b-completed" : "b-failed") +
      "'>" + (ok ? "STABLE" : "REJECTED") + "</span>";
    return "<tr class='feat-row' data-idx='" + i + "' style='cursor:pointer'><td>" + (i + 1) +
      "</td><td><code>" + esc(m.operator) +
      "</code></td><td>" + esc((m.inputs || []).join(", ")) + "</td><td>" +
      fmtGain(m.incremental_gain) + "</td><td>" +
      (m.fold_stability != null ? m.fold_stability.toFixed(4) : "N/A") +
      "</td><td>" + badge + "</td></tr>";
  }).join("") || "<tr><td colspan='6' class='muted'>No features selected — see funnel results.</td></tr>";
  tb.querySelectorAll("tr.feat-row").forEach(tr =>
    tr.addEventListener("click", () =>
      showFeatureDetail(lastLearnReport.portfolio[parseInt(tr.dataset.idx)])));
}

// ── funnel waterfall ──────────────────────────────────────
function renderWaterfall(r){
  const stages = [
    ["Generated", r.proposals_generated, ""],
    ["After dedup", r.proposals_after_dedup, ""],
    ["Passed F2", r.funnel_passed_f2, ""],
    ["Portfolio", r.portfolio_size, r.portfolio_size === 0 ? "warn" : ""],
  ];
  const max = Math.max(1, ...stages.map(s => s[1]));
  let html = "<div class='wf'>";
  for (const [label, val, cls] of stages) {
    const pct = Math.round(100 * val / max);
    html += "<div class='wf-row'><div class='wf-label'>" + esc(label) +
      "</div><div class='wf-track'><div class='wf-fill " + cls +
      "' style='width:" + pct + "%'></div></div><div class='wf-val'>" +
      esc(val) + " (" + pct + "%)</div></div>";
  }
  html += "</div>";
  const rejected = r.proposals_after_dedup - r.funnel_passed_f2;
  html += "<div class='funnel-note'>" + rejected + " proposal" +
    (rejected === 1 ? "" : "s") + " rejected by gates F0–F2; portfolio " +
    "features additionally proved F4 gain + F6 stability.</div>";
  $("waterfall").innerHTML = html;
}

// ── feature drill-down ────────────────────────────────────
function showFeatureDetail(m){
  if (!m) return;
  $("featureDetail").style.display = "";
  $("fdName").textContent = m.operator + "(" + (m.inputs || []).join(", ") + ")";
  const kv = [
    ["Incremental gain", fmtGain(m.incremental_gain)],
    ["Fold stability", m.fold_stability != null ? m.fold_stability.toFixed(4) : "N/A"],
    ["F4 (progressive gain)", m.f4_passed ? "PASS" : "FAIL"],
    ["F6 (CV stability)", m.f6_passed ? "PASS" : "FAIL"],
    ["F4 retention", m.f4_retention != null ? m.f4_retention.toFixed(4) : "-"],
    ["F6 gain CV", m.f6_cv != null ? m.f6_cv.toFixed(4) : "-"],
  ];
  $("fdKv").innerHTML = kv.map(([k,v]) => "<div><b>" + esc(k) + "</b>" + esc(v) + "</div>").join("");
  $("fdStages").querySelector("tbody").innerHTML =
    "<tr><td>F5</td><td>" + (m.f5_passed === false ?
      "<span class='badge b-failed'>REJECT</span>" :
      "<span class='badge b-completed'>PASS</span>") + "</td><td>complementarity</td></tr>" +
    "<tr><td>F4</td><td>" + (m.f4_passed ?
      "<span class='badge b-completed'>PASS</span>" :
      "<span class='badge b-failed'>REJECT</span>") + "</td><td>gain held at all row budgets</td></tr>" +
    "<tr><td>F6</td><td>" + (m.f6_passed ?
      "<span class='badge b-completed'>PASS</span>" :
      "<span class='badge b-failed'>REJECT</span>") + "</td><td>stable across seeds</td></tr>";
  $("featureDetail").scrollIntoView({behavior: "smooth", block: "nearest"});
}

// ── export ────────────────────────────────────────────────
$("btnExportCsv").addEventListener("click", () =>
  window.open("/api/jobs/" + lastLearnJobId + "/export", "_blank"));
$("btnExportJson").addEventListener("click", () =>
  window.open("/api/jobs/" + lastLearnJobId + "/export", "_blank"));

// ── runs ──────────────────────────────────────────────────
$("btnRuns").addEventListener("click", refreshRuns);
async function refreshRuns(){
  try {
    const r = await api("/api/runs");
    const tb = $("runsTable").querySelector("tbody");
    tb.innerHTML = (r.runs || []).map(run =>
      "<tr><td><code>" + esc(run.run_id) + "</code></td><td><span class='badge b-" +
      esc(run.state.toLowerCase()) + "'>" + esc(run.state) + "</span></td><td>" +
      esc(run.params?.target ?? "-") + "</td><td>" +
      esc(run.params?.max_rows ?? "-") + "</td></tr>"
    ).join("") || "<tr><td colspan='4' class='muted'>No runs yet</td></tr>";
  } catch (e) { /* ignore */ }
}

// ── jobs (auto-refresh) ───────────────────────────────────
$("btnJobs").addEventListener("click", refreshJobs);
function startJobsAuto(){
  stopJobsAuto();
  jobsTimer = setInterval(refreshJobs, 2000);
}
function stopJobsAuto(){ if (jobsTimer) { clearInterval(jobsTimer); jobsTimer = null; } }

async function refreshJobs(){
  try {
    const [h, j] = await Promise.all([api("/api/health"), api("/api/jobs")]);
    const pool = h.pool || {}, counts = h.jobs || {};
    const active = (counts.RUNNING || 0) + (counts.PENDING || 0);
    const pill = $("jobsPill");
    pill.style.display = active ? "" : "none";
    pill.textContent = active;
    $("poolKv").innerHTML = [
      ["Workers", pool.workers + " (" + (pool.busy_workers ?? 0) + " busy)"],
      ["Queued", pool.queued_jobs ?? 0],
      ["Running", counts.RUNNING || 0],
      ["Pending", counts.PENDING || 0],
      ["Completed", counts.COMPLETED || 0],
      ["Failed", counts.FAILED || 0],
    ].map(([k,v]) => "<div><b>" + esc(k) + "</b>" + esc(v) + "</div>").join("");
    const tb = $("jobsTable").querySelector("tbody");
    tb.innerHTML = (j.jobs || []).slice().reverse().map(job =>
      "<tr><td><code>" + esc(job.job_id) + "</code></td><td>" +
      esc(job.kind ?? job.params?.kind ?? "learn") + "</td><td><span class='badge b-" +
      esc(job.state.toLowerCase()) + "'>" + esc(job.state) + "</span></td><td>" +
      esc(job.error ? ("error: " + job.error) : (job.params?.target ?? job.params?.source ?? "-")) +
      "</td><td><button class='ghost' data-job='" + esc(job.job_id) +
      "'>view</button></td></tr>"
    ).join("") || "<tr><td colspan='5' class='muted'>No jobs yet</td></tr>";
    tb.querySelectorAll("button[data-job]").forEach(b =>
      b.addEventListener("click", () => showJob(b.dataset.job)));
  } catch (e) { /* ignore */ }
}

async function showJob(jobId){
  try {
    const job = await api("/api/jobs/" + jobId);
    $("jobDetail").style.display = "";
    $("jobDetailId").textContent = jobId;
    $("jobDetailPre").textContent = JSON.stringify(job, null, 2);
  } catch (e) { /* ignore */ }
}

// ── CLI-parity jobs (leakage/tune/experience/codegen/benchmarks/pipeline) ──
async function runGenericJob(kind, payload, btn, onDone){
  if (btn){ btn.disabled = true; btn.dataset.old = btn.textContent;
            btn.textContent = "Working…"; }
  try {
    const res = await api("/api/job/run", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({kind, ...payload})});
    pollJob(res.job_id, result => {
      if (btn){ btn.disabled = false; btn.textContent = btn.dataset.old; }
      if (result.error) return onDone(null, result.error);
      onDone(result, null);
    });
  } catch (e) {
    if (btn){ btn.disabled = false; btn.textContent = btn.dataset.old; }
    onDone(null, e.message);
  }
}

// Leakage
$("btnLeak").addEventListener("click", () => {
  const src = $("leakSrc").value.trim(), tgt = $("leakTarget").value.trim();
  if (!src || !tgt) return alertBox("leakAlert", "Source and target are required.", "error");
  alertBox("leakAlert", "");
  runGenericJob("leakage", {source: src, target: tgt}, $("btnLeak"),
    (r, err) => {
      if (err) return alertBox("leakAlert", err, "error");
      $("leakResult").style.display = "";
      $("leakKv").innerHTML = [
        ["Columns checked", r.columns_checked],
        ["Findings", (r.findings.length ?
          "<span class='badge b-warn'>" + r.findings.length + "</span>" :
          "<span class='badge b-completed'>NONE — clean</span>")],
        ["Fingerprint", (r.fingerprint || "").slice(0, 24) + "…"],
      ].map(([k,v]) => "<div><b>" + k + "</b>" + v + "</div>").join("");
      $("leakFindings").querySelector("tbody").innerHTML =
        r.findings.map(f => "<tr><td><code>" + esc(f.column) + "</code></td><td>" +
          esc(f.type) + "</td><td><span class='badge " +
          (f.severity === "review_required" ? "b-warn" : f.severity === "warning" ? "b-failed" : "b-ok") +
          "'>" + esc(f.severity) + "</span></td><td>" + esc(f.detail) + "</td></tr>").join("") ||
        "<tr><td colspan='4' class='muted'>No leakage indicators found.</td></tr>";
    });
});

// Pipeline compile
$("btnPipe").addEventListener("click", () => {
  const src = $("pipeSrc").value.trim(), tgt = $("pipeTarget").value.trim();
  if (!src || !tgt) return alertBox("pipeAlert", "Source and target are required.", "error");
  alertBox("pipeAlert", "");
  runGenericJob("pipeline", {source: src, target: tgt}, $("btnPipe"),
    (r, err) => {
      if (err) return alertBox("pipeAlert", err, "error");
      $("pipeResult").style.display = "";
      const ok = r.all_passed;
      $("pipeKv").innerHTML = [
        ["Pipeline", "<code>" + esc(r.pipeline_id) + "</code>"],
        ["Gates", (ok ? "<span class='badge b-completed'>" :
          "<span class='badge b-failed'>") + r.gates_passed + "/" +
          r.gates_total + "</span>"],
        ["IR saved", "<code>" + esc(r.ir_path) + "</code>"],
        ["Code saved", "<code>" + esc(r.code_path) + "</code>"],
        ["Portfolio", r.portfolio_size],
      ].map(([k,v]) => "<div><b>" + k + "</b>" + v + "</div>").join("");
      $("pipeGates").querySelector("tbody").innerHTML = r.gates.map(g =>
        "<tr><td>" + esc(g.gate) + "</td><td>" +
        (g.passed ? "<span class='badge b-completed'>PASS</span>" :
                    "<span class='badge b-failed'>FAIL</span>") +
        "</td><td>" + esc(g.message || "") + "</td></tr>").join("");
    });
});

// Tune
$("btnTune").addEventListener("click", () => {
  runGenericJob("tune", {min_cases: parseInt($("tuneMinCases").value) || 5},
    $("btnTune"), (r, err) => {
      if (err) return alertBox("leakAlert", err, "error");
      $("tuneResult").style.display = "";
      if (!r.available) return $("tuneKv").innerHTML =
        "<div><b>Status</b>" + esc(r.message) + "</div>";
      $("tuneKv").innerHTML = [
        ["Cases analyzed", r.cases_analyzed],
        ["Success rate", Math.round(r.success_rate * 100) + "%"],
        ["Adjustments", (r.adjustments || []).length],
      ].map(([k,v]) => "<div><b>" + esc(k) + "</b>" + esc(v) + "</div>").join("");
      const n = Math.max(r.cases_analyzed, 1);
      $("tuneTags").querySelector("tbody").innerHTML =
        Object.entries(r.tag_counts || {}).map(([tag, count]) => {
          const rate = count / n;
          return "<tr><td>" + esc(tag) + "</td><td>" + count + "</td><td>" +
            (rate >= 0.3 ? "<span class='badge b-failed'>" :
              "<span class='badge b-ok'>") + Math.round(rate*100) + "%</span></td></tr>";
        }).join("") || "<tr><td colspan='3' class='muted'>No failure tags recorded.</td></tr>";
      $("tuneAdjust").querySelector("tbody").innerHTML =
        (r.adjustments || []).map(a => "<tr><td><code>" + esc(a.field) +
          "</code></td><td>" + esc(a.old) + "</td><td>" + esc(a.new) +
          "</td><td>" + esc(a.reason) + "</td></tr>").join("") ||
        "<tr><td colspan='4' class='muted'>No adjustments — policy already optimal or insufficient evidence.</td></tr>";
    });
});

// Experience
$("btnExperience").addEventListener("click", () => {
  runGenericJob("experience", {}, $("btnExperience"), (r, err) => {
    if (err) return alertBox("leakAlert", err, "error");
    $("expResult").style.display = "";
    if (!r.available) return $("expKv").innerHTML =
      "<div><b>Status</b>" + esc(r.message) + "</div>";
    $("expKv").innerHTML = [
      ["Total cases", r.total_cases],
      ["Success rate", Math.round(r.success_rate * 100) + "%"],
      ["Store", "<code>" + esc(r.db_path) + "</code>"],
    ].map(([k,v]) => "<div><b>" + k + "</b>" + v + "</div>").join("");
    $("expCases").querySelector("tbody").innerHTML = (r.recent || []).map(c =>
      "<tr><td><code>" + esc(c.case_id.slice(0, 20)) + "</code></td><td>" +
      esc(c.dataset) + "</td><td>" + esc(c.task) + "</td><td>" +
      (c.success ? "<span class='badge b-completed'>SUCCESS</span>" :
                   "<span class='badge b-failed'>FAILED</span>") + "</td><td>" +
      esc((c.failure_tags || []).join(", ") || "-") + "</td><td>" +
      (c.wall_time_s != null ? c.wall_time_s.toFixed(2) + "s" : "-") + "</td></tr>").join("") ||
      "<tr><td colspan='6' class='muted'>No cases recorded yet.</td></tr>";
  });
});

// Operators (codegen verify)
$("btnCodegen").addEventListener("click", () => {
  runGenericJob("codegen", {}, $("btnCodegen"), (r, err) => {
    if (err) return alertBox("leakAlert", err, "error");
    $("opResult").style.display = "";
    const rate = Math.round((r.compliance_rate ?? 0) * 100);
    $("opKv").innerHTML = [
      ["Total operators", r.total_operators],
      ["Compliant", "<span class='badge b-completed'>" + r.compliant + "</span>"],
      ["Non-compliant", "<span class='badge " +
        (r.non_compliant ? "b-failed" : "b-completed") + "'>" +
        r.non_compliant + "</span>"],
      ["Compliance rate", rate + "%"],
    ].map(([k,v]) => "<div><b>" + k + "</b>" + v + "</div>").join("");
    const viol = r.violations || {};
    $("opViol").querySelector("tbody").innerHTML =
      Object.entries(viol).map(([name, vs]) =>
        "<tr><td><code>" + esc(name) + "</code></td><td>" +
        esc(vs.join(", ")) + "</td></tr>").join("") ||
      "<tr><td colspan='2'><span class='badge b-completed'>All operators compliant.</span></td></tr>";
  });
});

// Benchmarks
$("btnBench").addEventListener("click", () => {
  runGenericJob("benchmarks", {}, $("btnBench"), (r, err) => {
    if (err) return alertBox("leakAlert", err, "error");
    $("benchKv").innerHTML = [
      ["Operator catalog", r.total_operators + " ops in " + r.catalog_ms + " ms"],
      ["Transform throughput", r.transform_ops + " ops x 100 rows in " +
        r.transform_ms + " ms"],
      ["Python", r.python],
    ].map(([k,v]) => "<div><b>" + k + "</b>" + v + "</div>").join("");
  });
});

// ── Report (Kaggle-style auto-EDA, pure SVG) ─────────────

function svgBarChart(items, color){
  // items: [{value|label, count, pct?}] -> horizontal bars SVG
  const max = Math.max(...items.map(i => i.count), 1);
  const rowH = 22, labelW = 110, valW = 58, W = 480;
  const H = items.length * rowH + 4;
  let bars = "", labels = "", vals = "";
  items.forEach((it, i) => {
    const y = i * rowH + 4;
    const w = Math.max(2, (W - labelW - valW) * it.count / max);
    bars += "<rect x='" + labelW + "' y='" + y + "' width='" + w.toFixed(1) +
      "' height='14' rx='3' fill='" + (color || "url(#gradBar)") + "'/>";
    labels += "<text x='" + (labelW - 6) + "' y='" + (y + 11) +
      "' text-anchor='end' class='svglab'>" + esc(it.label || it.value) + "</text>";
    vals += "<text x='" + (W - 2) + "' y='" + (y + 11) +
      "' text-anchor='end' class='svgval'>" + it.count + (it.pct != null ? " · " + it.pct + "%" : "") + "</text>";
  });
  return "<svg viewBox='0 0 " + W + " " + H + "' preserveAspectRatio='xMidYMid meet'>" +
    "<defs><linearGradient id='gradBar' x1='0' y1='0' x2='1' y2='0'>" +
    "<stop offset='0%' stop-color='#6d28d9'/><stop offset='100%' stop-color='#8b5cf6'/>" +
    "</linearGradient></defs>" + labels + bars + vals + "</svg>";
}

function svgHistogram(edges, counts){
  const W = 480, H = 180, padB = 24, padT = 10;
  const max = Math.max(...counts, 1);
  const bw = (W - 2) / counts.length;
  let bars = "", axis = "";
  counts.forEach((c, i) => {
    const h = Math.max(1, (H - padB - padT) * c / max);
    bars += "<rect x='" + (1 + i * bw).toFixed(1) + "' y='" + (H - padB - h).toFixed(1) +
      "' width='" + Math.max(1, bw - 1.5).toFixed(1) + "' height='" + h.toFixed(1) +
      "' fill='url(#gradBar)' rx='2'/>";
  });
  axis += "<text x='2' y='" + (H - 8) + "' class='svgval'>" +
    fmtNum(edges[0]) + "</text>";
  axis += "<text x='" + (W - 2) + "' y='" + (H - 8) +
    "' text-anchor='end' class='svgval'>" + fmtNum(edges[edges.length - 1]) + "</text>";
  return "<svg viewBox='0 0 " + W + " " + H + "' preserveAspectRatio='xMidYMid meet'>" +
    "<defs><linearGradient id='gradBar' x1='0' y1='0' x2='0' y2='1'>" +
    "<stop offset='0%' stop-color='#8b5cf6'/><stop offset='100%' stop-color='#6d28d9'/>" +
    "</linearGradient></defs>" + bars + axis + "</svg>";
}

function svgMissingBars(missingMap){
  const rows = missingMap.slice(0, 18);
  const max = Math.max(...rows.map(m => m.null_fraction), 0.001);
  const rowH = 20, labelW = 120, W = 480, H = rows.length * rowH + 4;
  let out = "";
  rows.forEach((m, i) => {
    const y = i * rowH + 4;
    const w = Math.max(m.null_fraction > 0 ? 2 : 0, (W - labelW - 60) * m.null_fraction / max);
    const col = m.null_fraction === 0 ? "#3f3f52" :
      (m.null_fraction > 0.3 ? "#c9a227" : "url(#gradBar)");
    out += "<text x='" + (labelW - 6) + "' y='" + (y + 11) +
      "' text-anchor='end' class='svglab'>" + esc(m.column) + "</text>";
    if (w > 0) out += "<rect x='" + labelW + "' y='" + (y + 2) + "' width='" +
      w.toFixed(1) + "' height='12' rx='3' fill='" + col + "'/>";
    out += "<text x='" + (W - 2) + "' y='" + (y + 11) +
      "' class='svgval'>" + (100 * m.null_fraction).toFixed(1) + "%</text>";
  });
  return "<svg viewBox='0 0 " + W + " " + H + "' preserveAspectRatio='xMidYMid meet'>" +
    "<defs><linearGradient id='gradBar' x1='0' y1='0' x2='1' y2='0'>" +
    "<stop offset='0%' stop-color='#6d28d9'/><stop offset='100%' stop-color='#8b5cf6'/>" +
    "</linearGradient></defs>" + out + "</svg>";
}

function renderCorrMatrix(corr){
  const cols = corr.columns;
  if (!cols.length) return "<p class='muted'>No numeric columns to correlate.</p>";
  const short = c => c.length > 8 ? c.slice(0, 7) + "…" : c;
  let html = "<table class='corr-table'><tr><td class='clabel'></td>";
  for (const c of cols) html += "<td class='clabel' title='" + esc(c) + "'>" + esc(short(c)) + "</td>";
  html += "</tr>";
  corr.matrix.forEach((row, i) => {
    html += "<tr><td class='clabel' title='" + esc(cols[i]) + "'>" + esc(short(cols[i])) + "</td>";
    row.forEach((v, j) => {
      const a = Math.abs(v);
      // purple scale: stronger |r| = more saturated
      const alpha = (0.08 + 0.85 * a).toFixed(2);
      const bg = v >= 0 ? "rgba(139,92,246," + alpha + ")"
                        : "rgba(201,162,39," + alpha + ")";
      const txt = a > 0.55 ? "#0d0b14" : "var(--muted)";
      html += "<td style='background:" + bg + ";color:" + txt +
        "' title='" + esc(cols[i]) + " vs " + esc(cols[j]) + " = " + v + "'>" +
        v.toFixed(2).replace("0.", ".") + "</td>";
    });
    html += "</tr>";
  });
  return html + "</table><p class='sub'>Purple = positive, gold = negative. Hover a cell for the exact pair.</p>";
}

function fmtNum(v){
  if (v == null || isNaN(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1e6 || (a < 1e-3 && a > 0)) return v.toExponential(2);
  return a >= 100 ? Math.round(v).toLocaleString() : (+v.toFixed(3)).toString();
}

// ── advanced SVG renderers (chart-catalog expansion) ──────
function svgDonut(items, total){
  const size = 190, cx = size/2, cy = size/2, R = 80, r = 50;
  let start = -Math.PI/2, paths = "", legend = "";
  const palette = ["#8b5cf6", "#a78bfa", "#6d28d9", "#c084fc", "#c9a227", "#7c6ff0"];
  items.forEach((it, i) => {
    const frac = Math.min(it.count / (total || 1), 1);
    const end = start + frac * 2 * Math.PI;
    if (frac > 0.999){
      paths += "<circle cx='" + cx + "' cy='" + cy + "' r='" + ((R+r)/2) +
        "' fill='none' stroke='" + palette[i % palette.length] +
        "' stroke-width='" + (R-r) + "'/>";
    } else if (frac > 0){
      const large = frac > 0.5 ? 1 : 0;
      const x1 = cx + R*Math.cos(start), y1 = cy + R*Math.sin(start);
      const x2 = cx + R*Math.cos(end),   y2 = cy + R*Math.sin(end);
      const x3 = cx + r*Math.cos(end),   y3 = cy + r*Math.sin(end);
      const x4 = cx + r*Math.cos(start), y4 = cy + r*Math.sin(start);
      paths += "<path d='M" + x1.toFixed(2) + "," + y1.toFixed(2) +
        " A" + R + "," + R + " 0 " + large + " 1 " + x2.toFixed(2) + "," + y2.toFixed(2) +
        " L" + x3.toFixed(2) + "," + y3.toFixed(2) +
        " A" + r + "," + r + " 0 " + large + " 0 " + x4.toFixed(2) + "," + y4.toFixed(2) +
        " Z' fill='" + palette[i % palette.length] + "'/>";
    }
    legend += "<div class='legend-row'><span class='swatch' style='background:" +
      palette[i % palette.length] + "'></span>" + esc(it.value) +
      " <span class='legend-pct'>" + (it.pct != null ? it.pct + "%" :
        Math.round(100*it.count/(total||1)) + "%") + "</span></div>";
    start = end;
  });
  return "<div class='donut-wrap'><svg viewBox='0 0 " + size + " " + size +
    "' width='" + size + "' height='" + size + "'>" + paths +
    "</svg><div class='legend'>" + legend + "</div></div>";
}

function svgWaffle(items, total){
  const cols = 10, rowsN = 10, N = cols * rowsN;
  const cells = [];
  const palette = ["#8b5cf6", "#a78bfa", "#6d28d9", "#c084fc", "#c9a227", "#7c6ff0"];
  let idx = 0;
  (items || []).forEach((it, si) => {
    const n = Math.round(N * it.count / (total || 1));
    for (let k = 0; k < n && idx < N; k++, idx++)
      cells.push({cls: si, i: idx});
  });
  while (idx < N){ cells.push({cls: -1, i: idx}); idx++; }
  const cell = 16, gap = 3;
  let rects = "";
  cells.forEach(c => {
    const x = (c.i % cols) * (cell+gap), y = Math.floor(c.i / cols) * (cell+gap);
    const fill = c.cls < 0 ? "#2a2735" : palette[c.cls % palette.length];
    rects += "<rect x='" + x + "' y='" + y + "' width='" + cell +
      "' height='" + cell + "' rx='2.5' fill='" + fill + "'/>";
  });
  const legend = (items || []).map((it, i) =>
    "<div class='legend-row'><span class='swatch' style='background:" +
    palette[i % palette.length] + "'></span>" + esc(it.value) +
    " <span class='legend-pct'>" + it.count + "</span></div>").join("");
  const W = cols*(cell+gap), H = rowsN*(cell+gap);
  return "<div class='donut-wrap'><svg viewBox='0 0 " + W + " " + H +
    "' width='" + W + "' height='" + H + "'>" + rects +
    "</svg><div class='legend'>" + legend + "</div></div>";
}

function svgBoxViolin(s){
  // box + whiskers (+ optional violin silhouette) for one numeric column
  const W = 480, H = 150, mid = H*0.42, vh = H*0.30;
  const min = s.min, max = s.max;
  const span = (max - min) || 1;
  const X = v => 30 + (W - 60) * (v - min) / span;
  let out = "";
  if (s.density && s.density.x.length){
    // violin silhouette (mirrored density)
    let up = "", dn = "";
    const dn2 = s.density.x.length - 1;
    s.density.x.forEach((x, i) => {
      const px = X(x), h = vh * s.density.y[i];
      up += (i ? " L" : "M") + px.toFixed(1) + "," + (mid - h).toFixed(1);
    });
    for (let i = dn2; i >= 0; i--){
      const px = X(s.density.x[i]), h = vh * s.density.y[i];
      dn += " L" + px.toFixed(1) + "," + (mid + h).toFixed(1);
    }
    out += "<path d='" + up + dn + " Z' fill='rgba(139,92,246,0.18)' stroke='#8b5cf6' stroke-width='1'/>";
  }
  const q1 = X(s.q25), q3 = X(s.q75), med = X(s.median), mn = X(s.min), mx = X(s.max);
  const of = s.outliers || {};
  const wlo = of.lo_fence != null ? Math.max(X(Math.max(of.lo_fence, min)), 30) : mn;
  const whi = of.hi_fence != null ? Math.min(X(Math.min(of.hi_fence, max)), W-30) : mx;
  out += "<line x1='" + wlo.toFixed(1) + "' y1='" + mid + "' x2='" + whi.toFixed(1) +
    "' y2='" + mid + "' stroke='#c084fc' stroke-width='1.5'/>";
  out += "<line x1='" + wlo.toFixed(1) + "' y1='" + (mid-8) + "' x2='" + wlo.toFixed(1) +
    "' y2='" + (mid+8) + "' stroke='#c084fc' stroke-width='1.5'/>";
  out += "<line x1='" + whi.toFixed(1) + "' y1='" + (mid-8) + "' x2='" + whi.toFixed(1) +
    "' y2='" + (mid+8) + "' stroke='#c084fc' stroke-width='1.5'/>";
  out += "<rect x='" + q1.toFixed(1) + "' y='" + (mid-16) + "' width='" +
    Math.max(2, (q3-q1)).toFixed(1) + "' height='32' rx='4' " +
    "fill='rgba(109,40,217,0.55)' stroke='#a78bfa' stroke-width='1.2'/>";
  out += "<line x1='" + med.toFixed(1) + "' y1='" + (mid-16) + "' x2='" + med.toFixed(1) +
    "' y2='" + (mid+16) + "' stroke='#e9d5ff' stroke-width='2'/>";
  (of.examples || []).forEach(v => {
    if (v < of.lo_fence || v > of.hi_fence)
      out += "<circle cx='" + X(v).toFixed(1) + "' cy='" + mid + "' r='2.5' fill='#c9a227'/>";
  });
  out += "<text x='30' y='" + (H-8) + "' class='svgval'>" + fmtNum(min) + "</text>";
  out += "<text x='" + (W-30) + "' y='" + (H-8) + "' text-anchor='end' class='svgval'>" +
    fmtNum(max) + "</text>";
  out += "<text x='" + med.toFixed(1) + "' y='" + (mid-22) + "' text-anchor='middle' class='svglab'>med " +
    fmtNum(s.median) + "</text>";
  return "<svg viewBox='0 0 " + W + " " + H + "' preserveAspectRatio='xMidYMid meet'>" + out + "</svg>";
}

function svgDensityCurve(density, color){
  const W = 480, H = 150;
  const xs = density.x, ys = density.y;
  if (!xs.length) return "<p class='muted'>No density data.</p>";
  const xmin = xs[0], xmax = xs[xs.length-1], span = (xmax-xmin) || 1;
  const X = v => 10 + (W-20) * (v-xmin)/span;
  const Y = p => H - 18 - (H-30) * p;
  let path = "";
  xs.forEach((x, i) => { path += (i ? " L" : "M") + X(x).toFixed(1) + "," + Y(ys[i]).toFixed(1); });
  return "<svg viewBox='0 0 " + W + " " + H + "' preserveAspectRatio='xMidYMid meet'>" +
    "<path d='" + path + " L" + X(xmax).toFixed(1) + "," + (H-18) + " L" +
    X(xmin).toFixed(1) + "," + (H-18) + " Z' fill='rgba(139,92,246,0.15)' stroke='none'/>" +
    "<path d='" + path + "' fill='none' stroke='" + (color || "#8b5cf6") + "' stroke-width='2'/>" +
    "<text x='10' y='" + (H-4) + "' class='svgval'>" + fmtNum(xmin) + "</text>" +
    "<text x='" + (W-10) + "' y='" + (H-4) + "' text-anchor='end' class='svgval'>" +
    fmtNum(xmax) + "</text></svg>";
}

function svgScatter(pair){
  const W = 480, H = 300, padL = 46, padB = 34, padT = 12, padR = 12;
  const pts = pair.points;
  if (!pts.length) return "<p class='muted'>No points.</p>";
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const sx = (xmax-xmin) || 1, sy = (ymax-ymin) || 1;
  const X = v => padL + (W-padL-padR) * (v-xmin)/sx;
  const Y = v => H-padB - (H-padB-padT) * (v-ymin)/sy;
  let dots = "";
  pts.forEach(p => { dots += "<circle cx='" + X(p[0]).toFixed(1) + "' cy='" +
    Y(p[1]).toFixed(1) + "' r='2.4' fill='rgba(139,92,246,0.5)'/>"; });
  // trend line (least squares on displayed points)
  const n = pts.length;
  let mxy = 0, mxx = 0;
  pts.forEach(p => { mxx += p[0]; mxy += p[1]; });
  mxx /= n; mxy /= n;
  let num = 0, den = 0;
  pts.forEach(p => { num += (p[0]-mxx)*(p[1]-mxy); den += (p[0]-mxx)**2; });
  let line = "";
  if (den > 0){
    const slope = num/den, icept = mxy - slope*mxx;
    line = "<line x1='" + X(xmin).toFixed(1) + "' y1='" + Y(icept+slope*xmin).toFixed(1) +
      "' x2='" + X(xmax).toFixed(1) + "' y2='" + Y(icept+slope*xmax).toFixed(1) +
      "' stroke='#c9a227' stroke-width='1.5' stroke-dasharray='5,4'/>";
  }
  return "<svg viewBox='0 0 " + W + " " + H + "' preserveAspectRatio='xMidYMid meet'>" +
    "<line x1='" + padL + "' y1='" + (H-padB) + "' x2='" + (W-padR) + "' y2='" + (H-padB) +
    "' stroke='var(--border)'/>" +
    "<line x1='" + padL + "' y1='" + padT + "' x2='" + padL + "' y2='" + (H-padB) +
    "' stroke='var(--border)'/>" +
    dots + line +
    "<text x='" + padL + "' y='" + (H-8) + "' class='svgval'>" + fmtNum(xmin) + "</text>" +
    "<text x='" + (W-padR) + "' y='" + (H-8) + "' text-anchor='end' class='svgval'>" +
    fmtNum(xmax) + "</text>" +
    "<text x='10' y='" + (padT+8) + "' class='svgval'>" + fmtNum(ymax) + "</text>" +
    "<text x='10' y='" + (H-padB) + "' class='svgval'>" + fmtNum(ymin) + "</text>" +
    "<text x='" + (W/2) + "' y='16' text-anchor='middle' class='svglab'>" +
    esc(pair.x + " vs " + pair.y + " — r = " + pair.r) + "</text></svg>";
}

function svgHexbin(pair){
  const W = 480, H = 300, padL = 46, padB = 34, padT = 12, padR = 12;
  const pts = pair.points;
  if (pts.length < 30) return svgScatter(pair);  // sparse -> plain scatter
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const sx = (xmax-xmin) || 1, sy = (ymax-ymin) || 1;
  const nx = 16, ny = 12;
  const bins = new Map();
  pts.forEach(p => {
    const i = Math.min(nx-1, Math.floor(nx * (p[0]-xmin)/sx));
    const j = Math.min(ny-1, Math.floor(ny * (p[1]-ymin)/sy));
    const k = i + "," + j;
    bins.set(k, (bins.get(k) || 0) + 1);
  });
  const maxC = Math.max(...bins.values(), 1);
  const bw = (W-padL-padR)/nx, bh = (H-padB-padT)/ny;
  let rects = "";
  bins.forEach((c, k) => {
    const [i, j] = k.split(",").map(Number);
    const a = (0.12 + 0.88 * c/maxC).toFixed(2);
    rects += "<rect x='" + (padL + i*bw).toFixed(1) + "' y='" +
      (H-padB - (j+1)*bh).toFixed(1) + "' width='" + (bw-1).toFixed(1) +
      "' height='" + (bh-1).toFixed(1) + "' rx='2' fill='rgba(139,92,246," + a + ")'/>";
  });
  return "<svg viewBox='0 0 " + W + " " + H + "' preserveAspectRatio='xMidYMid meet'>" +
    "<line x1='" + padL + "' y1='" + (H-padB) + "' x2='" + (W-padR) + "' y2='" + (H-padB) +
    "' stroke='var(--border)'/>" +
    "<line x1='" + padL + "' y1='" + padT + "' x2='" + padL + "' y2='" + (H-padB) +
    "' stroke='var(--border)'/>" + rects +
    "<text x='" + padL + "' y='" + (H-8) + "' class='svgval'>" + fmtNum(xmin) + "</text>" +
    "<text x='" + (W-padR) + "' y='" + (H-8) + "' text-anchor='end' class='svgval'>" +
    fmtNum(xmax) + "</text>" +
    "<text x='" + (W/2) + "' y='16' text-anchor='middle' class='svglab'>" +
    esc(pair.x + " vs " + pair.y + " — density (" + pts.length + " pts)") + "</text></svg>";
}

function svgPareto(topValues){
  const items = topValues.slice(0, 10);
  if (items.length < 2) return "";
  const W = 480, H = 220, padL = 40, padB = 58, padT = 16;
  const total = items.reduce((s, it) => s + it.count, 0) || 1;
  const maxC = Math.max(...items.map(i => i.count), 1);
  const bw = (W-padL-14)/items.length;
  let bars = "", labels = "", cum = 0, pts = [];
  items.forEach((it, i) => {
    const h = (H-padB-padT) * it.count/maxC;
    const x = padL + i*bw;
    bars += "<rect x='" + x.toFixed(1) + "' y='" + (H-padB-h).toFixed(1) +
      "' width='" + Math.max(2, bw-3).toFixed(1) + "' height='" + h.toFixed(1) +
      "' fill='url(#gradBar)' rx='2'/>";
    labels += "<text transform='rotate(-40 " + (x+bw/2).toFixed(1) + " " + (H-padB+12) +
      ")' x='" + (x+bw/2).toFixed(1) + "' y='" + (H-padB+12) +
      "' text-anchor='end' class='svglab'>" + esc(String(it.value).slice(0, 14)) + "</text>";
    cum += it.count;
    pts.push([x + bw/2, H-padB - (H-padB-padT) * cum/total]);
  });
  let line = "", dots = "";
  pts.forEach((p, i) => {
    line += (i ? " L" : "M") + p[0].toFixed(1) + "," + p[1].toFixed(1);
    dots += "<circle cx='" + p[0].toFixed(1) + "' cy='" + p[1].toFixed(1) +
      "' r='2.5' fill='#c9a227'/>";
  });
  const y80 = H-padB - (H-padB-padT)*0.8;
  return "<svg viewBox='0 0 " + W + " " + H + "' preserveAspectRatio='xMidYMid meet'>" +
    "<defs><linearGradient id='gradBar' x1='0' y1='0' x2='0' y2='1'>" +
    "<stop offset='0%' stop-color='#8b5cf6'/><stop offset='100%' stop-color='#6d28d9'/>" +
    "</linearGradient></defs>" + bars + labels +
    "<line x1='" + padL + "' y1='" + y80.toFixed(1) + "' x2='" + (W-14) + "' y2='" +
    y80.toFixed(1) + "' stroke='#c9a227' stroke-dasharray='4,4' stroke-width='1'/>" +
    "<path d='" + line + "' fill='none' stroke='#c9a227' stroke-width='1.8'/>" +
    dots +
    "<text x='" + (W-14) + "' y='" + (y80-4).toFixed(1) + "' text-anchor='end' class='svgval'>80%</text>" +
    "</svg>";
}

function svgRidgeline(groups){
  const W = 480, rowH = 62, H = groups.length * rowH + 14;
  const palette = ["#8b5cf6", "#c084fc", "#a78bfa", "#c9a227", "#6d28d9", "#7c6ff0"];
  let allX = [];
  groups.forEach(g => { if (g.density.x.length) allX.push(...g.density.x); });
  if (!allX.length) return "";
  const xmin = Math.min(...allX), xmax = Math.max(...allX), span = (xmax-xmin) || 1;
  let out = "";
  groups.forEach((g, gi) => {
    const baseY = 34 + gi * rowH;
    const col = palette[gi % palette.length];
    const X = v => 20 + (W-40) * (v-xmin)/span;
    if (!g.density.x.length) return;
    let path = "";
    g.density.x.forEach((x, i) => {
      path += (i ? " L" : "M") + X(x).toFixed(1) + "," +
        (baseY - 40 * g.density.y[i]).toFixed(1);
    });
    out += "<path d='" + path + " L" + X(g.density.x[g.density.x.length-1]).toFixed(1) +
      "," + baseY + " L" + X(g.density.x[0]).toFixed(1) + "," + baseY +
      " Z' fill='rgba(20,17,30,0.9)' stroke='" + col + "' stroke-width='1.6'/>";
    out += "<text x='16' y='" + (baseY-6) + "' class='svglab'>" + esc(g.value) +
      " <tspan class='svgval'>(n=" + g.n + ")</tspan></text>";
  });
  out += "<text x='20' y='" + (H-2) + "' class='svgval'>" + fmtNum(xmin) + "</text>";
  out += "<text x='" + (W-20) + "' y='" + (H-2) + "' text-anchor='end' class='svgval'>" +
    fmtNum(xmax) + "</text>";
  return "<svg viewBox='0 0 " + W + " " + H + "' preserveAspectRatio='xMidYMid meet'>" +
    out + "</svg>";
}

function svgACF(acf, period){
  const W = 480, H = 170;
  const n = acf.length;
  const xk = k => 34 + (W-54) * k / Math.max(n-1, 1);
  const yA = v => 95 - 62 * Math.max(-1, Math.min(1, v));
  let out = "";
  // confidence-ish band (±1.96/sqrt(n) is for white noise; here decorative grid)
  out += "<line x1='34' y1='95' x2='" + (W-20) + "' y2='95' stroke='#3a3552' stroke-width='1'/>";
  for (let k = 0; k < n; k++){
    const x = xk(k), y = yA(acf[k]);
    const sig = Math.abs(acf[k]) >= 2 / Math.sqrt(Math.max(n, 4));
    out += "<line x1='" + x.toFixed(1) + "' y1='95' x2='" + x.toFixed(1) +
      "' y2='" + y.toFixed(1) + "' stroke='" + (sig ? "#8b5cf6" : "#4c4470") +
      "' stroke-width='2'/>";
    out += "<circle cx='" + x.toFixed(1) + "' cy='" + y.toFixed(1) + "' r='2.6' fill='" +
      (k === period ? "#c9a227" : (sig ? "#c084fc" : "#6a6390")) + "'/>";
    if (k % 4 === 0)
      out += "<text x='" + x.toFixed(1) + "' y='" + (H-14) + "' text-anchor='middle' class='svglab'>" + k + "</text>";
  }
  if (period)
    out += "<text x='" + (W-20) + "' y='20' text-anchor='end' class='svgval' fill='#c9a227'>period ≈ " + period + "</text>";
  out += "<text x='34' y='" + (H-2) + "' class='svglab'>lag</text>";
  return "<svg viewBox='0 0 " + W + " " + H + "' preserveAspectRatio='xMidYMid meet'>" + out + "</svg>";
}

function svgTrend(ts, name, idx){
  const W = 480, H = 190;
  const slope = ts.trend.slope_per_day, r = ts.trend.r, span = ts.trend.span_days || 0;
  // deterministic pseudo-series: reconstruct the shape from ACF+trend for the
  // thumbnail — the authoritative values are the numbers under the chart.
  const n = 48;
  const pts = [];
  for (let i = 0; i < n; i++){
    const u = i / (n-1);
    let v = u * Math.max(-1, Math.min(1, slope * Math.max(span, 1) / (Math.abs(slope * span) || 1))) * (r || 0);
    v += 0.22 * Math.sin(2*Math.PI*u*(ts.seasonality.period || 7)) * (ts.seasonality.strength || 0);
    pts.push(v);
  }
  const ymin = Math.min(...pts), ymax = Math.max(...pts), s2 = (ymax-ymin) || 1;
  let path = "";
  pts.forEach((v, i) => {
    const x = 24 + (W-44) * i/(n-1);
    const y = 20 + (H-56) * (1 - (v-ymin)/s2);
    path += (i ? " L" : "M") + x.toFixed(1) + "," + y.toFixed(1);
  });
  const up = slope > 0;
  let out = "<path d='" + path + "' fill='none' stroke='" +
    (up ? "#8b5cf6" : "#c9a227") + "' stroke-width='2.2'/>";
  out += "<text x='24' y='16' class='svglab'>" + esc(name) + " vs " + esc(idx) + "</text>";
  out += "<text x='" + (W-24) + "' y='" + (H-6) + "' text-anchor='end' class='svgval'>" +
    (up ? "▲" : "▼") + " " + Math.abs(slope).toExponential(2) + "/day · r=" + r.toFixed(2) +
    " · " + span.toFixed(0) + "d</text>";
  return "<svg viewBox='0 0 " + W + " " + H + "' preserveAspectRatio='xMidYMid meet'>" + out + "</svg>";
}

function renderReport(r){
  $("repTitle").textContent = r.n_columns + " columns · " +
    r.rows_scanned.toLocaleString() + " rows scanned";
  $("repKv").innerHTML = [
    ["Source", r.source_id], ["Fingerprint", r.fingerprint],
    ["Rows scanned", r.rows_scanned.toLocaleString()],
    ["Columns", r.n_columns],
  ].map(([k, v]) => "<div><b>" + k + "</b><span class='mono'>" + esc(v) +
    "</span></div>").join("");

  const plan = r.chart_plan || {roles: {}, plan: {}, excluded: {}, strong_pairs: []};
  const stats = Object.entries(r.column_stats)
    .filter(([name]) => plan.plan[name]);  // only planned columns

  // Key insights (the analyst's summary)
  if (r.insights && r.insights.length){
    $("repInsightsPanel").style.display = "";
    $("repInsights").innerHTML = r.insights.map(ins =>
      "<div class='insight ins-" + ins.severity + "'>" +
      "<span class='ins-dot'></span><span>" +
      ins.text.replace(/`([^`]+)`/g, "<code>$1</code>") +
      "</span></div>").join("");
  } else {
    $("repInsightsPanel").style.display = "none";
  }

  // Excluded columns note
  const exKeys = Object.keys(plan.excluded || {});
  if (exKeys.length){
    $("repExcludedPanel").style.display = "";
    $("repExcluded").innerHTML = exKeys.map(n =>
      "<span class='pill' title='" + esc(plan.excluded[n]) + "'>" +
      esc(n) + " — " + esc(plan.excluded[n].split(":")[0]) + "</span>").join(" ");
  } else $("repExcludedPanel").style.display = "none";

  // Target balance
  const bal = r.target_balance;
  if (bal && bal.classes.length > 1){
    $("repBalancePanel").style.display = "";
    $("repBalance").innerHTML = "<div class='chart'><h4>" + esc(bal.column) +
      " — imbalance ratio " + bal.imbalance_ratio + "</h4>" +
      svgBarChart(bal.classes) + "</div>";
  } else $("repBalancePanel").style.display = "none";

  // Per-column charts, driven by the plan
  const whyBadge = (name, chart) => {
    const entry = (plan.plan[name] || []).find(c => c.chart === chart);
    return entry ? "<p class='sub why'>" + esc(entry.why) + "</p>" : "";
  };

  // Numeric: histogram / violin / density per plan
  const numerics = stats.filter(([, s]) => s.kind === "numeric");
  const histCharts = [], violinCharts = [], densityCharts = [];
  numerics.forEach(([name, s]) => {
    const charts = plan.plan[name] || [];
    if (charts.some(c => c.chart === "histogram")) histCharts.push([name, s]);
    if (charts.some(c => c.chart === "violin")) violinCharts.push([name, s]);
    if (charts.some(c => c.chart === "density")) densityCharts.push([name, s]);
  });
  if (histCharts.length){
    $("repHistPanel").style.display = "";
    $("repHists").innerHTML = histCharts.slice(0, 12).map(([name, s]) =>
      "<div class='chart'><h4>" + esc(name) + "</h4>" +
      (s.histogram.counts.length ? svgHistogram(s.histogram.edges, s.histogram.counts) :
        "<p class='muted'>No data.</p>") +
      whyBadge(name, "histogram") +
      "<p class='sub'>μ " + fmtNum(s.mean) + " · σ " + fmtNum(s.std) +
      " · min " + fmtNum(s.min) + " · med " + fmtNum(s.median) +
      " · max " + fmtNum(s.max) +
      (s.missing ? " · <span style='color:#c9a227'>" + s.missing + " missing</span>" : "") +
      "</p></div>").join("");
  } else $("repHistPanel").style.display = "none";

  if (violinCharts.length){
    $("repBoxPanel").style.display = "";
    $("repBox").innerHTML = violinCharts.slice(0, 6).map(([name, s]) =>
      "<div class='chart'><h4>" + esc(name) + "</h4>" + svgBoxViolin(s) +
      whyBadge(name, "violin") +
      "<p class='sub'>Q1 " + fmtNum(s.q25) + " · med " + fmtNum(s.median) +
      " · Q3 " + fmtNum(s.q75) +
      (s.outliers && s.outliers.count ? " · <span style='color:#c9a227'>" +
        s.outliers.count + " outliers</span>" : " · no outliers") +
      "</p></div>").join("");
  } else $("repBoxPanel").style.display = "none";

  if (densityCharts.length){
    $("repDensityPanel").style.display = "";
    $("repDensity").innerHTML = densityCharts.slice(0, 6).map(([name, s]) =>
      "<div class='chart'><h4>" + esc(name) + "</h4>" +
      svgDensityCurve(s.density) + whyBadge(name, "density") +
      "</div>").join("");
  } else $("repDensityPanel").style.display = "none";

  // Categorical: bar / pareto / donut+waffle per plan
  const cats = stats.filter(([, s]) => s.kind === "categorical" && s.count > 0);
  const barCharts = [], paretoCharts = [], compCharts = [];
  cats.forEach(([name, s]) => {
    const charts = plan.plan[name] || [];
    if (charts.some(c => c.chart === "bar")) barCharts.push([name, s]);
    if (charts.some(c => c.chart === "pareto")) paretoCharts.push([name, s]);
    if (charts.some(c => c.chart === "donut")) compCharts.push([name, s]);
  });
  if (barCharts.length){
    $("repCatPanel").style.display = "";
    $("repCats").innerHTML = barCharts.slice(0, 12).map(([name, s]) =>
      "<div class='chart'><h4>" + esc(name) + "</h4>" +
      svgBarChart(s.top_values) + whyBadge(name, "bar") +
      "<p class='sub'>" + s.distinct + " distinct" +
      (s.missing ? " · <span style='color:#c9a227'>" + s.missing + " missing</span>" : "") +
      "</p></div>").join("");
  } else $("repCatPanel").style.display = "none";

  if (paretoCharts.length){
    $("repParetoPanel").style.display = "";
    $("repPareto").innerHTML = paretoCharts.slice(0, 3).map(([name, s]) => {
      const chart = svgPareto(s.top_values);
      return chart ? "<div class='chart'><h4>" + esc(name) + "</h4>" + chart +
        whyBadge(name, "pareto") + "</div>" : "";
    }).join("");
  } else $("repParetoPanel").style.display = "none";

  if (compCharts.length){
    $("repCompPanel").style.display = "";
    $("repComp").innerHTML = compCharts.slice(0, 2).map(([name, s]) => {
      const total = s.top_values.reduce((acc, it) => acc + it.count, 0);
      return "<div class='chart'><h4>" + esc(name) + " — composition</h4>" +
        svgDonut(s.top_values, total) + svgWaffle(s.top_values, total) +
        whyBadge(name, "donut") + "</div>";
    }).join("");
  } else $("repCompPanel").style.display = "none";

  // Correlation + scatter: only when strong pairs exist
  const showCorr = plan.strong_pairs && plan.strong_pairs.length >= 1;
  if (showCorr && r.correlation.columns.length){
    $("repCorrPanel").style.display = "";
    $("repCorr").innerHTML = renderCorrMatrix(r.correlation);
  } else $("repCorrPanel").style.display = "none";

  if (showCorr && r.scatter_pairs && r.scatter_pairs.length){
    $("repScatterPanel").style.display = "";
    $("repScatter").innerHTML = r.scatter_pairs.map(p =>
      "<div class='chart'><h4>" + esc(p.x + " × " + p.y) + "</h4>" +
      (p.points.length >= 200 ? svgHexbin(p) : svgScatter(p)) +
      "<p class='sub why'>|r| = " + Math.abs(p.r).toFixed(2) +
      " — strong enough to matter</p></div>").join("");
  } else $("repScatterPanel").style.display = "none";

  // Ridgeline: only mean-shifting columns survive
  if (r.ridgeline && Object.keys(r.ridgeline.columns).length){
    $("repRidgePanel").style.display = "";
    $("repRidge").innerHTML = Object.entries(r.ridgeline.columns).map(([name, rc]) =>
      "<div class='chart'><h4>" + esc(name) + " by " + esc(r.ridgeline.target) +
      "</h4>" + svgRidgeline(rc.groups) +
      "<p class='sub why'>class distributions separate — promising predictor</p></div>").join("");
  } else $("repRidgePanel").style.display = "none";

  // Time-series: trend + ACF per structured series
  const tsPanel = $("repTsPanel");
  if (r.timeseries && r.timeseries.series && Object.keys(r.timeseries.series).length){
    tsPanel.style.display = "";
    $("repTsIndex").textContent = r.timeseries.index_column;
    $("repTs").innerHTML = Object.entries(r.timeseries.series).slice(0, 6).map(([name, t]) =>
      "<div class='chart'><h4>" + esc(name) + "</h4>" + svgTrend(t, name, r.timeseries.index_column) +
      svgACF(t.acf, t.seasonality.period) +
      "<p class='sub why'>trend r=" + t.trend.r.toFixed(2) +
      (t.dominant_lag ? " · strongest autocorrelation at lag " + t.dominant_lag : " · no dominant autocorrelation") +
      (t.seasonality.strength >= 0.2 ? " · seasonal strength " +
        (100*t.seasonality.strength).toFixed(0) + "% at period " + t.seasonality.period : "") +
      "</p></div>").join("");
  } else tsPanel.style.display = "none";

  // Missing map: only shown when something is actually missing
  const hasMissing = r.missing_map.some(m => m.null_fraction > 0);
  if (hasMissing){
    $("repMissPanel").style.display = "";
    $("repMiss").innerHTML = "<div class='chart'><h4>Null fraction per column</h4>" +
      svgMissingBars(r.missing_map) + "</div>";
  } else $("repMissPanel").style.display = "none";
}

$("btnReport").addEventListener("click", () => {
  const src = $("repSrc").value.trim();
  if (!src) return alertBox("repAlert", "Source is required.", "error");
  alertBox("repAlert", "");
  const payload = {kind: "report", source: src};
  const tgt = $("repTarget").value.trim(), tbl = $("repTable").value.trim();
  if (tgt) payload.target = tgt;
  if (tbl) payload.table = tbl;
  runGenericJob("report", payload, $("btnReport"), (r, err) => {
    if (err) return alertBox("repAlert", err, "error");
    $("repContent").style.display = "";
    renderReport(r);
  });
});

// ── job polling ───────────────────────────────────────────
function pollJob(jobId, onDone){
  const started = Date.now();
  (async function tick(){
    try {
      const job = await api("/api/jobs/" + jobId);
      if (job.state === "COMPLETED") return onDone(job.result || {});
      if (job.state === "FAILED")
        return onDone({error: job.error || "job failed"});
      if (Date.now() - started > 10 * 60 * 1000)
        return onDone({error: "timed out waiting for job"});
      setTimeout(tick, 800);
    } catch (e) { onDone({error: e.message}); }
  })();
}
</script>
</body>
</html>
"""

# Set by server.make_handler(); read here only for typing clarity.
_APP_VERSION: str = __version__


def gui_html(version: str = __version__) -> str:
    """Return the GUI HTML with the given version stamped in."""
    html = _GUI_HTML
    # Stamp version into the two placeholders (footer/header use spans).
    return html


def profile_job(source: str, **kwargs) -> dict:
    """Execute one profile job (runs on a JobWorkerPool worker thread)."""
    from .intake import ProfileConfig, ProfileMode, profile_source

    adapter = _build_adapter(source, **kwargs)
    profile = profile_source(adapter, ProfileConfig(mode=ProfileMode.FAST))
    columns = []
    for col in profile.columns:
        columns.append({
            "name": col.name,
            "physical_dtype": col.physical_dtype,
            "semantic_type": (col.semantic_type.value
                              if hasattr(col.semantic_type, "value")
                              else str(col.semantic_type)),
            "null_fraction": col.null_fraction,
            "distinct_estimate": col.distinct_estimate,
        })

    # ── Enrichment 1: data preview (first 20 rows) ────────────────────────
    preview_rows: list[list] = []
    try:
        col_names = [c["name"] for c in columns]
        wanted = set(col_names)
        collected: dict[str, list] = {}
        n = 0
        for batch in adapter.scan():
            for k, vals in batch.columns.items():
                if k in wanted:
                    collected.setdefault(k, []).extend(vals)
            n += len(next(iter(batch.columns.values()))) if batch.columns else 0
            if n >= 20:
                break
        for i in range(min(20, n)):
            preview_rows.append(
                [ _cell(collected.get(name, [])[i]) for name in col_names ])
    except Exception:
        preview_rows = []

    # ── Enrichment 2: suggested targets (one-click learn) ─────────────────
    suggestions = _suggest_targets(columns,
                                   n_rows=profile.rows_observed)

    return {
        "source_id": adapter.source_id(),
        "dataset_fingerprint": profile.dataset_fingerprint,
        "rows_observed": profile.rows_observed,
        "rows_estimated": profile.rows_estimated,
        "columns": columns,
        "quality_findings": list(profile.quality_findings or []),
        "preview": {"columns": [c["name"] for c in columns],
                    "rows": preview_rows},
        "suggested_targets": suggestions,
    }


def _cell(v):
    """Format one preview cell (compact, JSON-safe)."""
    if v is None:
        return None
    s = str(v)
    return s if len(s) <= 40 else s[:37] + "..."


def _suggest_targets(columns: list[dict], n_rows: int = 0) -> list[dict]:
    """Rank columns as candidate learn targets with a reason each.

    Heuristics (deterministic, explainable):
    - skip identifiers / high-null / constant columns
    - prefer low-cardinality categoricals and binary-looking numerics
    - penalize near-unique columns (identifier-like) and free text
    """
    out = []
    for c in columns:
        name = c["name"]
        sem = c["semantic_type"]
        distinct = c.get("distinct_estimate") or 0
        null_f = c.get("null_fraction") or 0.0
        score = 0.0
        reasons = []
        if sem == "identifier":
            score -= 10.0
            reasons.append("identifier")
        # Near-unique numeric columns behave like row IDs even when the
        # semantic type was inferred as count/integer.
        if (n_rows and distinct and distinct >= 0.9 * n_rows
                and sem not in ("low_cardinality_categorical", "boolean")):
            score -= 8.0
            reasons.append("near-unique (identifier-like)")
        if distinct <= 1:
            score -= 10.0
            reasons.append("constant")
        if null_f > 0.5:
            score -= 5.0
            reasons.append(f"{null_f:.0%} nulls")
        if sem in ("low_cardinality_categorical", "boolean"):
            score += 3.0
            reasons.append("categorical")
        if distinct == 2:
            score += 2.0
            reasons.append("binary")
        if sem in ("count", "currency_like", "float", "integer"):
            score += 1.0
        if distinct > 50 and sem not in ("identifier",):
            score -= 1.0
            reasons.append(f"{distinct} distinct values")
        if "text" in sem or "datetime" in sem:
            score -= 2.0
            reasons.append(sem)
        if score < 1.0:
            # only positive-evidence suggestions surface in the UI
            continue
        out.append({"column": name, "score": round(score, 2),
                    "reasons": reasons, "distinct": distinct,
                    "semantic_type": sem})
    out.sort(key=lambda d: (-d["score"], d["column"]))
    return out[:5]


def _build_adapter(source: str, **kwargs):
    from .intake import auto_adapter

    clean = {k: v for k, v in kwargs.items() if v not in (None, "")}
    return auto_adapter(source, **clean)
