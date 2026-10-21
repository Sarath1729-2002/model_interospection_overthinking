// Probe Evaluation Dashboard — client-side logic.
// Loads the JSON produced by evaluate_probes.py and renders a sortable list
// view, a per-token table with column toggles, and a chart.

(() => {
  // ---------- State ----------
  let DATA = null;            // parsed JSON from evaluate_probes.py
  let CURRENT_EXAMPLE = null; // reference to data.examples[i] while in detail view
  let CHART = null;           // Chart.js instance (destroyed & rebuilt on re-render)
  let COLUMN_VISIBLE = {};    // { columnKey -> bool }

  const $ = (id) => document.getElementById(id);
  const showView = (id) => {
    ["list-view", "detail-view"].forEach((v) => $(v).classList.toggle("hidden", v !== id));
  };
  const fmt = (v, digits = 3) => {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    if (typeof v !== "number") return String(v);
    return v.toFixed(digits);
  };
  const fmtPct = (v) => (v === null || v === undefined ? "—" : (v * 100).toFixed(2) + "%");

  // Multi-seed aggregation helpers. evaluate_probes.py wraps every aggregate
  // scalar in {mean, std} when run with multiple --probe-dir paths; legacy
  // single-seed JSONs store plain numbers. These helpers handle both shapes
  // so callers can read scalars without knowing which shape they're getting.
  const meanOf = (v) => (v && typeof v === "object" && "mean" in v) ? v.mean : v;
  const stdOf  = (v) => (v && typeof v === "object" && "std"  in v) ? v.std  : null;

  // Trailing rolling-mean window length (in tokens) for the smoothed companion
  // columns in the per-token table. Per-token MAE/MAPE/SMAPE/accuracy can be
  // very noisy; the smoothed column at position j is the mean of the metric
  // over positions [max(0, j - window + 1), j], skipping nulls.
  let ROLLING_WINDOW = 10;

  // Trailing rolling mean of `arr` ending at index j, over `window` samples.
  // Returns null if no valid samples in the window.
  function rollingMean(arr, j, window) {
    if (!arr || j == null || j < 0 || j >= arr.length) return null;
    const start = Math.max(0, j - window + 1);
    let sum = 0;
    let n = 0;
    for (let k = start; k <= j; k++) {
      const v = arr[k];
      if (v === null || v === undefined || Number.isNaN(v)) continue;
      sum += v;
      n += 1;
    }
    return n > 0 ? sum / n : null;
  }

  // 1D Gaussian smoothing of an array of (possibly null) values. Sigma is in
  // units of array indices; sigma=0 is a no-op and just returns the input.
  // Nulls are skipped (treated as "missing"), and the kernel is renormalized
  // over the present neighbors so the smoothed value tracks the local mean.
  function gaussianSmooth(values, sigma) {
    if (!sigma || sigma <= 0) return values.slice();
    const radius = Math.max(1, Math.ceil(3 * sigma));
    const kernel = [];
    for (let d = -radius; d <= radius; d++) {
      kernel.push(Math.exp(-(d * d) / (2 * sigma * sigma)));
    }
    const out = new Array(values.length).fill(null);
    for (let i = 0; i < values.length; i++) {
      let num = 0;
      let den = 0;
      for (let d = -radius; d <= radius; d++) {
        const j = i + d;
        if (j < 0 || j >= values.length) continue;
        const v = values[j];
        if (v === null || v === undefined || Number.isNaN(v)) continue;
        const w = kernel[d + radius];
        num += w * v;
        den += w;
      }
      out[i] = den > 0 ? num / den : null;
    }
    return out;
  }

  // ---------- File loading ----------
  $("file-input").addEventListener("change", async (ev) => {
    const f = ev.target.files[0];
    if (!f) return;
    $("loader-status").textContent = `Loading ${f.name}…`;
    const text = await f.text();
    DATA = JSON.parse(text);
    $("loader-status").textContent = `Loaded ${DATA.examples.length} examples.`;
    initListView();
    showView("list-view");
  });

  // Also allow ?data=path to auto-fetch (useful when served over a local HTTP server).
  (async () => {
    const qs = new URLSearchParams(location.search);
    const dataUrl = qs.get("data");
    if (!dataUrl) return;
    $("loader-status").textContent = `Fetching ${dataUrl}…`;
    try {
      const resp = await fetch(dataUrl);
      if (!resp.ok) throw new Error(resp.statusText);
      DATA = await resp.json();
      $("loader-status").textContent = `Loaded ${DATA.examples.length} examples.`;
      initListView();
      showView("list-view");
    } catch (err) {
      $("loader-status").textContent = `Failed: ${err.message}`;
    }
  })();

  // ---------- Sort metric menu ----------
  // Build the list of possible metrics to sort the example list by. This
  // iterates over every probe + baseline in the metadata and every numeric
  // aggregate metric they expose.
  function collectSortOptions() {
    const options = [];
    if (!DATA || DATA.examples.length === 0) return options;

    // Find first example with has_eos to discover metric keys
    const sample = DATA.examples.find((e) => e.has_eos) || DATA.examples[0];

    const push = (bucket, probeName, entry) => {
      if (!entry || !entry.aggregate) return;
      for (const mk of Object.keys(entry.aggregate)) {
        options.push({
          value: `${bucket}.${probeName}.${mk}`,
          label: `${probeName} / ${mk}`,
          bucket,
          probeName,
          metricKey: mk,
        });
      }
    };

    for (const [name, entry] of Object.entries(sample.probes || {})) push("probes", name, entry);
    for (const [name, entry] of Object.entries(sample.statistical_baselines || {})) push("statistical_baselines", name, entry);
    return options;
  }

  function getAggregate(example, option) {
    const entry = (example[option.bucket] || {})[option.probeName];
    if (!entry || !entry.aggregate) return null;
    const raw = entry.aggregate[option.metricKey];
    // For multi-seed JSONs `raw` is `{mean, std}`; sort by the mean.
    const v = meanOf(raw);
    return v === null || v === undefined || Number.isNaN(v) ? null : v;
  }

  // ---------- List view ----------
  function initListView() {
    const opts = collectSortOptions();
    const sel = $("sort-metric");
    sel.innerHTML = "";
    for (const o of opts) {
      const el = document.createElement("option");
      el.value = o.value;
      el.textContent = o.label;
      sel.appendChild(el);
    }
    // Sensible default: count MAE
    const defaultOpt = opts.find((o) => o.value === "probes.count.mae") || opts[0];
    if (defaultOpt) sel.value = defaultOpt.value;

    sel.onchange = renderList;
    $("sort-order").onchange = renderList;
    $("hide-no-eos").onchange = renderList;

    const md = DATA.metadata || {};
    $("list-meta").textContent =
      `Model: ${md.model_name} • ${md.num_examples} examples • ` +
      `probes: ${(md.probes || []).join(", ")} • ` +
      `baselines: ${(md.baselines || []).length}`;

    renderList();
  }

  function renderList() {
    const opts = collectSortOptions();
    const optionKey = $("sort-metric").value;
    const sortOpt = opts.find((o) => o.value === optionKey);
    const order = $("sort-order").value;
    const hideNoEos = $("hide-no-eos").checked;

    let examples = DATA.examples.slice();
    if (hideNoEos) examples = examples.filter((e) => e.has_eos);

    examples.sort((a, b) => {
      const va = getAggregate(a, sortOpt);
      const vb = getAggregate(b, sortOpt);
      if (va === null && vb === null) return 0;
      if (va === null) return 1;
      if (vb === null) return -1;
      return order === "asc" ? va - vb : vb - va;
    });

    // Header
    $("examples-thead").innerHTML =
      `<th>ID</th><th>Has EOS</th><th>Compl len</th>` +
      `<th>${sortOpt ? sortOpt.label : "(sort metric)"}</th>` +
      `<th>Preview</th>`;

    // Body
    const tbody = $("examples-tbody");
    tbody.innerHTML = "";
    for (const ex of examples) {
      const tr = document.createElement("tr");
      tr.classList.add("clickable");
      tr.onclick = () => openDetail(ex);
      const preview = (ex.prompt || "").slice(0, 80).replace(/\s+/g, " ");
      const v = sortOpt ? getAggregate(ex, sortOpt) : null;
      tr.innerHTML =
        `<td>${ex.id}</td>` +
        `<td>${ex.has_eos ? "yes" : "no"}</td>` +
        `<td>${ex.completion_length}</td>` +
        `<td>${fmt(v, 4)}</td>` +
        `<td class="preview">${escapeHtml(preview)}</td>`;
      tbody.appendChild(tr);
    }
  }

  // ---------- Detail view ----------
  function openDetail(example) {
    CURRENT_EXAMPLE = example;
    $("detail-title").textContent = `Example ${example.id}`;
    renderMetadata(example);
    renderColumnToggles(example);
    renderTokenTable(example);
    initChart(example);
    showView("detail-view");
  }

  // Rolling-window input: re-render the per-token table on change so the
  // smoothed companion columns reflect the new window length.
  $("token-rolling-window").addEventListener("input", () => {
    const v = parseInt($("token-rolling-window").value, 10);
    if (Number.isFinite(v) && v >= 1) {
      ROLLING_WINDOW = v;
      if (CURRENT_EXAMPLE) renderTokenTable(CURRENT_EXAMPLE);
    }
  });

  $("back-to-list").onclick = () => {
    if (CHART) {
      CHART.destroy();
      CHART = null;
    }
    showView("list-view");
  };

  function renderMetadata(ex) {
    const md = $("detail-metadata");
    md.innerHTML = "";
    const row = (label, value, pre = false) => {
      const p = document.createElement("div");
      p.className = "meta-row";
      p.innerHTML = `<strong>${label}:</strong> <${pre ? "pre" : "span"}>${escapeHtml(value)}</${pre ? "pre" : "span"}>`;
      md.appendChild(p);
    };
    row("Prompt length", ex.prompt_length);
    row("Completion length", ex.completion_length);
    row("Has EOS", ex.has_eos);
    row("Prompt", ex.prompt || "", true);
    row("Completion", ex.completion || "", true);
  }

  // ---------- Token table columns ----------
  // Build a flat list of columns based on the example's data. Each column has:
  //   key, label, group (for show/hide grouping), getValue(tokenIdxInFullSeq)
  function buildColumns(ex) {
    const cols = [];
    const promptLen = ex.prompt_length;
    const complLen = ex.completion_length;

    const completionIdx = (i) => i - promptLen; // negative for prompt tokens

    // Always-on columns
    // Position is relative to the completion start: prompt tokens show "—",
    // first generated token is 0, last completion token is completion_length-1.
    cols.push({
      key: "position",
      label: "Pos",
      group: "basic",
      alwaysOn: true,
      get: (i) => (i < promptLen ? "—" : i - promptLen),
    });
    cols.push({ key: "token", label: "Token", group: "basic", alwaysOn: true, get: (i) => JSON.stringify(ex.tokens_decoded[i]) });
    cols.push({ key: "type", label: "Type", group: "basic", alwaysOn: true, get: (i) => (i < promptLen ? "PROMPT" : "COMPL") });

    // Ground truth
    const gtCount = (ex.ground_truth && ex.ground_truth.remaining_count) || null;
    const gtPct = (ex.ground_truth && ex.ground_truth.percentage) || null;
    cols.push({
      key: "gt_count",
      label: "GT count",
      group: "ground_truth",
      get: (i) => {
        const j = completionIdx(i);
        if (j < 0 || j >= complLen || !gtCount) return "—";
        return fmt(gtCount[j], 0);
      },
    });
    cols.push({
      key: "gt_pct",
      label: "GT %",
      group: "ground_truth",
      get: (i) => {
        const j = completionIdx(i);
        if (j < 0 || j >= complLen || !gtPct) return "—";
        return fmt(gtPct[j], 3);
      },
    });

    // Per probe columns
    const addProbeColumns = (bucket, probeName, entry) => {
      const isCls = Array.isArray(entry.predictions_argmax);
      if (isCls) {
        // Classification probe
        cols.push({
          key: `${bucket}.${probeName}.argmax`,
          label: `${probeName} argmax`,
          group: probeName,
          get: (i) => {
            const j = completionIdx(i);
            if (j < 0 || j >= complLen) return "—";
            return String(entry.predictions_argmax[j]);
          },
        });
        cols.push({
          key: `${bucket}.${probeName}.gt_class`,
          label: `${probeName} GT class`,
          group: probeName,
          get: (i) => {
            const j = completionIdx(i);
            if (j < 0 || j >= complLen || !entry.ground_truth_argmax) return "—";
            return String(entry.ground_truth_argmax[j]);
          },
        });
        cols.push({
          key: `${bucket}.${probeName}.derived_pct`,
          label: `${probeName} derived %`,
          group: probeName,
          get: (i) => {
            const j = completionIdx(i);
            if (j < 0 || j >= complLen) return "—";
            return fmt(entry.derived_percentage[j], 3);
          },
        });
        cols.push({
          key: `${bucket}.${probeName}.derived_pct_mae`,
          label: `${probeName} deriv-% MAE`,
          group: probeName,
          get: (i) => {
            const j = completionIdx(i);
            if (j < 0 || j >= complLen || !entry.derived_pct_mae_per_token) return "—";
            return fmt(entry.derived_pct_mae_per_token[j], 3);
          },
        });
        cols.push({
          key: `${bucket}.${probeName}.derived_pct_mae_avg`,
          label: `${probeName} deriv-% MAE μ`,
          group: probeName,
          get: (i) => {
            const j = completionIdx(i);
            if (j < 0 || j >= complLen || !entry.derived_pct_mae_per_token) return "—";
            return fmt(rollingMean(entry.derived_pct_mae_per_token, j, ROLLING_WINDOW), 3);
          },
        });
        cols.push({
          key: `${bucket}.${probeName}.derived_pct_mse`,
          label: `${probeName} deriv-% MSE`,
          group: probeName,
          get: (i) => {
            const j = completionIdx(i);
            if (j < 0 || j >= complLen || !entry.derived_pct_mse_per_token) return "—";
            return fmt(entry.derived_pct_mse_per_token[j], 3);
          },
        });
        cols.push({
          key: `${bucket}.${probeName}.derived_pct_mse_avg`,
          label: `${probeName} deriv-% MSE μ`,
          group: probeName,
          get: (i) => {
            const j = completionIdx(i);
            if (j < 0 || j >= complLen || !entry.derived_pct_mse_per_token) return "—";
            return fmt(rollingMean(entry.derived_pct_mse_per_token, j, ROLLING_WINDOW), 3);
          },
        });
        cols.push({
          key: `${bucket}.${probeName}.correct`,
          label: `${probeName} correct`,
          group: probeName,
          get: (i) => {
            const j = completionIdx(i);
            if (j < 0 || j >= complLen || !entry.correct_per_token) return "—";
            return entry.correct_per_token[j] ? "✅" : "❌";
          },
        });
        cols.push({
          key: `${bucket}.${probeName}.correct_avg`,
          label: `${probeName} acc μ`,
          group: probeName,
          get: (i) => {
            const j = completionIdx(i);
            if (j < 0 || j >= complLen || !entry.correct_per_token) return "—";
            const m = rollingMean(entry.correct_per_token, j, ROLLING_WINDOW);
            return m == null ? "—" : fmtPct(m);
          },
        });
      } else {
        // Regression probe
        cols.push({
          key: `${bucket}.${probeName}.pred`,
          label: `${probeName} pred`,
          group: probeName,
          get: (i) => {
            // Prompt-end-trained probes (e.g. prompt_only_count) carry a
            // headline scalar prediction at the prompt's last position; show
            // it inline so the central number is visible in the table.
            // After multi-seed merge this is `{mean, std}`, so unwrap via
            // meanOf — `fmt` itself doesn't handle the object shape.
            if (i === promptLen - 1 && entry.prompt_end_prediction != null) {
              return fmt(meanOf(entry.prompt_end_prediction), 3);
            }
            const j = completionIdx(i);
            if (j < 0 || j >= complLen) return "—";
            return fmt(entry.predictions[j], 3);
          },
        });
        cols.push({
          key: `${bucket}.${probeName}.mae`,
          label: `${probeName} MAE`,
          group: probeName,
          get: (i) => {
            const j = completionIdx(i);
            if (j < 0 || j >= complLen || !entry.mae_per_token) return "—";
            return fmt(entry.mae_per_token[j], 3);
          },
        });
        cols.push({
          key: `${bucket}.${probeName}.mae_avg`,
          label: `${probeName} MAE μ`,
          group: probeName,
          get: (i) => {
            const j = completionIdx(i);
            if (j < 0 || j >= complLen || !entry.mae_per_token) return "—";
            return fmt(rollingMean(entry.mae_per_token, j, ROLLING_WINDOW), 3);
          },
        });
        cols.push({
          key: `${bucket}.${probeName}.mse`,
          label: `${probeName} MSE`,
          group: probeName,
          get: (i) => {
            const j = completionIdx(i);
            if (j < 0 || j >= complLen || !entry.mse_per_token) return "—";
            return fmt(entry.mse_per_token[j], 3);
          },
        });
        cols.push({
          key: `${bucket}.${probeName}.mse_avg`,
          label: `${probeName} MSE μ`,
          group: probeName,
          get: (i) => {
            const j = completionIdx(i);
            if (j < 0 || j >= complLen || !entry.mse_per_token) return "—";
            return fmt(rollingMean(entry.mse_per_token, j, ROLLING_WINDOW), 3);
          },
        });
        cols.push({
          key: `${bucket}.${probeName}.mape`,
          label: `${probeName} MAPE`,
          group: probeName,
          get: (i) => {
            const j = completionIdx(i);
            if (j < 0 || j >= complLen || !entry.mape_per_token) return "—";
            return fmtPct(entry.mape_per_token[j]);
          },
        });
        cols.push({
          key: `${bucket}.${probeName}.mape_avg`,
          label: `${probeName} MAPE μ`,
          group: probeName,
          get: (i) => {
            const j = completionIdx(i);
            if (j < 0 || j >= complLen || !entry.mape_per_token) return "—";
            const m = rollingMean(entry.mape_per_token, j, ROLLING_WINDOW);
            return m == null ? "—" : fmtPct(m);
          },
        });
        cols.push({
          key: `${bucket}.${probeName}.smape`,
          label: `${probeName} SMAPE`,
          group: probeName,
          get: (i) => {
            const j = completionIdx(i);
            if (j < 0 || j >= complLen || !entry.smape_per_token) return "—";
            return fmtPct(entry.smape_per_token[j]);
          },
        });
        cols.push({
          key: `${bucket}.${probeName}.smape_avg`,
          label: `${probeName} SMAPE μ`,
          group: probeName,
          get: (i) => {
            const j = completionIdx(i);
            if (j < 0 || j >= complLen || !entry.smape_per_token) return "—";
            const m = rollingMean(entry.smape_per_token, j, ROLLING_WINDOW);
            return m == null ? "—" : fmtPct(m);
          },
        });
      }
    };

    for (const [name, entry] of Object.entries(ex.probes || {})) addProbeColumns("probes", name, entry);
    for (const [name, entry] of Object.entries(ex.statistical_baselines || {})) addProbeColumns("statistical_baselines", name, entry);

    return cols;
  }

  // Aggregate summary row: each column can optionally provide a summary string
  function buildAggregateRow(ex, cols) {
    // Format a {mean, std}-or-scalar value with the given scalar formatter.
    // When the raw value is a multi-seed object, returns "mean ± std";
    // otherwise returns just the mean. `null` falls through to `null` so the
    // caller can decide how to display absent values.
    const fmtPair = (raw, formatter) => {
      if (raw === null || raw === undefined) return null;
      const m = meanOf(raw);
      const s = stdOf(raw);
      if (m === null || m === undefined || Number.isNaN(m)) return null;
      return s != null && !Number.isNaN(s)
        ? `${formatter(m)} ± ${formatter(s)}`
        : formatter(m);
    };

    const cells = [];
    for (const c of cols) {
      const [bucket, probeName, metric] = c.key.split(".");
      if (bucket === "position" || bucket === "token" || bucket === "type" || c.group === "basic" || c.group === "ground_truth") {
        cells.push("");
        continue;
      }
      const entry = (ex[bucket] || {})[probeName];
      if (!entry || !entry.aggregate) {
        cells.push("");
        continue;
      }
      const agg = entry.aggregate;

      // Prompt-only probes expose a prompt vs. completion split via aggregate
      // keys like prompt_mae / completion_mae_remaining. When those exist, show
      // both halves in the cell so the headline prompt-end number is visible
      // inline alongside the per-completion-position summary.
      const promptOnly = agg.prompt_mae != null;
      if (promptOnly && (metric === "mae" || metric === "mse" || metric === "mape" || metric === "smape" || metric === "pred")) {
        const m = metric === "pred" ? "mae" : metric;
        const promptKey = `prompt_${m}`;
        const compKey = `completion_${m}_remaining`;
        const compFallback = `completion_${m}_T`;
        const promptRaw = agg[promptKey];
        const compRaw = agg[compKey] != null ? agg[compKey] : agg[compFallback];
        const fmtVal = (v) => (m === "mape" || m === "smape") ? fmtPct(v) : fmt(v, 3);
        const promptStr = fmtPair(promptRaw, fmtVal);
        const compStr = fmtPair(compRaw, fmtVal);
        const parts = [];
        if (promptStr != null) parts.push(`prompt: ${promptStr}`);
        if (compStr != null) parts.push(`completion: ${compStr}`);
        cells.push(parts.join(" / "));
        continue;
      }

      // Map column metric key to aggregate metric key
      const m = {
        mae: "mae",
        mse: "mse",
        mape: "mape",
        smape: "smape",
        pred: "mae", // show overall MAE under the prediction column for context
        derived_pct: "derived_pct_mae",
        derived_pct_mae: "derived_pct_mae",
        derived_pct_mse: "derived_pct_mse",
        correct: "accuracy",
        argmax: "accuracy",
      }[metric];
      if (!m || agg[m] === null || agg[m] === undefined) {
        cells.push("");
        continue;
      }
      const fullStr = fmtPair(agg[m], (v) => fmt(v, 3));
      const last10Str = fmtPair(agg[`${m}_last_10`], (v) => fmt(v, 3));
      if (fullStr === null) {
        cells.push("");
        continue;
      }
      let cell = `full: ${fullStr}`;
      if (last10Str !== null) cell += ` / last10: ${last10Str}`;

      // For classification probes, fold Cohen's kappa into the accuracy cells
      // — kappa is otherwise only reachable via the sort menu, but it's the
      // metric reviewers actually look at for chance-corrected agreement.
      if ((metric === "correct" || metric === "argmax") && agg.cohen_kappa != null) {
        const kStr = fmtPair(agg.cohen_kappa, (v) => fmt(v, 3));
        const k10Str = fmtPair(agg.cohen_kappa_last_10, (v) => fmt(v, 3));
        cell += ` (κ: ${kStr}`;
        if (k10Str != null) cell += ` / last10: ${k10Str}`;
        cell += ")";
      }
      cells.push(cell);
    }
    return cells;
  }

  function renderColumnToggles(ex) {
    const cols = buildColumns(ex);
    const wrap = $("column-toggles");
    wrap.innerHTML = "";

    // Initialize visibility once per example
    if (!Object.keys(COLUMN_VISIBLE).length || !cols.every((c) => c.key in COLUMN_VISIBLE)) {
      COLUMN_VISIBLE = {};
      for (const c of cols) {
        // Default: show everything
        COLUMN_VISIBLE[c.key] = true;
      }
    }

    // Group toggles by column group
    const groups = {};
    for (const c of cols) {
      if (!(c.group in groups)) groups[c.group] = [];
      groups[c.group].push(c);
    }

    // "All" controls per group
    for (const [group, groupCols] of Object.entries(groups)) {
      const box = document.createElement("fieldset");
      box.className = "col-group";
      const legend = document.createElement("legend");
      legend.textContent = group;
      box.appendChild(legend);

      // group-level all/none
      const allBtn = document.createElement("button");
      allBtn.type = "button";
      allBtn.textContent = "all";
      allBtn.onclick = () => {
        groupCols.forEach((c) => (COLUMN_VISIBLE[c.key] = true));
        renderColumnToggles(ex);
        renderTokenTable(ex);
      };
      const noneBtn = document.createElement("button");
      noneBtn.type = "button";
      noneBtn.textContent = "none";
      noneBtn.onclick = () => {
        groupCols.forEach((c) => {
          if (!c.alwaysOn) COLUMN_VISIBLE[c.key] = false;
        });
        renderColumnToggles(ex);
        renderTokenTable(ex);
      };
      box.appendChild(allBtn);
      box.appendChild(noneBtn);

      for (const c of groupCols) {
        const lbl = document.createElement("label");
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = COLUMN_VISIBLE[c.key];
        cb.disabled = !!c.alwaysOn;
        cb.onchange = () => {
          COLUMN_VISIBLE[c.key] = cb.checked;
          renderTokenTable(ex);
        };
        lbl.appendChild(cb);
        lbl.appendChild(document.createTextNode(` ${c.label}`));
        box.appendChild(lbl);
      }
      wrap.appendChild(box);
    }
  }

  function renderTokenTable(ex) {
    const cols = buildColumns(ex);
    const visibleCols = cols.filter((c) => COLUMN_VISIBLE[c.key]);

    // Header
    const thead = $("token-thead");
    thead.innerHTML = "";
    for (const c of visibleCols) {
      const th = document.createElement("th");
      th.textContent = c.label;
      thead.appendChild(th);
    }

    // Aggregate summary row (sticky-ish, just a row right under header)
    const aggCells = buildAggregateRow(ex, visibleCols);

    const tbody = $("token-tbody");
    tbody.innerHTML = "";

    const aggTr = document.createElement("tr");
    aggTr.className = "aggregate-row";
    for (const v of aggCells) {
      const td = document.createElement("td");
      td.textContent = v;
      aggTr.appendChild(td);
    }
    tbody.appendChild(aggTr);

    // Token rows
    const seqLen = ex.tokens_decoded.length;
    for (let i = 0; i < seqLen; i++) {
      const tr = document.createElement("tr");
      tr.classList.add(i < ex.prompt_length ? "row-prompt" : "row-compl");
      for (const c of visibleCols) {
        const td = document.createElement("td");
        td.textContent = c.get(i);
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
  }

  // ---------- Chart ----------
  function collectChartSeries(ex) {
    // One option per per-token series we could plot as bars.
    const options = [];
    const push = (label, bucket, probeName, field, mapFn = (x) => x) => {
      const entry = (ex[bucket] || {})[probeName];
      if (!entry || !entry[field]) return;
      options.push({ value: `${bucket}.${probeName}.${field}`, label, bucket, probeName, field, mapFn });
    };

    for (const [name, entry] of Object.entries(ex.probes || {})) {
      if (Array.isArray(entry.predictions_argmax)) {
        // classification
        push(`${name} derived %`, "probes", name, "derived_percentage");
        push(`${name} derived % MAE`, "probes", name, "derived_pct_mae_per_token");
        push(`${name} derived % MSE`, "probes", name, "derived_pct_mse_per_token");
        push(`${name} argmax`, "probes", name, "predictions_argmax");
      } else {
        push(`${name} pred`, "probes", name, "predictions");
        push(`${name} MAE`, "probes", name, "mae_per_token");
        push(`${name} MSE`, "probes", name, "mse_per_token");
        push(`${name} MAPE`, "probes", name, "mape_per_token");
        push(`${name} SMAPE`, "probes", name, "smape_per_token");
        // Static-decay series: only present for prompt-only-trained probes,
        // i.e. when `static_decay_predictions` is in the entry.
        push(`${name} static-decay pred`, "probes", name, "static_decay_predictions");
        push(`${name} static-decay MAE`, "probes", name, "static_decay_mae_per_token");
        push(`${name} static-decay MSE`, "probes", name, "static_decay_mse_per_token");
        push(`${name} static-decay MAPE`, "probes", name, "static_decay_mape_per_token");
        push(`${name} static-decay SMAPE`, "probes", name, "static_decay_smape_per_token");
      }
    }
    for (const [name, entry] of Object.entries(ex.statistical_baselines || {})) {
      push(`${name} pred`, "statistical_baselines", name, "predictions");
      push(`${name} MAE`, "statistical_baselines", name, "mae_per_token");
      push(`${name} MSE`, "statistical_baselines", name, "mse_per_token");
      push(`${name} MAPE`, "statistical_baselines", name, "mape_per_token");
      push(`${name} SMAPE`, "statistical_baselines", name, "smape_per_token");
    }
    return options;
  }

  function initChart(ex) {
    const opts = collectChartSeries(ex);
    const sel = $("chart-series");
    sel.innerHTML = "";
    for (const o of opts) {
      const el = document.createElement("option");
      el.value = o.value;
      el.textContent = o.label;
      sel.appendChild(el);
    }
    // Default: count pred
    const defaultOpt = opts.find((o) => o.value === "probes.count.predictions") || opts[0];
    if (defaultOpt) sel.value = defaultOpt.value;

    sel.onchange = () => renderChart(ex);
    $("chart-groundtruth").onchange = () => renderChart(ex);
    $("chart-smoothing-sigma").oninput = () => renderChart(ex);

    renderChart(ex);
  }

  function renderChart(ex) {
    const opts = collectChartSeries(ex);
    const selected = opts.find((o) => o.value === $("chart-series").value);
    const gtChoice = $("chart-groundtruth").value;

    const complLen = ex.completion_length;
    const xs = Array.from({ length: complLen }, (_, k) => k);

    let bars = [];
    if (selected) {
      const arr = (ex[selected.bucket] || {})[selected.probeName][selected.field] || [];
      bars = arr.map((v) => (v === null || v === undefined ? null : v));
    }

    let gtData = null;
    if (gtChoice === "remaining_count") gtData = (ex.ground_truth && ex.ground_truth.remaining_count) || null;
    else if (gtChoice === "percentage") gtData = (ex.ground_truth && ex.ground_truth.percentage) || null;

    const datasets = [];
    datasets.push({
      type: "bar",
      label: selected ? selected.label : "(none)",
      data: bars,
      backgroundColor: "rgba(99,133,199,0.7)",
      borderColor: "rgba(99,133,199,1)",
      borderWidth: 1,
      order: 2,
      yAxisID: "y",
    });

    // Optional Gaussian-smoothed line over the bars (sigma in token-position units).
    const sigmaRaw = parseFloat($("chart-smoothing-sigma").value);
    const sigma = Number.isFinite(sigmaRaw) && sigmaRaw > 0 ? sigmaRaw : 0;
    if (sigma > 0 && bars.length) {
      const smoothed = gaussianSmooth(bars, sigma);
      datasets.push({
        type: "line",
        label: `${selected ? selected.label : "smoothed"} (Gaussian σ=${sigma})`,
        data: smoothed,
        borderColor: "rgba(220,60,60,1)",
        backgroundColor: "rgba(220,60,60,0.15)",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
        fill: false,
        order: 0,
        yAxisID: "y",
      });
    }

    if (gtData && gtData.length) {
      datasets.push({
        type: "line",
        label: `ground truth (${gtChoice})`,
        data: gtData,
        borderColor: "rgba(30,80,200,1)",
        backgroundColor: "rgba(30,80,200,0.2)",
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
        order: 1,
        yAxisID: gtChoice === "percentage" ? "y1" : "y",
      });
    }

    const config = {
      data: { labels: xs, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: { title: { display: true, text: "Token position in completion" } },
          y: {
            type: "linear",
            position: "left",
            title: { display: true, text: selected ? selected.label : "" },
          },
          y1: {
            type: "linear",
            position: "right",
            title: { display: true, text: "percentage" },
            grid: { drawOnChartArea: false },
            min: 0,
            max: 1.05,
            display: gtChoice === "percentage",
          },
        },
      },
    };

    if (CHART) CHART.destroy();
    CHART = new Chart($("detail-chart"), config);
  }

  function escapeHtml(s) {
    if (s === null || s === undefined) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }
})();
