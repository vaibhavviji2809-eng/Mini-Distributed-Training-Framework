const sampleMetrics = {
  sync_method: "ring",
  compression: "fp16",
  losses: [3.46, 3.41, 3.37, 3.31, 3.26, 3.19],
  throughput: [10234, 11890, 12600, 13120, 13655, 14110],
  communication_time: [0.34, 0.31, 0.29, 0.28, 0.27, 0.26],
  worker_health: { 0: "done", 1: "done", 2: "done", 3: "done" },
  restarts: 0,
};

const state = { metrics: sampleMetrics };

const summaryCards = document.getElementById("summary-cards");
const workerGrid = document.getElementById("worker-grid");
const jsonView = document.getElementById("json-view");
const syncChip = document.getElementById("sync-chip");
const compressionChip = document.getElementById("compression-chip");
const restartChip = document.getElementById("restart-chip");
const lossChart = document.getElementById("loss-chart");
const throughputChart = document.getElementById("throughput-chart");

function normalizeMetrics(payload) {
  if (payload && typeof payload === "object" && payload.metrics) {
    return payload.metrics;
  }
  return payload;
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  if (Math.abs(value) >= 1000) {
    return value.toFixed(0);
  }
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function latest(values) {
  return values.length ? values[values.length - 1] : null;
}

function average(values) {
  if (!values.length) return 0;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function makeStatCard(label, value, foot) {
  return `
    <article class="stat-card">
      <div class="stat-label">${label}</div>
      <div class="stat-value">${value}</div>
      <div class="stat-foot">${foot}</div>
    </article>
  `;
}

function renderSummary(metrics) {
  summaryCards.innerHTML = [
    makeStatCard("Final loss", formatNumber(latest(metrics.losses)), `${metrics.losses.length} recorded steps`),
    makeStatCard("Throughput", `${formatNumber(latest(metrics.throughput))}`, `${formatNumber(average(metrics.throughput))} avg samples/sec`),
    makeStatCard("Comm. time", `${formatNumber(latest(metrics.communication_time))}s`, `${formatNumber(average(metrics.communication_time))}s average`),
    makeStatCard("Restarts", `${metrics.restarts ?? 0}`, "Fault-tolerance counter"),
  ].join("");
}

function renderWorkers(metrics) {
  const workerIds = Object.keys(metrics.worker_health || {}).sort((left, right) => Number(left) - Number(right));
  const workers = workerIds.length ? workerIds : ["0", "1", "2", "3"];
  workerGrid.innerHTML = workers
    .map((rank) => {
      const status = metrics.worker_health?.[rank] || "offline";
      const statusClass = status === "started" ? "running" : ["healthy", "running", "done", "offline"].includes(status) ? status : "offline";
      return `
        <article class="worker-card">
          <h3>Worker ${rank}</h3>
          <div class="worker-pill ${statusClass}">${status}</div>
        </article>
      `;
    })
    .join("");
}

function drawChart(canvas, values, options) {
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const padding = 42;

  context.clearRect(0, 0, width, height);

  context.fillStyle = "#08111f";
  context.fillRect(0, 0, width, height);

  context.strokeStyle = "rgba(159, 177, 209, 0.25)";
  context.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick += 1) {
    const y = padding + ((height - padding * 2) / 4) * tick;
    context.beginPath();
    context.moveTo(padding, y);
    context.lineTo(width - padding, y);
    context.stroke();
  }

  if (!values.length) {
    context.fillStyle = "#9fb1d1";
    context.font = "16px sans-serif";
    context.fillText("No data available", padding, height / 2);
    return;
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const safeRange = Math.max(max - min, 1e-6);
  const graphWidth = width - padding * 2;
  const graphHeight = height - padding * 2;

  const points = values.map((value, index) => {
    const x = padding + (graphWidth * index) / Math.max(values.length - 1, 1);
    const normalized = (value - min) / safeRange;
    const y = height - padding - normalized * graphHeight;
    return { x, y, value };
  });

  const gradient = context.createLinearGradient(0, padding, width, height - padding);
  gradient.addColorStop(0, options.startColor);
  gradient.addColorStop(1, options.endColor);

  context.beginPath();
  points.forEach((point, index) => {
    if (index === 0) {
      context.moveTo(point.x, point.y);
    } else {
      context.lineTo(point.x, point.y);
    }
  });
  context.strokeStyle = gradient;
  context.lineWidth = 4;
  context.stroke();

  context.fillStyle = "rgba(118, 215, 255, 0.15)";
  context.beginPath();
  context.moveTo(points[0].x, height - padding);
  points.forEach((point) => context.lineTo(point.x, point.y));
  context.lineTo(points[points.length - 1].x, height - padding);
  context.closePath();
  context.fill();

  points.forEach((point) => {
    context.fillStyle = options.endColor;
    context.beginPath();
    context.arc(point.x, point.y, 4.5, 0, Math.PI * 2);
    context.fill();
  });

  context.fillStyle = "#9fb1d1";
  context.font = "13px ui-sans-serif, system-ui, sans-serif";
  context.fillText(options.labelLeft, padding, 24);
  context.fillText(options.labelRight, width - padding - 90, 24);
  context.fillText(options.minLabel, padding, height - 14);
  context.fillText(options.maxLabel, padding, 40);
}

function renderCharts(metrics) {
  const losses = metrics.losses || [];
  const throughput = metrics.throughput || [];
  drawChart(lossChart, metrics.losses || [], {
    startColor: "#76d7ff",
    endColor: "#7c8cff",
    labelLeft: "Loss",
    labelRight: "lower is better",
    minLabel: losses.length ? `min ${formatNumber(Math.min(...losses))}` : "no data",
    maxLabel: losses.length ? `max ${formatNumber(Math.max(...losses))}` : "no data",
  });

  drawChart(throughputChart, metrics.throughput || [], {
    startColor: "#56e39f",
    endColor: "#76d7ff",
    labelLeft: "Samples/sec",
    labelRight: "higher is better",
    minLabel: throughput.length ? `min ${formatNumber(Math.min(...throughput))}` : "no data",
    maxLabel: throughput.length ? `max ${formatNumber(Math.max(...throughput))}` : "no data",
  });
}

function renderMetadata(metrics) {
  syncChip.textContent = metrics.sync_method || "parameter_server";
  compressionChip.textContent = metrics.compression || "none";
  restartChip.textContent = `${metrics.restarts ?? 0} restarts`;
  jsonView.value = JSON.stringify(metrics, null, 2);
}

function render(payload) {
  const metrics = normalizeMetrics(payload);
  state.metrics = metrics;
  renderSummary(metrics);
  renderWorkers(metrics);
  renderCharts(metrics);
  renderMetadata(metrics);
}

async function readFile(file) {
  const text = await file.text();
  return JSON.parse(text);
}

document.getElementById("file-input").addEventListener("change", async (event) => {
  const [file] = event.target.files || [];
  if (!file) return;
  try {
    render(await readFile(file));
  } catch (error) {
    alert(`Could not load JSON: ${error.message}`);
  }
});

document.getElementById("load-sample").addEventListener("click", () => {
  render(sampleMetrics);
});

document.getElementById("copy-json").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(jsonView.value);
  } catch (error) {
    alert(`Copy failed: ${error.message}`);
  }
});

window.addEventListener("resize", () => render(state.metrics));

render(sampleMetrics);
