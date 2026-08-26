const form = document.querySelector("#chat-form");
const chat = document.querySelector("#chat");
const fileInput = document.querySelector("#file");
const messageInput = document.querySelector("#message");
const sendButton = document.querySelector("#send");
const chip = document.querySelector("#file-chip");
const fileName = document.querySelector("#file-name");
const documentStatus = document.querySelector("#document-status");
const sessionId = crypto.randomUUID();
const chartColors = ["#4f46e5", "#0891b2", "#db2777"];
let hasDocument = false;

function scrollToLatest() { chat.scrollTop = chat.scrollHeight; }

function textNode(tag, value) {
  const node = document.createElement(tag);
  node.textContent = String(value);
  return node;
}

function addMessage(className, content) {
  const element = document.createElement("article");
  element.className = `message ${className}`;
  element.append(content);
  chat.append(element);
  scrollToLatest();
  return element;
}

function addParagraphs(fragment, text, className = "") {
  String(text).split(/\n\s*\n/).filter(Boolean).forEach((paragraph) => {
    const item = textNode("p", paragraph.trim());
    if (className) item.className = className;
    fragment.append(item);
  });
}

function svgElement(tag, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function displayNumber(value) {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value);
}

function renderChart(chart) {
  const card = document.createElement("section");
  card.className = "chart-card";
  card.append(textNode("h3", chart.title));
  if (chart.description) addParagraphs(card, chart.description, "chart-description");

  const width = 640;
  const height = 290;
  const margin = { top: 20, right: 20, bottom: 70, left: 52 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const values = chart.datasets.flatMap((dataset) => dataset.values);
  const minValue = Math.min(0, ...values);
  const maxValue = Math.max(0, ...values);
  const spread = maxValue - minValue || 1;
  const yFor = (value) => margin.top + ((maxValue - value) / spread) * plotHeight;
  const svg = svgElement("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": chart.title });

  for (let tick = 0; tick <= 4; tick += 1) {
    const value = minValue + (spread * tick) / 4;
    const y = yFor(value);
    svg.append(svgElement("line", { x1: margin.left, y1: y, x2: width - margin.right, y2: y, class: "grid-line" }));
    const label = svgElement("text", { x: margin.left - 8, y: y + 4, class: "axis-label", "text-anchor": "end" });
    label.textContent = displayNumber(value);
    svg.append(label);
  }
  svg.append(svgElement("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, class: "axis-line" }));
  svg.append(svgElement("line", { x1: margin.left, y1: yFor(0), x2: width - margin.right, y2: yFor(0), class: "axis-line" }));

  const groupWidth = plotWidth / chart.labels.length;
  if (chart.type === "bar") {
    const barWidth = Math.max(4, (groupWidth * 0.74) / chart.datasets.length);
    chart.datasets.forEach((dataset, datasetIndex) => {
      dataset.values.forEach((value, index) => {
        const x = margin.left + index * groupWidth + groupWidth * 0.13 + datasetIndex * barWidth;
        const y = yFor(Math.max(value, 0));
        const zero = yFor(0);
        svg.append(svgElement("rect", {
          x, y, width: barWidth - 2, height: Math.abs(zero - yFor(value)), rx: 3,
          fill: chartColors[datasetIndex % chartColors.length],
        }));
      });
    });
  } else {
    chart.datasets.forEach((dataset, datasetIndex) => {
      const points = dataset.values.map((value, index) => `${margin.left + groupWidth * (index + 0.5)},${yFor(value)}`).join(" ");
      svg.append(svgElement("polyline", { points, fill: "none", stroke: chartColors[datasetIndex % chartColors.length], "stroke-width": 3, "stroke-linejoin": "round", "stroke-linecap": "round" }));
      dataset.values.forEach((value, index) => svg.append(svgElement("circle", { cx: margin.left + groupWidth * (index + 0.5), cy: yFor(value), r: 3.5, fill: chartColors[datasetIndex % chartColors.length] })));
    });
  }

  chart.labels.forEach((label, index) => {
    const x = margin.left + groupWidth * (index + 0.5);
    const text = svgElement("text", { x, y: height - margin.bottom + 18, class: "axis-label", "text-anchor": "end", transform: `rotate(-32 ${x} ${height - margin.bottom + 18})` });
    text.textContent = label.length > 18 ? `${label.slice(0, 17)}…` : label;
    svg.append(text);
  });
  card.append(svg);

  const legend = document.createElement("div");
  legend.className = "chart-legend";
  chart.datasets.forEach((dataset, index) => {
    const item = document.createElement("span");
    const dot = document.createElement("i");
    dot.style.background = chartColors[index % chartColors.length];
    item.append(dot, document.createTextNode(dataset.label));
    legend.append(item);
  });
  card.append(legend);
  return card;
}

function renderResponse(data) {
  const fragment = document.createDocumentFragment();
  fragment.append(textNode("h2", data.title || "Document analysis"));
  if (data.summary) addParagraphs(fragment, data.summary, "summary");
  if (data.detailed_explanation) {
    fragment.append(textNode("h3", "Detailed explanation"));
    addParagraphs(fragment, data.detailed_explanation);
  }
  if (Array.isArray(data.key_points) && data.key_points.length) {
    fragment.append(textNode("h3", "Key points"));
    const list = document.createElement("ul");
    data.key_points.forEach((item) => list.append(textNode("li", item)));
    fragment.append(list);
  }
  if (Array.isArray(data.charts)) data.charts.forEach((chart) => fragment.append(renderChart(chart)));
  if (Array.isArray(data.citations) && data.citations.length) {
    fragment.append(textNode("h3", "Sources used"));
    const citations = document.createElement("div");
    citations.className = "citations";
    data.citations.forEach((citation) => {
      const card = document.createElement("div");
      card.className = "citation";
      card.append(textNode("strong", `${citation.id} · ${citation.modality} · page ${citation.page}`));
      if (citation.excerpt) card.append(textNode("span", citation.excerpt));
      citations.append(card);
    });
    fragment.append(citations);
  }
  if (data.evaluation) {
    const evaluation = document.createElement("p");
    evaluation.className = `evaluation evaluation-${data.evaluation.groundedness || "medium"}`;
    const route = Array.isArray(data.evaluation.route) ? data.evaluation.route.join(" + ") : "text";
    evaluation.textContent = `RAG evaluation: ${data.evaluation.groundedness || "unknown"} grounding · ${data.evaluation.retrieved_sources || 0} sources retrieved · route: ${route}`;
    fragment.append(evaluation);
  }
  if (Array.isArray(data.possible_questions) && data.possible_questions.length) {
    fragment.append(textNode("h3", "Explore further"));
    const suggestions = document.createElement("div");
    suggestions.className = "suggestions";
    data.possible_questions.forEach((question) => {
      const button = textNode("button", question);
      button.type = "button";
      button.addEventListener("click", () => { messageInput.value = question; messageInput.focus(); });
      suggestions.append(button);
    });
    fragment.append(suggestions);
  }
  if (data.notice) { const note = textNode("p", data.notice); note.className = "notice"; fragment.append(note); }
  return fragment;
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  chip.hidden = !file;
  fileName.textContent = file?.name || "";
});
document.querySelector("#remove-file").addEventListener("click", () => { fileInput.value = ""; chip.hidden = true; });
document.querySelectorAll("[data-prompt]").forEach((button) => button.addEventListener("click", () => { messageInput.value = button.dataset.prompt; messageInput.focus(); }));

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  const message = messageInput.value.trim();
  if (!file && !hasDocument) {
    addMessage("error", textNode("p", "Please attach a PDF first. You can ask follow-up questions without uploading it again."));
    return;
  }

  const userText = `${message || "Explain this document clearly"}${file ? `\n📎 ${file.name}` : ""}`;
  addMessage("user", textNode("p", userText));
  const dots = document.createElement("div"); dots.className = "typing"; dots.innerHTML = "<i></i><i></i><i></i>";
  const loading = addMessage("assistant", dots);
  sendButton.disabled = true;

  const body = new FormData();
  if (file) body.append("file", file);
  body.append("message", message);
  body.append("session_id", sessionId);
  try {
    const response = await fetch("/explain", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || data.error || "Something went wrong.");
    hasDocument = Boolean(data.document_loaded) || hasDocument;
    if (data.document_name) {
      const counts = data.ingestion || {};
      const indexSummary = `${counts.text || 0} text · ${counts.table || 0} tables · ${counts.image || 0} figures`;
      documentStatus.textContent = `📄 ${data.document_name} · ${indexSummary}`;
      documentStatus.hidden = false;
    }
    loading.classList.add("response"); loading.replaceChildren(renderResponse(data));
  } catch (error) {
    loading.className = "message error"; loading.replaceChildren(textNode("p", error.message));
  } finally {
    sendButton.disabled = false; messageInput.value = ""; fileInput.value = ""; chip.hidden = true; scrollToLatest();
  }
});
