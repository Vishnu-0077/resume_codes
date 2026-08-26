const form = document.querySelector("#chat-form");
const chat = document.querySelector("#chat");
const fileInput = document.querySelector("#file");
const fileBInput = document.querySelector("#file-b");
const messageInput = document.querySelector("#message");
const sendButton = document.querySelector("#send");
const chip = document.querySelector("#file-chip");
const chipB = document.querySelector("#file-b-chip");
const fileName = document.querySelector("#file-name");
const fileBName = document.querySelector("#file-b-name");
const documentStatus = document.querySelector("#document-status");
const researchMode = document.querySelector("#research-mode");
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

function selectedAnswerMode() {
  return document.querySelector('input[name="answer-mode"]:checked')?.value || "document_only";
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

function linkifyCitations(text, citations) {
  const fragment = document.createDocumentFragment();
  const byDisplay = new Map();
  (citations || []).forEach((citation) => {
    const key = citation.display || `${citation.label || citation.modality}, Page ${citation.page}`;
    byDisplay.set(key, citation);
  });
  const pattern = /\[([^\]]+Page\s+\d+[^\]]*)\]/g;
  let last = 0;
  let match;
  const source = String(text);
  while ((match = pattern.exec(source))) {
    if (match.index > last) fragment.append(document.createTextNode(source.slice(last, match.index)));
    const button = document.createElement("button");
    button.type = "button";
    button.className = "cite-link";
    button.textContent = match[0];
    const citation = byDisplay.get(match[1]) || citations?.find((item) => match[0].includes(`Page ${item.page}`));
    if (citation) {
      button.title = citation.excerpt || citation.display;
      button.addEventListener("click", () => {
        const target = document.querySelector(`[data-cite-id="${citation.id}"]`);
        if (target) {
          target.classList.add("citation-flash");
          target.scrollIntoView({ behavior: "smooth", block: "nearest" });
          setTimeout(() => target.classList.remove("citation-flash"), 1200);
        }
      });
    }
    fragment.append(button);
    last = match.index + match[0].length;
  }
  if (last < source.length) fragment.append(document.createTextNode(source.slice(last)));
  return fragment;
}

function renderResponse(data) {
  const fragment = document.createDocumentFragment();
  fragment.append(textNode("h2", data.title || "Document analysis"));

  if (data.confidence) {
    const panel = document.createElement("div");
    panel.className = `confidence confidence-${(data.confidence.evidence_strength || "LOW").toLowerCase()}`;
    const pages = Array.isArray(data.confidence.pages) ? data.confidence.pages.join(", ") : "";
    panel.append(
      textNode("strong", `Evidence strength: ${data.confidence.evidence_strength || "LOW"}`),
      textNode("span", `Sources used: ${data.confidence.sources_used || 0}`),
      textNode("span", pages ? `Pages: ${pages}` : "Pages: —"),
      textNode("span", `Retrieval confidence: ${data.confidence.retrieval_confidence ?? "—"}`),
    );
    if (data.confidence.reason) panel.append(textNode("em", data.confidence.reason));
    fragment.append(panel);
  }

  if (data.summary) addParagraphs(fragment, data.summary, "summary");
  if (data.detailed_explanation) {
    fragment.append(textNode("h3", "Detailed explanation"));
    String(data.detailed_explanation).split(/\n\s*\n/).filter(Boolean).forEach((paragraph) => {
      const item = document.createElement("p");
      item.append(linkifyCitations(paragraph.trim(), data.citations));
      fragment.append(item);
    });
  }
  if (Array.isArray(data.key_points) && data.key_points.length) {
    fragment.append(textNode("h3", "Key points"));
    const list = document.createElement("ul");
    data.key_points.forEach((item) => list.append(textNode("li", item)));
    fragment.append(list);
  }
  if (data.research_mode && Array.isArray(data.related_sections) && data.related_sections.length) {
    fragment.append(textNode("h3", "Related sections"));
    const list = document.createElement("ul");
    data.related_sections.forEach((item) => list.append(textNode("li", item)));
    fragment.append(list);
  }
  if (data.general_knowledge_notes) {
    fragment.append(textNode("h3", "General knowledge (separated)"));
    addParagraphs(fragment, data.general_knowledge_notes, "gk-notes");
  }
  if (Array.isArray(data.charts)) data.charts.forEach((chart) => fragment.append(renderChart(chart)));

  if (Array.isArray(data.citations) && data.citations.length) {
    fragment.append(textNode("h3", "Sources used"));
    const citations = document.createElement("div");
    citations.className = "citations";
    data.citations.forEach((citation) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "citation";
      card.dataset.citeId = citation.id;
      const heading = citation.display || `${citation.label || citation.modality} · Page ${citation.page}`;
      card.append(textNode("strong", `${citation.id} · ${heading}`));
      if (citation.excerpt) card.append(textNode("span", citation.excerpt));
      if (citation.bbox) card.append(textNode("small", `bbox: ${citation.bbox.map((n) => Math.round(n)).join(", ")}`));
      citations.append(card);
    });
    fragment.append(citations);
  }

  if (Array.isArray(data.evidence_bundle) && data.evidence_bundle.length) {
    fragment.append(textNode("h3", "Evidence bundle"));
    const bundle = document.createElement("div");
    bundle.className = "bundle";
    data.evidence_bundle.slice(0, 8).forEach((item) => {
      const chip = textNode("span", `${item.id} · ${item.label || item.modality} · p${item.page}`);
      chip.className = `bundle-chip modality-${item.modality}`;
      bundle.append(chip);
    });
    fragment.append(bundle);
  }

  if (data.evaluation) {
    const evaluation = document.createElement("p");
    evaluation.className = `evaluation evaluation-${data.evaluation.groundedness || "medium"}`;
    const route = Array.isArray(data.evaluation.route) ? data.evaluation.route.join(" + ") : "text";
    const latency = data.latency ? ` · ${Math.round(data.latency.total_ms)} ms` : "";
    evaluation.textContent = `RAG evaluation: ${data.evaluation.groundedness || "unknown"} grounding · ${data.evaluation.retrieved_sources || 0} sources · route: ${route}${latency}`;
    fragment.append(evaluation);
  }

  if (data.agent) {
    const agent = document.createElement("p");
    agent.className = "agent-meta";
    agent.textContent = `Agent: intent=${data.agent.intent} · route=${data.agent.route_label} · mode=${data.answer_mode || "document_only"}`;
    fragment.append(agent);
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
fileBInput.addEventListener("change", () => {
  const file = fileBInput.files[0];
  chipB.hidden = !file;
  fileBName.textContent = file ? `Compare: ${file.name}` : "";
});
document.querySelector("#remove-file").addEventListener("click", () => { fileInput.value = ""; chip.hidden = true; });
document.querySelector("#remove-file-b").addEventListener("click", () => { fileBInput.value = ""; chipB.hidden = true; });
document.querySelectorAll("[data-prompt]").forEach((button) => button.addEventListener("click", () => { messageInput.value = button.dataset.prompt; messageInput.focus(); }));

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  const fileB = fileBInput.files[0];
  const message = messageInput.value.trim();
  if (!file && !hasDocument) {
    addMessage("error", textNode("p", "Please attach PDF A first. You can ask follow-up questions without uploading it again."));
    return;
  }

  const attachments = [file ? `📎 A: ${file.name}` : "", fileB ? `📎 B: ${fileB.name}` : ""].filter(Boolean).join(" · ");
  const userText = `${message || "Explain this document clearly"}${attachments ? `\n${attachments}` : ""}`;
  addMessage("user", textNode("p", userText));
  const dots = document.createElement("div"); dots.className = "typing"; dots.innerHTML = "<i></i><i></i><i></i>";
  const loading = addMessage("assistant", dots);
  sendButton.disabled = true;

  const body = new FormData();
  if (file) body.append("file", file);
  if (fileB) body.append("file_b", fileB);
  body.append("message", message);
  body.append("session_id", sessionId);
  body.append("answer_mode", selectedAnswerMode());
  body.append("research_mode", researchMode.checked ? "true" : "false");
  try {
    const response = await fetch("/explain", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || data.error || "Something went wrong.");
    hasDocument = Boolean(data.document_loaded) || hasDocument;
    if (data.document_name) {
      const counts = data.ingestion || {};
      const docs = data.documents ? Object.values(data.documents).join(" · ") : data.document_name;
      const indexSummary = `${counts.text || 0} text · ${counts.table || 0} tables · ${counts.image || 0} figures`;
      documentStatus.textContent = `📄 ${docs} · ${indexSummary}`;
      documentStatus.hidden = false;
    }
    loading.classList.add("response"); loading.replaceChildren(renderResponse(data));
  } catch (error) {
    loading.className = "message error"; loading.replaceChildren(textNode("p", error.message));
  } finally {
    sendButton.disabled = false; messageInput.value = ""; fileInput.value = ""; fileBInput.value = "";
    chip.hidden = true; chipB.hidden = true; scrollToLatest();
  }
});
