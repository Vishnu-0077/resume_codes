const form = document.querySelector("#chat-form");
const chat = document.querySelector("#chat");
const fileInput = document.querySelector("#file");
const messageInput = document.querySelector("#message");
const sendButton = document.querySelector("#send");
const chip = document.querySelector("#file-chip");
const fileName = document.querySelector("#file-name");
const sessionId = crypto.randomUUID();
let hasDocument = false;

function scrollToLatest() { chat.scrollTop = chat.scrollHeight; }

function addMessage(className, content) {
  const element = document.createElement("article");
  element.className = `message ${className}`;
  element.append(content);
  chat.append(element);
  scrollToLatest();
  return element;
}

function textNode(tag, value) { const node = document.createElement(tag); node.textContent = value; return node; }

function renderResponse(data) {
  const fragment = document.createDocumentFragment();
  fragment.append(textNode("h2", data.title || "PDF explanation"));
  if (data.summary) fragment.append(textNode("p", data.summary));
  if (data.detailed_explanation) fragment.append(textNode("p", data.detailed_explanation));
  for (const [heading, items] of [["Key points", data.key_points], ["Possible follow-up questions", data.possible_questions]]) {
    if (Array.isArray(items) && items.length) {
      fragment.append(textNode("h3", heading));
      const list = document.createElement("ul");
      items.forEach((item) => list.append(textNode("li", item)));
      fragment.append(list);
    }
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
    loading.classList.add("response"); loading.replaceChildren(renderResponse(data));
  } catch (error) {
    loading.className = "message error"; loading.replaceChildren(textNode("p", error.message));
  } finally {
    sendButton.disabled = false; messageInput.value = ""; fileInput.value = ""; chip.hidden = true; scrollToLatest();
  }
});
