const SESSION_KEY = "legal_assistant_session_id";

const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chatForm");
const inputEl = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const charCounter = document.querySelector("#charCounter");
const systemStatus = document.querySelector("#systemStatus");
const newSessionButton = document.querySelector("#newSessionButton");
const copyLastButton = document.querySelector("#copyLastButton");
const sessionHistoryEl = document.querySelector("#sessionHistory");
const refreshHistoryButton = document.querySelector("#refreshHistoryButton");

let sessionId = localStorage.getItem(SESSION_KEY) || "";
let lastAnswer = "";
let lastQuestion = "";
let historyLoadToken = 0;

function setStatus(text) {
  systemStatus.textContent = text;
}

function autoGrowInput() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${inputEl.scrollHeight}px`;
}

function clearEmptyState() {
  const empty = messagesEl.querySelector(".empty-state");
  if (empty) empty.remove();
}

function emptyStateMarkup() {
  return `
    <div class="empty-state">
      <h3>Bạn cần hỗ trợ vấn đề pháp lý nào?</h3>
      <p>Nhập tình huống cụ thể, thời điểm và chủ thể liên quan để hệ thống tra cứu căn cứ phù hợp.</p>
      <div class="prompt-grid" aria-label="Gợi ý câu hỏi">
        <button type="button" data-prompt="Công ty tôi phát hiện một lô hàng nhập khẩu nghi xâm phạm quyền tác giả và muốn yêu cầu hải quan kiểm soát, đồng thời cần thuê giám định viên để xác minh thì phải thực hiện nghĩa vụ bảo đảm tài chính ra sao và hợp đồng giám định cần soạn thảo những nội dung chính nào, đặc biệt nếu đối tượng bị xâm phạm là người tiêu dùng dễ bị tổn thương thì công ty có trách nhiệm gì trong việc giải quyết tranh chấp?">
          Yêu cầu hải quan kiểm soát hàng nghi xâm phạm quyền tác giả và thuê giám định viên cần lưu ý gì?
        </button>
        <button type="button" data-prompt="Công ty tôi muốn đặt tên thương mại cho chuỗi cửa hàng mới nhưng tên này bị một đối thủ trong cùng khu vực phản đối vì cho là xâm phạm quyền, vậy làm sao để xác định tên của tôi có khả năng phân biệt không và căn cứ nào để kết luận tôi có đang xâm phạm tên thương mại của họ hay không, đồng thời tôi cần lưu ý những dấu hiệu nào không được bảo hộ làm nhãn hiệu để tránh rủi ro pháp lý?">
          Đặt tên thương mại cho chuỗi cửa hàng mới có thể xâm phạm quyền của đối thủ không?
        </button>
        <button type="button" data-prompt="Công ty tôi đang áp dụng trả lương theo sản phẩm cho nhân viên, nhưng hiện tại đang trả thấp hơn mức lương tối thiểu và chưa đóng BHXH cho một số lao động không thuộc đối tượng bắt buộc, vậy công ty cần xử lý việc trả lương, khắc phục sai phạm về chi phí bảo hiểm và mức lương này như thế nào, đồng thời tiền lương đóng BHXH của những nhân viên này sau này sẽ được điều chỉnh ra sao?">
          Trả lương theo sản phẩm thấp hơn lương tối thiểu và vấn đề BHXH xử lý thế nào?
        </button>
        <button type="button" data-prompt="Trong trường hợp công ty trách nhiệm hữu hạn hai thành viên trở lên muốn thay đổi thành viên do chuyển nhượng phần vốn góp và đồng thời muốn tìm hiểu về điều kiện để một doanh nghiệp nhỏ và vừa được Quỹ phát triển doanh nghiệp nhỏ và vừa tiếp tục xem xét cho vay sau khi đã trả hết nợ, công ty cần chuẩn bị hồ sơ đăng ký thay đổi gồm những gì và điều kiện vay vốn tiếp theo là gì?">
          Công ty TNHH thay đổi thành viên do chuyển nhượng vốn và vay tiếp Quỹ DNNVV cần gì?
        </button>
        <button type="button" data-prompt="Công ty tôi đang nộp đơn đăng ký nhãn hiệu, kiểu dáng công nghiệp và thiết kế bố trí cho sản phẩm mới, vậy nếu Cục Sở hữu trí tuệ nghi ngờ tính xác thực thông tin trong đơn kiểu dáng hoặc thông báo đơn thiết kế bố trí có thiếu sót thì công ty phải xử lý thế nào để không bị từ chối cấp văn bằng, và nhãn hiệu cần đáp ứng điều kiện gì về khả năng phân biệt để được bảo hộ?">
          Đơn nhãn hiệu, kiểu dáng công nghiệp và thiết kế bố trí cần xử lý thiếu sót ra sao?
        </button>
      </div>
    </div>
  `;
}

function scrollToBottom(force = false) {
  const threshold = 50;
  const isNearBottom = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight <= threshold;
  if (force || isNearBottom) {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

function createMessage(role, text = "") {
  clearEmptyState();
  const el = document.createElement("article");
  el.className = `message ${role}`;
  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = text;
  el.appendChild(content);
  messagesEl.appendChild(el);
  scrollToBottom(true);
  return el;
}

function messageContent(el) {
  return el.querySelector(".message-content");
}

function setMessageText(el, text) {
  const content = messageContent(el);
  content.textContent = text;
  scrollToBottom();
}

function createProcessPanel(el) {
  const panel = document.createElement("details");
  panel.className = "process-panel";
  panel.open = true;
  panel._sections = {};
  panel._seenSteps = new Set();
  panel._stepElements = {};
  panel._stageTimes = {};
  panel.dataset.startTime = Date.now();

  const summary = document.createElement("summary");
  summary.className = "process-main-title";
  summary.textContent = "Đang suy nghĩ (0s)";

  const body = document.createElement("div");
  body.className = "process-body";

  panel.append(summary, body);
  el.insertBefore(panel, messageContent(el));

  panel._timer = setInterval(() => {
    const elapsed = Math.floor((Date.now() - parseInt(panel.dataset.startTime, 10)) / 1000);
    summary.textContent = `Đang suy nghĩ (${elapsed}s)`;

    const activeSections = panel.querySelectorAll(".process-section[data-active='true']");
    activeSections.forEach(section => updateProcessSectionTime(section));
  }, 1000);

  return panel;
}

function processBody(panel) {
  return panel.querySelector(".process-body");
}

function markProcessStage(panel, stage) {
  if (!panel || !stage) return;
  const now = Date.now();
  panel._stageTimes = panel._stageTimes || {};
  if (!panel._stageTimes[stage]) {
    panel._stageTimes[stage] = now;
  }

  if (stage === "running" && !panel._stageTimes.planner) {
    panel._stageTimes.planner = now;
  }
  if (stage === "compression" && !panel._stageTimes.reasoning) {
    panel._stageTimes.reasoning = now;
  }
}

function processSectionStartTime(panel, key) {
  const stageTimes = panel?._stageTimes || {};
  if (key === "planner" || key === "planner_json") {
    return stageTimes.planner || stageTimes.running || Date.now();
  }
  if (key === "reasoning") {
    return stageTimes.reasoning || stageTimes.compression || Date.now();
  }
  return Date.now();
}

function updateProcessSectionTime(section, endTime = Date.now()) {
  if (!section) return;
  const startTime = parseInt(section.dataset.startTime || "", 10);
  const timeSpan = section.querySelector(".process-time");
  if (!timeSpan || !startTime) return;
  const elapsed = Math.max(0, Math.floor((endTime - startTime) / 1000));
  timeSpan.textContent = ` (${elapsed}s)`;
}

function deactivateProcessSection(section, endTime = Date.now()) {
  if (!section) return;
  updateProcessSectionTime(section, endTime);
  section.dataset.active = "false";
  section.dataset.endTime = endTime;
}

function freezeProcessSection(panel, key, endTime = Date.now()) {
  const stream = panel?._sections?.[key];
  if (!stream) return;
  deactivateProcessSection(stream.closest(".process-section"), endTime);
}

function freezeActiveProcessSections(panel, endTime = Date.now()) {
  if (!panel) return;
  panel.querySelectorAll(".process-section[data-active='true']").forEach(section => {
    deactivateProcessSection(section, endTime);
  });
}

function deactivateProcessSections(panel, activeKey = "") {
  if (!panel?._sections) return;
  const now = Date.now();
  for (const key in panel._sections) {
    if (key === activeKey) continue;
    const section = panel._sections[key].closest(".process-section");
    if (section && section.dataset.active === "true") {
      deactivateProcessSection(section, now);
    }
  }
}

function addProcessStep(panel, text, stage = "") {
  if (!panel || !text) return;
  markProcessStage(panel, stage);
  if (panel._seenSteps.has(text)) return;
  panel._seenSteps.add(text);
  const item = document.createElement("div");
  item.className = "process-step";
  item.textContent = text;
  processBody(panel).appendChild(item);
  if (stage) panel._stepElements[stage] = item;
  scrollToBottom();
}

function insertProcessSection(panel, key, section) {
  const body = processBody(panel);
  if (key === "planner") {
    const plannerStep = panel._stepElements?.planner;
    if (plannerStep && plannerStep.parentNode === body) {
      body.insertBefore(section, plannerStep);
      return;
    }
    const runningStep = panel._stepElements?.running;
    if (runningStep && runningStep.parentNode === body) {
      runningStep.after(section);
      return;
    }
  }
  body.appendChild(section);
}

function ensureProcessSection(panel, key, label, options = {}) {
  const activate = options.activate !== false;
  if (panel._sections[key]) {
    const existingSection = panel._sections[key].closest(".process-section");
    if (existingSection && activate) {
      existingSection.dataset.active = "true";
      deactivateProcessSections(panel, key);
      updateProcessSectionTime(existingSection);
    } else if (existingSection) {
      updateProcessSectionTime(existingSection, parseInt(existingSection.dataset.endTime || "", 10) || Date.now());
    }
    return panel._sections[key];
  }
  const section = document.createElement("details");
  section.className = "process-section";
  section.dataset.active = activate ? "true" : "false";
  section.dataset.startTime = processSectionStartTime(panel, key);

  const title = document.createElement("summary");
  title.className = "process-title";
  
  const titleText = document.createElement("span");
  titleText.textContent = label;
  
  const timeText = document.createElement("span");
  timeText.className = "process-time";
  timeText.textContent = " (0s)";

  title.append(titleText, timeText);

  const stream = document.createElement("pre");
  stream.className = "process-token-stream";

  section.append(title, stream);
  insertProcessSection(panel, key, section);
  panel._sections[key] = stream;
  updateProcessSectionTime(section);
  if (activate) {
    deactivateProcessSections(panel, key);
  } else {
    deactivateProcessSection(section);
  }

  return stream;
}

function appendProcessToken(panel, key, label, text) {
  if (!panel || !text) return;
  const stream = ensureProcessSection(panel, key, label);
  stream.textContent += text;
  scrollToBottom();
}

function setProcessJson(panel, key, label, value) {
  if (!panel) return;
  const stream = ensureProcessSection(panel, key, label, { activate: false });
  stream.classList.add("process-json");
  stream.textContent = JSON.stringify(value || {}, null, 2);
  deactivateProcessSection(stream.closest(".process-section"));
  scrollToBottom();
}

function sourceLabel(source, docId) {
  if (!source) return "Căn cứ";
  return source.label || source.title || source.article_no || "Căn cứ";
}

function createCitationChip(docId, sources) {
  const source = sources?.[docId] || {};
  const chip = source.url ? document.createElement("a") : document.createElement("span");
  chip.className = "citation-chip";
  chip.dataset.docId = docId;
  if (source.url) {
    chip.href = source.url;
    chip.target = "_blank";
    chip.rel = "noopener noreferrer";
    chip.title = [docId, source.title, source.domain].filter(Boolean).join(" | ");
  } else {
    chip.title = docId;
  }

  if (source.favicon) {
    const icon = document.createElement("img");
    icon.className = "citation-favicon";
    icon.src = source.favicon;
    icon.alt = "";
    icon.addEventListener("error", () => icon.remove());
    chip.appendChild(icon);
  }

  const text = document.createElement("span");
  text.textContent = sourceLabel(source, docId);
  chip.appendChild(text);
  return chip;
}

function splitSegmentsIntoLines(segments) {
  const lines = [[]];
  (segments || []).forEach((segment) => {
    if (segment.type === "text") {
      String(segment.text || "")
        .split("\n")
        .forEach((part, index) => {
          if (index > 0) lines.push([]);
          if (part) lines[lines.length - 1].push({ type: "text", text: part });
        });
      return;
    }
    if (segment.type === "citation") {
      lines[lines.length - 1].push({
        type: "citation",
        doc_ids: segment.doc_ids || [],
      });
    }
  });
  return lines;
}

function lineText(tokens) {
  return (tokens || [])
    .filter((token) => token.type === "text")
    .map((token) => token.text || "")
    .join("");
}

function stripLinePrefix(tokens, pattern) {
  const match = lineText(tokens).match(pattern);
  if (!match) return tokens;
  let remaining = match[0].length;
  const output = [];
  tokens.forEach((token) => {
    if (token.type !== "text") {
      if (remaining <= 0) output.push(token);
      return;
    }
    let text = token.text || "";
    if (remaining > 0) {
      const count = Math.min(remaining, text.length);
      text = text.slice(count);
      remaining -= count;
    }
    if (text) output.push({ type: "text", text });
  });
  return output;
}

function appendMarkdownText(parent, text) {
  const pattern = /(\*\*[^*]+?\*\*|__[^_]+?__|\[[^\]]+?\]\(https?:\/\/[^)\s]+?\))/g;
  let lastIndex = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parent.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
    }
    const token = match[0];
    if (token.startsWith("**") || token.startsWith("__")) {
      const strong = document.createElement("strong");
      appendMarkdownText(strong, token.slice(2, -2));
      parent.appendChild(strong);
    } else {
      const linkMatch = token.match(/^\[([^\]]+?)\]\((https?:\/\/[^)\s]+?)\)$/);
      if (linkMatch) {
        const link = document.createElement("a");
        link.href = linkMatch[2];
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = linkMatch[1].replace(/\\([\[\]])/g, "$1");
        parent.appendChild(link);
      } else {
        parent.appendChild(document.createTextNode(token));
      }
    }
    lastIndex = pattern.lastIndex;
  }
  if (lastIndex < text.length) {
    parent.appendChild(document.createTextNode(text.slice(lastIndex)));
  }
}

function appendInlineTokens(parent, tokens, sources) {
  (tokens || []).forEach((token) => {
    if (token.type === "text") {
      appendMarkdownText(parent, token.text || "");
      return;
    }
    if (token.type !== "citation") return;
    const group = document.createElement("span");
    group.className = "citation-chip-group";
    (token.doc_ids || []).forEach((docId) => {
      group.appendChild(createCitationChip(docId, sources));
    });
    parent.appendChild(group);
  });
}

function isBlankLine(tokens) {
  return lineText(tokens).trim() === "" && !tokens.some((token) => token.type === "citation");
}

function headingLevel(tokens) {
  const match = lineText(tokens).match(/^\s*(#{1,4})\s+/);
  return match ? match[1].length : 0;
}

function isRuleLine(tokens) {
  return /^-{3,}$/.test(lineText(tokens).trim());
}

function listMatch(tokens) {
  return lineText(tokens).match(/^\s*((?:[-*+])|\d+[.)])\s+/);
}

function splitTableCells(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed.startsWith("|") || !trimmed.endsWith("|")) return [];
  return trimmed
    .slice(1, -1)
    .split("|")
    .map((cell) => cell.trim());
}

function isTableSeparatorLine(tokens) {
  const cells = splitTableCells(lineText(tokens));
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function isTableStart(lines, index) {
  if (index + 1 >= lines.length) return false;
  const headerCells = splitTableCells(lineText(lines[index]));
  return headerCells.length > 0 && isTableSeparatorLine(lines[index + 1]);
}

function appendTableCell(row, tagName, cellText, sources) {
  const cell = document.createElement(tagName);
  appendInlineTokens(cell, [{ type: "text", text: cellText }], sources);
  row.appendChild(cell);
}

function appendMarkdownTable(content, lines, start, sources) {
  const headerCells = splitTableCells(lineText(lines[start]));
  const table = document.createElement("table");
  table.className = "markdown-table";

  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  headerCells.forEach((cellText) => appendTableCell(headerRow, "th", cellText, sources));
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  let index = start + 2;
  while (index < lines.length) {
    const cells = splitTableCells(lineText(lines[index]));
    if (!cells.length) break;
    const row = document.createElement("tr");
    for (let cellIndex = 0; cellIndex < headerCells.length; cellIndex += 1) {
      appendTableCell(row, "td", cells[cellIndex] || "", sources);
    }
    tbody.appendChild(row);
    index += 1;
  }
  table.appendChild(tbody);

  const wrap = document.createElement("div");
  wrap.className = "markdown-table-wrap";
  wrap.appendChild(table);
  content.appendChild(wrap);
  return index;
}

function appendParagraph(content, lines, start, end, sources) {
  const paragraph = document.createElement("p");
  for (let index = start; index < end; index += 1) {
    if (index > start) paragraph.appendChild(document.createTextNode(" "));
    appendInlineTokens(paragraph, lines[index], sources);
  }
  content.appendChild(paragraph);
}

function renderMarkdownSegments(content, segments, sources) {
  content.textContent = "";
  const lines = splitSegmentsIntoLines(segments);
  let index = 0;

  while (index < lines.length) {
    const tokens = lines[index];
    if (isBlankLine(tokens)) {
      index += 1;
      continue;
    }

    const level = headingLevel(tokens);
    if (level) {
      const heading = document.createElement(level <= 2 ? "h3" : "h4");
      heading.className = "markdown-heading";
      appendInlineTokens(heading, stripLinePrefix(tokens, /^\s*#{1,4}\s+/), sources);
      content.appendChild(heading);
      index += 1;
      continue;
    }

    if (isRuleLine(tokens)) {
      content.appendChild(document.createElement("hr"));
      index += 1;
      continue;
    }

    if (isTableStart(lines, index)) {
      index = appendMarkdownTable(content, lines, index, sources);
      continue;
    }

    const match = listMatch(tokens);
    if (match) {
      const ordered = /^\s*\d+[.)]\s+/.test(lineText(tokens));
      const list = document.createElement(ordered ? "ol" : "ul");
      if (ordered) {
        const numMatch = lineText(tokens).match(/^\s*(\d+)[.)]\s+/);
        if (numMatch) {
          list.start = parseInt(numMatch[1], 10);
        }
      }
      while (index < lines.length) {
        const currentMatch = listMatch(lines[index]);
        if (!currentMatch || /^\s*\d+[.)]\s+/.test(lineText(lines[index])) !== ordered) {
          break;
        }
        const item = document.createElement("li");
        appendInlineTokens(item, stripLinePrefix(lines[index], /^\s*(?:[-*+]|\d+[.)])\s+/), sources);
        list.appendChild(item);
        index += 1;
      }
      content.appendChild(list);
      continue;
    }

    const paragraphStart = index;
    index += 1;
    while (
      index < lines.length &&
      !isBlankLine(lines[index]) &&
      !headingLevel(lines[index]) &&
      !isRuleLine(lines[index]) &&
      !isTableStart(lines, index) &&
      !listMatch(lines[index])
    ) {
      index += 1;
    }
    appendParagraph(content, lines, paragraphStart, index, sources);
  }
}

function appendAnswerText(el, streamState, text) {
  if (!text) return;
  const content = messageContent(el);
  if (!streamState.answerStarted) {
    content.textContent = "";
    streamState.answerStarted = true;
  }
  const lastSegment = streamState.segments[streamState.segments.length - 1];
  if (lastSegment && lastSegment.type === "text") {
    lastSegment.text += text;
  } else {
    streamState.segments.push({ type: "text", text });
  }
  renderMarkdownSegments(content, streamState.segments, streamState.sources);
  scrollToBottom();
}

function appendCitation(el, streamState, docIds) {
  if (!docIds || docIds.length === 0) return;
  const content = messageContent(el);
  if (!streamState.answerStarted) {
    content.textContent = "";
    streamState.answerStarted = true;
  }
  streamState.segments.push({ type: "citation", doc_ids: docIds });
  renderMarkdownSegments(content, streamState.segments, streamState.sources);
  scrollToBottom();
}

function removeStructuredSections(el) {
  el.querySelectorAll(".citations, .legal-basis, .answer-check").forEach((node) => node.remove());
}

function resetAssistantAnswer(el, streamState) {
  const content = messageContent(el);
  content.textContent = "";
  removeStructuredSections(el);
  streamState.answerStarted = false;
  streamState.answerDone = false;
  streamState.segments = [];
  streamState.sources = {};
}

function createSourceIconLink(source) {
  if (!source || !source.url) return null;
  const link = document.createElement("a");
  link.className = "legal-basis-icon-link";
  link.href = source.url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.title = [source.doc_id, source.title, source.domain].filter(Boolean).join(" | ");

  if (source.favicon) {
    const icon = document.createElement("img");
    icon.className = "legal-basis-favicon";
    icon.src = source.favicon;
    icon.alt = "";
    icon.addEventListener("error", () => {
      icon.remove();
      link.textContent = "Link";
    });
    link.appendChild(icon);
  } else {
    link.textContent = "Link";
  }
  return link;
}

function renderLegalBasis(el, legalBasis, disclaimer, sources = {}) {
  if ((!legalBasis || legalBasis.length === 0) && !disclaimer) return;

  const wrap = document.createElement("details");
  wrap.className = "legal-basis";
  wrap.open = true;

  const summary = document.createElement("summary");
  summary.textContent = "Cơ Sở Pháp Lý và Lưu Ý";
  wrap.appendChild(summary);

  const body = document.createElement("div");
  body.className = "legal-basis-body";

  if (legalBasis && legalBasis.length) {
    const title = document.createElement("h3");
    title.textContent = "Cơ Sở Pháp Lý";
    const list = document.createElement("ul");
    const sourceValues = Object.values(sources || {});
    legalBasis.forEach((item, index) => {
      const li = document.createElement("li");
      appendMarkdownText(li, item);
      const iconLink = createSourceIconLink(sourceValues[index]);
      if (iconLink) li.appendChild(iconLink);
      list.appendChild(li);
    });
    body.append(title, list);
  }

  if (disclaimer) {
    const noteTitle = document.createElement("h3");
    noteTitle.textContent = "Lưu Ý";
    const note = document.createElement("p");
    appendMarkdownText(note, disclaimer);
    body.append(noteTitle, note);
  }

  wrap.appendChild(body);
  el.appendChild(wrap);
}

function checkerStatusLabel(status) {
  if (status === "pass") return "Đã thông qua";
  if (status === "corrected") return "Đã sửa";
  if (status === "failed") return "Cần lưu ý";
  return "Chưa kiểm tra";
}

function checkerConfidenceLabel(confidence) {
  if (confidence === "high") return "cao";
  if (confidence === "medium") return "trung bình";
  if (confidence === "low") return "thấp";
  return "không xác định";
}

function renderAnswerCheck(el, answerCheck) {
  if (!answerCheck || !answerCheck.status) return;

  const wrap = document.createElement("section");
  wrap.className = `answer-check answer-check-${answerCheck.status}`;

  const title = document.createElement("div");
  title.className = "answer-check-title";
  title.textContent = `Kiểm tra độ tin cậy: ${checkerStatusLabel(answerCheck.status)} · độ tin cậy ${checkerConfidenceLabel(answerCheck.confidence)}`;
  wrap.appendChild(title);

  const issues = Array.isArray(answerCheck.issues) ? answerCheck.issues : [];
  if (issues.length) {
    const list = document.createElement("ul");
    list.className = "answer-check-issues";
    issues.slice(0, 6).forEach((issue) => {
      const item = document.createElement("li");
      const type = issue.type ? `[${issue.type}] ` : "";
      const quote = issue.quote ? `"${issue.quote}" — ` : "";
      item.textContent = `${type}${quote}${issue.reason || ""}`.trim();
      list.appendChild(item);
    });
    wrap.appendChild(list);
  }

  el.appendChild(wrap);
}

function renderCitations(el, citations) {
  if (!citations || citations.length === 0) return;

  const wrap = document.createElement("section");
  wrap.className = "citations";

  const button = document.createElement("button");
  button.className = "citation-toggle";
  button.type = "button";
  button.textContent = `Xem ${citations.length} căn cứ`;

  const list = document.createElement("div");
  list.className = "citation-list";

  citations.forEach((citation) => {
    const item = document.createElement("div");
    item.className = "citation-item";

    const title = document.createElement("div");
    title.className = "citation-title";
    title.textContent = citation.article_no || "Căn cứ pháp lý";

    const meta = document.createElement("div");
    meta.className = "citation-meta";
    meta.textContent = [citation.document_number, citation.doc_name, citation.article_title]
      .filter(Boolean)
      .join(" | ");

    const snippet = document.createElement("div");
    snippet.className = "citation-snippet";
    snippet.textContent = citation.snippet || "";

    item.append(title, meta, snippet);
    list.appendChild(item);
  });

  button.addEventListener("click", () => {
    list.classList.toggle("open");
    button.textContent = list.classList.contains("open")
      ? "Ẩn căn cứ"
      : `Xem ${citations.length} căn cứ`;
  });

  wrap.append(button, list);
  el.appendChild(wrap);
}

function renderStructuredAnswer(el, data) {
  removeStructuredSections(el);
  const content = messageContent(el);
  const segments = data.segments || [];
  const sources = data.sources || {};

  if (segments.length) {
    renderMarkdownSegments(content, segments, sources);
  } else {
    renderMarkdownSegments(
      content,
      [{ type: "text", text: data.answer_text || data.answer || "" }],
      sources
    );
  }

  renderAnswerCheck(el, data.answer_check || null);
  renderLegalBasis(el, data.legal_basis || [], data.disclaimer || "", sources);
  if (!segments.length) {
    renderCitations(el, data.citations || []);
  }
  scrollToBottom();
}

function addRetryButton(el, question) {
  if (!question) return;
  const retry = document.createElement("button");
  retry.className = "retry-button";
  retry.type = "button";
  retry.textContent = "Thử lại";
  retry.addEventListener("click", () => sendMessage(question));
  el.appendChild(retry);
}

async function ensureSession() {
  if (sessionId) {
    return sessionId;
  }
  const response = await fetch("/api/chat/sessions", { method: "POST" });
  if (!response.ok) throw new Error("Không tạo được phiên chat.");
  const data = await response.json();
  sessionId = data.session_id;
  localStorage.setItem(SESSION_KEY, sessionId);
  return sessionId;
}

function formatHistoryTime(timestamp) {
  const date = new Date(Number(timestamp || 0) * 1000);
  if (Number.isNaN(date.getTime())) return "";

  const now = new Date();
  const sameDay = date.toDateString() === now.toDateString();
  if (sameDay) {
    return date.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
  }

  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (date.toDateString() === yesterday.toDateString()) return "Hôm qua";

  return date.toLocaleDateString("vi-VN", { day: "2-digit", month: "2-digit" });
}

function renderSessionHistory(sessions) {
  if (!sessionHistoryEl) return;
  sessionHistoryEl.replaceChildren();

  if (!sessions.length) {
    const empty = document.createElement("p");
    empty.className = "history-empty";
    empty.textContent = "Chưa có cuộc trò chuyện.";
    sessionHistoryEl.appendChild(empty);
    return;
  }

  sessions.forEach((session) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-item";
    button.dataset.sessionId = session.session_id;
    button.title = session.title || "Cuộc trò chuyện";
    if (session.session_id === sessionId) button.classList.add("active");

    const title = document.createElement("span");
    title.className = "history-item-title";
    title.textContent = session.title || "Cuộc trò chuyện mới";

    const meta = document.createElement("span");
    meta.className = "history-item-meta";
    const turnLabel = `${session.turn_count || 0} lượt`;
    const timeLabel = formatHistoryTime(session.updated_at);
    meta.textContent = [turnLabel, timeLabel].filter(Boolean).join(" · ");

    if (session.last_status === "running") {
      const status = document.createElement("span");
      status.className = "history-item-status";
      status.textContent = "Đang xử lý";
      meta.append(" · ", status);
    }

    button.append(title, meta);
    sessionHistoryEl.appendChild(button);
  });
}

async function refreshSessionHistory() {
  if (!sessionHistoryEl) return;
  try {
    const response = await fetch("/api/chat/sessions?limit=50");
    if (!response.ok) throw new Error("Không tải được danh sách phiên chat.");
    const data = await response.json();
    renderSessionHistory(data.sessions || []);
  } catch {
    sessionHistoryEl.innerHTML = '<p class="history-empty">Không tải được lịch sử.</p>';
  }
}

async function loadHistory(targetSessionId = sessionId) {
  if (!targetSessionId) return false;
  const loadToken = ++historyLoadToken;
  try {
    const response = await fetch(`/api/chat/messages?session_id=${encodeURIComponent(targetSessionId)}`);
    if (response.status === 404) return false;
    if (!response.ok) throw new Error("Không tải được lịch sử.");
    const data = await response.json();
    if (loadToken !== historyLoadToken || targetSessionId !== sessionId) return false;

    messagesEl.innerHTML = emptyStateMarkup();
    lastAnswer = "";
    lastQuestion = "";
    const messages = data.messages || [];
    if (!messages.length) return true;

    clearEmptyState();
    let runningMessage = null;
    messages.forEach((message) => {
      const el = createMessage(message.role, "");
      if (message.role === "user") {
        lastQuestion = message.content || lastQuestion;
      }

      if (message.role === "assistant" && message.status === "completed") {
        lastAnswer = message.content || lastAnswer;

        const elapsed = message.elapsed_sec ? Math.round(message.elapsed_sec) : 0;
        if (elapsed > 0) {
          const processPanel = createProcessPanel(el);
          processPanel.open = false;
          if (processPanel._timer) clearInterval(processPanel._timer);
          const summary = processPanel.querySelector("summary.process-main-title");
          if (summary) summary.textContent = `Đã suy nghĩ trong ${elapsed}s`;
        }

        renderStructuredAnswer(el, message);
      } else if (message.role === "assistant" && message.status === "running") {
        const processPanel = createProcessPanel(el);
        addProcessStep(processPanel, "Đang khôi phục tiến trình...", "running");
        runningMessage = { message, el, processPanel };
      } else {
        setMessageText(el, message.content || message.error || "Không thể tạo câu trả lời.");
      }
      if (message.status === "failed") {
        el.classList.add("failed");
      }
    });

    scrollToBottom(true);
    if (runningMessage) {
      sendButton.disabled = true;
      inputEl.disabled = true;
      const { message, el, processPanel } = runningMessage;
      const streamUrl = `/api/chat/stream?session_id=${encodeURIComponent(targetSessionId)}&message_id=${encodeURIComponent(message.message_id)}`;
      handleStream(streamUrl, el, processPanel);
    } else {
      setStatus("Đang sẵn sàng");
    }
    return true;
  } catch {
    setStatus("Không tải được lịch sử");
    return false;
  }
}

async function openChatSession(nextSessionId) {
  if (!nextSessionId || nextSessionId === sessionId) {
    document.body.classList.remove("sidebar-open");
    return;
  }
  if (sendButton.disabled) {
    setStatus("Vui lòng đợi lượt hiện tại hoàn tất");
    return;
  }

  sessionId = nextSessionId;
  localStorage.setItem(SESSION_KEY, sessionId);
  messagesEl.innerHTML = emptyStateMarkup();
  setStatus("Đang tải lịch sử");
  await Promise.all([loadHistory(sessionId), refreshSessionHistory()]);
  document.body.classList.remove("sidebar-open");
}

function handleStream(streamUrl, assistantEl, processPanel) {
  const source = new EventSource(streamUrl);
  const streamState = {
    answerStarted: false,
    answerDone: false,
    segments: [],
    sources: {},
    answerCheck: null,
  };
  setStatus("Đang xử lý");

  source.addEventListener("status", () => {});

  source.addEventListener("process_step", (event) => {
    const data = JSON.parse(event.data);
    const stage = data.stage || "";
    addProcessStep(processPanel, data.message || "Đang xử lý...", data.stage || "");
    if (stage && stage !== "queued" && stage !== "running") {
      freezeActiveProcessSections(processPanel);
    }
  });

  source.addEventListener("planner_token", (event) => {
    const data = JSON.parse(event.data);
    appendProcessToken(processPanel, "planner", "Planner", data.delta || "");
  });

  source.addEventListener("reasoning_token", (event) => {
    const data = JSON.parse(event.data);
    appendProcessToken(processPanel, "reasoning", "Reasoning", data.delta || "");
  });

  source.addEventListener("answer_delta", (event) => {
    const data = JSON.parse(event.data);
    appendAnswerText(assistantEl, streamState, data.delta || "");

    if (processPanel && processPanel.dataset.startTime && !processPanel.dataset.timeFrozen) {
      processPanel.dataset.timeFrozen = "true";
      if (processPanel._timer) clearInterval(processPanel._timer);
      processPanel.querySelectorAll(".process-section").forEach(s => s.dataset.active = "false");
      const summary = processPanel.querySelector("summary.process-main-title");
      if (summary) {
        const elapsed = Math.floor((Date.now() - parseInt(processPanel.dataset.startTime, 10)) / 1000);
        summary.textContent = `Đã suy nghĩ trong ${elapsed}s`;
      }
    }
  });

  source.addEventListener("source", (event) => {
    const data = JSON.parse(event.data);
    const docId = data.doc_id;
    if (docId) streamState.sources[docId] = data;
    if (streamState.answerStarted && !streamState.answerDone) {
      renderMarkdownSegments(messageContent(assistantEl), streamState.segments, streamState.sources);
    }
  });

  source.addEventListener("citation", (event) => {
    const data = JSON.parse(event.data);
    appendCitation(assistantEl, streamState, data.doc_ids || []);
  });

  source.addEventListener("answer_reset", () => {
    resetAssistantAnswer(assistantEl, streamState);
  });

  source.addEventListener("checker_result", (event) => {
    const data = JSON.parse(event.data);
    streamState.answerCheck = data;
  });

  source.addEventListener("answer_done", (event) => {
    const data = JSON.parse(event.data);
    streamState.answerDone = true;
    streamState.sources = data.sources || streamState.sources;
    if (streamState.answerCheck && (!data.answer_check || !data.answer_check.status)) {
      data.answer_check = streamState.answerCheck;
    }
    lastAnswer = data.answer || data.answer_text || "";
    renderStructuredAnswer(assistantEl, data);
    if (processPanel) {
      processPanel.open = false;
      if (processPanel._timer) clearInterval(processPanel._timer);
      processPanel.querySelectorAll(".process-section").forEach(s => s.dataset.active = "false");
      if (!processPanel.dataset.timeFrozen) {
        processPanel.dataset.timeFrozen = "true";
        const summary = processPanel.querySelector("summary.process-main-title");
        if (summary && processPanel.dataset.startTime) {
          const elapsed = Math.floor((Date.now() - parseInt(processPanel.dataset.startTime, 10)) / 1000);
          summary.textContent = `Đã suy nghĩ trong ${elapsed}s`;
        }
      }
    }
    setStatus("Đã hoàn tất");
  });

  source.addEventListener("planner", (event) => {
    const data = JSON.parse(event.data);
    if (data.plan) {
      freezeProcessSection(processPanel, "planner");
      setProcessJson(processPanel, "planner_json", "Planner JSON", data.plan);
    }
  });

  source.addEventListener("tool_call", (event) => {
    const data = JSON.parse(event.data);
    const toolCall = data.tool_call || {};
    const callId = toolCall.id || `${Date.now()}`;
    const label = toolCall.name ? `Tool Call Args: ${toolCall.name}` : "Tool Call Args";
    setProcessJson(processPanel, `tool_call_${callId}`, label, toolCall);
  });

  source.addEventListener("retrieval", () => {});

  source.addEventListener("compression", () => {});

  source.addEventListener("final", (event) => {
    if (streamState.answerDone) return;
    const data = JSON.parse(event.data);
    lastAnswer = data.answer || data.answer_text || "";
    renderStructuredAnswer(assistantEl, data);
    if (processPanel) {
      processPanel.open = false;
      if (processPanel._timer) clearInterval(processPanel._timer);
      processPanel.querySelectorAll(".process-section").forEach(s => s.dataset.active = "false");
      if (!processPanel.dataset.timeFrozen) {
        processPanel.dataset.timeFrozen = "true";
        const summary = processPanel.querySelector("summary.process-main-title");
        if (summary && processPanel.dataset.startTime) {
          const elapsed = Math.floor((Date.now() - parseInt(processPanel.dataset.startTime, 10)) / 1000);
          summary.textContent = `Đã suy nghĩ trong ${elapsed}s`;
        }
      }
    }
    setStatus("Đã hoàn tất");
  });

  source.addEventListener("error", (event) => {
    if (!event.data) {
      setStatus("Mất kết nối stream");
      return;
    }
    const data = JSON.parse(event.data);
    assistantEl.classList.add("failed");
    setMessageText(assistantEl, data.message || "Có lỗi khi xử lý câu hỏi.");
    addRetryButton(assistantEl, lastQuestion);
    setStatus("Có lỗi");
  });

  source.addEventListener("done", () => {
    source.close();
    sendButton.disabled = false;
    inputEl.disabled = false;
    refreshSessionHistory();
    if (window.matchMedia("(min-width: 841px)").matches) inputEl.focus();
  });
}

async function sendMessage(question) {
  await ensureSession();
  lastQuestion = question;
  createMessage("user", question);
  const assistantEl = createMessage("assistant", "");
  const processPanel = createProcessPanel(assistantEl);

  sendButton.disabled = true;
  inputEl.disabled = true;

  const response = await fetch("/api/chat/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message: question }),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    assistantEl.classList.add("failed");
    setMessageText(assistantEl, data.detail || "Không gửi được câu hỏi.");
    addRetryButton(assistantEl, question);
    sendButton.disabled = false;
    inputEl.disabled = false;
    setStatus("Có lỗi");
    return;
  }

  const data = await response.json();
  sessionId = data.session_id;
  localStorage.setItem(SESSION_KEY, sessionId);
  refreshSessionHistory();
  handleStream(data.stream_url, assistantEl, processPanel);
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = inputEl.value.trim();
  if (!question) return;
  inputEl.value = "";
  charCounter.textContent = "0/4000";
  autoGrowInput();
  try {
    await sendMessage(question);
  } catch (error) {
    const errorEl = createMessage("assistant", error.message || "Không thể gửi câu hỏi.");
    errorEl.classList.add("failed");
    addRetryButton(errorEl, question);
    sendButton.disabled = false;
    inputEl.disabled = false;
  }
});

inputEl.addEventListener("input", () => {
  charCounter.textContent = `${inputEl.value.length}/4000`;
  autoGrowInput();
});

inputEl.addEventListener("keydown", (event) => {
  if (event.isComposing || event.key !== "Enter") return;
  if (event.shiftKey) return;
  if (!event.ctrlKey && !event.metaKey) {
    event.preventDefault();
    formEl.requestSubmit();
    return;
  }
  if (event.ctrlKey || event.metaKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

newSessionButton.addEventListener("click", async () => {
  if (sendButton.disabled) {
    setStatus("Vui lòng đợi lượt hiện tại hoàn tất");
    return;
  }
  historyLoadToken += 1;
  localStorage.removeItem(SESSION_KEY);
  sessionId = "";
  lastAnswer = "";
  lastQuestion = "";
  inputEl.value = "";
  charCounter.textContent = "0/4000";
  autoGrowInput();
  messagesEl.innerHTML = emptyStateMarkup();
  await ensureSession();
  await refreshSessionHistory();
  setStatus("Đang sẵn sàng");
  document.body.classList.remove("sidebar-open");
});

sessionHistoryEl?.addEventListener("click", (event) => {
  if (!(event.target instanceof Element)) return;
  const item = event.target.closest("[data-session-id]");
  if (!item) return;
  openChatSession(item.dataset.sessionId || "");
});

refreshHistoryButton?.addEventListener("click", () => refreshSessionHistory());

document.getElementById("sidebarToggle")?.addEventListener("click", () => {
  document.body.classList.add("sidebar-open");
});

document.getElementById("sidebarBackdrop")?.addEventListener("click", () => {
  document.body.classList.remove("sidebar-open");
});

messagesEl.addEventListener("click", (event) => {
  if (!(event.target instanceof Element)) return;
  const promptButton = event.target.closest("[data-prompt]");
  if (!promptButton) return;
  const prompt = promptButton.dataset.prompt || promptButton.textContent || "";
  inputEl.value = prompt.trim();
  charCounter.textContent = `${inputEl.value.length}/4000`;
  autoGrowInput();
  inputEl.focus();
});

copyLastButton.addEventListener("click", async () => {
  if (!lastAnswer) return;
  await navigator.clipboard.writeText(lastAnswer);
  setStatus("Đã copy câu trả lời");
});

autoGrowInput();

async function initializeChat() {
  if (sessionId) {
    const loaded = await loadHistory(sessionId);
    if (!loaded) {
      localStorage.removeItem(SESSION_KEY);
      sessionId = "";
    }
  }
  await ensureSession();
  await refreshSessionHistory();
}

initializeChat().catch(() => setStatus("Không khởi tạo được lịch sử chat"));
