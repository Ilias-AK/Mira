/* ==========================================================================
   Mira 1.0 — логика интерфейса чата
   Без зависимостей и сборки. API совместим с прежним app.js:
     GET  /api/status  → { ok, sessionId, maxQuestionCharacters }
     POST /api/message → { ok, text, seconds }      тело: { message }
     POST /api/reset   → { ok, sessionId }
     GET  /api/memory  → { ok, items: [{ label, value }] }   (окно «Память»)
   В ответе /api/message сервер Mira дополнительно присылает route и memoryUpdated.
   Демо без сервера: откройте страницу с ?demo
   ========================================================================== */
(() => {
  "use strict";

  /* ---------- Настройки ---------- */
  const CONFIG = {
    model: "Mira 1.0",
    // Все кнопки из макета включены. Поиск и «Подумать глубже» уходят на сервер в options,
    // вложения вставляют текст файла в сообщение, голос работает через Web Speech API браузера.
    features: { attach: true, search: true, think: true, voice: true },
    typewriter: true,          // постепенный вывод ответа (пока API не стримит)
    minThinkingMs: 450,        // чтобы индикатор «думает» не мигал
    historyLimit: 50,
    counterFrom: 420,          // с какого числа символов показывать счётчик
    maxFileBytes: 512 * 1024,  // вложения: только текстовые файлы до 512 КБ
    followups: ["Расскажи подробнее", "Приведи пример", "Объясни проще"],
  };

  const KEYS = {
    session: "mira-session-v2",
    history: "mira-history-v1",
    theme: "mira-theme",
    sidebar: "mira-sidebar-collapsed",
  };

  const DEMO = new URLSearchParams(location.search).has("demo");
  const IS_MAC = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
  const mqMobile = window.matchMedia("(max-width: 860px)");
  const mqDark = window.matchMedia("(prefers-color-scheme: dark)");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- Элементы ---------- */
  const $ = (id) => document.getElementById(id);
  const els = {
    app: $("app"),
    sidebar: $("sidebar"),
    scrim: $("scrim"),
    sidebarOpen: $("sidebarOpen"),
    sidebarClose: $("sidebarClose"),
    newChat: $("newChatButton"),
    newChatTop: $("newChatTop"),
    newChatKbd: $("newChatKbd"),
    projects: $("projectsButton"),
    rowMenu: $("rowMenu"),
    search: $("historySearch"),
    history: $("history"),
    settingsButton: $("settingsButton"),
    settingsMenu: $("settingsMenu"),
    clearHistory: $("clearHistory"),
    memoryButton: $("memoryButton"),
    connectionLabel: $("connectionLabel"),
    modelButton: $("modelButton"),
    modelMenu: $("modelMenu"),
    modelName: $("modelName"),
    chatTitle: $("chatTitle"),
    shareButton: $("shareButton"),
    moreButton: $("moreButton"),
    moreMenu: $("moreMenu"),
    notice: $("notice"),
    noticeText: $("noticeText"),
    noticeActions: $("noticeActions"),
    scroller: $("scroller"),
    greeting: $("greeting"),
    heroSlot: $("heroSlot"),
    messages: $("messages"),
    toBottom: $("toBottom"),
    dock: $("dock"),
    archiveBar: $("archiveBar"),
    archiveNewChat: $("archiveNewChat"),
    composer: $("composer"),
    input: $("composerInput"),
    charCount: $("charCount"),
    send: $("sendButton"),
    attach: $("attachButton"),
    fileInput: $("fileInput"),
    searchToggle: $("searchToggle"),
    thinkToggle: $("thinkToggle"),
    voice: $("voiceButton"),
    voiceMode: $("voiceModeButton"),
    themeColor: $("themeColor"),
    renameDialog: $("renameDialog"),
    renameForm: $("renameForm"),
    renameInput: $("renameInput"),
    renameCancel: $("renameCancel"),
    memoryDialog: $("memoryDialog"),
    memoryClose: $("memoryClose"),
    memoryList: $("memoryList"),
  };

  /* ---------- Состояние ---------- */
  const state = {
    connected: false,
    sessionId: null,
    chatId: newId(),
    maxChars: 500,
    messages: [],          // [{ role, text, seconds?, feedback?, route?, think?, memory?, at }]
    customTitle: null,     // название текущего чата, если его переименовали
    history: [],           // [{ id, title, updatedAt, messages }]
    archiveId: null,       // открыт сохранённый чат (только чтение)
    busy: false,
    controller: null,      // AbortController текущего запроса
    typing: null,          // { finish } — идёт постепенный вывод
    composing: false,      // ввод через IME
    stick: true,           // держать ленту прижатой к низу
    noticeTimer: null,
    rowTarget: null,       // id чата, для которого открыто меню «…»
    rowButton: null,
    renameTarget: null,
    speaking: null,        // { msg, utterance, button } — идёт озвучка
    recognition: null,     // идёт распознавание речи
    voiceAuto: false,      // голосовой режим: отправить сказанное и озвучить ответ
    speakNext: false,
    searchHintShown: false,
  };

  /* ---------- Иконки (статичные строки) ---------- */
  const MARK = '<svg class="mark" viewBox="0 0 64 64" aria-hidden="true"><path d="M10 50V31a11 11 0 0 1 22 0a11 11 0 0 1 22 0V50"/><circle cx="32" cy="44" r="5"/></svg>';
  const ICON = {
    copy: '<svg class="i-main" viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2.5"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></svg>',
    done: '<svg class="i-done" viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 5 5 9-10"/></svg>',
    up: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 10v10H4V10zM7 10l4-7a2 2 0 0 1 3 2l-1 4h6a2 2 0 0 1 2 2.3l-1.2 7A2 2 0 0 1 17.8 20H7"/></svg>',
    down: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 14V4H4v10zM7 14l4 7a2 2 0 0 0 3-2l-1-4h6a2 2 0 0 0 2-2.3l-1.2-7A2 2 0 0 0 17.8 4H7"/></svg>',
    regen: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 12a8 8 0 1 1-2.3-5.7L20 8.5"/><path d="M20 3.5v5h-5"/></svg>',
    close: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>',
    speak: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9v6h4l5 4V5L8 9z"/><path d="M16.5 8.5a5 5 0 0 1 0 7M19 6a8.5 8.5 0 0 1 0 12"/></svg>',
    dots: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="5" cy="12" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="19" cy="12" r="1.6"/></svg>',
    chevron: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 6 6 6-6 6"/></svg>',
    arrow: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
    memory: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 4.8A1.8 1.8 0 0 1 7.8 3h8.4A1.8 1.8 0 0 1 18 4.8V21l-6-3.8L6 21z"/><path d="m9 10.5 2 2 4-4"/></svg>',
  };

  /* ---------- Хранилище ---------- */
  const store = {
    read(storage, key, fallback) {
      try { const raw = storage.getItem(key); return raw == null ? fallback : JSON.parse(raw); }
      catch { return fallback; }
    },
    write(storage, key, value) {
      try { storage.setItem(key, JSON.stringify(value)); } catch { /* приватный режим или квота */ }
    },
    remove(storage, key) { try { storage.removeItem(key); } catch { /* ignore */ } },
  };

  /* ---------- API ---------- */
  async function http(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
    const payload = await response.json().catch(() => ({ ok: false, error: "Некорректный ответ сервера" }));
    if (!response.ok || !payload.ok) throw new Error(payload.error || "Ошибка соединения");
    return payload;
  }

  const api = DEMO ? demoApi() : {
    status: () => http("/api/status"),
    message: (message, signal, options = {}) => http("/api/message", { method: "POST", body: JSON.stringify({ message, options }), signal }),
    reset: () => http("/api/reset", { method: "POST", body: "{}" }),
    memory: () => http("/api/memory"),
  };

  /* ---------- Утилиты ---------- */
  function newId() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 8); }

  function sleep(ms, signal) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(resolve, ms);
      if (signal) signal.addEventListener("abort", () => {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      }, { once: true });
    });
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function titleFrom(text) {
    const compact = String(text || "").replace(/\s+/g, " ").trim();
    if (!compact) return "Новый чат";
    return compact.length > 40 ? `${compact.slice(0, 40).trimEnd()}…` : compact;
  }

  function greetingText() {
    const h = new Date().getHours();
    const hello = h >= 5 && h < 12 ? "Доброе утро" : h >= 12 && h < 18 ? "Добрый день" : h >= 18 && h < 23 ? "Добрый вечер" : "Доброй ночи";
    return `${hello}. Чем займёмся?`;
  }

  function plural(n, forms) {
    const a = Math.abs(n) % 100;
    const b = a % 10;
    if (a > 10 && a < 20) return forms[2];
    if (b > 1 && b < 5) return forms[1];
    if (b === 1) return forms[0];
    return forms[2];
  }

  function safeFileName(title) {
    const name = String(title || "").replace(/[\\/:*?"<>|\u0000-\u001f]+/g, " ").replace(/\s+/g, " ").trim().slice(0, 60);
    return name || "Mira";
  }

  function formatSeconds(value) {
    const n = Number(value);
    if (!Number.isFinite(n) || n <= 0) return "";
    return `${n < 10 ? n.toFixed(1).replace(".0", "").replace(".", ",") : Math.round(n)} с`;
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      const area = el("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.cssText = "position:fixed;opacity:0;top:0;left:0";
      document.body.appendChild(area);
      area.select();
      let ok = false;
      try { ok = document.execCommand("copy"); } catch { ok = false; }
      area.remove();
      return ok;
    }
  }

  function flashDone(button, label) {
    const prev = button.getAttribute("aria-label");
    button.classList.add("is-done");
    if (label) button.setAttribute("aria-label", label);
    setTimeout(() => {
      button.classList.remove("is-done");
      if (prev) button.setAttribute("aria-label", prev);
    }, 1400);
  }

  /* ---------- Безопасный мини-Markdown ----------
     Поддержка: ```код```, # заголовки, списки (- * • и 1.), **жирный**, `код`, [ссылка](https://…), ---
     Всё экранируется до разметки — HTML из ответа никогда не исполняется. */
  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function inline(text) {
    let html = escapeHtml(text);
    const codes = [];
    html = html.replace(/`([^`\n]+)`/g, (_, code) => `\u0000${codes.push(code) - 1}\u0000`);
    html = html.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    html = html.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[Number(i)]}</code>`);
    return html;
  }

  /* Лёгкая подсветка кода без зависимостей. Цвета — токены --code-* (макет тёмной темы). */
  const LANG = {
    py: "python", python: "python",
    js: "js", javascript: "js", mjs: "js", jsx: "js", ts: "js", typescript: "js", tsx: "js",
    json: "json",
    sh: "bash", bash: "bash", shell: "bash", zsh: "bash", console: "bash", powershell: "bash", ps1: "bash", bat: "bash", cmd: "bash",
    sql: "sql",
    c: "c", h: "c", cpp: "c", "c++": "c", cs: "c", csharp: "c", java: "c", go: "c", rust: "c", rs: "c", kotlin: "c", kt: "c", swift: "c", php: "c",
  };
  const KEYWORDS = {
    python: "and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield None True False match case self",
    js: "await break case catch class const continue debugger default delete do else export extends finally for function if import in instanceof let new of return super switch this throw try typeof var void while with yield async static get set null undefined true false interface type enum implements from as",
    json: "true false null",
    bash: "if then else elif fi for while do done case esac in function return local export echo cd exit set unset readonly shift source alias sudo",
    sql: "select from where and or not insert into values update set delete create table drop alter join left right inner outer on group by order having limit as distinct null is in like between union all primary key foreign references index view case when then else end",
    c: "auto break case catch char class const continue default do double else enum extern float for goto if inline int long new private protected public return short signed sizeof static struct switch template this throw try typedef union unsigned using virtual void volatile while bool true false null nullptr func package import var fn let mut impl pub use mod match trait where self string namespace override fun val when interface final abstract extends implements",
  };
  const BUILTINS = {
    python: "print len range set list dict tuple str int float bool sum min max sorted enumerate zip map filter open input abs any all isinstance type super object iter next reversed round",
    js: "console Math JSON Object Array String Number Boolean Promise Map Set Date RegExp Error document window parseInt parseFloat require module process fetch setTimeout",
    sql: "count sum avg min max coalesce now",
    c: "printf scanf malloc free std cout cin endl fmt println System String Vec Option Some None Ok Err",
    bash: "ls grep cat mkdir rm cp mv git python pip npm node curl",
    json: "",
  };
  const highlighters = {};
  function highlighter(key) {
    if (highlighters[key]) return highlighters[key];
    const comment = key === "python" || key === "bash" ? String.raw`#[^\n]*`
      : key === "sql" ? String.raw`--[^\n]*`
      : key === "json" ? null
      : String.raw`\/\/[^\n]*|\/\*[\s\S]*?(?:\*\/|$)`;
    const quotes = String.raw`"(?:\\.|[^"\\\n])*"?|'(?:\\.|[^'\\\n])*'?`;
    const strings = key === "python" ? String.raw`"""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$)|` + quotes
      : key === "js" ? String.raw`\x60(?:\\[\s\S]|[^\x60\\])*\x60?|` + quotes
      : quotes;
    const parts = [];
    if (comment) parts.push(`(?<com>${comment})`);
    parts.push(`(?<str>${strings})`);
    parts.push(String.raw`(?<num>\b(?:0x[\da-fA-F]+|\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?)\b)`);
    parts.push(String.raw`(?<word>[A-Za-z_$][\w$]*)`);
    const result = {
      re: new RegExp(parts.join("|"), "g"),
      kw: new Set(KEYWORDS[key].split(" ")),
      bi: new Set((BUILTINS[key] || "").split(" ").filter(Boolean)),
      ci: key === "sql",
    };
    highlighters[key] = result;
    return result;
  }
  function highlight(code, lang) {
    const key = LANG[String(lang || "").toLowerCase()];
    if (!key) return escapeHtml(code);
    const { re, kw, bi, ci } = highlighter(key);
    re.lastIndex = 0;
    let out = "";
    let last = 0;
    let m;
    const span = (cls, text) => `<span class="tok-${cls}">${escapeHtml(text)}</span>`;
    while ((m = re.exec(code))) {
      if (!m[0]) { re.lastIndex += 1; continue; }
      out += escapeHtml(code.slice(last, m.index));
      const g = m.groups;
      if (g.com) out += span("com", m[0]);
      else if (g.str) out += span("str", m[0]);
      else if (g.num) out += span("num", m[0]);
      else {
        const word = m[0];
        const test = ci ? word.toLowerCase() : word;
        if (kw.has(test)) out += span("kw", word);
        else if (bi.has(word)) out += span("bi", word);
        else if (/^\s*\(/.test(code.slice(re.lastIndex))) out += span("fn", word);
        else out += escapeHtml(word);
      }
      last = re.lastIndex;
    }
    return out + escapeHtml(code.slice(last));
  }

  function codeBlock(code, lang) {
    const figure = el("figure", "code");
    const head = el("div", "code-head");
    head.appendChild(el("span", "", lang || "код"));
    const copy = el("button", "code-copy");
    copy.type = "button";
    copy.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2.5"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></svg><span>Копировать</span>';
    copy.addEventListener("click", async () => {
      const ok = await copyText(code);
      const label = copy.querySelector("span");
      label.textContent = ok ? "Скопировано" : "Не вышло";
      setTimeout(() => { label.textContent = "Копировать"; }, 1400);
    });
    head.appendChild(copy);
    const pre = el("pre");
    const codeEl = el("code");
    codeEl.innerHTML = highlight(code, lang);
    if (lang) codeEl.dataset.lang = lang;
    pre.appendChild(codeEl);
    figure.append(head, pre);
    return figure;
  }

  /* Таблицы GitHub-Markdown: | шапка | … |, строка |---|:---:|---:| и строки данных */
  function splitRow(line) {
    let t = line.trim();
    if (t.startsWith("|")) t = t.slice(1);
    if (t.endsWith("|") && !t.endsWith("\\|")) t = t.slice(0, -1);
    return t.split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, "|"));
  }
  const isTableSep = (s) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(s || "");
  // Пока ответ печатается, строка-разделитель может быть недописана — таблицу показываем уже по её началу
  const isTableStart = (lines, i) => lines[i].includes("|")
    && (isTableSep(lines[i + 1]) || (i + 2 === lines.length && /^\s*\|\s*:?-[-|:\s]*$/.test(lines[i + 1] || "")));

  function tableBlock(head, aligns, rows) {
    const wrap = el("div", "table-wrap");
    const table = el("table");
    const thead = el("thead");
    const tr = el("tr");
    head.forEach((cell, i) => {
      const th = el("th");
      th.innerHTML = inline(cell);
      if (aligns[i]) th.dataset.align = aligns[i];
      tr.appendChild(th);
    });
    thead.appendChild(tr);
    const tbody = el("tbody");
    rows.forEach((row) => {
      const r = el("tr");
      head.forEach((_, i) => {
        const td = el("td");
        td.innerHTML = inline(row[i] || "");
        if ((row[i] || "").length <= 12) td.classList.add("is-short"); // «1 неделя», «3 дня» не переносим
        if (aligns[i]) td.dataset.align = aligns[i];
        r.appendChild(td);
      });
      tbody.appendChild(r);
    });
    table.append(thead, tbody);
    wrap.appendChild(table);
    return wrap;
  }

  function renderMarkdown(src) {
    const frag = document.createDocumentFragment();
    const lines = String(src).replace(/\r\n?/g, "\n").split("\n");
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];

      const fence = line.match(/^\s*```\s*([\w+#.-]*)\s*$/);
      if (fence) {
        const buf = [];
        i += 1;
        while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) { buf.push(lines[i]); i += 1; }
        i += 1; // закрывающий ``` (или конец текста, если ответ ещё печатается)
        frag.appendChild(codeBlock(buf.join("\n"), fence[1]));
        continue;
      }

      if (!line.trim()) { i += 1; continue; }

      const heading = line.match(/^(#{1,4})\s+(.+)$/);
      if (heading) {
        const h = el(heading[1].length <= 2 ? "h3" : "h4");
        h.innerHTML = inline(heading[2]);
        frag.appendChild(h);
        i += 1;
        continue;
      }

      if (isTableStart(lines, i)) {
        const head = splitRow(line);
        const aligns = splitRow(lines[i + 1]).map((c) => (c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : ""));
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].trim() && lines[i].includes("|")) { rows.push(splitRow(lines[i])); i += 1; }
        frag.appendChild(tableBlock(head, aligns, rows));
        continue;
      }

      if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) { frag.appendChild(el("hr")); i += 1; continue; }

      const isUl = (s) => /^\s*[-*•]\s+/.test(s);
      const isOl = (s) => /^\s*\d+[.)]\s+/.test(s);
      if (isUl(line) || isOl(line)) {
        const ordered = isOl(line);
        const list = el(ordered ? "ol" : "ul");
        while (i < lines.length && (ordered ? isOl(lines[i]) : isUl(lines[i]))) {
          const li = el("li");
          li.innerHTML = inline(lines[i].replace(ordered ? /^\s*\d+[.)]\s+/ : /^\s*[-*•]\s+/, ""));
          list.appendChild(li);
          i += 1;
        }
        frag.appendChild(list);
        continue;
      }

      const buf = [];
      while (i < lines.length && lines[i].trim() && !/^\s*```/.test(lines[i]) && !/^#{1,4}\s/.test(lines[i]) && !isUl(lines[i]) && !isOl(lines[i])
        && !/^\s*(-{3,}|\*{3,})\s*$/.test(lines[i]) && !isTableStart(lines, i)) {
        buf.push(lines[i]);
        i += 1;
      }
      const p = el("p");
      p.innerHTML = inline(buf.join("\n"));
      frag.appendChild(p);
    }
    return frag;
  }

  /* ---------- Уведомления ---------- */
  function notify(message, { error = false, actions = [], sticky = false } = {}) {
    clearTimeout(state.noticeTimer);
    els.noticeText.textContent = message;
    els.noticeActions.replaceChildren(...actions.map(({ label, onClick }) => {
      const b = el("button", "btn", label);
      b.type = "button";
      b.addEventListener("click", onClick);
      return b;
    }));
    els.notice.classList.toggle("is-error", error);
    els.notice.hidden = false;
    if (!sticky) state.noticeTimer = setTimeout(hideNotice, 4800);
  }
  function hideNotice() { els.notice.hidden = true; }

  /* ---------- Тема ---------- */
  function getThemePref() {
    try { return localStorage.getItem(KEYS.theme) || "system"; } catch { return "system"; }
  }
  function applyTheme(pref) {
    if (pref === "light" || pref === "dark") document.documentElement.dataset.theme = pref;
    else delete document.documentElement.dataset.theme;
    try { pref === "system" ? localStorage.removeItem(KEYS.theme) : localStorage.setItem(KEYS.theme, pref); } catch { /* ignore */ }
    els.settingsMenu.querySelectorAll("[data-theme-option]").forEach((b) => {
      b.setAttribute("aria-checked", String(b.dataset.themeOption === pref));
    });
    const dark = pref === "dark" || (pref === "system" && mqDark.matches);
    els.themeColor.setAttribute("content", dark ? "#0e0f12" : "#ffffff");
  }

  /* ---------- Меню ---------- */
  const menus = [
    { button: els.modelButton, menu: els.modelMenu },
    { button: els.settingsButton, menu: els.settingsMenu },
    { button: els.moreButton, menu: els.moreMenu },
  ];
  function closeRowMenu() {
    els.rowMenu.hidden = true;
    if (state.rowButton) state.rowButton.setAttribute("aria-expanded", "false");
    state.rowButton = null;
  }
  function closeMenus(except) {
    menus.forEach(({ button, menu }) => {
      if (menu === except) return;
      menu.hidden = true;
      button.setAttribute("aria-expanded", "false");
    });
    if (except !== els.rowMenu) closeRowMenu();
  }

  /* Меню «…» у чата в истории. Позиционируется у кнопки, не выходя за край окна. */
  function openRowMenu(button, chatId) {
    const wasOpen = !els.rowMenu.hidden && state.rowButton === button;
    closeMenus();
    if (wasOpen) return;
    state.rowTarget = chatId;
    state.rowButton = button;
    button.setAttribute("aria-expanded", "true");
    const menu = els.rowMenu;
    menu.hidden = false;
    const r = button.getBoundingClientRect();
    const w = menu.offsetWidth;
    const h = menu.offsetHeight;
    const left = Math.max(8, Math.min(r.left - 6, window.innerWidth - w - 8));
    let top = r.bottom + 4;
    if (top + h > window.innerHeight - 8) top = Math.max(8, r.top - h - 4);
    menu.style.left = `${left}px`;
    menu.style.top = `${top}px`;
    const first = menu.querySelector("button");
    if (first) first.focus({ preventScroll: true });
  }
  function toggleMenu(button, menu) {
    const open = menu.hidden;
    closeMenus(menu);
    menu.hidden = !open;
    button.setAttribute("aria-expanded", String(open));
    if (open) {
      const first = menu.querySelector('[aria-checked="true"], button');
      if (first) first.focus({ preventScroll: true });
    }
  }

  /* ---------- Боковая панель ---------- */
  function setDrawer(open) {
    els.app.classList.toggle("drawer-open", open);
    els.scrim.hidden = !open;
    els.sidebarOpen.setAttribute("aria-expanded", String(open));
    if (open) els.newChat.focus({ preventScroll: true });
  }
  function setCollapsed(collapsed) {
    els.app.classList.toggle("sidebar-collapsed", collapsed);
    els.sidebarOpen.setAttribute("aria-expanded", String(!collapsed));
    store.write(localStorage, KEYS.sidebar, collapsed);
  }
  function openSidebar() { mqMobile.matches ? setDrawer(true) : setCollapsed(false); }
  function closeSidebar() { mqMobile.matches ? setDrawer(false) : setCollapsed(true); }

  /* ---------- Раскладка ---------- */
  function isEmptyView() {
    return !state.archiveId && state.messages.length === 0 && !state.busy;
  }

  function updateLayout() {
    const empty = isEmptyView();
    els.app.classList.toggle("is-empty", empty);
    const hadFocus = document.activeElement === els.input;
    const target = empty ? els.heroSlot : els.dock;
    if (els.composer.parentElement !== target) {
      if (empty) els.heroSlot.appendChild(els.composer);
      else els.dock.insertBefore(els.composer, els.dock.querySelector(".disclaimer"));
      if (hadFocus) els.input.focus({ preventScroll: true });
    }
    els.composer.hidden = Boolean(state.archiveId);
    els.archiveBar.hidden = !state.archiveId;
    if (!(state.recognition && state.voiceAuto)) els.input.placeholder = empty ? "Спросите что угодно…" : "Ответить Mira…";
    requestAnimationFrame(measureDock);
  }

  function measureDock() {
    els.app.style.setProperty("--dock-h", `${els.dock.offsetHeight}px`);
  }

  function setTitle(text) {
    const title = titleFrom(text);
    els.chatTitle.textContent = title;
    document.title = title === "Новый чат" ? "Mira" : `${title} — Mira`;
  }

  function currentTitle() {
    if (state.customTitle) return titleFrom(state.customTitle);
    const firstUser = state.messages.find((m) => m.role === "user");
    return titleFrom(firstUser ? firstUser.text : "");
  }

  function setConnected(connected) {
    state.connected = connected;
    els.app.dataset.connected = String(connected);
    els.connectionLabel.textContent = connected ? "Mira на связи" : "Нет связи с сервером";
    updateComposer();
  }

  /* ---------- Поле ввода ---------- */
  function updateComposer() {
    const value = els.input.value;
    const hasText = value.trim().length > 0;
    const working = state.busy || Boolean(state.typing);
    els.composer.dataset.state = working ? "busy" : hasText ? "ready" : "idle";
    els.send.disabled = !working && (!hasText || !state.connected);
    els.send.setAttribute("aria-label", working ? "Остановить ответ" : "Отправить");

    const len = value.length;
    els.charCount.textContent = len >= CONFIG.counterFrom ? `${len}/${state.maxChars}` : "";
    els.charCount.classList.toggle("is-limit", len >= state.maxChars);

    els.input.style.height = "auto";
    els.input.style.height = `${Math.min(els.input.scrollHeight, 220)}px`;
    measureDock();
  }

  function fillComposer(text) {
    els.input.value = text;
    updateComposer();
    els.input.focus();
    els.input.setSelectionRange(text.length, text.length);
  }

  /* ---------- Прокрутка ---------- */
  function distanceToBottom() {
    const s = els.scroller;
    return s.scrollHeight - s.scrollTop - s.clientHeight;
  }
  function scrollToBottom(smooth = true) {
    els.scroller.scrollTo({ top: els.scroller.scrollHeight, behavior: smooth && !reducedMotion ? "smooth" : "auto" });
  }
  function stickToBottom() {
    if (state.stick) els.scroller.scrollTop = els.scroller.scrollHeight;
  }
  function onScroll() {
    const d = distanceToBottom();
    state.stick = d < 120;
    els.toBottom.hidden = isEmptyView() || d < 200;
  }

  /* ---------- Сообщения ---------- */
  function renderUser(msg) {
    const row = el("div", "msg msg-user");
    row.appendChild(el("div", "bubble", msg.text));
    return row;
  }

  function actionButton(className, label, icon, onClick) {
    const b = el("button", `act ${className}`);
    b.type = "button";
    b.setAttribute("aria-label", label);
    b.title = label;
    b.innerHTML = icon;
    b.addEventListener("click", onClick);
    return b;
  }

  function thinkLabel(seconds) {
    const n = Math.round(Number(seconds) || 0);
    if (n < 1) return "Размышляла меньше секунды";
    return `Размышляла ${n} ${plural(n, ["секунду", "секунды", "секунд"])}`;
  }

  function routeLabel(route) {
    const r = String(route || "");
    if (r.startsWith("bank")) return "Mira нашла ответ в своей базе знаний.";
    if (r.startsWith("retrieval")) return "Mira подобрала ответ поиском по своей базе диалогов.";
    if (r.startsWith("fallback")) return "Точного ответа Mira не нашла и ответила своими словами.";
    if (/memory|repeat/.test(r)) return "Mira ответила по памяти этого разговора.";
    if (/^(tool|clock|counting)/.test(r)) return "Mira посчитала ответ встроенным инструментом.";
    if (r.startsWith("introduction")) return "Mira запомнила то, что вы о себе рассказали.";
    if (r.startsWith("pair_choice")) return "Mira сравнила варианты из вопроса.";
    if (r.startsWith("demo")) return "Это демонстрационный ответ-заготовка.";
    return "Mira разобрала вопрос по своим правилам и собрала ответ.";
  }

  function renderAssistant(msg, { readonly = false } = {}) {
    const row = el("article", "msg msg-mira");
    const avatar = el("div", "msg-avatar");
    avatar.innerHTML = MARK;
    const body = el("div", "msg-body");

    // «Размышляла N секунд» — появляется, если был включён режим «Подумать глубже»
    if (msg.think) {
      const toggle = el("button", "think", thinkLabel(msg.seconds));
      toggle.type = "button";
      toggle.setAttribute("aria-expanded", "false");
      toggle.insertAdjacentHTML("beforeend", ICON.chevron);
      const panel = el("div", "think-panel", routeLabel(msg.route));
      panel.hidden = true;
      toggle.addEventListener("click", () => {
        const open = panel.hidden;
        panel.hidden = !open;
        toggle.setAttribute("aria-expanded", String(open));
      });
      body.append(toggle, panel);
    }

    // «Память обновлена» — сервер сообщил, что Mira что-то запомнила
    if (msg.memory) {
      const chip = el("button", "memory-chip");
      chip.type = "button";
      chip.innerHTML = ICON.memory;
      chip.appendChild(document.createTextNode("Память обновлена"));
      chip.addEventListener("click", openMemory);
      body.appendChild(chip);
    }

    const answer = el("div", "answer");
    answer.appendChild(renderMarkdown(msg.text));
    body.appendChild(answer);

    const actions = el("div", "msg-actions");
    const copy = actionButton("act-copy", "Скопировать", ICON.copy + ICON.done, async () => {
      if (await copyText(msg.text)) flashDone(copy, "Скопировано");
      else notify("Не получилось скопировать ответ", { error: true });
    });
    actions.appendChild(copy);

    if (!readonly) {
      const up = actionButton("act-up", "Хороший ответ", ICON.up, () => setFeedback(msg, "up", up, down));
      const down = actionButton("act-down", "Плохой ответ", ICON.down, () => setFeedback(msg, "down", up, down));
      up.setAttribute("aria-pressed", String(msg.feedback === "up"));
      down.setAttribute("aria-pressed", String(msg.feedback === "down"));
      const regen = actionButton("act-regen", "Ответить заново", ICON.regen, () => regenerate());
      actions.append(up, down, regen);
    }

    const speakBtn = actionButton("act-speak", "Озвучить", ICON.speak, () => speak(msg, speakBtn));
    speakBtn.setAttribute("aria-pressed", "false");
    actions.appendChild(speakBtn);

    const meta = el("span", "msg-meta", [CONFIG.model, formatSeconds(msg.seconds)].filter(Boolean).join(" · "));
    actions.appendChild(meta);
    body.appendChild(actions);
    row.append(avatar, body);
    return row;
  }

  function setFeedback(msg, value, up, down) {
    msg.feedback = msg.feedback === value ? null : value;
    up.setAttribute("aria-pressed", String(msg.feedback === "up"));
    down.setAttribute("aria-pressed", String(msg.feedback === "down"));
    persistSession();
    // Точка интеграции: отправить оценку на бэкенд, когда появится эндпоинт.
  }

  function appendMessage(msg, opts) {
    const node = msg.role === "user" ? renderUser(msg) : renderAssistant(msg, opts);
    els.messages.appendChild(node);
    return node;
  }

  function renderThread(messages, opts = {}) {
    els.messages.replaceChildren(...messages.map((m) => (m.role === "user" ? renderUser(m) : renderAssistant(m, opts))));
    updateFollowups();
  }

  /* «Продолжить» — подсказки под последним ответом текущего чата */
  function updateFollowups() {
    els.messages.querySelectorAll(".followups").forEach((node) => node.remove());
    if (state.archiveId || state.busy || state.typing) return;
    const last = els.messages.lastElementChild;
    if (!last || !last.classList.contains("msg-mira") || last.classList.contains("thinking")) return;
    const box = el("div", "followups");
    box.appendChild(el("span", "followups-label", "Продолжить"));
    const list = el("div", "followups-list");
    CONFIG.followups.forEach((text) => {
      const b = el("button", "followup");
      b.type = "button";
      b.innerHTML = ICON.arrow;
      b.appendChild(document.createTextNode(text));
      b.addEventListener("click", () => sendMessage(text));
      list.appendChild(b);
    });
    box.appendChild(list);
    last.querySelector(".msg-body").appendChild(box);
  }

  function showThinking() {
    const row = el("div", "msg msg-mira thinking");
    row.id = "thinkingRow";
    const avatar = el("div", "msg-avatar");
    avatar.innerHTML = MARK;
    const label = el("span", "thinking-label", "Mira думает…");
    label.setAttribute("role", "status");
    row.append(avatar, label);
    els.messages.appendChild(row);
    stickToBottom();
  }
  function hideThinking() {
    const row = document.getElementById("thinkingRow");
    if (row) row.remove();
  }

  /* Постепенный вывод ответа. Каждый кадр перерисовывает Markdown частичного текста,
     поэтому заголовки и код выглядят правильно уже во время «печати». */
  function typeOut(row, text) {
    const answer = row.querySelector(".answer");
    if (!CONFIG.typewriter || reducedMotion || text.length < 24) return Promise.resolve();
    row.classList.add("is-typing");
    return new Promise((resolve) => {
      let shown = 0;
      let raf = 0;
      const step = Math.max(2, Math.ceil(text.length / 110));
      const finish = () => {
        cancelAnimationFrame(raf);
        answer.replaceChildren(renderMarkdown(text));
        row.classList.remove("is-typing");
        state.typing = null;
        updateComposer();
        stickToBottom();
        resolve();
      };
      const tick = () => {
        shown = Math.min(text.length, shown + step);
        answer.replaceChildren(renderMarkdown(text.slice(0, shown)));
        // Курсор стоит в конце последней строки текста, а не отдельной строкой
        let target = answer.lastElementChild;
        if (target && (target.tagName === "UL" || target.tagName === "OL")) target = target.lastElementChild;
        if (!target || !/^(P|LI|H3|H4)$/.test(target.tagName)) target = answer;
        target.appendChild(el("span", "caret"));
        stickToBottom();
        if (shown < text.length) raf = requestAnimationFrame(tick);
        else finish();
      };
      state.typing = { finish };
      updateComposer();
      raf = requestAnimationFrame(tick);
    });
  }

  /* ---------- Отправка ---------- */
  function setBusy(busy) {
    state.busy = busy;
    els.input.readOnly = false;
    updateComposer();
  }

  async function sendMessage(raw, { regenerate: isRegen = false } = {}) {
    const message = String(raw ?? "").trim();
    if (!message || state.busy || state.typing || state.archiveId) return;
    if (!state.connected) {
      notify("Нет связи с сервером Mira", { error: true });
      return;
    }
    if (message.length > state.maxChars) {
      notify(`Сообщение должно быть не длиннее ${state.maxChars} символов`, { error: true });
      return;
    }

    hideNotice();
    if (!isRegen) {
      const userMsg = { role: "user", text: message, at: Date.now() };
      state.messages.push(userMsg);
      if (state.messages.filter((m) => m.role === "user").length === 1) setTitle(currentTitle());
      els.input.value = "";
    }

    stopSpeaking();
    setBusy(true);
    updateFollowups();
    updateLayout();
    if (!isRegen) appendMessage(state.messages[state.messages.length - 1]);
    state.stick = true;
    showThinking();
    scrollToBottom(false);
    persistSession();
    renderHistory();
    if (mqMobile.matches) setDrawer(false);

    const controller = new AbortController();
    state.controller = controller;
    const started = performance.now();
    // Переключатели из поля ввода уходят на сервер — он может учитывать их, когда научится
    const options = { search: isPressed(els.searchToggle), think: isPressed(els.thinkToggle) };

    try {
      const answer = await api.message(message, controller.signal, options);
      const wait = Math.max(0, CONFIG.minThinkingMs - (performance.now() - started));
      if (wait) await sleep(wait, controller.signal);
      hideThinking();
      const msg = {
        role: "assistant",
        text: String(answer.text ?? ""),
        seconds: Number(answer.seconds) || (performance.now() - started) / 1000,
        route: answer.route ? String(answer.route) : undefined,
        think: options.think || undefined,
        memory: answer.memoryUpdated === true || undefined,
        at: Date.now(),
      };
      state.messages.push(msg);
      persistSession();
      state.controller = null;
      setBusy(false);
      const row = appendMessage(msg);
      if (state.speakNext) {
        state.speakNext = false;
        speak(msg, row.querySelector(".act-speak"));
      }
      await typeOut(row, msg.text);
    } catch (error) {
      hideThinking();
      if (error && error.name === "AbortError") {
        notify("Ответ остановлен");
      } else {
        notify(error && error.message ? error.message : "Mira сейчас не отвечает", {
          error: true,
          actions: [{ label: "Повторить", onClick: () => { hideNotice(); sendMessage(message, { regenerate: true }); } }],
        });
      }
    } finally {
      state.controller = null;
      state.speakNext = false;
      setBusy(false);
      persistSession();
      renderHistory();
      updateLayout();
      updateFollowups();
      if (!mqMobile.matches) els.input.focus({ preventScroll: true });
    }
  }

  function stop() {
    if (state.controller) state.controller.abort();
    else if (state.typing) state.typing.finish();
  }

  function regenerate() {
    if (state.busy || state.typing) return;
    const last = state.messages[state.messages.length - 1];
    if (!last || last.role !== "assistant") return;
    const prevUser = [...state.messages].reverse().find((m) => m.role === "user");
    if (!prevUser) return;
    state.messages.pop();
    const rows = els.messages.querySelectorAll(".msg-mira");
    if (rows.length) rows[rows.length - 1].remove();
    sendMessage(prevUser.text, { regenerate: true });
  }

  /* ---------- Сессия и история ---------- */
  function persistSession() {
    store.write(sessionStorage, KEYS.session, {
      sessionId: state.sessionId,
      chatId: state.chatId,
      customTitle: state.customTitle,
      messages: state.messages,
    });
  }

  function saveHistory() {
    state.history = state.history
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, CONFIG.historyLimit);
    store.write(localStorage, KEYS.history, state.history);
  }

  function archiveMessages(chatId, messages, customTitle) {
    if (!messages.length) return;
    const firstUser = messages.find((m) => m.role === "user");
    const entry = {
      id: chatId,
      title: titleFrom(customTitle || (firstUser ? firstUser.text : "")),
      updatedAt: messages[messages.length - 1].at || Date.now(),
      messages: messages.slice(-200),
    };
    state.history = state.history.filter((h) => h.id !== chatId);
    state.history.push(entry);
    saveHistory();
  }

  function groupLabel(ts) {
    const day = 864e5;
    const start = new Date(); start.setHours(0, 0, 0, 0);
    const t = start.getTime();
    if (ts >= t) return "Сегодня";
    if (ts >= t - day) return "Вчера";
    if (ts >= t - 6 * day) return "Последние 7 дней";
    return "Ранее";
  }

  function renderHistory() {
    const query = els.search.value.trim().toLowerCase();
    const items = [];
    if (state.messages.length) {
      items.push({ id: state.chatId, title: currentTitle(), updatedAt: Date.now(), current: true });
    }
    state.history.filter((h) => h.id !== state.chatId).forEach((h) => items.push(h));

    const filtered = query
      ? items.filter((it) => it.title.toLowerCase().includes(query)
        || (it.messages || (it.current ? state.messages : [])).some((m) => m.text.toLowerCase().includes(query)))
      : items;

    if (!filtered.length) {
      els.history.replaceChildren(el("p", "history-empty", query ? "Ничего не нашлось" : "Здесь появятся ваши чаты"));
      return;
    }

    const groups = new Map();
    filtered.forEach((it) => {
      const label = groupLabel(it.updatedAt);
      if (!groups.has(label)) groups.set(label, []);
      groups.get(label).push(it);
    });

    const nodes = [];
    groups.forEach((list, label) => {
      const group = el("div", "history-group");
      group.appendChild(el("p", "history-label", label));
      list.forEach((it) => {
        const row = el("div", "history-row");
        const btn = el("button", "history-item", it.title);
        btn.type = "button";
        btn.title = it.title;
        const active = it.current ? !state.archiveId : state.archiveId === it.id;
        if (active) btn.setAttribute("aria-current", "page");
        btn.addEventListener("click", () => (it.current ? openCurrent() : openArchive(it.id)));
        row.appendChild(btn);
        const more = el("button", "history-more");
        more.type = "button";
        more.setAttribute("aria-label", `Действия с чатом «${it.title}»`);
        more.setAttribute("aria-haspopup", "menu");
        more.setAttribute("aria-expanded", "false");
        more.setAttribute("aria-controls", "rowMenu");
        more.innerHTML = ICON.dots;
        more.addEventListener("click", (event) => { event.stopPropagation(); openRowMenu(more, it.id); });
        row.appendChild(more);
        group.appendChild(row);
      });
      nodes.push(group);
    });
    els.history.replaceChildren(...nodes);
  }

  function openCurrent() {
    if (!state.archiveId) { if (mqMobile.matches) setDrawer(false); return; }
    state.archiveId = null;
    renderThread(state.messages);
    setTitle(currentTitle());
    updateLayout();
    renderHistory();
    scrollToBottom(false);
    if (mqMobile.matches) setDrawer(false);
  }

  function openArchive(id) {
    if (state.busy || state.typing) return;
    const entry = state.history.find((h) => h.id === id);
    if (!entry) return;
    state.archiveId = id;
    renderThread(entry.messages, { readonly: true });
    setTitle(entry.title);
    updateLayout();
    renderHistory();
    els.scroller.scrollTop = 0;
    if (mqMobile.matches) setDrawer(false);
  }

  function deleteArchive(id) {
    state.history = state.history.filter((h) => h.id !== id);
    saveHistory();
    if (state.archiveId === id) openCurrent();
    renderHistory();
  }

  function newChat() {
    if (!state.messages.length && !state.archiveId) {
      closeMenus();
      if (mqMobile.matches) setDrawer(false);
      els.input.focus();
      return Promise.resolve();
    }
    return startFresh({ archive: true });
  }

  /* Начать чистый чат. archive: false — текущая переписка удаляется, а не уходит в историю. */
  async function startFresh({ archive }) {
    if (state.busy || state.typing) stop();
    stopSpeaking();
    stopDictation();
    closeMenus();
    try {
      const result = await api.reset();
      if (archive) archiveMessages(state.chatId, state.messages, state.customTitle);
      state.sessionId = result.sessionId;
      state.chatId = newId();
      state.messages = [];
      state.customTitle = null;
      state.archiveId = null;
      els.messages.replaceChildren();
      els.input.value = "";
      setTitle("");
      els.greeting.textContent = greetingText();
      persistSession();
      updateComposer();
      updateLayout();
      renderHistory();
      if (mqMobile.matches) setDrawer(false);
      els.input.focus();
    } catch (error) {
      notify(error.message || "Не удалось начать новый чат", { error: true });
    }
  }

  function restoreSession(serverSessionId) {
    const saved = store.read(sessionStorage, KEYS.session, null);
    if (!saved || !Array.isArray(saved.messages)) return;
    const messages = saved.messages.filter((m) => m && (m.role === "user" || m.role === "assistant") && typeof m.text === "string");
    if (saved.sessionId === serverSessionId) {
      state.chatId = saved.chatId || state.chatId;
      state.customTitle = typeof saved.customTitle === "string" ? saved.customTitle : null;
      state.messages = messages;
      renderThread(messages);
      setTitle(currentTitle());
    } else if (messages.length) {
      // Сервер перезапустился — контекст потерян, но переписку сохраняем в историю.
      archiveMessages(saved.chatId || newId(), messages, saved.customTitle);
    }
  }

  /* ---------- Чаты: поделиться, скачать, переименовать, удалить ---------- */
  function chatById(id) {
    if (id === state.chatId) {
      return { id, title: currentTitle(), messages: state.messages, current: true };
    }
    const entry = state.history.find((h) => h.id === id);
    return entry ? { id, title: entry.title, messages: entry.messages, current: false } : null;
  }

  function shownChat() {
    return chatById(state.archiveId || state.chatId);
  }

  function chatMarkdown(chat) {
    const parts = [`# ${chat.title}`];
    chat.messages.forEach((m) => {
      parts.push(m.role === "user" ? `**Вы:** ${m.text}` : `**Mira:**\n\n${m.text}`);
    });
    return `${parts.join("\n\n")}\n`;
  }

  function downloadChat(chat) {
    if (!chat || !chat.messages.length) { notify("В этом чате пока ничего нет"); return; }
    const blob = new Blob([chatMarkdown(chat)], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = el("a");
    link.href = url;
    link.download = `${safeFileName(chat.title)}.md`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1500);
  }

  async function shareChat(chat) {
    closeMenus();
    if (!chat || !chat.messages.length) { notify("В этом чате пока нечем поделиться"); return; }
    const text = chatMarkdown(chat);
    if (navigator.share && window.matchMedia("(pointer: coarse)").matches) {
      try { await navigator.share({ title: chat.title, text }); return; } catch (error) {
        if (error && error.name === "AbortError") return;
      }
    }
    const ok = await copyText(text);
    notify(ok ? "Диалог скопирован — его можно вставить куда угодно" : "Не получилось скопировать диалог", {
      error: !ok,
      actions: [{ label: "Скачать .md", onClick: () => { hideNotice(); downloadChat(chat); } }],
    });
  }

  function openRename(chat) {
    closeMenus();
    if (!chat) return;
    const current = chat.title === "Новый чат" ? "" : chat.title;
    if (typeof els.renameDialog.showModal !== "function") {
      const value = window.prompt("Название чата", current);
      if (value != null) renameChat(chat.id, value);
      return;
    }
    state.renameTarget = chat.id;
    els.renameInput.value = current;
    els.renameDialog.showModal();
    els.renameInput.select();
  }

  function renameChat(id, raw) {
    const title = String(raw || "").replace(/\s+/g, " ").trim().slice(0, 80);
    if (!title) return;
    if (id === state.chatId) {
      state.customTitle = title;
      persistSession();
      if (!state.archiveId) setTitle(title);
    } else {
      const entry = state.history.find((h) => h.id === id);
      if (!entry) return;
      entry.title = title;
      saveHistory();
      if (state.archiveId === id) setTitle(title);
    }
    renderHistory();
  }

  function deleteChat(id) {
    closeMenus();
    if (id === state.chatId) {
      if (!state.messages.length) return;
      if (!window.confirm("Удалить этот чат? Восстановить его не получится.")) return;
      startFresh({ archive: false });
      return;
    }
    deleteArchive(id);
  }

  /* ---------- Память ---------- */
  function renderMemory(items) {
    if (!items.length) {
      const empty = el("li", "memory-empty", "Пока Mira ничего о вас не запомнила. Скажите, например: «Меня зовут Аня» или «Запомни, что я учу Python».");
      els.memoryList.replaceChildren(empty);
      return;
    }
    els.memoryList.replaceChildren(...items.map((item) => {
      const li = el("li");
      li.append(el("span", "memory-key", String(item.label || "")), el("span", "memory-value", String(item.value || "")));
      return li;
    }));
  }

  async function openMemory() {
    closeMenus();
    if (mqMobile.matches) setDrawer(false);
    els.memoryList.replaceChildren(el("li", "memory-empty", "Загружаю…"));
    if (typeof els.memoryDialog.showModal === "function") els.memoryDialog.showModal();
    else els.memoryDialog.setAttribute("open", "");
    try {
      const data = await api.memory();
      renderMemory(Array.isArray(data.items) ? data.items : []);
    } catch (error) {
      els.memoryList.replaceChildren(el("li", "memory-empty", error.message || "Не удалось открыть память"));
    }
  }

  /* ---------- Озвучка ответа (speechSynthesis работает без интернета) ---------- */
  const synth = window.speechSynthesis;

  function speechText(text) {
    return String(text)
      .replace(/```[\s\S]*?(```|$)/g, " Фрагмент кода. ")
      .replace(/`([^`]+)`/g, "$1")
      .replace(/\*\*([^*]+)\*\*/g, "$1")
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .replace(/^\s*#{1,4}\s+/gm, "")
      .replace(/^\s*[-*•]\s+/gm, "")
      .replace(/^\s*\|?\s*:?-{3,}.*$/gm, "")
      .replace(/\|/g, ", ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function setSpeakButton(button, on) {
    if (!button) return;
    button.setAttribute("aria-pressed", String(on));
    button.setAttribute("aria-label", on ? "Остановить озвучку" : "Озвучить");
    button.title = on ? "Остановить озвучку" : "Озвучить";
  }

  function stopSpeaking() {
    if (!state.speaking) return;
    const { button } = state.speaking;
    state.speaking = null;
    setSpeakButton(button, false);
    if (synth) synth.cancel();
  }

  function speak(msg, button) {
    if (!synth || typeof window.SpeechSynthesisUtterance !== "function") {
      notify("Озвучка не поддерживается в этом браузере", { error: true });
      return;
    }
    const same = state.speaking && state.speaking.msg === msg;
    stopSpeaking();
    if (same) return;
    const utterance = new SpeechSynthesisUtterance(speechText(msg.text));
    utterance.lang = "ru-RU";
    const voice = synth.getVoices().find((v) => /^ru/i.test(v.lang));
    if (voice) utterance.voice = voice;
    const done = () => {
      if (state.speaking && state.speaking.utterance === utterance) {
        state.speaking = null;
        setSpeakButton(button, false);
      }
    };
    utterance.onend = done;
    utterance.onerror = done;
    state.speaking = { msg, utterance, button };
    setSpeakButton(button, true);
    synth.speak(utterance);
  }

  /* ---------- Голосовой ввод и голосовой режим (Web Speech API) ---------- */
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const RECOGNITION_ERRORS = {
    "not-allowed": "Нет доступа к микрофону — разрешите его в настройках браузера",
    "service-not-allowed": "Браузер не разрешает распознавание речи",
    network: "Для распознавания речи браузеру нужен интернет",
    "no-speech": "Не расслышала — попробуйте ещё раз",
    "audio-capture": "Микрофон не найден",
    "language-not-supported": "Браузер не распознаёт русскую речь",
  };

  function setListening(on, auto) {
    [[els.voice, !auto], [els.voiceMode, auto]].forEach(([button, mine]) => {
      const active = on && mine;
      button.setAttribute("aria-pressed", String(active));
      button.classList.toggle("is-listening", active);
    });
    els.voice.setAttribute("aria-label", on && !auto ? "Остановить голосовой ввод" : "Голосовой ввод");
    els.voiceMode.setAttribute("aria-label", on && auto ? "Остановить голосовой режим" : "Голосовой режим");
    if (on && auto) els.input.placeholder = "Слушаю…";
  }

  function stopDictation() {
    if (state.recognition) {
      state.recognition.cancelled = true;
      try { state.recognition.abort(); } catch { /* ignore */ }
    }
  }

  function startDictation({ auto = false } = {}) {
    if (state.recognition) {
      try { state.recognition.stop(); } catch { /* ignore */ }
      return;
    }
    if (!Recognition) {
      notify("Голосовой ввод не поддерживается в этом браузере — попробуйте Chrome или Edge", { error: true });
      return;
    }
    if (state.busy || state.typing || state.archiveId) return;
    stopSpeaking();
    const rec = new Recognition();
    rec.lang = "ru-RU";
    rec.interimResults = true;
    rec.continuous = !auto;
    rec.maxAlternatives = 1;
    const base = auto ? "" : els.input.value.replace(/\s+$/, "");
    let finalText = "";

    rec.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        if (result.isFinal) finalText += result[0].transcript;
        else interim += result[0].transcript;
      }
      const spoken = `${finalText}${interim}`.replace(/\s+/g, " ").trim();
      if (auto) {
        els.input.placeholder = spoken || "Слушаю…";
      } else {
        els.input.value = [base, spoken].filter(Boolean).join(" ").slice(0, state.maxChars);
        updateComposer();
      }
    };
    rec.onerror = (event) => {
      if (event.error === "aborted") return;
      notify(RECOGNITION_ERRORS[event.error] || "Голосовой ввод не сработал", { error: event.error !== "no-speech" });
    };
    rec.onend = () => {
      state.recognition = null;
      state.voiceAuto = false;
      setListening(false, auto);
      updateLayout();
      const text = finalText.replace(/\s+/g, " ").trim();
      if (auto && text && !rec.cancelled) {
        state.speakNext = true;
        sendMessage(text);
      } else if (!auto) {
        els.input.focus();
      }
    };

    state.recognition = rec;
    state.voiceAuto = auto;
    setListening(true, auto);
    try {
      rec.start();
    } catch {
      state.recognition = null;
      state.voiceAuto = false;
      setListening(false, auto);
      updateLayout();
      notify("Не удалось включить микрофон", { error: true });
    }
  }

  /* ---------- Вложения: текст файла вставляется в сообщение ---------- */
  const TEXT_FILE = /\.(txt|md|markdown|csv|tsv|json|log|py|m?js|tsx?|jsx|html?|css|xml|ya?ml|ini|toml|sql|sh|bat|ps1|java|c|cpp|h|cs|go|rs|rb|php|kt|swift)$/i;

  async function attachFile(file) {
    if (!file) return;
    if (!(/^text\//.test(file.type) || file.type === "application/json" || TEXT_FILE.test(file.name))) {
      notify("Mira пока читает только текстовые файлы", { error: true });
      return;
    }
    if (file.size > CONFIG.maxFileBytes) {
      notify("Файл слишком большой — можно до 512 КБ", { error: true });
      return;
    }
    let text = "";
    try { text = await file.text(); } catch {
      notify("Не получилось прочитать файл", { error: true });
      return;
    }
    text = text.replace(/\r\n?/g, "\n").replace(/\u0000/g, "").trim();
    if (!text) { notify(`Файл «${file.name}» пустой`); return; }
    const current = els.input.value.replace(/\s+$/, "");
    const room = state.maxChars - current.length - (current ? 2 : 0);
    if (room <= 0) {
      notify("В сообщении уже нет места для текста файла", { error: true });
      return;
    }
    const piece = text.length > room ? text.slice(0, room) : text;
    fillComposer(current ? `${current}\n\n${piece}` : piece);
    notify(text.length > room
      ? `Из «${file.name}» вставлено начало — сообщение не длиннее ${state.maxChars} символов`
      : `Текст из «${file.name}» добавлен в сообщение`);
  }

  function isPressed(button) {
    return Boolean(button) && !button.hidden && button.getAttribute("aria-pressed") === "true";
  }

  /* ---------- Демо-API (без сервера) ---------- */
  function demoApi() {
    const id = () => `demo-${Math.random().toString(36).slice(2, 10)}`;
    let facts = [];
    return {
      async status() { await sleep(120); return { ok: true, sessionId: store.read(sessionStorage, KEYS.session, {})?.sessionId || id(), maxQuestionCharacters: 500 }; },
      async reset() { await sleep(120); facts = []; return { ok: true, sessionId: id() }; },
      async memory() { await sleep(120); return { ok: true, items: facts.slice() }; },
      async message(message, signal) {
        const ms = 700 + Math.random() * 900;
        await sleep(ms, signal);
        const name = message.match(/меня зовут\s+([А-ЯЁA-Z][а-яёa-z-]+)/i);
        const memo = message.match(/^\s*запомни,?\s*(?:что\s+)?(.{2,160}?)\s*[.!]?$/i);
        if (name) {
          facts = facts.filter((f) => f.label !== "Имя").concat({ label: "Имя", value: name[1] });
          return { ok: true, text: `Приятно познакомиться, ${name[1]}! Запомнила.`, seconds: ms / 1000, route: "demo:memory", memoryUpdated: true };
        }
        if (memo) {
          facts = facts.concat({ label: "Просили запомнить", value: memo[1] });
          return { ok: true, text: "Запомнила. Теперь буду учитывать это в ответах.", seconds: ms / 1000, route: "demo:memory", memoryUpdated: true };
        }
        return { ok: true, text: demoAnswer(message), seconds: ms / 1000, route: "demo" };
      },
    };
  }

  function demoAnswer(message) {
    const t = message.toLowerCase();
    if (/таблиц|план|провер/.test(t)) {
      return [
        "Вот план проверки на две недели:",
        "",
        "| Что проверяем | Как | Срок |",
        "|---|---|---|",
        "| Поток людей | Посчитать прохожих у входа утром, в обед и вечером | 1 неделя |",
        "| Готовность платить | Поставить временную стойку на пять учебных дней | 1 неделя |",
        "| Аренда | Запросить ставки у трёх арендодателей рядом | 3 дня |",
        "",
        "Хотите, соберу из этого финансовую модель на первый год?",
      ].join("\n");
    }
    if (/код|python|функци|javascript|скрипт/.test(t)) {
      return [
        "Держим множество уже встреченных элементов и добавляем в результат только новые — проход линейный:",
        "",
        "```python",
        "def unique(items):",
        "    seen = set()",
        "    result = []",
        "    for item in items:",
        "        if item not in seen:",
        "            seen.add(item)",
        "            result.append(item)",
        "    return result",
        "```",
        "",
        "Короче: `list(dict.fromkeys(items))` — словарь тоже помнит порядок вставки.",
      ].join("\n");
    }
    if (/иде|разбер/.test(t)) {
      return [
        "Это демо-режим, поэтому отвечаю заготовкой — но в том формате, в каком Mira отвечает по-настоящему.",
        "",
        "### Что говорит в пользу",
        "- **Понятная проблема.** Её легко объяснить в одном предложении.",
        "- **Быстрая проверка.** Первую версию можно собрать за выходные.",
        "",
        "### Где слабые места",
        "1. **Спрос не проверен.** Пока это гипотеза, а не факт.",
        "2. **Нет плана на первые сто пользователей.**",
        "",
        "Хотите, разложу это в план проверки на две недели?",
      ].join("\n");
    }
    return [
      "Это демо-режим интерфейса — настоящий ответ придёт от сервера Mira.",
      "",
      "Обратите внимание на формат: вопрос — в «пузыре», ответ — открытым текстом на всю колонку. Так длинный ответ читается как статья, а не как переписка.",
    ].join("\n");
  }

  /* ---------- События ---------- */
  els.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.busy || state.typing) stop();
    else sendMessage(els.input.value);
  });

  els.input.addEventListener("input", updateComposer);
  els.input.addEventListener("compositionstart", () => { state.composing = true; });
  els.input.addEventListener("compositionend", () => { state.composing = false; });
  els.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !state.composing && !event.isComposing) {
      // На телефоне Enter — перенос строки, отправка кнопкой
      if (mqMobile.matches && window.matchMedia("(pointer: coarse)").matches) return;
      event.preventDefault();
      if (!state.busy && !state.typing) sendMessage(els.input.value);
    }
  });

  document.querySelectorAll("[data-fill]").forEach((b) => b.addEventListener("click", () => fillComposer(b.dataset.fill || "")));
  document.querySelectorAll("[data-feature]").forEach((b) => { b.hidden = !CONFIG.features[b.dataset.feature]; });
  document.querySelectorAll(".tool-btn.chip").forEach((b) => b.addEventListener("click", () => {
    b.setAttribute("aria-pressed", String(b.getAttribute("aria-pressed") !== "true"));
  }));
  els.searchToggle.addEventListener("click", () => {
    if (!isPressed(els.searchToggle) || state.searchHintShown) return;
    state.searchHintShown = true;
    notify("Mira работает без интернета — поиск идёт по её собственной базе знаний");
  });

  els.attach.addEventListener("click", () => {
    if (state.archiveId) return;
    els.fileInput.value = "";
    els.fileInput.click();
  });
  els.fileInput.addEventListener("change", () => attachFile(els.fileInput.files && els.fileInput.files[0]));
  els.voice.addEventListener("click", () => startDictation({ auto: false }));
  els.voiceMode.addEventListener("click", () => startDictation({ auto: true }));

  els.projects.addEventListener("click", () => {
    notify("Проекты скоро появятся — в них можно будет собирать чаты по темам");
  });

  els.shareButton.addEventListener("click", () => shareChat(shownChat()));
  els.moreButton.addEventListener("click", () => toggleMenu(els.moreButton, els.moreMenu));
  els.moreMenu.querySelectorAll("[data-chat-action]").forEach((b) => b.addEventListener("click", () => {
    const chat = shownChat();
    const action = b.dataset.chatAction;
    closeMenus();
    if (!chat) return;
    if (action === "rename") openRename(chat);
    else if (action === "download") downloadChat(chat);
    else if (action === "delete") deleteChat(chat.id);
  }));
  els.rowMenu.querySelectorAll("[data-row-action]").forEach((b) => b.addEventListener("click", () => {
    const chat = chatById(state.rowTarget);
    const action = b.dataset.rowAction;
    closeMenus();
    if (!chat) return;
    if (action === "share") shareChat(chat);
    else if (action === "rename") openRename(chat);
    else if (action === "delete") deleteChat(chat.id);
  }));
  els.history.addEventListener("scroll", closeRowMenu, { passive: true });

  els.renameForm.addEventListener("submit", (event) => {
    event.preventDefault();
    renameChat(state.renameTarget, els.renameInput.value);
    els.renameDialog.close();
  });
  els.renameCancel.addEventListener("click", () => els.renameDialog.close());
  els.memoryButton.addEventListener("click", openMemory);
  els.memoryClose.addEventListener("click", () => els.memoryDialog.close());
  [els.renameDialog, els.memoryDialog].forEach((dialog) => {
    // Клик по затемнению вокруг окна закрывает его
    dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
  });

  els.newChat.addEventListener("click", newChat);
  els.newChatTop.addEventListener("click", newChat);
  els.archiveNewChat.addEventListener("click", newChat);
  els.sidebarOpen.addEventListener("click", openSidebar);
  els.sidebarClose.addEventListener("click", closeSidebar);
  els.scrim.addEventListener("click", () => setDrawer(false));
  els.search.addEventListener("input", renderHistory);

  els.modelButton.addEventListener("click", () => toggleMenu(els.modelButton, els.modelMenu));
  els.settingsButton.addEventListener("click", () => toggleMenu(els.settingsButton, els.settingsMenu));
  els.settingsMenu.querySelectorAll("[data-theme-option]").forEach((b) => b.addEventListener("click", () => {
    applyTheme(b.dataset.themeOption);
  }));
  els.clearHistory.addEventListener("click", () => {
    closeMenus();
    if (!state.history.length) { notify("История уже пуста"); return; }
    if (!window.confirm("Удалить все сохранённые чаты? Текущий чат останется.")) return;
    state.history = [];
    saveHistory();
    if (state.archiveId) openCurrent();
    renderHistory();
  });

  els.scroller.addEventListener("scroll", onScroll, { passive: true });
  els.toBottom.addEventListener("click", () => { state.stick = true; scrollToBottom(); });

  document.addEventListener("click", (event) => {
    if (!event.target.closest(".model-picker, .sidebar-foot, .more-picker, .row-menu, .history-more")) closeMenus();
  });

  window.addEventListener("keydown", (event) => {
    if (document.querySelector("dialog[open]")) return; // Esc и фокус внутри окна обрабатывает сам <dialog>
    const mod = IS_MAC ? event.metaKey : event.ctrlKey;
    if (mod && event.key.toLowerCase() === "k") { event.preventDefault(); newChat(); return; }
    if (event.key === "Escape") {
      if (!els.rowMenu.hidden) { const button = state.rowButton; closeMenus(); if (button) button.focus(); return; }
      const openMenu = menus.find(({ menu }) => !menu.hidden);
      if (openMenu) { closeMenus(); openMenu.button.focus(); return; }
      if (state.recognition) { stopDictation(); return; }
      if (els.app.classList.contains("drawer-open")) { setDrawer(false); els.sidebarOpen.focus(); return; }
      if (state.busy || state.typing) stop();
    }
  });

  mqMobile.addEventListener("change", () => { setDrawer(false); updateLayout(); });
  mqDark.addEventListener("change", () => applyTheme(getThemePref()));
  window.addEventListener("resize", () => { measureDock(); closeRowMenu(); });
  window.addEventListener("pagehide", () => { stopSpeaking(); stopDictation(); });
  if (synth) synth.getVoices(); // Chrome подгружает голоса лениво — запрашиваем заранее

  /* ---------- Запуск ---------- */
  async function connect() {
    try {
      const status = await api.status();
      state.sessionId = status.sessionId;
      state.maxChars = Number(status.maxQuestionCharacters) || 500;
      els.input.maxLength = state.maxChars;
      restoreSession(status.sessionId);
      setConnected(true);
      hideNotice();
    } catch {
      setConnected(false);
      notify("Локальный сервер Mira недоступен", {
        error: true,
        sticky: true,
        actions: [
          { label: "Повторить", onClick: () => { hideNotice(); connect(); } },
          { label: "Демо", onClick: () => { location.search = "?demo"; } },
        ],
      });
    }
    renderHistory();
    updateLayout();
    updateComposer();
    scrollToBottom(false);
  }

  function init() {
    els.modelName.textContent = CONFIG.model;
    els.newChatKbd.textContent = IS_MAC ? "⌘K" : "Ctrl K";
    els.greeting.textContent = greetingText();
    state.history = store.read(localStorage, KEYS.history, []).filter((h) => h && h.id && Array.isArray(h.messages));
    applyTheme(getThemePref());
    if (store.read(localStorage, KEYS.sidebar, false) && !mqMobile.matches) setCollapsed(true);
    updateLayout();
    updateComposer();
    if (!mqMobile.matches) els.input.focus();
    connect();
  }

  init();
})();
