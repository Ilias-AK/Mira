(() => {
  "use strict";

  const app = document.getElementById("app");
  const conversation = document.getElementById("conversation");
  const welcome = document.getElementById("welcome");
  const chatScroll = document.getElementById("chatScroll");
  const composer = document.getElementById("composer");
  const input = document.getElementById("composerInput");
  const sendButton = document.getElementById("sendButton");
  const scrollBottomButton = document.getElementById("scrollBottomButton");
  const characterCount = document.getElementById("characterCount");
  const newChatButton = document.getElementById("newChatButton");
  const sidebarButton = document.getElementById("sidebarButton");
  const sidebarBackdrop = document.getElementById("sidebarBackdrop");
  const modelButton = document.getElementById("modelButton");
  const modelMenu = document.getElementById("modelMenu");
  const modelMode = document.getElementById("modelMode");
  const notice = document.getElementById("notice");
  const userTemplate = document.getElementById("userMessageTemplate");
  const assistantTemplate = document.getElementById("assistantMessageTemplate");
  const thinkingTemplate = document.getElementById("thinkingTemplate");

  const STORAGE_KEY = "mira-ui-session-v1";
  const state = {
    busy: false,
    connected: false,
    sessionId: null,
    maxQuestionCharacters: 500,
    messages: [],
    composing: false,
    noticeTimer: null,
  };

  async function request(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
    const payload = await response.json().catch(() => ({ ok: false, error: "Некорректный ответ сервера" }));
    if (!response.ok || !payload.ok) throw new Error(payload.error || "Ошибка соединения");
    return payload;
  }

  function setConnection(connected, text) {
    state.connected = connected;
    const local = document.getElementById("localStatus");
    local.querySelector(".status-dot").style.background = connected ? "" : "var(--danger)";
    local.querySelector("strong").textContent = connected ? "Mira готова" : "Нет связи";
    local.querySelector("small").textContent = connected ? "локальная модель" : "сервер недоступен";
  }

  function showNotice(message) {
    clearTimeout(state.noticeTimer);
    notice.textContent = message;
    notice.classList.add("visible");
    state.noticeTimer = setTimeout(() => notice.classList.remove("visible"), 4800);
  }

  function saveState() {
    if (!state.sessionId) return;
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ sessionId: state.sessionId, messages: state.messages }));
  }

  function restoreState(sessionId) {
    try {
      const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "null");
      if (!saved || saved.sessionId !== sessionId || !Array.isArray(saved.messages)) return;
      state.messages = saved.messages.filter((item) => item && ["user", "assistant"].includes(item.role));
      state.messages.forEach(renderSavedMessage);
    } catch {
      sessionStorage.removeItem(STORAGE_KEY);
    }
  }

  function updateWelcome() {
    welcome.classList.toggle("hidden", state.messages.length > 0 || state.busy);
  }

  function updateComposer() {
    const length = input.value.length;
    sendButton.disabled = state.busy || !state.connected || input.value.trim().length === 0;
    characterCount.textContent = length >= 420 ? `${length}/${state.maxQuestionCharacters}` : "";
    characterCount.classList.toggle("warning", length >= state.maxQuestionCharacters);
    input.style.height = "44px";
    input.style.height = `${Math.min(input.scrollHeight, 138)}px`;
  }

  function closeSidebar() {
    app.classList.remove("sidebar-open");
    sidebarButton.setAttribute("aria-expanded", "false");
  }

  function scrollToBottom(behavior = "smooth") {
    requestAnimationFrame(() => {
      chatScroll.scrollTo({ top: chatScroll.scrollHeight, behavior });
      setTimeout(updateScrollButton, behavior === "smooth" ? 220 : 0);
    });
  }

  function updateScrollButton() {
    const distance = chatScroll.scrollHeight - chatScroll.scrollTop - chatScroll.clientHeight;
    const visible = state.messages.length > 0 && distance > 140;
    scrollBottomButton.hidden = !visible;
    scrollBottomButton.classList.toggle("visible", visible);
  }

  function appendUser(text, save = true) {
    const row = userTemplate.content.firstElementChild.cloneNode(true);
    row.querySelector(".user-bubble").textContent = text;
    conversation.appendChild(row);
    if (save) state.messages.push({ role: "user", text });
    updateWelcome();
    scrollToBottom();
    return row;
  }

  function appendAssistant(text, seconds, save = true) {
    const row = assistantTemplate.content.firstElementChild.cloneNode(true);
    row.querySelector(".assistant-text").textContent = text;
    const time = row.querySelector(".response-time");
    const value = Number(seconds);
    time.textContent = Number.isFinite(value) ? `${value.toFixed(1).replace(".0", "")} с` : "";
    const copy = row.querySelector(".copy-button");
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(text);
        copy.setAttribute("aria-label", "Скопировано");
        setTimeout(() => copy.setAttribute("aria-label", "Скопировать ответ"), 1200);
      } catch {
        showNotice("Не получилось скопировать ответ");
      }
    });
    conversation.appendChild(row);
    if (save) state.messages.push({ role: "assistant", text, seconds: value || 0 });
    updateWelcome();
    scrollToBottom();
    return row;
  }

  function appendThinking() {
    const row = thinkingTemplate.content.firstElementChild.cloneNode(true);
    conversation.appendChild(row);
    updateWelcome();
    scrollToBottom();
    return row;
  }

  function renderSavedMessage(message) {
    if (message.role === "user") appendUser(String(message.text || ""), false);
    else appendAssistant(String(message.text || ""), message.seconds, false);
  }

  async function sendMessage(text = input.value) {
    const message = String(text).trim();
    if (!message || state.busy || !state.connected) return;
    if (message.length > state.maxQuestionCharacters) {
      showNotice(`Сообщение должно быть короче ${state.maxQuestionCharacters + 1} символа`);
      return;
    }

    state.busy = true;
    input.value = "";
    input.disabled = true;
    appendUser(message);
    const thinking = appendThinking();
    updateComposer();
    saveState();
    closeSidebar();
    const started = performance.now();

    try {
      const answer = await request("/api/message", {
        method: "POST",
        body: JSON.stringify({ message }),
      });
      const remaining = Math.max(0, 480 - (performance.now() - started));
      if (remaining) await new Promise((resolve) => setTimeout(resolve, remaining));
      thinking.remove();
      appendAssistant(answer.text, answer.seconds);
      setConnection(true, "Mira готова");
    } catch (error) {
      thinking.remove();
      showNotice(error.message || "Mira сейчас не отвечает");
      setConnection(false, "Нет связи");
    } finally {
      state.busy = false;
      input.disabled = false;
      updateComposer();
      saveState();
      input.focus();
    }
  }

  async function resetConversation() {
    if (state.busy) return;
    try {
      const result = await request("/api/reset", { method: "POST", body: "{}" });
      state.sessionId = result.sessionId;
      state.messages = [];
      conversation.querySelectorAll(".message-row").forEach((row) => row.remove());
      sessionStorage.removeItem(STORAGE_KEY);
      updateWelcome();
      closeSidebar();
      input.value = "";
      updateComposer();
      input.focus();
    } catch (error) {
      showNotice(error.message || "Не удалось начать новый диалог");
    }
  }

  composer.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage();
  });

  input.addEventListener("input", updateComposer);
  input.addEventListener("compositionstart", () => { state.composing = true; });
  input.addEventListener("compositionend", () => { state.composing = false; });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !state.composing) {
      event.preventDefault();
      sendMessage();
    }
  });

  newChatButton.addEventListener("click", resetConversation);
  sidebarButton.addEventListener("click", () => {
    const open = app.classList.toggle("sidebar-open");
    sidebarButton.setAttribute("aria-expanded", String(open));
  });
  sidebarBackdrop.addEventListener("click", closeSidebar);
  chatScroll.addEventListener("scroll", updateScrollButton, { passive: true });
  scrollBottomButton.addEventListener("click", () => scrollToBottom());

  modelButton.addEventListener("click", () => {
    const open = modelMenu.classList.toggle("open");
    modelButton.setAttribute("aria-expanded", String(open));
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest(".model-picker")) {
      modelMenu.classList.remove("open");
      modelButton.setAttribute("aria-expanded", "false");
    }
  });

  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => sendMessage(button.dataset.prompt));
  });

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeSidebar();
      modelMenu.classList.remove("open");
      modelButton.setAttribute("aria-expanded", "false");
    }
  });

  async function init() {
    updateComposer();
    try {
      const status = await request("/api/status");
      state.sessionId = status.sessionId;
      state.maxQuestionCharacters = status.maxQuestionCharacters || 500;
      input.maxLength = state.maxQuestionCharacters;
      modelMode.textContent = status.modeLabel;
      setConnection(true, "Mira готова");
      restoreState(status.sessionId);
      updateWelcome();
    } catch {
      setConnection(false, "Нет связи");
      showNotice("Локальный сервер Mira недоступен");
    }
    updateComposer();
    input.focus();
  }

  init();
})();
