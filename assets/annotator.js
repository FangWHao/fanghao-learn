(function () {
  var root = null;
  var annotations = [];
  var lastSelection = null;
  var toolbar = null;
  var panel = null;
  var list = null;
  var editor = null;
  var editorText = null;
  var editingNoteId = null;
  var storeKey = "fhlearn:annotations:v1:" + location.pathname;
  var deniedSelector = "script,style,textarea,pre,code,button,svg,mjx-container,.pr-cell,.fh-annot-toolbar,.fh-annot-panel,.fh-annot-editor,.fh-annot-fab";

  function readStore() {
    try {
      return JSON.parse(localStorage.getItem(storeKey) || "[]").filter(function (item) {
        return Number.isFinite(item.start) && Number.isFinite(item.end) && item.end > item.start;
      });
    } catch (_) {
      return [];
    }
  }

  function writeStore() {
    try {
      localStorage.setItem(storeKey, JSON.stringify(annotations));
    } catch (_) {}
  }

  function uid() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  }

  function blocked(node) {
    var el = node.nodeType === 1 ? node : node.parentElement;
    return !el || Boolean(el.closest(deniedSelector));
  }

  function clearMarks() {
    root.querySelectorAll("span.fh-ann").forEach(function (span) {
      span.replaceWith(document.createTextNode(span.textContent || ""));
    });
    root.normalize();
  }

  function textNodes() {
    var nodes = [];
    var offset = 0;
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: function (node) {
        if (!node.nodeValue || blocked(node)) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    var node = walker.nextNode();
    while (node) {
      nodes.push({ node: node, start: offset, end: offset + node.nodeValue.length });
      offset += node.nodeValue.length;
      node = walker.nextNode();
    }
    return nodes;
  }

  function applyMarks() {
    clearMarks();
    if (!annotations.length) return;
    textNodes().forEach(function (entry) {
      var hits = annotations.filter(function (ann) {
        return ann.start < entry.end && ann.end > entry.start;
      });
      if (!hits.length) return;
      var cuts = [0, entry.node.nodeValue.length];
      hits.forEach(function (ann) {
        cuts.push(Math.max(0, ann.start - entry.start));
        cuts.push(Math.min(entry.node.nodeValue.length, ann.end - entry.start));
      });
      cuts = Array.from(new Set(cuts)).sort(function (a, b) { return a - b; });
      var frag = document.createDocumentFragment();
      for (var i = 0; i < cuts.length - 1; i += 1) {
        var from = cuts[i];
        var to = cuts[i + 1];
        var part = entry.node.nodeValue.slice(from, to);
        if (!part) continue;
        var globalStart = entry.start + from;
        var globalEnd = entry.start + to;
        var covers = hits.filter(function (ann) {
          return ann.start < globalEnd && ann.end > globalStart;
        });
        if (!covers.length) {
          frag.appendChild(document.createTextNode(part));
          continue;
        }
        var span = document.createElement("span");
        var types = covers.map(function (ann) { return ann.type; });
        span.className = "fh-ann" +
          (types.includes("highlight") ? " fh-ann-highlight" : "") +
          (types.includes("underline") ? " fh-ann-underline" : "") +
          (types.includes("note") ? " fh-ann-note" : "");
        span.dataset.ids = covers.map(function (ann) { return ann.id; }).join(",");
        var note = covers.find(function (ann) { return ann.type === "note" && ann.note; });
        if (note) span.title = note.note;
        span.textContent = part;
        frag.appendChild(span);
      }
      entry.node.replaceWith(frag);
    });
  }

  function selectionInfo() {
    var sel = window.getSelection();
    if (!sel || sel.rangeCount === 0 || sel.isCollapsed) return null;
    var range = sel.getRangeAt(0);
    if (!root.contains(range.commonAncestorContainer) || blocked(range.commonAncestorContainer)) return null;
    var nodes = textNodes();
    var start = null;
    var end = null;
    nodes.forEach(function (entry) {
      if (entry.node === range.startContainer) start = entry.start + range.startOffset;
      if (entry.node === range.endContainer) end = entry.start + range.endOffset;
    });
    if (start === null || end === null || end <= start) return null;
    var text = sel.toString().replace(/\s+/g, " ").trim();
    if (!text) return null;
    return { start: start, end: end, text: text, rect: range.getBoundingClientRect() };
  }

  function hideToolbar() {
    toolbar.classList.remove("open");
  }

  function showToolbar(info) {
    lastSelection = info;
    var left = Math.max(10, Math.min(Math.max(10, window.innerWidth - 260), info.rect.left + info.rect.width / 2 - 120));
    var top = Math.max(10, Math.min(Math.max(10, window.innerHeight - 56), info.rect.top - 48));
    toolbar.style.left = left + "px";
    toolbar.style.top = top + "px";
    toolbar.classList.add("open");
  }

  function saveAnnotation(type, note) {
    if (!lastSelection) return;
    annotations.push({
      id: uid(),
      type: type,
      start: lastSelection.start,
      end: lastSelection.end,
      text: lastSelection.text.slice(0, 280),
      note: note || "",
      createdAt: new Date().toISOString()
    });
    writeStore();
    applyMarks();
    renderList();
    hideToolbar();
    window.getSelection().removeAllRanges();
  }

  function typeLabel(type) {
    if (type === "highlight") return "高亮";
    if (type === "underline") return "下划线";
    return "注释";
  }

  function renderList() {
    if (!list) return;
    list.innerHTML = "";
    if (!annotations.length) {
      var empty = document.createElement("div");
      empty.className = "fh-annot-empty";
      empty.textContent = "选中文本后可以加高亮、下划线或注释。内容只保存在这个浏览器里。";
      list.appendChild(empty);
      return;
    }
    annotations.slice().reverse().forEach(function (ann) {
      var item = document.createElement("div");
      item.className = "fh-annot-item";
      item.innerHTML =
        '<div class="fh-annot-item-text"></div>' +
        (ann.note ? '<div class="fh-annot-item-note"></div>' : '') +
        '<div class="fh-annot-item-meta"><span></span><div class="fh-annot-mini-actions"><button class="fh-annot-mini" data-action="jump">定位</button><button class="fh-annot-mini" data-action="edit">编辑</button><button class="fh-annot-mini" data-action="delete">删除</button></div></div>';
      item.querySelector(".fh-annot-item-text").textContent = ann.text;
      if (ann.note) item.querySelector(".fh-annot-item-note").textContent = ann.note;
      item.querySelector(".fh-annot-item-meta span").textContent = typeLabel(ann.type);
      item.querySelector('[data-action="jump"]').onclick = function () { jumpTo(ann.id); };
      item.querySelector('[data-action="edit"]').onclick = function () { openEditor(ann.id); };
      item.querySelector('[data-action="delete"]').onclick = function () { removeAnnotation(ann.id); };
      list.appendChild(item);
    });
  }

  function jumpTo(id) {
    var mark = Array.from(root.querySelectorAll(".fh-ann")).find(function (el) {
      return (el.dataset.ids || "").split(",").includes(id);
    });
    if (!mark) return;
    mark.scrollIntoView({ behavior: "smooth", block: "center" });
    mark.animate([{ outline: "2px solid #2563eb" }, { outline: "2px solid transparent" }], { duration: 1200 });
  }

  function removeAnnotation(id) {
    annotations = annotations.filter(function (ann) { return ann.id !== id; });
    writeStore();
    applyMarks();
    renderList();
  }

  function openEditor(id) {
    editingNoteId = id || null;
    var ann = editingNoteId ? annotations.find(function (item) { return item.id === editingNoteId; }) : null;
    editorText.value = ann ? ann.note || "" : "";
    var rect = lastSelection ? lastSelection.rect : { left: window.innerWidth / 2 - 180, bottom: 160 };
    editor.style.left = Math.max(10, Math.min(window.innerWidth - 380, rect.left)) + "px";
    editor.style.top = Math.max(10, Math.min(window.innerHeight - 260, rect.bottom + 10)) + "px";
    editor.classList.add("open");
    editorText.focus();
  }

  function saveEditor() {
    var note = editorText.value.trim();
    if (editingNoteId) {
      annotations = annotations.map(function (ann) {
        if (ann.id === editingNoteId) ann.note = note;
        return ann;
      });
      writeStore();
      applyMarks();
      renderList();
    } else if (note) {
      saveAnnotation("note", note);
    }
    editor.classList.remove("open");
    editingNoteId = null;
  }

  function buildUi() {
    toolbar = document.createElement("div");
    toolbar.className = "fh-annot-toolbar";
    toolbar.innerHTML =
      '<button class="fh-annot-btn hot" data-type="highlight">高亮</button>' +
      '<button class="fh-annot-btn" data-type="underline">下划线</button>' +
      '<button class="fh-annot-btn" data-type="note">注释</button>';
    toolbar.addEventListener("mousedown", function (event) { event.preventDefault(); });
    toolbar.querySelector('[data-type="highlight"]').onclick = function () { saveAnnotation("highlight"); };
    toolbar.querySelector('[data-type="underline"]').onclick = function () { saveAnnotation("underline"); };
    toolbar.querySelector('[data-type="note"]').onclick = function () { openEditor(null); };
    document.body.appendChild(toolbar);

    var fab = document.createElement("button");
    fab.className = "fh-annot-fab";
    fab.textContent = "笔记";
    document.body.appendChild(fab);

    panel = document.createElement("aside");
    panel.className = "fh-annot-panel";
    panel.innerHTML =
      '<div class="fh-annot-panel-head"><div class="fh-annot-panel-title">本页笔记</div><button class="fh-annot-mini" data-action="export">导出</button><button class="fh-annot-mini" data-action="clear">清空</button><button class="fh-annot-mini" data-action="close">收起</button></div><div class="fh-annot-list"></div>';
    list = panel.querySelector(".fh-annot-list");
    panel.querySelector('[data-action="close"]').onclick = function () { panel.classList.remove("open"); };
    panel.querySelector('[data-action="clear"]').onclick = function () {
      if (!annotations.length || !confirm("清空本页所有笔记？")) return;
      annotations = [];
      writeStore();
      applyMarks();
      renderList();
    };
    panel.querySelector('[data-action="export"]').onclick = function () {
      var blob = new Blob([JSON.stringify(annotations, null, 2)], { type: "application/json" });
      var a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "learn-notes-" + location.pathname.replace(/[^\w]+/g, "-") + ".json";
      a.click();
      URL.revokeObjectURL(a.href);
    };
    document.body.appendChild(panel);
    fab.onclick = function () { panel.classList.toggle("open"); };

    editor = document.createElement("div");
    editor.className = "fh-annot-editor";
    editor.innerHTML =
      '<div class="fh-annot-editor-head">写注释</div><textarea placeholder="写一点自己的理解、疑问、例子..."></textarea><div class="fh-annot-editor-actions"><button class="fh-annot-btn" data-action="cancel">取消</button><button class="fh-annot-btn hot" data-action="save">保存</button></div>';
    editorText = editor.querySelector("textarea");
    editor.querySelector('[data-action="cancel"]').onclick = function () {
      editor.classList.remove("open");
      editingNoteId = null;
    };
    editor.querySelector('[data-action="save"]').onclick = saveEditor;
    document.body.appendChild(editor);
  }

  function init() {
    root = document.querySelector("article") || document.querySelector(".doc-main");
    if (!root) return;
    buildUi();
    annotations = readStore();
    applyMarks();
    renderList();

    document.addEventListener("mouseup", function () {
      setTimeout(function () {
        var info = selectionInfo();
        if (info) showToolbar(info);
      }, 0);
    });
    document.addEventListener("keyup", function () {
      var info = selectionInfo();
      if (info) showToolbar(info);
    });
    document.addEventListener("mousedown", function (event) {
      if (!event.target.closest(".fh-annot-toolbar,.fh-annot-editor")) hideToolbar();
    });
    root.addEventListener("click", function (event) {
      var mark = event.target.closest(".fh-ann-note");
      if (!mark) return;
      var ids = (mark.dataset.ids || "").split(",");
      var ann = annotations.find(function (item) { return ids.includes(item.id) && item.type === "note"; });
      if (ann) {
        event.preventDefault();
        event.stopPropagation();
        openEditor(ann.id);
      }
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
