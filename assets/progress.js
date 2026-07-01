(function () {
  var storageKey = "fhlearn:readmap:v1";
  var bucketCount = 72;
  var tickMs = 1000;
  var lastTick = Date.now();
  var saveTimer = 0;

  function pageKey() {
    var path = location.pathname.replace(/^\/+/, "") || "index.html";
    return path.replace(/\/$/, "/index.html");
  }

  function loadState() {
    try {
      var data = JSON.parse(localStorage.getItem(storageKey) || "{}");
      if (!data.pages) data.pages = {};
      return data;
    } catch (_) {
      return { pages: {} };
    }
  }

  function saveState(state) {
    try {
      localStorage.setItem(storageKey, JSON.stringify(state));
    } catch (_) {}
  }

  function pageHeight() {
    var body = document.body;
    var doc = document.documentElement;
    return Math.max(body.scrollHeight, body.offsetHeight, doc.clientHeight, doc.scrollHeight, doc.offsetHeight);
  }

  function scrollMax() {
    return Math.max(1, pageHeight() - window.innerHeight);
  }

  function currentBucket() {
    var focusY = window.scrollY + window.innerHeight * 0.38;
    return Math.max(0, Math.min(bucketCount - 1, Math.floor(focusY / Math.max(1, pageHeight()) * bucketCount)));
  }

  function ensurePage(state, key) {
    if (!state.pages[key]) {
      state.pages[key] = {
        title: document.title || key,
        visits: 0,
        totalMs: 0,
        lastY: 0,
        maxRatio: 0,
        lastAt: "",
        buckets: []
      };
    }
    while (state.pages[key].buckets.length < bucketCount) state.pages[key].buckets.push(0);
    if (state.pages[key].buckets.length > bucketCount) state.pages[key].buckets.length = bucketCount;
    return state.pages[key];
  }

  function formatTime(ms) {
    var sec = Math.round(ms / 1000);
    if (sec < 60) return sec + " 秒";
    var min = Math.floor(sec / 60);
    var left = sec % 60;
    if (min < 60) return min + " 分 " + left + " 秒";
    return Math.floor(min / 60) + " 小时 " + (min % 60) + " 分";
  }

  function saveSoon(state) {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(function () { saveState(state); }, 250);
  }

  function heatColor(value, max) {
    if (!max) return 0.13;
    return Math.max(0.13, Math.min(1, 0.18 + Math.log(1 + value) / Math.log(1 + max) * 0.82));
  }

  function hottestRange(page) {
    var max = Math.max.apply(null, page.buckets);
    if (!max) return "还没有明显热点";
    var index = page.buckets.indexOf(max);
    var start = Math.round(index / bucketCount * 100);
    var end = Math.round((index + 1) / bucketCount * 100);
    return start + "% - " + end + "%";
  }

  function renderPanel(panel, page, state, key) {
    var max = Math.max.apply(null, page.buckets);
    var maxRatio = Math.round((page.maxRatio || 0) * 100);
    panel.innerHTML =
      '<div class="fh-readmap-panel-head">' +
      '<strong>阅读轨迹</strong>' +
      '<button type="button" class="fh-readmap-close" aria-label="关闭">×</button>' +
      '</div>' +
      '<div class="fh-readmap-panel-body">' +
      '<div class="fh-readmap-row"><span>本页停留</span><b>' + formatTime(page.totalMs || 0) + '</b></div>' +
      '<div class="fh-readmap-row"><span>最深位置</span><b>' + maxRatio + '%</b></div>' +
      '<div class="fh-readmap-row"><span>停留最多</span><b>' + hottestRange(page) + '</b></div>' +
      '<div class="fh-readmap-mini" aria-label="本页停留热图"></div>' +
      '<div class="fh-readmap-actions">' +
      '<button type="button" class="fh-readmap-restore">回到上次位置</button>' +
      '<button type="button" class="fh-readmap-clear">清空本页</button>' +
      '</div>' +
      '</div>';
    var mini = panel.querySelector(".fh-readmap-mini");
    page.buckets.forEach(function (value) {
      var block = document.createElement("span");
      block.style.opacity = heatColor(value, max);
      mini.appendChild(block);
    });
    panel.querySelector(".fh-readmap-close").addEventListener("click", function () {
      panel.classList.remove("open");
    });
    panel.querySelector(".fh-readmap-restore").addEventListener("click", function () {
      window.scrollTo({ top: page.lastY || 0, behavior: "smooth" });
      panel.classList.remove("open");
    });
    panel.querySelector(".fh-readmap-clear").addEventListener("click", function () {
      delete state.pages[key];
      saveState(state);
      location.reload();
    });
  }

  function ready(run) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", run);
    else run();
  }

  ready(function () {
    if (!document.body) return;

    var key = pageKey();
    var state = loadState();
    var page = ensurePage(state, key);
    page.title = document.title || page.title || key;
    page.visits = (page.visits || 0) + 1;
    page.lastAt = new Date().toISOString();
    saveState(state);

    var rail = document.createElement("button");
    rail.type = "button";
    rail.className = "fh-readmap-rail";
    rail.setAttribute("aria-label", "打开阅读轨迹热图");
    rail.title = "阅读轨迹";
    for (var i = 0; i < bucketCount; i += 1) rail.appendChild(document.createElement("span"));

    var panel = document.createElement("div");
    panel.className = "fh-readmap-panel";
    document.body.appendChild(rail);
    document.body.appendChild(panel);

    var draw = function () {
      var max = Math.max.apply(null, page.buckets);
      var active = currentBucket();
      Array.from(rail.children).forEach(function (segment, index) {
        segment.style.opacity = heatColor(page.buckets[index], max);
        segment.classList.toggle("active", index === active);
      });
      if (panel.classList.contains("open")) renderPanel(panel, page, state, key);
    };

    var record = function () {
      var now = Date.now();
      var elapsed = Math.min(5000, Math.max(0, now - lastTick));
      lastTick = now;
      if (document.visibilityState !== "visible") return;
      var index = currentBucket();
      page.buckets[index] += elapsed;
      page.totalMs = (page.totalMs || 0) + elapsed;
      page.lastY = Math.round(window.scrollY);
      page.maxRatio = Math.max(page.maxRatio || 0, Math.min(1, (window.scrollY + window.innerHeight) / Math.max(1, pageHeight())));
      page.lastAt = new Date().toISOString();
      draw();
      saveSoon(state);
    };

    rail.addEventListener("click", function () {
      record();
      renderPanel(panel, page, state, key);
      panel.classList.toggle("open");
    });
    document.addEventListener("click", function (event) {
      if (!panel.classList.contains("open")) return;
      if (event.target.closest(".fh-readmap-panel,.fh-readmap-rail")) return;
      panel.classList.remove("open");
    });
    document.addEventListener("visibilitychange", function () {
      record();
      saveState(state);
    });
    window.addEventListener("pagehide", function () {
      record();
      saveState(state);
    });
    window.addEventListener("beforeunload", function () {
      record();
      saveState(state);
    });
    window.addEventListener("scroll", draw, { passive: true });
    window.addEventListener("resize", draw);

    draw();
    setInterval(record, tickMs);
  });
})();
