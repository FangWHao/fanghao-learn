(function () {
  var key = "fhlearn:theme";
  var saved = "";
  try {
    saved = localStorage.getItem(key) || "";
  } catch (_) {}
  var dark = saved ? saved === "dark" : window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = dark ? "dark" : "light";

  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "theme-toggle";
    document.body.appendChild(btn);

    var draw = function () {
      var isDark = document.documentElement.dataset.theme === "dark";
      btn.textContent = isDark ? "☀ 日间" : "☾ 夜间";
      btn.setAttribute("aria-label", isDark ? "切换到日间模式" : "切换到黑夜模式");
    };

    btn.onclick = function () {
      var next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try {
        localStorage.setItem(key, next);
      } catch (_) {}
      draw();
    };

    draw();
  });
})();
