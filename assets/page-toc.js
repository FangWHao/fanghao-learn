(function () {
  function ready(run) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", run);
    else run();
  }

  ready(function () {
    var article = document.querySelector("article");
    if (!article) return;

    var headings = Array.from(article.querySelectorAll("h2, h3")).filter(function (heading) {
      return !heading.closest(".toc");
    });
    if (headings.length < 2) return;

    var used = new Set(Array.from(document.querySelectorAll("[id]")).map(function (node) {
      return node.id;
    }));

    headings.forEach(function (heading, index) {
      if (heading.id) return;
      var id = "section-" + (index + 1);
      while (used.has(id)) id = id + "-x";
      heading.id = id;
      used.add(id);
    });

    var nav = document.createElement("nav");
    nav.className = "page-toc";
    nav.setAttribute("aria-label", "本页目录");
    nav.innerHTML = '<div class="page-toc-title">本页目录</div><ol class="page-toc-list"></ol>';

    var list = nav.querySelector(".page-toc-list");
    headings.forEach(function (heading) {
      var item = document.createElement("li");
      var link = document.createElement("a");
      item.className = "page-toc-item page-toc-" + heading.tagName.toLowerCase();
      link.className = "page-toc-link";
      link.href = "#" + heading.id;
      link.textContent = heading.textContent.trim();
      item.appendChild(link);
      list.appendChild(item);
    });

    var toggle = document.createElement("button");
    toggle.className = "page-toc-toggle";
    toggle.type = "button";
    toggle.textContent = "目录";
    toggle.setAttribute("aria-label", "打开本页目录");
    toggle.setAttribute("aria-expanded", "false");

    document.body.appendChild(nav);
    document.body.appendChild(toggle);

    toggle.addEventListener("click", function () {
      var isOpen = nav.classList.toggle("open");
      toggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
    });

    nav.addEventListener("click", function (event) {
      if (!event.target.closest("a")) return;
      nav.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
    });

    document.addEventListener("click", function (event) {
      if (!nav.classList.contains("open")) return;
      if (event.target.closest(".page-toc,.page-toc-toggle")) return;
      nav.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
    });

    var links = Array.from(nav.querySelectorAll(".page-toc-link"));
    var setActive = function (id) {
      links.forEach(function (link) {
        link.classList.toggle("active", link.getAttribute("href") === "#" + id);
      });
    };

    if ("IntersectionObserver" in window) {
      var visible = new Map();
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) visible.set(entry.target.id, entry.boundingClientRect.top);
          else visible.delete(entry.target.id);
        });
        var current = Array.from(visible.entries()).sort(function (a, b) {
          return a[1] - b[1];
        })[0];
        if (current) setActive(current[0]);
      }, { rootMargin: "-88px 0px -70% 0px", threshold: 0.01 });
      headings.forEach(function (heading) { observer.observe(heading); });
    }

    var updateFromScroll = function () {
      var current = headings[0];
      headings.forEach(function (heading) {
        if (heading.getBoundingClientRect().top <= 110) current = heading;
      });
      setActive(current.id);
    };
    updateFromScroll();
    window.addEventListener("scroll", updateFromScroll, { passive: true });
  });
})();
