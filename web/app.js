/* SignalBoard — minimalist frontend */

(function () {
    "use strict";

    var page = document.body.dataset.page;

    if (page === "list") initListPage();
    else if (page === "detail") initDetailPage();

    // ── List Page ────────────────────────────────────────────

    function initListPage() {
        var container = document.getElementById("ticker-container");
        var aboutBtn = document.getElementById("about-btn");
        var aboutOverlay = document.getElementById("about-overlay");
        var aboutClose = document.getElementById("about-close");

        // About toggle
        aboutBtn.addEventListener("click", function () {
            aboutOverlay.classList.add("open");
        });

        function closeAbout() {
            aboutOverlay.classList.remove("open");
        }

        aboutClose.addEventListener("click", closeAbout);
        aboutOverlay.addEventListener("click", function (e) {
            if (e.target === aboutOverlay) closeAbout();
        });

        // Fetch top 3 signals
        fetchSignals();

        async function fetchSignals() {
            try {
                var res = await fetch("/signals");
                var data = await res.json();
                var signals = data.signals || [];
                var top3 = signals.slice(0, 3);
                renderTickers(top3);
            } catch (err) {
                container.innerHTML =
                    '<div class="empty-state">' +
                    '<div class="icon">&#9888;</div>' +
                    '<div class="title">Connection Error</div>' +
                    '<div class="subtitle">Could not reach the SignalBoard API.</div>' +
                    "</div>";
            }
        }

        function renderTickers(signals) {
            if (signals.length === 0) {
                container.innerHTML =
                    '<div class="empty-state">' +
                    '<div class="icon">&#8212;</div>' +
                    '<div class="title">No Signals</div>' +
                    '<div class="subtitle">The algorithm has not generated signals yet.</div>' +
                    "</div>";
                return;
            }

            var html = '<hr class="rule">';
            for (var i = 0; i < signals.length; i++) {
                html += renderTickerRow(signals[i]);
                html += '<hr class="rule">';
            }
            container.innerHTML = html;
        }

        function renderTickerRow(s) {
            return (
                '<a class="ticker-row" href="/detail.html?ticker=' + encodeURIComponent(s.ticker) + '">' +
                '<span class="symbol">' + escapeHtml(s.ticker) + "</span>" +
                '<span class="industry">' + escapeHtml(s.sector || "") + "</span>" +
                "</a>"
            );
        }
    }

    // ── Detail Page ──────────────────────────────────────────

    function initDetailPage() {
        var contentEl = document.getElementById("detail-content");
        var titleEl = document.getElementById("detail-title");
        var backBtn = document.getElementById("back-btn");

        backBtn.addEventListener("click", function (e) {
            e.preventDefault();
            if (window.history.length > 1) {
                window.history.back();
            } else {
                window.location.href = "/";
            }
        });

        var params = new URLSearchParams(window.location.search);
        var ticker = params.get("ticker");

        if (!ticker) {
            contentEl.innerHTML =
                '<div class="empty-state"><div class="icon">&#10067;</div>' +
                '<div class="title">No Ticker</div>' +
                '<div class="subtitle">No ticker specified in the URL.</div></div>';
            return;
        }

        titleEl.textContent = ticker.toUpperCase();
        fetchDetail(ticker);

        async function fetchDetail(ticker) {
            contentEl.innerHTML = '<div class="loading"></div>';
            try {
                var res = await fetch("/signals/" + encodeURIComponent(ticker));
                if (!res.ok) throw new Error("Not found");
                var data = await res.json();
                renderDetail(data);
            } catch (err) {
                contentEl.innerHTML =
                    '<div class="empty-state"><div class="icon">&#9888;</div>' +
                    '<div class="title">Signal Not Found</div>' +
                    '<div class="subtitle">No data for ' + escapeHtml(ticker) + ".</div></div>";
            }
        }

        function renderDetail(s) {
            var conf = Math.round(s.confidence * 100);
            var html = "";

            // Header
            html += '<div class="detail-header">';
            html += '<div class="ticker">' + escapeHtml(s.ticker) + "</div>";
            html += '<span class="name">' + escapeHtml(s.short_name || "") + "</span>";
            html += '<span class="sector-label">' + escapeHtml(s.sector || "") + "</span>";
            html += '<div class="confidence-bar">';
            html += '<span class="confidence-value">' + conf + "% confidence</span>";
            html += "</div></div>";

            // ML Insight — lead with what the model saw
            if (s.ml_insight) {
                html += renderTextSection("Why This Stock", "ml", s.ml_insight);
            }

            // Fundamental Analysis
            if (s.fundamental && s.fundamental.points) {
                html += renderSection("Fundamentals", "fundamental", s.fundamental.points);
            }

            // Macro / Industry context
            if (s.macro && s.macro.points) {
                html += renderSection("Industry & Macro Context", "macro", s.macro.points);
            }

            // Risk context
            if (s.risk_context) {
                html += renderTextSection("Risk Considerations", "risk", s.risk_context);
            }

            // Historical context
            if (s.historical_context) {
                html += renderTextSection("Historical Precedent", "history", s.historical_context);
            }

            contentEl.innerHTML = html;
        }

        function renderSection(title, cls, points) {
            var html = '<div class="explanation-section section-' + cls + '">';
            html += '<div class="section-title">' + title + "</div>";
            html += "<ul>";
            for (var i = 0; i < points.length; i++) {
                html += "<li>" + escapeHtml(points[i]) + "</li>";
            }
            html += "</ul></div>";
            return html;
        }

        function renderTextSection(title, cls, text) {
            return (
                '<div class="explanation-section section-' + cls + '">' +
                '<div class="section-title">' + title + "</div>" +
                "<p>" + escapeHtml(text) + "</p></div>"
            );
        }
    }

    // ── Helpers ──────────────────────────────────────────────

    function escapeHtml(str) {
        var div = document.createElement("div");
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }
})();
