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

        fetchScreenedStocks();

        async function fetchScreenedStocks() {
            try {
                var res = await fetch("/screened");
                var data = await res.json();
                var stocks = data.stocks || [];
                if (stocks.length > 0) {
                    renderStocks(stocks);
                } else {
                    // Fallback to signals if no screened stocks yet
                    fetchSignals();
                }
            } catch (err) {
                fetchSignals();
            }
        }

        async function fetchSignals() {
            try {
                var res = await fetch("/signals");
                var data = await res.json();
                var signals = data.signals || [];
                var top3 = signals.slice(0, 3);
                renderLegacyTickers(top3);
            } catch (err) {
                container.innerHTML =
                    '<div class="empty-state">' +
                    '<div class="icon">&#9888;</div>' +
                    '<div class="title">Connection Error</div>' +
                    '<div class="subtitle">Could not reach the SignalBoard API.</div>' +
                    "</div>";
            }
        }

        function renderStocks(stocks) {
            if (stocks.length === 0) {
                container.innerHTML =
                    '<div class="empty-state">' +
                    '<div class="icon">&#8212;</div>' +
                    '<div class="title">No Stocks Screened</div>' +
                    '<div class="subtitle">The screener has not run yet.</div>' +
                    "</div>";
                return;
            }

            var html = '<hr class="rule">';
            for (var i = 0; i < stocks.length; i++) {
                html += renderStockRow(stocks[i], i);
                html += '<hr class="rule">';
            }
            container.innerHTML = html;
        }

        function renderStockRow(s, index) {
            var score = s.composite_score != null ? Math.round(s.composite_score * 100) : "—";
            var rank = s.rank || (index + 1);

            var html = '<a class="ticker-row" href="/detail.html?ticker=' + encodeURIComponent(s.ticker) + '&source=screened">';
            html += '<div class="row-main">';
            html += '<span class="rank">#' + rank + '</span>';
            html += '<div class="row-info">';
            html += '<span class="symbol">' + escapeHtml(s.ticker) + '</span>';
            html += '<span class="name-label">' + escapeHtml(s.short_name || "") + '</span>';
            if (s.market_cap) {
                html += '<span class="market-cap-label">' + formatMarketCap(s.market_cap) + '</span>';
            }
            html += '</div>';
            html += '<span class="score">' + score + '</span>';
            html += '</div>';

            // Mini score bars
            html += '<div class="score-bars">';
            html += miniBar("Piotroski", s.piotroski_score);
            html += miniBar("Cash Flow", s.cash_flow_score);
            html += miniBar("DCF", s.dcf_score);
            html += miniBar("Balance Sheet", s.balance_sheet_score);
            html += '</div>';

            html += '<span class="industry">' + escapeHtml(s.sector || "") + '</span>';
            html += '</a>';
            return html;
        }

        function miniBar(label, value) {
            var pct = value != null ? Math.round(value * 100) : 0;
            return (
                '<div class="mini-bar">' +
                '<span class="mini-label">' + label + '</span>' +
                '<div class="mini-track"><div class="mini-fill" style="width:' + pct + '%"></div></div>' +
                '</div>'
            );
        }

        function renderLegacyTickers(signals) {
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
                html += '<a class="ticker-row" href="/detail.html?ticker=' + encodeURIComponent(signals[i].ticker) + '">';
                html += '<div class="row-main">';
                html += '<div class="row-info">';
                html += '<span class="symbol">' + escapeHtml(signals[i].ticker) + '</span>';
                html += '</div>';
                html += '</div>';
                html += '<span class="industry">' + escapeHtml(signals[i].sector || "") + '</span>';
                html += '</a><hr class="rule">';
            }
            container.innerHTML = html;
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
        var source = params.get("source");

        if (!ticker) {
            contentEl.innerHTML =
                '<div class="empty-state"><div class="icon">&#10067;</div>' +
                '<div class="title">No Ticker</div>' +
                '<div class="subtitle">No ticker specified in the URL.</div></div>';
            return;
        }

        titleEl.textContent = ticker.toUpperCase();

        if (source === "screened") {
            fetchScreenedDetail(ticker);
        } else {
            fetchLegacyDetail(ticker);
        }

        // ── Screened stock detail ──────────────────────────────

        async function fetchScreenedDetail(ticker) {
            contentEl.innerHTML = '<div class="loading"></div>';
            try {
                var res = await fetch("/screened/" + encodeURIComponent(ticker));
                if (!res.ok) throw new Error("Not found");
                var data = await res.json();
                renderScreenedDetail(data);
            } catch (err) {
                // Fallback to legacy detail
                fetchLegacyDetail(ticker);
            }
        }

        function renderScreenedDetail(s) {
            var analysis = s.analysis || {};
            var html = "";

            // Header with composite score
            html += '<div class="detail-header">';
            html += '<div class="ticker">' + escapeHtml(s.ticker) + '</div>';
            html += '<span class="name">' + escapeHtml(s.short_name || "") + '</span>';
            if (s.market_cap) {
                html += '<span class="market-cap-label">' + formatMarketCap(s.market_cap) + '</span>';
            }
            html += '<span class="sector-label">' + escapeHtml(s.sector || "") + '</span>';
            if (s.industry && s.industry !== s.sector) {
                html += '<span class="sector-label">' + escapeHtml(s.industry) + '</span>';
            }
            html += '<div class="composite-score">';
            html += '<span class="composite-value">' + Math.round(s.composite_score * 100) + '</span>';
            html += '<span class="composite-label">Composite Score</span>';
            html += '</div>';
            html += '</div>';

            // Why This Stock Was Selected
            var reasons = analysis.reasons || [];
            if (reasons.length > 0) {
                html += '<div class="explanation-section section-reasons">';
                html += '<div class="section-title">Why This Stock Was Selected</div>';
                html += '<ul>';
                for (var i = 0; i < reasons.length; i++) {
                    html += '<li>' + escapeHtml(reasons[i]) + '</li>';
                }
                html += '</ul></div>';
            }

            // Component Score Breakdown
            html += '<div class="explanation-section section-scores">';
            html += '<div class="section-title">Score Breakdown</div>';
            html += '<div class="score-grid">';
            html += scoreCard("Piotroski", s.piotroski_score, fmtFScore(analysis.piotroski_f_score));
            html += scoreCard("ROIC Spread", s.roic_spread_score, fmtPct(analysis.roic_vs_wacc_spread));
            html += scoreCard("Cash Flow", s.cash_flow_score, fmtRatio(analysis.fcf_to_net_income, "x NI"));
            html += scoreCard("Balance Sheet", s.balance_sheet_score, fmtAltman(analysis.altman_z_score));
            html += scoreCard("DCF Valuation", s.dcf_score, fmtPct(analysis.dcf_upside_pct, true));
            html += scoreCard("Under-Radar", s.blindspot_score, fmtAnalysts(analysis.analyst_count));
            html += scoreCard("Margin Trend", s.margin_score, null);
            html += '</div></div>';

            // DCF Valuation Detail
            if (analysis.intrinsic_value_per_share != null) {
                html += '<div class="explanation-section section-dcf">';
                html += '<div class="section-title">DCF Valuation</div>';
                html += '<div class="metric-grid">';
                html += metricRow("Intrinsic Value", "$" + analysis.intrinsic_value_per_share.toFixed(2));
                if (analysis.dcf_margin_of_safety != null)
                    html += metricRow("Margin of Safety", (analysis.dcf_margin_of_safety * 100).toFixed(1) + "%");
                if (analysis.dcf_upside_pct != null)
                    html += metricRow("Upside Potential", (analysis.dcf_upside_pct > 0 ? "+" : "") + (analysis.dcf_upside_pct * 100).toFixed(1) + "%");
                if (analysis.wacc != null)
                    html += metricRow("WACC", (analysis.wacc * 100).toFixed(1) + "%");
                if (analysis.fcf_yield != null)
                    html += metricRow("FCF Yield", (analysis.fcf_yield * 100).toFixed(1) + "%");
                if (analysis.ev_to_fcf != null)
                    html += metricRow("EV/FCF", analysis.ev_to_fcf.toFixed(1) + "x");
                html += '</div></div>';
            }

            // Key Metrics
            html += '<div class="explanation-section section-metrics">';
            html += '<div class="section-title">Key Metrics</div>';
            html += '<div class="metric-grid">';
            if (analysis.pe_ratio != null) html += metricRow("P/E Ratio", analysis.pe_ratio.toFixed(1));
            if (analysis.pb_ratio != null) html += metricRow("P/B Ratio", analysis.pb_ratio.toFixed(2));
            if (analysis.roe != null) html += metricRow("ROE", (analysis.roe * 100).toFixed(1) + "%");
            if (analysis.debt_to_equity != null) html += metricRow("D/E Ratio", analysis.debt_to_equity.toFixed(2));
            if (analysis.dividend_yield != null) html += metricRow("Div Yield", (analysis.dividend_yield * 100).toFixed(2) + "%");
            if (analysis.market_cap != null) html += metricRow("Market Cap", fmtMarketCap(analysis.market_cap));
            if (analysis.interest_coverage != null) html += metricRow("Interest Coverage", analysis.interest_coverage.toFixed(1) + "x");
            if (analysis.accruals_ratio != null) html += metricRow("Accruals Ratio", analysis.accruals_ratio.toFixed(3));
            html += '</div></div>';

            // Institutional Signals
            if (analysis.analyst_count != null || analysis.inst_ownership_pct != null || analysis.insider_cluster_buy) {
                html += '<div class="explanation-section section-inst">';
                html += '<div class="section-title">Institutional Signals</div>';
                html += '<div class="metric-grid">';
                if (analysis.analyst_count != null) html += metricRow("Analyst Coverage", analysis.analyst_count + " analysts");
                if (analysis.inst_ownership_pct != null) html += metricRow("Inst. Ownership", (analysis.inst_ownership_pct * 100).toFixed(1) + "%");
                if (analysis.insider_cluster_buy) html += metricRow("Insider Activity", "Cluster buying detected");
                html += '</div></div>';
            }

            contentEl.innerHTML = html;
        }

        // ── Score card helpers ─────────────────────────────────

        function scoreCard(label, score, detail) {
            var pct = score != null ? Math.round(score * 100) : 0;
            var html = '<div class="score-card">';
            html += '<div class="sc-header">';
            html += '<span class="sc-label">' + label + '</span>';
            html += '<span class="sc-pct">' + pct + '</span>';
            html += '</div>';
            html += '<div class="sc-bar"><div class="sc-fill" style="width:' + pct + '%"></div></div>';
            if (detail) {
                html += '<span class="sc-detail">' + escapeHtml(detail) + '</span>';
            }
            html += '</div>';
            return html;
        }

        function metricRow(label, value) {
            return '<div class="metric-row"><span class="metric-label">' + label + '</span><span class="metric-value">' + escapeHtml(String(value)) + '</span></div>';
        }

        function fmtFScore(v) {
            return v != null ? v + "/9" : null;
        }

        function fmtPct(v, showPlus) {
            if (v == null) return null;
            var s = (v * 100).toFixed(1) + "%";
            return (showPlus && v > 0 ? "+" : "") + s;
        }

        function fmtRatio(v, suffix) {
            if (v == null) return null;
            return v.toFixed(1) + suffix;
        }

        function fmtAltman(v) {
            if (v == null) return null;
            return "Z=" + v.toFixed(1);
        }

        function fmtAnalysts(v) {
            if (v == null) return null;
            return v + " analysts";
        }

        function fmtMarketCap(v) {
            if (v == null) return "—";
            if (v >= 1e12) return "$" + (v / 1e12).toFixed(1) + "T";
            if (v >= 1e9) return "$" + (v / 1e9).toFixed(1) + "B";
            if (v >= 1e6) return "$" + (v / 1e6).toFixed(0) + "M";
            return "$" + v.toLocaleString();
        }

        // ── Legacy signal detail ───────────────────────────────

        async function fetchLegacyDetail(ticker) {
            contentEl.innerHTML = '<div class="loading"></div>';
            try {
                var res = await fetch("/signals/" + encodeURIComponent(ticker));
                if (!res.ok) throw new Error("Not found");
                var data = await res.json();
                renderLegacyDetail(data);
            } catch (err) {
                contentEl.innerHTML =
                    '<div class="empty-state"><div class="icon">&#9888;</div>' +
                    '<div class="title">Signal Not Found</div>' +
                    '<div class="subtitle">No data for ' + escapeHtml(ticker) + '.</div></div>';
            }
        }

        function renderLegacyDetail(s) {
            var conf = Math.round(s.confidence * 100);
            var html = "";

            html += '<div class="detail-header">';
            html += '<div class="ticker">' + escapeHtml(s.ticker) + '</div>';
            html += '<span class="name">' + escapeHtml(s.short_name || "") + '</span>';
            html += '<span class="sector-label">' + escapeHtml(s.sector || "") + '</span>';
            html += '<div class="confidence-bar">';
            html += '<span class="confidence-value">' + conf + '% confidence</span>';
            html += '</div></div>';

            if (s.ml_insight) {
                html += renderTextSection("Why This Stock", "ml", s.ml_insight);
            }
            if (s.fundamental && s.fundamental.points) {
                html += renderSection("Fundamentals", "fundamental", s.fundamental.points);
            }
            if (s.macro && s.macro.points) {
                html += renderSection("Industry & Macro Context", "macro", s.macro.points);
            }
            if (s.risk_context) {
                html += renderTextSection("Risk Considerations", "risk", s.risk_context);
            }
            if (s.historical_context) {
                html += renderTextSection("Historical Precedent", "history", s.historical_context);
            }

            contentEl.innerHTML = html;
        }

        function renderSection(title, cls, points) {
            var html = '<div class="explanation-section section-' + cls + '">';
            html += '<div class="section-title">' + title + '</div>';
            html += '<ul>';
            for (var i = 0; i < points.length; i++) {
                html += '<li>' + escapeHtml(points[i]) + '</li>';
            }
            html += '</ul></div>';
            return html;
        }

        function renderTextSection(title, cls, text) {
            return (
                '<div class="explanation-section section-' + cls + '">' +
                '<div class="section-title">' + title + '</div>' +
                '<p>' + escapeHtml(text) + '</p></div>'
            );
        }
    }

    // ── Helpers ──────────────────────────────────────────────

    function formatMarketCap(cap) {
        if (cap == null) return "";
        if (cap >= 1e9) return "$" + (cap / 1e9).toFixed(1) + "B";
        if (cap >= 1e6) return "$" + (cap / 1e6).toFixed(0) + "M";
        return "";
    }

    function escapeHtml(str) {
        var div = document.createElement("div");
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }
})();
