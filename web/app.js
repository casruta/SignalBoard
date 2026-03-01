/* SignalBoard — minimalist frontend */

(function () {
    "use strict";

    var page = document.body.dataset.page;

    if (page === "list") initListPage();
    else if (page === "detail") initDetailPage();

    // ── List Page ────────────────────────────────────────────

    function initListPage() {
        var container = document.getElementById("ticker-container");
        // Wire nav buttons to overlays
        document.querySelectorAll(".nav-btn").forEach(function (btn) {
            btn.addEventListener("click", function () {
                var id = btn.getAttribute("data-overlay");
                var overlay = document.getElementById(id);
                if (overlay) overlay.classList.add("open");
            });
        });

        document.querySelectorAll(".overlay").forEach(function (overlay) {
            var closeBtn = overlay.querySelector(".overlay-close");
            if (closeBtn) {
                closeBtn.addEventListener("click", function () {
                    overlay.classList.remove("open");
                });
            }
            overlay.addEventListener("click", function (e) {
                if (e.target === overlay) overlay.classList.remove("open");
            });
        });

        // Animate methodology overlay components on open
        document.querySelectorAll(".nav-btn").forEach(function (btn) {
            btn.addEventListener("click", function () {
                var id = btn.getAttribute("data-overlay");
                if (id === "methodology-overlay") {
                    setTimeout(function () {
                        // Animate scoring dimension bars
                        document.querySelectorAll(".dim-fill").forEach(function (fill) {
                            var w = fill.getAttribute("data-width");
                            if (w) fill.style.width = w + "%";
                        });
                        // Stagger funnel rows
                        document.querySelectorAll(".funnel-row").forEach(function (row, i) {
                            setTimeout(function () {
                                row.style.transition = "opacity 0.4s ease, transform 0.4s ease";
                                row.style.opacity = "1";
                                row.style.transform = "translateY(0)";
                            }, i * 100);
                        });
                    }, 150);
                }
            });
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

            // Pick generation date
            if (stocks.length > 0 && stocks[0].generated_at) {
                var d = new Date(stocks[0].generated_at);
                var months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
                var dateStr = months[d.getMonth()] + " " + d.getDate() + ", " + d.getFullYear();
                html += '<div class="pick-date">Picks generated ' + dateStr + '</div>';
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
            html += miniBar("ROIC Spread", s.roic_spread_score);
            html += miniBar("Balance Sheet", s.balance_sheet_score);
            html += miniBar("DCF", s.dcf_score);
            html += miniBar("Growth", s.growth_score);
            html += miniBar("Margins", s.margin_score);
            html += miniBar("Blindspot", s.blindspot_score);
            if (s.stock_category === 'dividend' && s.analysis && s.analysis.dividend_yield != null) {
                html += '<div class="mini-bar"><span class="div-yield-indicator">DIV YIELD ' + (s.analysis.dividend_yield * 100).toFixed(1) + '%</span></div>';
            } else if (s.stock_category === 'dividend' && s.dividend_yield != null) {
                html += '<div class="mini-bar"><span class="div-yield-indicator">DIV YIELD ' + (s.dividend_yield * 100).toFixed(1) + '%</span></div>';
            }
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

        fetchReport(ticker);

        async function fetchReport(ticker) {
            contentEl.innerHTML = '<div class="loading"></div>';
            try {
                var res = await fetch("/report/" + encodeURIComponent(ticker));
                if (!res.ok) throw new Error("Not found");
                var data = await res.json();
                contentEl.innerHTML = renderReport(data);
                // Section summary toggle on header click
                contentEl.addEventListener('click', function(e) {
                    var header = e.target.closest('.report-section-header[data-summary]');
                    if (!header) return;
                    var id = header.getAttribute('data-summary');
                    var panel = document.getElementById('summary-' + id);
                    if (!panel) return;
                    panel.classList.toggle('open');
                    header.classList.toggle('summary-active');
                });
                var tocEl = document.getElementById("report-toc");
                if (tocEl) renderTOC(tocEl);
            } catch (err) {
                console.error("Report render failed:", err);
                // Fallback: try screened detail, then legacy
                if (source === "screened") {
                    fetchScreenedDetail(ticker);
                } else {
                    fetchLegacyDetail(ticker);
                }
            }
        }

        // ── Report rendering ─────────────────────────────────────

        function renderReport(data) {
            var sections = [
                { id: "thesis", title: "Investment Thesis", html: renderThesis(data.thesis) },
                { id: "snapshot", title: "Market Snapshot", html: renderSnapshot(data.snapshot) },
                { id: "income", title: "Income Statement", html: renderIncomeStatement(data.income_statement) },
                { id: "balance", title: "Balance Sheet", html: renderBalanceSheet(data.balance_sheet) },
                { id: "capital", title: "Capital Structure", html: renderCapitalStructure(data.capital_structure) },
                { id: "cashflow", title: "Cash Flow Statement", html: renderCashFlow(data.cash_flow) },
                { id: "dcf", title: "DCF Valuation", html: renderDCF(data.dcf) },
                { id: "ddm", title: "DDM Valuation", html: renderDDM(data.ddm) },
                { id: "comps", title: "Comparable Companies", html: renderComps(data.comps) },
                { id: "impliedval", title: "Implied Valuation from Comps", html: renderImpliedValuation(data.comps) },
                { id: "profitability", title: "Profitability & Efficiency", html: renderProfitability(data.profitability) },
                { id: "ceoinfo", title: "CEO & Leadership", html: renderCEOInfo(data.ceo_info) },
                { id: "compensation", title: "Compensation Structure", html: renderCompensation(data.compensation) },
                { id: "roi", title: "ROI Analysis", html: renderROI(data.roi_analysis) },
                { id: "scoring", title: "Scoring Model Breakdown", html: renderScoringBreakdown(data.scoring_breakdown) },
                { id: "catalysts", title: "Catalysts & Events", html: renderCatalysts(data.catalysts) },
                { id: "moat", title: "Competitive Moat", html: renderMoat(data.moat) },
                { id: "risks", title: "Risk Factors", html: renderRisks(data.risks) },
                { id: "viewchangers", title: "View Changers", html: renderViewChangers(data.view_changers) },
                { id: "pricetarget", title: "Price Target Derivation", html: renderPriceTarget(data.price_target) },
                { id: "verdict", title: "Verdict", html: renderVerdict(data.verdict, data.header) }
            ];

            var html = renderReportHeader(data.header);
            html += '<div class="report-disclaimer" style="font-size:0.6rem;color:#999;text-align:center;padding:8px 0;letter-spacing:0.08em;text-transform:uppercase">Generated by SignalBoard Analytics Engine \u00b7 For Research Purposes Only</div>';
            var summaries = data.section_summaries || {};
            var sectionIndex = 0;
            for (var i = 0; i < sections.length; i++) {
                var s = sections[i];
                if (!s.html) continue;
                html += wrapSection(s.id, s.title, s.html, sectionIndex, summaries[s.id] || '');
                sectionIndex++;
            }

            // Store sections for TOC building
            window._reportSections = sections.filter(function (s) { return !!s.html; });

            return html;
        }

        function wrapSection(id, title, content, index, summary) {
            var num = String(index).padStart(2, '0');
            var summaryHtml = summary
                ? '<div class="section-summary" id="summary-' + id + '">' +
                  '<div class="section-summary-inner">' +
                  '<span class="summary-label">Key Takeaway</span>' +
                  escapeHtml(summary) +
                  '</div></div>'
                : '';
            return (
                '<section class="report-section" id="section-' + id + '">' +
                '<div class="report-section-header" data-summary="' + id + '">' +
                '<span class="section-num">' + num + '</span> ' + escapeHtml(title) +
                '</div>' +
                summaryHtml +
                '<div class="report-section-body">' +
                content +
                '</div></section>'
            );
        }

        function renderTOC(tocEl) {
            var sections = window._reportSections || [];
            var html = '<ul>';
            for (var i = 0; i < sections.length; i++) {
                var s = sections[i];
                var num = String(i + 1).padStart(2, '0');
                html += '<li><a href="#section-' + s.id + '" id="toc-' + s.id + '" data-section="' + s.id + '">' + escapeHtml(s.title) + '</a></li>';
            }
            html += '</ul>';
            tocEl.innerHTML = html;

            // Set up Intersection Observer for active tracking
            setupScrollTracking();
        }

        function setupScrollTracking() {
            var sectionEls = document.querySelectorAll('.report-section');
            if (!sectionEls.length) return;

            var observer = new IntersectionObserver(function (entries) {
                entries.forEach(function (entry) {
                    if (entry.isIntersecting) {
                        // Remove active from all
                        var links = document.querySelectorAll('.report-toc a');
                        for (var i = 0; i < links.length; i++) {
                            links[i].classList.remove('active');
                        }
                        // Add active to current
                        var id = entry.target.id.replace('section-', '');
                        var activeLink = document.getElementById('toc-' + id);
                        if (activeLink) {
                            activeLink.classList.add('active');
                            // Scroll TOC to make active item visible
                            activeLink.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
                        }
                    }
                });
            }, {
                rootMargin: '-80px 0px -70% 0px',
                threshold: 0
            });

            for (var i = 0; i < sectionEls.length; i++) {
                observer.observe(sectionEls[i]);
            }

            // Smooth scroll on TOC click
            var tocLinks = document.querySelectorAll('.report-toc a');
            for (var j = 0; j < tocLinks.length; j++) {
                tocLinks[j].addEventListener('click', function (e) {
                    e.preventDefault();
                    var targetId = 'section-' + this.getAttribute('data-section');
                    var target = document.getElementById(targetId);
                    if (target) {
                        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                });
            }
        }

        function renderReportHeader(h) {
            if (!h) return "";
            var ratingClass = (h.rating || "").toLowerCase().replace(/\s+/g, "-");
            // Map rating to gauge position (0-100)
            var gaugeMap = {"strong-sell": 10, "sell": 25, "hold": 50, "buy": 75, "strong-buy": 90};
            var gaugePos = gaugeMap[ratingClass] || 50;
            var upsideClass = (h.upside_pct || 0) >= 0 ? "positive" : "negative";

            var html = '<div class="report-header">';
            html += '<div class="report-header-top">';
            html += '<div class="report-header-info">';
            html += '<h1>' + escapeHtml(h.name || h.ticker) + '</h1>';
            html += '<div class="header-meta">' + escapeHtml(h.ticker) + ' · ' + escapeHtml(h.sector || '') + ' · ' + escapeHtml(h.exchange || '') + '</div>';
            html += '</div>';
            html += '<div class="header-price"><span class="pt-value">' + fmtUSD(h.current_price) + '</span></div>';
            html += '</div>';

            // Confidence gauge
            html += '<div class="confidence-gauge">';
            html += '<div class="gauge-track">';
            html += '<div class="gauge-fill ' + ratingClass + '" style="width:' + gaugePos + '%"></div>';
            html += '<div class="gauge-indicator ' + ratingClass + '" style="left:' + gaugePos + '%;background:' + (gaugePos > 50 ? '#2d6a4f' : gaugePos < 50 ? '#c1292e' : '#adb5bd') + '"></div>';
            html += '</div>';
            html += '<div class="gauge-labels">';
            html += '<span class="gauge-label' + (ratingClass === "strong-sell" ? " active" : "") + '">Strong Sell</span>';
            html += '<span class="gauge-label' + (ratingClass === "sell" ? " active" : "") + '">Sell</span>';
            html += '<span class="gauge-label' + (ratingClass === "hold" ? " active" : "") + '">Hold</span>';
            html += '<span class="gauge-label' + (ratingClass === "buy" ? " active" : "") + '">Buy</span>';
            html += '<span class="gauge-label' + (ratingClass === "strong-buy" ? " active" : "") + '">Strong Buy</span>';
            html += '</div>';
            html += '</div>';

            // Key metrics row: Target | Upside | Date
            html += '<div class="price-target-row">';
            html += '<div class="pt-item"><span class="pt-label">Target</span><span class="pt-value">' + fmtUSD(h.price_target) + '</span></div>';
            html += '<div class="pt-item"><span class="pt-label">Upside</span><span class="pt-value ' + upsideClass + '">' + fmtPctReport(h.upside_pct) + '</span></div>';
            html += '<div class="pt-item"><span class="pt-label">Date</span><span class="pt-value" style="font-size:0.8rem">' + escapeHtml(h.date || '') + '</span></div>';
            html += '</div>';
            html += '</div>';
            return html;
        }

        function renderThesis(t) {
            if (!t || !t.text) return "";
            return '<div class="thesis-section"><span class="thesis-label">Core Thesis</span><p class="thesis-text">' + escapeHtml(t.text) + '</p></div>';
        }

        function renderSnapshot(s) {
            if (!s) return "";
            var items = [
                { label: "Market Cap", value: fmtB(s.market_cap) },
                { label: "Enterprise Value", value: fmtB(s.enterprise_value) },
                { label: "Shares Outstanding", value: fmtNum(s.shares_outstanding) },
                { label: "52-Week Range", value: fmtUSD(s.range_52w_low) + " \u2013 " + fmtUSD(s.range_52w_high) },
                { label: "Avg Volume (3M)", value: fmtNum(s.avg_volume_3m) },
                { label: "Dividend Yield", value: fmtPctReport(s.dividend_yield) },
                { label: "Beta", value: fmtX(s.beta) },
                { label: "Short Interest", value: fmtPctReport(s.short_interest_pct) }
            ];
            var html = '<div class="snapshot-grid">';
            for (var i = 0; i < items.length; i++) {
                html += '<div class="snapshot-item"><span class="snapshot-label">' + items[i].label + '</span><span class="snapshot-value">' + items[i].value + '</span></div>';
            }
            html += '</div>';
            return html;
        }

        function renderFinTable(title, sectionId, headers, rows, options) {
            options = options || {};
            var html = '<div class="fin-table-wrapper">';
            if (title) html += '<h4>' + escapeHtml(title) + '</h4>';
            html += '<table class="fin-table">';
            html += '<thead><tr><th></th>';
            for (var h = 0; h < headers.length; h++) {
                html += '<th>' + escapeHtml(String(headers[h])) + '</th>';
            }
            html += '</tr></thead><tbody>';
            for (var r = 0; r < rows.length; r++) {
                var row = rows[r];
                var cls = "";
                if (row.type === "header") cls = "row-header";
                else if (row.type === "subtotal") cls = "row-subtotal";
                else if (row.type === "pct") cls = "row-pct";
                html += '<tr class="' + cls + '"><td>' + escapeHtml(row.label) + '</td>';
                var vals = row.values || [];
                for (var v = 0; v < headers.length; v++) {
                    html += '<td>' + (v < vals.length ? formatCellValue(vals[v], row.format) : "\u2014") + '</td>';
                }
                html += '</tr>';
            }
            html += '</tbody></table></div>';
            return html;
        }

        function formatCellValue(val, format) {
            if (val == null) return "\u2014";
            if (format === "pct") return fmtPctReport(val);
            if (format === "usd") return fmtUSD(val);
            if (format === "millions") return fmtM(val);
            if (format === "billions") return fmtB(val);
            if (format === "x") return fmtX(val);
            if (typeof val === "number") {
                if (Math.abs(val) >= 1e9) return fmtB(val);
                if (Math.abs(val) >= 1e6) return fmtM(val);
                if (Math.abs(val) < 1 && val !== 0) return fmtPctReport(val);
                return fmtNum(val);
            }
            return escapeHtml(String(val));
        }

        function renderIncomeStatement(data) {
            if (!data || data.length === 0) return "";
            var headers = data.map(function (d) { return d.year; });
            var rows = [
                { label: "Revenue", values: data.map(function (d) { return d.revenue; }), type: "header", format: "millions" },
                { label: "YoY Growth", values: data.map(function (d) { return d.yoy_growth; }), type: "pct", format: "pct" },
                { label: "Gross Margin", values: data.map(function (d) { return d.gross_margin; }), type: "pct", format: "pct" },
                { label: "EBITDA", values: data.map(function (d) { return d.ebitda; }), format: "millions" },
                { label: "EBITDA Margin", values: data.map(function (d) { return d.ebitda_margin; }), type: "pct", format: "pct" },
                { label: "Net Income", values: data.map(function (d) { return d.net_income; }), type: "subtotal", format: "millions" },
                { label: "Diluted EPS", values: data.map(function (d) { return d.diluted_eps; }), format: "usd" }
            ];
            return renderFinTable(null, "income", headers, rows);
        }

        function renderBalanceSheet(data) {
            if (!data || data.length === 0) return "";
            var headers = data.map(function (d) { return d.year; });
            var rows = [
                { label: "Cash & Equivalents", values: data.map(function (d) { return d.cash; }), format: "millions" },
                { label: "Total Assets", values: data.map(function (d) { return d.total_assets; }), type: "header", format: "millions" },
                { label: "Total Debt", values: data.map(function (d) { return (d.long_term_debt || 0) + (d.short_term_debt || 0); }), format: "millions" },
                { label: "Total Equity", values: data.map(function (d) { return d.total_equity; }), type: "subtotal", format: "millions" },
                { label: "Book Value / Share", values: data.map(function (d) { return d.book_value_per_share; }), format: "usd" }
            ];
            return renderFinTable(null, "balance", headers, rows);
        }

        function renderCapitalStructure(data) {
            if (!data) return "";
            var rows = [
                { label: "Net Debt", values: [data.net_debt], format: "millions" },
                { label: "Net Debt / EBITDA", values: [data.net_debt_ebitda], format: "x" },
                { label: "Debt / Equity", values: [data.debt_to_equity], format: "x" },
                { label: "Interest Coverage", values: [data.interest_coverage], format: "x" },
                { label: "Current Ratio", values: [data.current_ratio], format: "x" },
                { label: "Quick Ratio", values: [data.quick_ratio], format: "x" }
            ];
            return renderFinTable(null, "capital", ["Current"], rows);
        }

        function renderCashFlow(data) {
            if (!data || data.length === 0) return "";
            var headers = data.map(function (d) { return d.year; });
            var rows = [
                { label: "Cash from Operations", values: data.map(function (d) { return d.cfo; }), type: "header", format: "millions" },
                { label: "CapEx", values: data.map(function (d) { return d.capex; }), format: "millions" },
                { label: "Free Cash Flow", values: data.map(function (d) { return d.fcf; }), type: "subtotal", format: "millions" },
                { label: "FCF Margin", values: data.map(function (d) { return d.fcf_margin; }), type: "pct", format: "pct" },
                { label: "FCF Yield", values: data.map(function (d) { return d.fcf_yield; }), type: "pct", format: "pct" },
                { label: "FCF / Share", values: data.map(function (d) { return d.fcf_per_share; }), format: "usd" }
            ];
            return renderFinTable(null, "cashflow", headers, rows);
        }

        function renderDCF(dcf) {
            if (!dcf) return "";
            var html = "";

            // Key assumptions — compact
            if (dcf.assumptions) {
                var a = dcf.assumptions;
                var aRows = [
                    { label: "WACC", values: [a.wacc], format: "pct" },
                    { label: "Terminal Growth", values: [a.terminal_growth], format: "pct" },
                    { label: "Revenue CAGR", values: [a.revenue_cagr], format: "pct" },
                    { label: "Terminal EBITDA Margin", values: [a.terminal_ebitda_margin], format: "pct" }
                ];
                html += '<div class="dcf-sub-table">';
                html += renderFinTable("Assumptions", "dcf-assumptions", ["Value"], aRows);
                html += '</div>';
            }

            // DCF Output — the money numbers
            if (dcf.output) {
                var o = dcf.output;
                var oRows = [
                    { label: "PV of Cash Flows", values: [o.pv_fcfs], format: "millions" },
                    { label: "PV of Terminal Value", values: [o.pv_terminal], format: "millions" },
                    { label: "Implied Enterprise Value", values: [o.implied_ev], type: "subtotal", format: "millions" },
                    { label: "Implied Price / Share", values: [o.implied_price], type: "header", format: "usd" },
                    { label: "Upside / (Downside)", values: [o.upside_pct], type: "pct", format: "pct" }
                ];
                html += '<div class="dcf-sub-table">';
                html += renderFinTable("Valuation Output", "dcf-output", ["Value"], oRows);
                html += '</div>';
            }

            // Sensitivity matrix — separate heading with clear visual break
            if (dcf.sensitivity) {
                html += '<div class="sensitivity-wrapper">';
                html += renderSensitivityMatrix(dcf.sensitivity, dcf.output ? dcf.output.current_price : null);
                html += '</div>';
            }

            return html;
        }

        function renderDDM(ddm) {
            if (!ddm) return '';
            var html = '<div class="report-card">';
            html += '<div class="card-accent"></div>';

            // Assumptions table
            html += '<table class="data-tbl"><tbody>';
            html += '<tr class="sub-header"><td colspan="2">DDM ASSUMPTIONS</td></tr>';
            html += '<tr><td>Current Dividend (D0)</td><td>' + fmtUSD(ddm.assumptions.current_dividend) + '</td></tr>';
            html += '<tr><td>Next Year Dividend (D1)</td><td>' + fmtUSD(ddm.assumptions.next_year_dividend) + '</td></tr>';
            html += '<tr><td>Sustainable Growth Rate</td><td>' + fmtPctReport(ddm.assumptions.sustainable_growth) + '</td></tr>';
            html += '<tr><td>Growth Rate Used</td><td>' + fmtPctReport(ddm.assumptions.growth_rate_used) + '</td></tr>';
            html += '<tr><td>Required Return (CAPM)</td><td>' + fmtPctReport(ddm.assumptions.required_return) + '</td></tr>';
            html += '</tbody></table>';

            // Output
            html += '<table class="data-tbl"><tbody>';
            html += '<tr class="sub-header"><td colspan="2">DDM OUTPUT</td></tr>';
            html += '<tr><td>Intrinsic Value</td><td>' + fmtUSD(ddm.output.intrinsic_value) + '</td></tr>';
            html += '<tr><td>Current Price</td><td>' + fmtUSD(ddm.output.current_price) + '</td></tr>';
            html += '<tr><td>Upside</td><td>' + fmtPctReport(ddm.output.upside) + '</td></tr>';
            html += '<tr><td>Margin of Safety</td><td>' + fmtPctReport(ddm.output.margin_of_safety) + '</td></tr>';
            html += '</tbody></table>';

            // Sensitivity matrix
            if (ddm.sensitivity && ddm.sensitivity.rows) {
                html += '<table class="data-tbl">';
                html += '<thead><tr><th>' + ddm.sensitivity.row_label + ' \\ ' + ddm.sensitivity.col_label + '</th>';
                ddm.sensitivity.g_values.forEach(function(g) {
                    html += '<th>' + fmtPctReport(g) + '</th>';
                });
                html += '</tr></thead><tbody>';
                ddm.sensitivity.rows.forEach(function(row) {
                    html += '<tr><td>' + fmtPctReport(row.r) + '</td>';
                    row.values.forEach(function(v) {
                        html += '<td>' + (v != null ? fmtUSD(v) : '\u2014') + '</td>';
                    });
                    html += '</tr>';
                });
                html += '</tbody></table>';
            }

            html += '</div>';
            return html;
        }

        function renderCEOInfo(ceoInfo) {
            if (!ceoInfo) return '';
            var html = '<div class="report-card">';
            html += '<div class="card-accent"></div>';
            html += '<table class="data-tbl"><tbody>';
            html += '<tr class="sub-header"><td colspan="2">CEO STATUS</td></tr>';

            if (ceoInfo.has_data) {
                var status = ceoInfo.ceo_changed_recently ? 'CHANGED' : 'STABLE';
                var statusClass = ceoInfo.ceo_changed_recently ? 'color: #8b5e3c' : 'color: #1a1a1a';
                html += '<tr><td>Leadership Status</td><td style="' + statusClass + '">' + status + '</td></tr>';
                if (ceoInfo.change_date) {
                    html += '<tr><td>Change Date</td><td>' + ceoInfo.change_date + '</td></tr>';
                }
                if (ceoInfo.filing_url) {
                    html += '<tr><td>SEC Filing</td><td><a href="' + ceoInfo.filing_url + '" target="_blank" style="color: #666; text-decoration: underline">View 8-K</a></td></tr>';
                }
            } else {
                html += '<tr><td colspan="2" style="color: #999">CEO data unavailable</td></tr>';
            }

            html += '</tbody></table>';
            if (ceoInfo.note) {
                html += '<p style="font-size: 0.78rem; color: #666; margin-top: 8px; font-style: italic">' + ceoInfo.note + '</p>';
            }
            html += '</div>';
            return html;
        }

        function renderCompensation(comp) {
            if (!comp || !comp.has_data) {
                return '<div class="report-card"><div class="card-accent"></div><p style="color: #999; font-size: 0.78rem">Executive compensation data unavailable</p></div>';
            }
            var html = '<div class="report-card">';
            html += '<div class="card-accent"></div>';
            html += '<table class="data-tbl"><tbody>';
            html += '<tr class="sub-header"><td colspan="2">EXECUTIVE COMPENSATION</td></tr>';

            if (comp.equity_pct != null) {
                html += '<tr><td>Equity-Based (%)</td><td>' + fmtPctReport(comp.equity_pct) + '</td></tr>';
            }
            if (comp.cash_pct != null) {
                html += '<tr><td>Cash-Based (%)</td><td>' + fmtPctReport(comp.cash_pct) + '</td></tr>';
            }
            if (comp.total_ceo_compensation) {
                html += '<tr><td>Total CEO Compensation</td><td>' + fmtUSD(comp.total_ceo_compensation) + '</td></tr>';
            }

            var alignment = comp.equity_heavy ? 'Equity-heavy \u2014 aligned with shareholders' : 'Cash-heavy \u2014 review alignment';
            html += '<tr><td>Incentive Alignment</td><td>' + alignment + '</td></tr>';

            if (comp.latest_proxy_date) {
                html += '<tr><td>Latest Proxy Date</td><td>' + comp.latest_proxy_date + '</td></tr>';
            }
            if (comp.filing_url) {
                html += '<tr><td>SEC Filing</td><td><a href="' + comp.filing_url + '" target="_blank" style="color: #666; text-decoration: underline">View DEF 14A</a></td></tr>';
            }

            html += '</tbody></table>';
            if (comp.alignment_note) {
                html += '<p style="font-size: 0.78rem; color: #666; margin-top: 8px; font-style: italic">' + comp.alignment_note + '</p>';
            }
            html += '</div>';
            return html;
        }

        function renderROI(roi) {
            if (!roi || roi.total_roi_pct == null) return '';
            var html = '<div class="report-card">';
            html += '<div class="card-accent"></div>';
            html += '<table class="data-tbl"><tbody>';
            html += '<tr class="sub-header"><td colspan="2">RETURN ON INVESTMENT</td></tr>';
            html += '<tr><td>Total Projected ROI</td><td style="font-weight: 500">' + fmtPctReport(roi.total_roi_pct) + '</td></tr>';
            html += '<tr><td>Capital Gain</td><td>' + fmtPctReport(roi.capital_gain_pct) + '</td></tr>';
            html += '<tr><td>Income Return (Dividends)</td><td>' + fmtPctReport(roi.income_return_pct) + '</td></tr>';
            html += '<tr><td>Risk-Adjusted ROI (\u00F7 Beta)</td><td>' + fmtPctReport(roi.risk_adjusted_roi) + '</td></tr>';
            if (roi.target_price) {
                html += '<tr><td>Target Price</td><td>' + fmtUSD(roi.target_price) + '</td></tr>';
            }
            if (roi.current_price) {
                html += '<tr><td>Current Price</td><td>' + fmtUSD(roi.current_price) + '</td></tr>';
            }
            html += '</tbody></table></div>';
            return html;
        }

        function renderScoringBreakdown(scoring) {
            if (!scoring || !scoring.has_data || !scoring.components) return '';
            var comps = scoring.components;
            var html = '<div class="report-card">';
            html += '<div class="card-accent"></div>';
            html += '<table class="data-tbl scoring-breakdown"><tbody>';
            html += '<tr class="sub-header"><td>Component</td><td>Score</td><td>Weight</td><td>Contribution</td><td></td></tr>';
            for (var i = 0; i < comps.length; i++) {
                var c = comps[i];
                var pct = Math.round(c.score * 100);
                html += '<tr>';
                html += '<td>' + escapeHtml(c.label) + '</td>';
                html += '<td class="num">' + c.score.toFixed(2) + '</td>';
                html += '<td class="num">' + (c.weight * 100).toFixed(1) + '%</td>';
                html += '<td class="num">' + c.contribution.toFixed(3) + '</td>';
                html += '<td class="bar-cell"><div class="scoring-bar-track"><div class="scoring-bar-fill" style="width:' + pct + '%"></div></div></td>';
                html += '</tr>';
            }
            // Composite total row
            html += '<tr class="composite-row">';
            html += '<td>Composite Score</td>';
            html += '<td class="num" style="font-weight:500">' + (scoring.composite_total != null ? scoring.composite_total.toFixed(3) : '—') + '</td>';
            html += '<td></td><td></td>';
            html += '<td class="num">Rank #' + (scoring.rank || '—') + '</td>';
            html += '</tr>';
            html += '</tbody></table></div>';
            return html;
        }

        function renderSensitivityMatrix(sens, currentPrice) {
            if (!sens || !sens.matrix) return "";
            var waccVals = sens.wacc_values || [];
            var growthVals = sens.growth_values || [];
            var matrix = sens.matrix || [];
            var midRow = Math.floor(matrix.length / 2);
            var midCol = Math.floor(waccVals.length / 2);

            var html = '<h4>Sensitivity Analysis (WACC vs Terminal Growth)</h4>';
            html += '<div class="fin-table-wrapper"><table class="fin-table sensitivity-grid">';
            html += '<thead><tr><th>WACC \\ g</th>';
            for (var g = 0; g < growthVals.length; g++) {
                html += '<th>' + fmtPctReport(growthVals[g]) + '</th>';
            }
            html += '</tr></thead><tbody>';
            for (var r = 0; r < matrix.length; r++) {
                html += '<tr><td class="row-header">' + fmtPctReport(waccVals[r]) + '</td>';
                for (var c = 0; c < matrix[r].length; c++) {
                    var val = matrix[r][c];
                    var cellClass = "";
                    if (currentPrice != null) {
                        cellClass = val > currentPrice ? "positive" : "negative";
                    }
                    if (r === midRow && c === midCol) cellClass += " base-case";
                    html += '<td class="' + cellClass.trim() + '">' + fmtUSD(val) + '</td>';
                }
                html += '</tr>';
            }
            html += '</tbody></table></div>';
            return html;
        }

        function renderComps(comps) {
            if (!comps) return "";
            var html = "";

            // Peer comparison table
            if (comps.peers && comps.peers.length > 0) {
                var compHeaders = ["Company", "Ticker", "EV", "EV/Rev", "EV/EBITDA", "P/E (Fwd)", "PEG", "Rev Growth", "EBITDA Margin", "Net Margin", "ROIC"];
                html += '<div class="fin-table-wrapper"><table class="fin-table comps-table">';
                html += '<thead><tr>';
                for (var h = 0; h < compHeaders.length; h++) {
                    html += '<th>' + compHeaders[h] + '</th>';
                }
                html += '</tr></thead><tbody>';

                // Peer rows
                for (var p = 0; p < comps.peers.length; p++) {
                    html += compRow(comps.peers[p], "");
                }
                // Peer median
                if (comps.peer_median) {
                    html += compRow(comps.peer_median, "peer-median-row");
                }
                // Subject
                if (comps.subject) {
                    html += compRow(comps.subject, "subject-row");
                }
                // Premium/discount
                if (comps.premium_discount) {
                    var pd = comps.premium_discount;
                    html += '<tr class="premium-discount-row">';
                    html += '<td colspan="3">Premium / (Discount)</td>';
                    html += '<td>' + fmtPctReport(pd.ev_revenue) + '</td>';
                    html += '<td>' + fmtPctReport(pd.ev_ebitda) + '</td>';
                    html += '<td>' + fmtPctReport(pd.pe) + '</td>';
                    html += '<td></td><td></td><td></td><td></td>';
                    html += '<td>' + fmtPctReport(pd.roic) + '</td>';
                    html += '</tr>';
                }

                html += '</tbody></table></div>';
            }

            return html;
        }

        function renderImpliedValuation(comps) {
            if (!comps || !comps.implied_valuation || comps.implied_valuation.length === 0) return "";
            var ivHeaders = ["Method", "Peer Median", "Subject Metric", "Implied EV", "Implied Price"];
            var html = '<div class="fin-table-wrapper"><table class="fin-table">';
            html += '<thead><tr>';
            for (var ih = 0; ih < ivHeaders.length; ih++) {
                html += '<th>' + ivHeaders[ih] + '</th>';
            }
            html += '</tr></thead><tbody>';
            for (var iv = 0; iv < comps.implied_valuation.length; iv++) {
                var row = comps.implied_valuation[iv];
                html += '<tr>';
                html += '<td>' + escapeHtml(row.method || "") + '</td>';
                html += '<td>' + fmtX(row.peer_median) + '</td>';
                html += '<td>' + fmtM(row.subject_metric) + '</td>';
                html += '<td>' + fmtM(row.implied_ev) + '</td>';
                html += '<td>' + fmtUSD(row.implied_price) + '</td>';
                html += '</tr>';
            }
            html += '</tbody></table></div>';
            return html;
        }

        function compRow(peer, cls) {
            var html = '<tr class="' + cls + '">';
            html += '<td>' + escapeHtml(peer.name || "") + '</td>';
            html += '<td>' + escapeHtml(peer.ticker || "") + '</td>';
            html += '<td>' + fmtB(peer.ev) + '</td>';
            html += '<td>' + fmtX(peer.ev_revenue) + '</td>';
            html += '<td>' + fmtX(peer.ev_ebitda) + '</td>';
            html += '<td>' + fmtX(peer.pe_fwd) + '</td>';
            html += '<td>' + fmtX(peer.peg) + '</td>';
            html += '<td>' + fmtPctReport(peer.rev_growth) + '</td>';
            html += '<td>' + fmtPctReport(peer.ebitda_margin) + '</td>';
            html += '<td>' + fmtPctReport(peer.net_margin) + '</td>';
            html += '<td>' + fmtPctReport(peer.roic) + '</td>';
            html += '</tr>';
            return html;
        }

        function renderProfitability(p) {
            if (!p) return "";
            var items = [
                { label: "ROE", value: fmtPctReport(p.roe) },
                { label: "ROA", value: fmtPctReport(p.roa) },
                { label: "ROIC", value: fmtPctReport(p.roic) },
                { label: "Asset Turnover", value: fmtX(p.asset_turnover) },
                { label: "Inventory Turnover", value: fmtX(p.inventory_turnover) },
                { label: "DSO", value: p.dso != null ? Math.round(p.dso) + " days" : "\u2014" },
                { label: "DPO", value: p.dpo != null ? Math.round(p.dpo) + " days" : "\u2014" },
                { label: "Cash Conversion Cycle", value: p.cash_conversion_cycle != null ? Math.round(p.cash_conversion_cycle) + " days" : "\u2014" }
            ];
            var html = '<div class="snapshot-grid">';
            for (var i = 0; i < items.length; i++) {
                html += '<div class="snapshot-item"><span class="snapshot-label">' + items[i].label + '</span><span class="snapshot-value">' + items[i].value + '</span></div>';
            }
            html += '</div>';
            return html;
        }

        function renderCatalysts(c) {
            if (!c || c.length === 0) return "";
            var html = '<div class="fin-table-wrapper"><table class="fin-table">';
            html += '<thead><tr><th>Date</th><th>Event</th><th>Impact</th></tr></thead><tbody>';
            for (var i = 0; i < c.length; i++) {
                html += '<tr>';
                html += '<td>' + escapeHtml(c[i].date || "") + '</td>';
                html += '<td>' + escapeHtml(c[i].event || "") + '</td>';
                html += '<td>' + escapeHtml(c[i].impact || "") + '</td>';
                html += '</tr>';
            }
            html += '</tbody></table></div>';
            return html;
        }

        function renderMoat(m) {
            if (!m) return "";
            var ratingClass = (m.rating || "none").toLowerCase();
            var html = '<div class="moat-rating ' + ratingClass + '">' + escapeHtml(m.rating || "None") + ' Moat</div>';
            if (m.description) html += '<p>' + escapeHtml(m.description) + '</p>';
            if (m.tam || m.market_share) {
                html += '<div class="snapshot-grid">';
                if (m.tam) html += '<div class="snapshot-item"><span class="snapshot-label">TAM</span><span class="snapshot-value">' + fmtB(m.tam) + '</span></div>';
                if (m.market_share) html += '<div class="snapshot-item"><span class="snapshot-label">Market Share</span><span class="snapshot-value">' + fmtPctReport(m.market_share) + '</span></div>';
                html += '</div>';
            }
            return html;
        }

        function renderRisks(risks) {
            if (!risks || risks.length === 0) return "";
            var html = '<div class="fin-table-wrapper"><table class="fin-table">';
            html += '<thead><tr><th>Factor</th><th>Severity</th><th>Probability</th><th>Detail</th></tr></thead><tbody>';
            for (var i = 0; i < risks.length; i++) {
                var r = risks[i];
                var sevClass = (r.severity || "").toLowerCase();
                var probClass = (r.probability || "").toLowerCase();
                html += '<tr>';
                html += '<td>' + escapeHtml(r.factor || "") + '</td>';
                html += '<td><span class="risk-badge ' + sevClass + '">' + escapeHtml(r.severity || "") + '</span></td>';
                html += '<td><span class="risk-badge ' + probClass + '">' + escapeHtml(r.probability || "") + '</span></td>';
                html += '<td>' + escapeHtml(r.detail || "") + '</td>';
                html += '</tr>';
            }
            html += '</tbody></table></div>';
            return html;
        }

        function renderViewChangers(vc) {
            if (!vc) return "";
            var html = "";
            if (vc.bullish && vc.bullish.length > 0) {
                html += '<ul class="trigger-list bullish">';
                for (var i = 0; i < vc.bullish.length; i++) {
                    html += '<li>&#9650; ' + escapeHtml(vc.bullish[i]) + '</li>';
                }
                html += '</ul>';
            }
            if (vc.bearish && vc.bearish.length > 0) {
                html += '<ul class="trigger-list bearish">';
                for (var i = 0; i < vc.bearish.length; i++) {
                    html += '<li>&#9660; ' + escapeHtml(vc.bearish[i]) + '</li>';
                }
                html += '</ul>';
            }
            return html;
        }

        function renderPriceTarget(pt) {
            if (!pt) return "";
            var rows = [
                { label: "DCF", weight: pt.dcf_weight, value: pt.dcf_value },
                { label: "Comps", weight: pt.comps_weight, value: pt.comps_value },
                { label: "Technical", weight: pt.technical_weight, value: pt.technical_value }
            ];

            var html = '<div class="fin-table-wrapper"><table class="fin-table">';
            html += '<thead><tr><th>Method</th><th>Weight</th><th>Value</th><th>Contribution</th></tr></thead><tbody>';
            for (var i = 0; i < rows.length; i++) {
                var r = rows[i];
                var contribution = (r.weight != null && r.value != null) ? r.weight * r.value : null;
                html += '<tr>';
                html += '<td>' + r.label + '</td>';
                html += '<td>' + fmtPctReport(r.weight) + '</td>';
                html += '<td>' + fmtUSD(r.value) + '</td>';
                html += '<td>' + fmtUSD(contribution) + '</td>';
                html += '</tr>';
            }
            html += '<tr class="row-subtotal"><td>Blended Target</td><td></td><td></td><td>' + fmtUSD(pt.blended) + '</td></tr>';
            html += '</tbody></table></div>';

            // Visual weight bar
            var segmentClasses = ["dcf-segment", "comps-segment", "tech-segment"];
            html += '<div class="pt-bar">';
            for (var j = 0; j < rows.length; j++) {
                var widthPct = (rows[j].weight || 0) * 100;
                if (widthPct > 0) {
                    var segCls = j < segmentClasses.length ? segmentClasses[j] : "";
                    html += '<div class="pt-bar-segment ' + segCls + '" style="width:' + widthPct + '%" title="' + rows[j].label + ': ' + fmtPctReport(rows[j].weight) + '">' + rows[j].label + '</div>';
                }
            }
            html += '</div>';

            return html;
        }

        function renderVerdict(v, header) {
            if (!v) return "";
            var ratingClass = (v.rating || "").toLowerCase().replace(/\s+/g, "-");
            var html = '<div class="verdict-box">';
            html += '<div class="verdict-rating ' + ratingClass + '">' + escapeHtml(v.rating || "") + '</div>';
            if (header && header.upside_pct != null) {
                var upsideVal = header.upside_pct;
                var upsideCls = upsideVal >= 0 ? "positive" : "negative";
                html += '<div class="verdict-upside ' + upsideCls + '" style="font-size:1.8rem;font-weight:700;margin:8px 0">' + fmtPctReport(upsideVal) + ' upside</div>';
            }
            html += '<div class="verdict-target">Price Target: ' + fmtUSD(v.price_target) + '</div>';
            if (v.confidence != null) {
                html += '<div class="verdict-confidence">Confidence: ' + fmtPctReport(v.confidence) + '</div>';
            }
            if (v.summary) {
                html += '<p class="verdict-summary">' + escapeHtml(v.summary) + '</p>';
            }
            html += '</div>';
            return html;
        }

        // ── Report formatting helpers ────────────────────────────

        function fmtM(n) {
            if (n == null || typeof n === "string" || isNaN(n)) return typeof n === "string" ? escapeHtml(n) : "\u2014";
            return "$" + (n / 1e6).toLocaleString(undefined, { maximumFractionDigits: 1 }) + "M";
        }

        function fmtB(n) {
            if (n == null || typeof n === "string" || isNaN(n)) return typeof n === "string" ? escapeHtml(n) : "\u2014";
            if (Math.abs(n) >= 1e12) return "$" + (n / 1e12).toLocaleString(undefined, { maximumFractionDigits: 1 }) + "T";
            return "$" + (n / 1e9).toLocaleString(undefined, { maximumFractionDigits: 1 }) + "B";
        }

        function fmtPctReport(n) {
            if (n == null) return "\u2014";
            if (typeof n === "string") return escapeHtml(n);
            if (isNaN(n)) return "\u2014";
            var pct = (Math.abs(n) <= 1 && n !== 0) ? n * 100 : n;
            return pct.toFixed(1) + "%";
        }

        function fmtX(n) {
            if (n == null || typeof n === "string" || isNaN(n)) return typeof n === "string" ? escapeHtml(n) : "\u2014";
            return Number(n).toFixed(1) + "x";
        }

        function fmtUSD(n) {
            if (n == null || typeof n === "string" || isNaN(n)) return typeof n === "string" ? escapeHtml(n) : "\u2014";
            return "$" + Number(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        }

        function fmtNum(n) {
            if (n == null || typeof n === "string" || isNaN(n)) return typeof n === "string" ? escapeHtml(n) : "\u2014";
            return Number(n).toLocaleString();
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
            html += '<div class="ticker">' + escapeHtml(s.ticker);
            if (s.stock_category) {
                var catLabel = s.stock_category === 'growth' ? 'GROWTH' : s.stock_category === 'dividend' ? 'DIVIDEND' : s.stock_category.toUpperCase();
                html += '<span style="font-size:0.6rem;letter-spacing:0.08em;color:#999;background:rgba(26,26,26,0.04);padding:2px 8px;margin-left:10px;vertical-align:middle">' + catLabel + '</span>';
            }
            html += '</div>';
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

            // Forward Outlook
            html += renderForwardOutlook(analysis);

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

            // DDM Valuation (dividend stocks)
            if (analysis.ddm) {
                html += '<div class="explanation-section section-ddm">';
                html += '<div class="section-title">DDM Valuation</div>';
                html += '<div class="metric-grid">';
                if (analysis.ddm.output) {
                    if (analysis.ddm.output.intrinsic_value != null)
                        html += metricRow("Intrinsic Value (DDM)", "$" + analysis.ddm.output.intrinsic_value.toFixed(2));
                    if (analysis.ddm.output.upside != null)
                        html += metricRow("DDM Upside", (analysis.ddm.output.upside > 0 ? "+" : "") + (analysis.ddm.output.upside * 100).toFixed(1) + "%");
                    if (analysis.ddm.output.margin_of_safety != null)
                        html += metricRow("Margin of Safety", (analysis.ddm.output.margin_of_safety * 100).toFixed(1) + "%");
                }
                if (analysis.ddm.assumptions) {
                    if (analysis.ddm.assumptions.growth_rate_used != null)
                        html += metricRow("Growth Rate", (analysis.ddm.assumptions.growth_rate_used * 100).toFixed(1) + "%");
                    if (analysis.ddm.assumptions.required_return != null)
                        html += metricRow("Required Return", (analysis.ddm.assumptions.required_return * 100).toFixed(1) + "%");
                }
                html += '</div></div>';
            }

            // CEO Info
            if (analysis.ceo_info) {
                html += '<div class="explanation-section section-ceo">';
                html += '<div class="section-title">CEO & Leadership</div>';
                html += '<div class="metric-grid">';
                if (analysis.ceo_info.has_data) {
                    var ceoStatus = analysis.ceo_info.ceo_changed_recently ? 'Changed' : 'Stable';
                    html += metricRow("Leadership Status", ceoStatus);
                    if (analysis.ceo_info.change_date)
                        html += metricRow("Change Date", analysis.ceo_info.change_date);
                } else {
                    html += metricRow("Status", "Data unavailable");
                }
                html += '</div>';
                if (analysis.ceo_info.note) {
                    html += '<p style="font-size:0.78rem;color:#666;margin-top:8px;font-style:italic">' + escapeHtml(analysis.ceo_info.note) + '</p>';
                }
                html += '</div>';
            }

            // Compensation
            if (analysis.compensation) {
                html += '<div class="explanation-section section-comp">';
                html += '<div class="section-title">Compensation Structure</div>';
                html += '<div class="metric-grid">';
                if (analysis.compensation.has_data) {
                    if (analysis.compensation.equity_pct != null)
                        html += metricRow("Equity-Based", (analysis.compensation.equity_pct * 100).toFixed(1) + "%");
                    if (analysis.compensation.cash_pct != null)
                        html += metricRow("Cash-Based", (analysis.compensation.cash_pct * 100).toFixed(1) + "%");
                    if (analysis.compensation.total_ceo_compensation)
                        html += metricRow("Total CEO Comp", "$" + Number(analysis.compensation.total_ceo_compensation).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}));
                    var alignLabel = analysis.compensation.equity_heavy ? 'Equity-heavy' : 'Cash-heavy';
                    html += metricRow("Alignment", alignLabel);
                } else {
                    html += metricRow("Status", "Data unavailable");
                }
                html += '</div>';
                if (analysis.compensation.alignment_note) {
                    html += '<p style="font-size:0.78rem;color:#666;margin-top:8px;font-style:italic">' + escapeHtml(analysis.compensation.alignment_note) + '</p>';
                }
                html += '</div>';
            }

            // ROI Analysis
            if (analysis.roi && analysis.roi.total_roi_pct != null) {
                html += '<div class="explanation-section section-roi">';
                html += '<div class="section-title">ROI Analysis</div>';
                html += '<div class="metric-grid">';
                html += metricRow("Total Projected ROI", (analysis.roi.total_roi_pct * 100).toFixed(1) + "%");
                if (analysis.roi.capital_gain_pct != null)
                    html += metricRow("Capital Gain", (analysis.roi.capital_gain_pct * 100).toFixed(1) + "%");
                if (analysis.roi.income_return_pct != null)
                    html += metricRow("Income Return", (analysis.roi.income_return_pct * 100).toFixed(1) + "%");
                if (analysis.roi.risk_adjusted_roi != null)
                    html += metricRow("Risk-Adjusted ROI", (analysis.roi.risk_adjusted_roi * 100).toFixed(1) + "%");
                if (analysis.roi.target_price != null)
                    html += metricRow("Target Price", "$" + analysis.roi.target_price.toFixed(2));
                if (analysis.roi.current_price != null)
                    html += metricRow("Current Price", "$" + analysis.roi.current_price.toFixed(2));
                html += '</div></div>';
            }

            contentEl.innerHTML = html;
        }

        // ── Forward Outlook renderer ─────────────────────────

        function renderForwardOutlook(a) {
            var outlook = a.forward_outlook;
            var hasTrends = a.roe_4q_trend != null || a.roic_4q_trend != null ||
                            a.gross_margin_trend != null || a.operating_margin_trend != null ||
                            a.net_margin_4q_trend != null || a.ocf_margin_4q_trend != null;
            var hasForwardSignals = a.implied_growth_rate != null || a.fcf_growth_3yr_cagr != null ||
                                   a.earnings_persistence != null || a.continuous_piotroski != null;

            if (!outlook && !hasTrends && !hasForwardSignals) return "";

            var html = '<div class="explanation-section section-outlook">';
            html += '<div class="section-title">Forward Outlook</div>';

            // Direction header
            if (outlook) {
                var dir = outlook.direction || "stable";
                var dirLabel = dir.toUpperCase();
                var dirClass = "outlook-dir-" + dir;
                var arrow = dir === "improving" ? "\u2197" : dir === "deteriorating" ? "\u2198" : "\u2192";
                html += '<div class="outlook-header">';
                html += '<div class="outlook-direction ' + dirClass + '">';
                html += '<span class="outlook-arrow">' + arrow + '</span> ';
                html += dirLabel;
                html += '</div>';
                html += '<span class="outlook-agreement">' + outlook.trend_agreement + ' of ' + outlook.trend_count + ' trends agree</span>';
                html += '</div>';
            }

            // Trend grid
            if (hasTrends) {
                html += '<div class="outlook-trends">';
                html += '<div class="outlook-subheading">Trajectory (4-Quarter Trend Slopes)</div>';
                html += trendRow("ROE", a.roe_4q_trend);
                html += trendRow("ROIC", a.roic_4q_trend);
                html += trendRow("Gross Margin", a.gross_margin_trend);
                html += trendRow("Operating Margin", a.operating_margin_trend);
                html += trendRow("Net Margin", a.net_margin_4q_trend);
                html += trendRow("OCF Margin", a.ocf_margin_4q_trend);
                html += '</div>';
            }

            // Forward signals
            if (hasForwardSignals) {
                html += '<div class="outlook-signals">';
                html += '<div class="outlook-subheading">Forward Signals</div>';
                html += '<div class="metric-grid">';
                if (a.implied_growth_rate != null)
                    html += metricRow("Implied Growth Rate", (a.implied_growth_rate * 100).toFixed(1) + "%");
                if (a.fcf_growth_3yr_cagr != null)
                    html += metricRow("FCF 3yr CAGR", (a.fcf_growth_3yr_cagr * 100).toFixed(1) + "%");
                if (a.continuous_piotroski != null)
                    html += metricRow("Piotroski (Continuous)", Math.round(a.continuous_piotroski) + " / 100");
                if (a.earnings_persistence != null)
                    html += metricRow("Earnings Persistence", a.earnings_persistence.toFixed(2));
                if (a.revenue_acceleration != null)
                    html += metricRow("Revenue Acceleration", (a.revenue_acceleration > 0 ? "+" : "") + (a.revenue_acceleration * 100).toFixed(1) + " pp");
                if (a.blindspot_score != null)
                    html += metricRow("Blindspot Score", a.blindspot_score.toFixed(2));
                html += '</div></div>';
            }

            // Insider conviction
            if (a.insider_cluster_buy) {
                html += '<div class="outlook-insider">';
                html += '<span class="outlook-insider-label">Insider Conviction</span> ';
                html += 'Cluster buying detected \u2014 management is buying their own stock';
                html += '</div>';
            }

            html += '</div>';
            return html;
        }

        function trendRow(label, value) {
            if (value == null) return "";
            var arrow, cls;
            if (value > 0.01) { arrow = "\u25B2"; cls = "trend-up"; }
            else if (value < -0.01) { arrow = "\u25BC"; cls = "trend-down"; }
            else { arrow = "\u25C6"; cls = "trend-flat"; }
            var display = (value > 0 ? "+" : "") + (value * 100).toFixed(2) + "%";
            return '<div class="trend-row">' +
                '<span class="trend-label">' + label + '</span>' +
                '<span class="trend-indicator ' + cls + '">' + arrow + '</span>' +
                '<span class="trend-value">' + display + '</span>' +
                '</div>';
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
