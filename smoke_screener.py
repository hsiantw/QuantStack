"""Exercise the screener with real browser controls and deterministic market rows.

Run against a running dashboard with:
    .venv/Scripts/python.exe smoke_screener.py --url http://127.0.0.1:8765

The live API is checked first. Browser routes then provide boundary-case market
data without changing the database or the user's browser profile. Static mode
uses the actual static adapter and gzip price format, without building the site.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import io
import json
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import Page, expect, sync_playwright


ROOT = Path(__file__).resolve().parent
ALL_SYMBOLS = {"AAPL", "MSFT", "TWD1", "EUR1", "MISS", "ZERO", "FORMULA"}


def open_screener(page: Page) -> None:
    if page.locator('#toggleScreener').get_attribute('aria-expanded') != 'true':
        page.locator('#toggleScreener').click()
    expect(page.locator('#screenerContent')).to_be_visible()
    if page.locator('#workspaceScreenerFilters').get_attribute('open') is None:
        page.locator('.workspace-screen-toolbar button').click()


def chart_save(page: Page) -> None:
    if page.locator('#toggleScreener').get_attribute('aria-expanded') == 'true':
        page.locator('#collapseScreener').click()
    page.locator('#save').click()
    open_screener(page)


def fixture() -> dict:
    def stock(symbol: str, **values) -> dict:
        return dict(symbol=symbol, name=symbol + " Research", currency="USD",
                    country="United States", exchange="NASDAQ", sector="Technology",
                    industry="Software", date="2026-09-24", metadata_date="2026-09-23",
                    has_history=symbol in {"AAPL", "MSFT"}, **values)

    rows = [
        stock("AAPL", market_cap=300e9, close=100, pe=30, dividend_yield_pct=1,
              rsi14=60, relative_volume=3, return_1m_pct=10,
              sma50_distance_pct=5, sma200_distance_pct=8, volume=2e6,
              revenue_growth_pct=20, profit_margin_pct=20, distance_52w_high_pct=-1,
              adx14=30, macd_histogram_pct=1, bollinger_width_pct=9,
              avg_volume20=2e6, avg_turnover20=200e6),
        stock("MSFT", market_cap=100e9, close=70, pe=15, dividend_yield_pct=3,
              rsi14=25, relative_volume=2, return_1m_pct=-5,
              sma50_distance_pct=-3, sma200_distance_pct=-2, volume=1e6,
              revenue_growth_pct=10, profit_margin_pct=15, distance_52w_high_pct=-20,
              adx14=20, macd_histogram_pct=-1, bollinger_width_pct=5,
              avg_volume20=1e6, avg_turnover20=70e6),
        stock("TWD1", market_cap=900e9, close=1000, pe=10, dividend_yield_pct=4,
              rsi14=80, relative_volume=1, return_1m_pct=20,
              sma50_distance_pct=8, sma200_distance_pct=6, volume=5e6,
              revenue_growth_pct=18, profit_margin_pct=12, distance_52w_high_pct=-2,
              adx14=40, macd_histogram_pct=2, bollinger_width_pct=12,
              avg_volume20=5e6, avg_turnover20=5e9),
        stock("EUR1", market_cap=50e9, close=20, pe=50, dividend_yield_pct=0,
              rsi14=20, relative_volume=1, return_1m_pct=-10,
              sma50_distance_pct=-4, sma200_distance_pct=-5, volume=3e6),
        stock("MISS", market_cap=None, close=None, pe=None, rsi14=None),
        stock("ZERO", market_cap=0, close=0, pe=0, rsi14=0, volume=0),
        stock("FORMULA", market_cap=1e6, close=5, pe=None, rsi14=50),
    ]
    rows[2].update(currency="TWD", country="Taiwan", exchange="TAI")
    rows[3].update(currency="EUR", country="Germany", exchange="XETRA", date="2026-09-23")
    rows[4].update(currency=None, country=None, sector=None, industry=None, date=None)
    rows[6]["name"] = '=SUM(1,2) <img src=x onerror="window.fixtureInjected=true">'
    return dict(rows=rows, generated_at="2026-09-25T00:00:00Z",
                data_dates={"earliest": "2026-09-23", "latest": "2026-09-24"},
                universe={"scope": "Deterministic browser test universe"})


def symbols(page: Page) -> list[str]:
    return page.locator("#screenerTableBody [data-screen-symbol]").evaluate_all(
        "buttons => buttons.map(button => button.dataset.screenSymbol)")


def expect_symbols(page: Page, expected: set[str]) -> None:
    expect(page.locator("#screenerTableBody [data-screen-symbol]")).to_have_count(len(expected))
    assert set(symbols(page)) == expected, (symbols(page), expected)


def reset(page: Page) -> None:
    page.locator("#resetScreener").click()
    expect(page.locator("#screenerRules [data-screen-rule]")).to_have_count(0)


def add_rule(page: Page, field: str, operator: str, value: str = "",
             maximum: str = "", currency: str | None = None) -> None:
    page.locator("#addScreenerRule").click()
    rule = page.locator("[data-screen-rule]").last
    rule.locator('[data-rule-property="field"]').select_option(field)
    rule.locator('[data-rule-property="operator"]').select_option(operator)
    if operator not in {"missing", "present"}:
        rule.locator('[data-rule-property="value"]').fill(value)
        if operator == "between":
            rule.locator('[data-rule-property="maximum"]').fill(maximum)
        if currency:
            rule.locator('[data-rule-property="currency"]').select_option(currency)


def read_export(page: Page) -> tuple[list[str], list[dict]]:
    with page.expect_download() as download:
        page.locator("#exportScreener").click()
    assert download.value.suggested_filename.startswith("quantstack-screen-")
    reader = csv.DictReader(io.StringIO(Path(download.value.path()).read_text(encoding="utf-8-sig")))
    records = list(reader)
    return reader.fieldnames, records


def exercise_filters(page: Page) -> None:
    expect_symbols(page, ALL_SYMBOLS)
    expect(page.locator('[data-screen-symbol="MISS"]')).to_be_disabled()
    assert not page.locator("#stockScreener img").count(), "Metadata must be escaped"
    assert page.evaluate("window.fixtureInjected !== true")

    for preset, expected in {
        "mega": {"AAPL"}, "momentum": {"AAPL", "TWD1"},
        "oversold": {"MSFT", "EUR1", "ZERO"}, "volume": {"AAPL", "MSFT"},
        "value": {"MSFT", "TWD1"}, "growth": {"AAPL", "TWD1"},
        "highs": {"AAPL", "TWD1"}, "trend": {"AAPL", "TWD1"},
        "squeeze": {"AAPL", "MSFT"}, "liquid": {"AAPL", "MSFT"},
    }.items():
        page.locator(f'[data-screen-preset="{preset}"]').click()
        expect_symbols(page, expected)

    reset(page)
    page.locator("#screenerSearch").fill("aapl research")
    expect_symbols(page, {"AAPL"})
    page.locator('[data-clear-filter="search"]').click()
    expect_symbols(page, ALL_SYMBOLS)
    reset(page)
    page.locator("#screenerCountry").select_option("Taiwan")
    expect_symbols(page, {"TWD1"})
    page.locator("#screenerCountry").select_option("__missing__")
    expect_symbols(page, {"MISS"})
    reset(page)
    page.locator("#screenerOnlyHistory").check()
    expect_symbols(page, {"AAPL", "MSFT"})
    reset(page)
    page.locator("#screenerSector").select_option("Technology")
    page.locator("#screenerIndustry").select_option("Software")
    expect_symbols(page, ALL_SYMBOLS - {"MISS"})

    reset(page)
    page.locator("#screenerCapTier").select_option("mega")
    expect_symbols(page, {"AAPL"})
    page.locator("#screenerCurrency").select_option("EUR")
    expect_symbols(page, {"EUR1"})
    expect(page.locator("#screenerCapTier")).to_have_value("")
    reset(page)
    page.locator("#screenerPriceSince").fill("2026-09-24")
    page.locator("#screenerPriceSince").dispatch_event("change")
    expect_symbols(page, ALL_SYMBOLS - {"MISS", "EUR1"})
    page.locator("#screenerPriceSince").fill("2026-10-01")
    page.locator("#screenerPriceSince").dispatch_event("change")
    expect_symbols(page, set())
    expect(page.locator("#exportScreener")).to_be_disabled()

    reset(page)
    add_rule(page, "pe", "gt", "20")
    add_rule(page, "rsi14", "lt", "30")
    expect_symbols(page, {"EUR1"})
    page.locator("#screenerMatchMode").select_option("any")
    expect_symbols(page, {"AAPL", "MSFT", "EUR1", "ZERO"})
    page.locator("#screenerCurrency").select_option("USD")
    expect_symbols(page, {"AAPL", "MSFT", "ZERO"})

    reset(page)
    add_rule(page, "pe", "missing")
    expect_symbols(page, {"MISS", "FORMULA"})
    page.locator('[data-rule-property="operator"]').select_option("present")
    expect_symbols(page, ALL_SYMBOLS - {"MISS", "FORMULA"})
    page.locator('[data-rule-property="operator"]').select_option("eq")
    page.locator('[data-rule-property="value"]').fill("0")
    expect_symbols(page, {"ZERO"})

    reset(page)
    add_rule(page, "market_cap", "gte", "200", currency="USD")
    expect_symbols(page, {"AAPL"})
    page.locator('[data-rule-property="currency"]').select_option("TWD")
    expect_symbols(page, {"TWD1"})
    reset(page)
    add_rule(page, "volume", "gte", "3")
    expect_symbols(page, {"TWD1", "EUR1"})
    reset(page)
    add_rule(page, "close", "between", "50", "100", "USD")
    expect_symbols(page, {"AAPL", "MSFT"})
    page.locator('[data-rule-property="maximum"]').fill("40")
    expect(page.locator("#screenerRuleError")).to_be_visible()
    expect(page.locator("#exportScreener")).to_be_disabled()
    expect_symbols(page, set())
    page.locator('[data-rule-property="value"]').fill("")
    expect(page.locator("#screenerRuleError")).to_contain_text("numeric value")
    page.locator("[data-remove-rule]").click()
    expect(page.locator("#screenerRuleError")).to_be_hidden()
    expect_symbols(page, ALL_SYMBOLS)


def exercise_views(page: Page) -> None:
    reset(page)
    assert symbols(page) == ["AAPL", "MSFT", "FORMULA", "ZERO", "EUR1", "TWD1", "MISS"]
    expect(page.locator("#screenerSortNote")).to_contain_text("grouped by currency")
    page.locator('[data-screen-sort="market_cap"]').click()
    assert symbols(page) == ["ZERO", "FORMULA", "MSFT", "AAPL", "EUR1", "TWD1", "MISS"]
    page.locator('[data-screen-sort="symbol"]').click()
    assert symbols(page) == sorted(ALL_SYMBOLS)

    page.locator('[data-screen-columns="fundamentals"]').click()
    expect(page.locator('[data-screen-sort="pe"]')).to_be_visible()
    page.locator("#screenerColumnsDetails summary").click()
    page.locator('[data-custom-column="rsi14"]').check()
    expect(page.locator('[data-screen-sort="rsi14"]')).to_be_visible()
    page.locator('[data-custom-column="pe"]').uncheck()
    expect(page.locator('[data-screen-sort="pe"]')).to_have_count(0)
    page.locator("#screenerColumnsDetails summary").click()
    add_rule(page, "rsi14", "lt", "30")
    page.locator("#screenerMatchMode").select_option("any")
    page.locator("#screenName").fill("Browser QA <saved>")
    page.locator("#saveScreener").click()
    expect(page.locator("#savedScreenSelect")).to_have_value("Browser QA <saved>")
    reset(page)
    page.locator("#savedScreenSelect").select_option("Browser QA <saved>")
    expect_symbols(page, {"MSFT", "EUR1", "ZERO"})
    expect(page.locator("#screenerMatchMode")).to_have_value("any")
    expect(page.locator('[data-screen-sort="rsi14"]')).to_be_visible()
    expect(page.locator('[data-screen-sort="pe"]')).to_have_count(0)
    page.reload()
    open_screener(page)
    expect_symbols(page, {"MSFT", "EUR1", "ZERO"})
    expect(page.locator("#screenerMatchMode")).to_have_value("any")
    page.locator("#savedScreenSelect").select_option("Browser QA <saved>")
    page.locator("#deleteScreener").click()
    expect(page.locator("#savedScreenSelect option")).to_have_count(1)

    reset(page)
    page.locator('#collapseScreener').click()
    page.wait_for_function("selected && rows.length > 0")
    page.locator('[data-interval="1h"]').click()
    expect(page.locator('[data-interval="1h"]')).to_have_class("active")
    open_screener(page)
    page.locator('[data-screen-symbol="MSFT"]').click()
    expect(page.locator("#symbol")).to_have_text("MSFT")
    expect(page.locator("#screenerNotice")).to_contain_text("Opened the MSFT chart")
    expect(page.locator("#rows tr")).to_have_count(15)
    expect(page.locator('[data-interval="1d"]')).to_have_class("active")
    expect(page.locator('[data-period="1Y"]')).to_have_class("active")
    chart_save(page)
    page.locator("#screenerOnlyFavorites").check()
    expect_symbols(page, {"MSFT"})
    page.locator('[data-screen-favorite="MSFT"]').click()
    expect_symbols(page, set())
    expect(page.locator("#save")).to_have_attribute("aria-pressed", "false")
    chart_save(page)
    expect_symbols(page, {"MSFT"})
    chart_save(page)
    expect_symbols(page, set())
    reset(page)
    page.locator('[data-screen-favorite="MISS"]').click()
    page.locator("#screenerOnlyFavorites").check()
    expect_symbols(page, {"MISS"})
    page.locator('[data-screen-favorite="MISS"]').click()
    expect_symbols(page, set())
    reset(page)
    header, rows = read_export(page)
    assert len(rows) == 7
    assert {row["symbol"] for row in rows} == ALL_SYMBOLS
    assert {"currency", "date", "metadata_date", "has_history"} <= set(header)
    formula = next(row for row in rows if row["symbol"] == "FORMULA")
    assert formula["name"].startswith("'=SUM(1,2)"), formula
    assert next(row for row in rows if row["symbol"] == "MISS")["market_cap"] == ""
    page.locator("#screenerExportColumns").select_option("all")
    header, rows = read_export(page)
    assert {"return_6m_pct", "return_ytd_pct", "adx14", "macd_histogram_pct",
            "bollinger_width_pct", "avg_turnover20"} <= set(header)
    assert next(row for row in rows if row["symbol"] == "AAPL")["avg_turnover20"] == "200000000"
    page.locator("#screenerExportColumns").select_option("visible")


def exercise_pagination_and_errors(page: Page, source: dict) -> None:
    base = copy.deepcopy(source["payload"])
    source["payload"]["rows"].extend(
        dict(symbol=f"QA{i:03}", name=f"Pagination {i}", currency="USD", has_history=False)
        for i in range(60))
    page.locator("#refreshScreener").click()
    expect(page.locator("#screenerResultCount")).to_have_text("67 matches of 67 stocks")
    expect(page.locator("#screenerTableBody [data-screen-symbol]")).to_have_count(50)
    page.locator("#screenerNext").click()
    expect(page.locator("#screenerTableBody [data-screen-symbol]")).to_have_count(17)
    expect(page.locator("#screenerNext")).to_be_disabled()
    assert len(read_export(page)[1]) == 67, "Export must include matches on every page"
    page.locator("#screenerPageSize").select_option("20")
    expect(page.locator("#screenerTableBody [data-screen-symbol]")).to_have_count(20)
    expect(page.locator("#screenerPrevious")).to_be_disabled()
    page.locator("#screenerPageSize").select_option("100")
    expect(page.locator("#screenerTableBody [data-screen-symbol]")).to_have_count(67)
    expect(page.locator("#screenerNext")).to_be_disabled()
    page.locator("#screenerSearch").fill("AAPL")
    expect_symbols(page, {"AAPL"})
    expect(page.locator("#screenerPrevious")).to_be_disabled()
    reset(page)
    source["status"] = 503
    page.locator("#refreshScreener").click()
    expect(page.locator("#screenerError")).to_contain_text("previous snapshot")
    expect(page.locator("#screenerResultCount")).to_have_text("67 matches of 67 stocks")
    source.update(status=200, payload=base)
    page.locator("#refreshScreener").click()
    expect(page.locator("#screenerError")).to_be_hidden()
    expect_symbols(page, ALL_SYMBOLS)
    page.locator("#collapseScreener").click()
    expect(page.locator("#screenerContent")).to_be_hidden()
    page.reload()
    expect(page.locator("#screenerContent")).to_be_hidden()
    page.locator("#toggleScreener").click()
    expect_symbols(page, ALL_SYMBOLS)


def mobile_check(page: Page) -> None:
    page.set_viewport_size({"width": 390, "height": 844})
    open_screener(page)
    page.locator('[data-screen-preset="momentum"]').click()
    page.locator("#screenerColumnsDetails summary").click()
    page.locator("#stockScreener").scroll_into_view_if_needed()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Mobile page overflow"
    assert page.locator(".screener-table-wrap").evaluate(
        "element => element.scrollWidth > element.clientWidth"), "Wide tables must scroll within their container"
    artifacts = ROOT / "data"
    artifacts.mkdir(exist_ok=True)
    page.screenshot(path=str(artifacts / "screener-mobile.png"), full_page=True)


def exercise_static(browser, url: str) -> None:
    context = browser.new_context(viewport={"width": 1440, "height": 1000})
    context.add_init_script((ROOT / "web" / "static-data.js").read_text(encoding="utf-8-sig"))
    page = context.new_page()
    errors, requests = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: requests.append(urlparse(request.url).path))
    payload = fixture()
    catalog = [dict(row, kind="Stocks", change=0) for row in payload["rows"] if row["has_history"]]
    page.route("**/symbols.json", lambda route: route.fulfill(json=catalog))
    source = {"status": 200, "payload": payload}
    page.route("**/screener.json", lambda route: route.fulfill(status=source["status"], json=source["payload"]))
    prices = {"columns": ["date", "open", "high", "low", "close", "adjusted_close", "volume"],
              "rows": [[f"2026-09-{day:02}", 100, 102, 99, 101, 101, 1000] for day in range(1, 25)]}
    page.route("**/prices/*.json.gz", lambda route: route.fulfill(
        content_type="application/gzip", body=gzip.compress(json.dumps(prices).encode())))
    page.goto(url)
    open_screener(page)
    expect_symbols(page, ALL_SYMBOLS)
    expect(page.locator("#rows tr")).to_have_count(15)
    page.locator('[data-screen-preset="mega"]').click()
    expect_symbols(page, {"AAPL"})
    source["status"] = 404
    page.locator("#refreshScreener").click()
    expect(page.locator("#screenerError")).to_contain_text("previous snapshot")
    expect_symbols(page, {"AAPL"})
    assert "/screener.json" in requests and "/api/screener" not in requests
    page.locator('#workspaceOpenStrategy').click()
    page.locator('#st-fast').fill('2')
    page.locator('#st-slow').fill('3')
    page.locator('#strategyRun').click()
    expect(page.locator('#strategyResults')).to_be_visible()
    expect(page.locator('#strategyError')).not_to_be_visible()
    page.locator('#strategyCustomize').click()
    expect(page.locator('#st-kind')).to_have_value('custom')
    page.locator('#strategyRun').click()
    expect(page.locator('#strategyResults')).to_be_visible()
    page.locator('#st-kind').select_option('golden_cross')
    page.locator('#strategyRun').click()
    expect(page.locator('#strategyError')).to_contain_text('203 bars')
    page.locator('#st-kind').select_option('roc')
    page.locator('#st-lookback').fill('2')
    page.locator('#strategyRun').click()
    expect(page.locator('#strategyResults')).to_be_visible()
    expect(page.locator('#strategyError')).not_to_be_visible()
    page.locator('#strategyPerformanceTab').click()
    expect(page.locator('#strategyPerformance')).to_contain_text('2026-09')
    page.locator('#workspaceTheme').select_option('dark')
    expect(page.locator('html')).to_have_attribute('data-theme', 'dark')
    assert not errors, errors
    context.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1100}, accept_downloads=True)
        live = context.request.get(args.url.rstrip("/") + "/api/screener", timeout=120_000)
        assert live.ok, f"Live screener API returned {live.status}"
        response = live.json()
        assert isinstance(response.get("rows"), list) and response["rows"], "Live stock universe is empty"
        assert len({row["symbol"] for row in response["rows"]}) == len(response["rows"])
        added_metrics = {"return_6m_pct", "return_ytd_pct", "avg_turnover20", "ema20_distance_pct",
                         "ema50_distance_pct", "macd_pct", "macd_signal_pct", "macd_histogram_pct",
                         "adx14", "stochastic_k", "stochastic_d", "bollinger_position_pct",
                         "bollinger_width_pct", "distance_52w_low_pct", "range_52w_position_pct"}
        assert all(added_metrics <= set(row) for row in response["rows"])
        live_page = context.new_page()
        live_page.goto(args.url)
        open_screener(live_page)
        expect(live_page.locator("#screenerContent")).to_have_attribute("aria-busy", "false", timeout=30_000)
        expect(live_page.locator("#screenerTableBody [data-screen-symbol]").first).to_be_visible()
        (ROOT / "data").mkdir(exist_ok=True)
        live_page.locator("#stockScreener").screenshot(path=str(ROOT / "data" / "screener-desktop.png"))
        live_page.close()
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        source = {"status": 200, "payload": fixture()}
        page.route("**/api/screener", lambda route: route.fulfill(
            status=source["status"], json=source["payload"] if source["status"] == 200 else {"error": "Test refresh failure"}))
        page.goto(args.url)
        open_screener(page)
        expect(page.locator("#symbol")).to_have_text("AAPL")
        exercise_filters(page)
        exercise_views(page)
        exercise_pagination_and_errors(page, source)
        mobile_check(page)
        assert not errors, errors
        context.close()
        exercise_static(browser, args.url)
        browser.close()
    print("PASS: live API, presets, AND/OR rules, null/zero handling, currency safety, sorting, custom columns, "
          "saved screens, chart navigation, favorites, full CSV export, pagination, refresh recovery, "
          "static snapshot, mobile overflow; no JavaScript errors.")


if __name__ == "__main__":
    main()
