"""Check chart scale and navigation controls against the real local dataset.

Run with .venv/Scripts/python.exe smoke_chart_navigation.py --url http://127.0.0.1:8765.
Uses a fresh headless Edge profile, real UI events, and read-only geometry checks.
No market data or user browser state is changed.
"""
from __future__ import annotations

import argparse
import math
import re

from playwright.sync_api import Page, expect, sync_playwright


def chart_state(page: Page) -> dict:
    return page.evaluate("""() => {
        const g = geometry(), middle = g.top + g.ph / 2;
        return {
            count: viewCount, start: viewStart, origin: viewStart - chartScale.offset, total: rows.length,
            first: g.data[0]?.date, last: g.data.at(-1)?.date,
            latest: rows.at(-1)?.date,
            upper: g.price(g.top), lower: g.price(g.top + g.ph),
            middle: g.price(middle), midpointY: middle,
            roundtrip: g.y(g.price(middle)),
            width: g.pw, height: g.ph,
            finite: [g.price(g.top), g.price(middle), g.price(g.top + g.ph),
                     g.y(g.price(middle))].every(Number.isFinite)
        };
    }""")


def chart_point(page: Page, x: float = .5, y: float = .5, *, axis: bool = False) -> dict:
    page.locator("#chart").scroll_into_view_if_needed()
    return page.evaluate("""([x, y, axis]) => {
        const g = geometry();
        return {
            x: g.box.x + (axis ? g.left + g.pw + g.right * .45 : g.left + g.pw * x),
            y: g.box.y + g.top + g.ph * y
        };
    }""", [x, y, axis])


def drag(page: Page, point: dict, dx: float, dy: float) -> None:
    page.mouse.move(point["x"], point["y"])
    page.mouse.down()
    page.mouse.move(point["x"] + dx, point["y"] + dy, steps=12)
    page.mouse.up()


def assert_geometry(state: dict) -> None:
    assert state["finite"], f"Scale geometry contains non-finite values: {state}"
    assert state["upper"] > state["lower"], f"Price axis must increase upward: {state}"
    assert math.isclose(state["roundtrip"], state["midpointY"], abs_tol=1e-6), state
    assert state["width"] > 40 and state["height"] > 40, state


def axis_labels(page: Page) -> list[str]:
    """Observe rendered labels, then restore the canvas method immediately."""
    return page.evaluate("""() => {
        const g = geometry(), context = g.canvas.getContext('2d');
        const original = context.fillText, labels = [];
        context.fillText = function(text, x, y, ...rest) {
            if (x > g.left + g.pw && y >= g.top && y <= g.top + g.ph + 5)
                labels.push(String(text));
            return original.call(this, text, x, y, ...rest);
        };
        try { draw(); } finally { context.fillText = original; }
        return labels;
    }""")


def set_auto(page: Page, enabled: bool) -> None:
    button = page.locator("#chartAutoScale")
    if (button.get_attribute("aria-pressed") == "true") != enabled:
        button.click()
    expect(button).to_have_attribute("aria-pressed", str(enabled).lower())


def set_mode(page: Page, mode: str) -> None:
    page.locator("#chartScaleMenuButton").click()
    button = page.locator(f'[data-scale-mode="{mode}"]')
    expect(button).to_be_visible()
    button.click()
    expect(page.locator("#chartLogScale")).to_have_attribute(
        "aria-pressed", "true" if mode == "log" else "false")
    assert_geometry(chart_state(page))


def exercise_modes(page: Page) -> None:
    set_mode(page, "linear")
    set_auto(page, True)
    linear = chart_state(page)
    assert math.isclose(linear["middle"], (linear["upper"] + linear["lower"]) / 2,
                        rel_tol=1e-8), linear

    page.locator("#chartLogScale").click()
    expect(page.locator("#chartLogScale")).to_have_attribute("aria-pressed", "true")
    logarithmic = chart_state(page)
    assert_geometry(logarithmic)
    assert logarithmic["lower"] > 0, logarithmic
    assert math.isclose(logarithmic["middle"],
                        math.sqrt(logarithmic["upper"] * logarithmic["lower"]),
                        rel_tol=1e-8), "Log scale must use multiplicative price spacing"

    # Alternate scales must keep anchors in raw price coordinates. A price at
    # any sampled pixel must round-trip to that pixel regardless of its label.
    for mode in ("percent", "indexed", "log", "linear"):
        set_mode(page, mode)
        assert page.evaluate("""() => {
            const g = geometry();
            return [.1, .3, .7, .9].every(fraction => {
                const y = g.top + fraction * g.ph;
                return Math.abs(g.y(g.price(y)) - y) < 1e-6;
            });
        }"""), f"Drawing/crosshair coordinates failed on {mode} scale"
        labels = axis_labels(page)
        assert len(labels) >= 3, f"Missing price-axis labels in {mode} mode: {labels}"
        if mode == "percent":
            assert all("%" in label for label in labels), labels
        elif mode == "indexed":
            values = [float(re.sub(r"[^0-9.+-]", "", label.replace("−", "-"))) for label in labels]
            assert min(values) <= 100 <= max(values), \
                f"An indexed chart must include its visible starting-price baseline of 100: {labels}"
        else:
            assert all("%" not in label for label in labels), labels


def exercise_auto_panning(page: Page) -> None:
    point = chart_point(page)
    before_zoom = chart_state(page)
    page.mouse.move(point["x"], point["y"])
    page.mouse.wheel(0, -350)
    page.wait_for_function("count => viewCount < count", arg=before_zoom["count"])

    for mode in ("linear", "log", "percent", "indexed"):
        set_mode(page, mode)
        before = chart_state(page)
        point = chart_point(page)
        page.mouse.move(point["x"], point["y"])
        page.mouse.down()
        for dx, dy in ((20, 24), (45, -30)):
            page.mouse.move(point["x"] + dx, point["y"] + dy, steps=4)
            expect(page.locator("#chartAutoScale")).to_have_attribute("aria-pressed", "true")
            assert page.evaluate("""() => {
                const g = geometry();
                const shown = g.data.filter((_, i) => g.x(i) >= g.left && g.x(i) <= g.left + g.pw);
                return chartScale.bounds === null && shown.length > 0 && shown.every(bar =>
                    g.y(bar.high) >= g.top - 1 && g.y(bar.low) <= g.top + g.ph + 1);
            }"""), f"Auto must fit visible candles during diagonal dragging in {mode} mode"
        page.mouse.up()
        after = chart_state(page)
        assert after["origin"] < before["origin"], "Auto dragging must reveal earlier bars"
        assert after["count"] == before["count"], "Auto panning must preserve time zoom"
        expect(page.locator("#chartAutoScale")).to_have_attribute("aria-pressed", "true")

        # A purely vertical movement must neither disable Auto nor freeze bounds.
        drag(page, chart_point(page), 0, 35)
        vertical = chart_state(page)
        expect(page.locator("#chartAutoScale")).to_have_attribute("aria-pressed", "true")
        for key in ("origin", "upper", "lower"):
            assert math.isclose(vertical[key], after[key], abs_tol=1e-8), \
                f"Vertical dragging must preserve automatic price fitting: {mode}, {key}"
    page.locator("#chartResetView").click()


def exercise_extended_zoom(page: Page) -> None:
    page.locator('#chartResetView').click()
    before = chart_state(page)
    point = chart_point(page)
    page.mouse.move(point['x'], point['y'])
    page.mouse.wheel(0, 300)
    page.wait_for_function('count => viewCount > count', arg=before['count'])
    zoomed = chart_state(page)
    assert zoomed['count'] > zoomed['total'], 'Wheel zoom must extend beyond the loaded bar count'
    expect(page.locator('#chartAutoScale')).to_have_attribute('aria-pressed', 'true')
    assert_geometry(zoomed)

    point = page.evaluate('''() => {
        const g = geometry();
        return {x: g.box.x + g.left + g.pw * .6, y: g.box.y + g.top + g.ph + 15};
    }''')
    page.keyboard.down('Shift')
    try:
        drag(page, point, -120, 0)
    finally:
        page.keyboard.up('Shift')
    axis_zoom = chart_state(page)
    assert axis_zoom['count'] > zoomed['count'], 'Time-axis drag must use the extended zoom range'
    drag(page, chart_point(page), 60, 25)
    expect(page.locator('#chartAutoScale')).to_have_attribute('aria-pressed', 'true')
    assert chart_state(page)['count'] == axis_zoom['count']

    # At the widest zoom, even extreme panning must keep real data on screen.
    page.evaluate('zoomChartTime(100); draw()')
    assert page.evaluate('viewCount === rows.length * 5')
    for target in (-1e9, 1e9):
        page.evaluate('target => { moveChartTime(target); draw(); }', target)
        assert page.evaluate('''() => {
            const g = geometry();
            return g.data.some((bar, i) => g.x(i) >= g.left && g.x(i) <= g.left + g.pw);
        }'''), 'Extended zoom must not strand the chart in empty space'
        assert_geometry(chart_state(page))
    point = chart_point(page)
    page.mouse.move(point['x'], point['y'])
    page.mouse.wheel(0, -300)
    page.wait_for_function('viewCount < rows.length * 5')
    page.locator('#chartResetView').click()
    assert chart_state(page)['count'] == before['count']


def exercise_navigation(page: Page) -> None:
    set_mode(page, "linear")
    set_auto(page, True)
    before_zoom = chart_state(page)
    point = chart_point(page, .65, .45)
    page.mouse.move(point["x"], point["y"])
    page.mouse.wheel(0, -350)
    page.wait_for_function("count => viewCount < count", arg=before_zoom["count"])
    zoomed = chart_state(page)
    assert_geometry(zoomed)
    assert zoomed["count"] < zoomed["total"], "Zoom must leave history available for panning"

    set_auto(page, False)
    before_pan = chart_state(page)
    drag(page, chart_point(page), 45, 30)
    panned = chart_state(page)
    assert panned["start"] < before_pan["start"], "Dragging right must reveal earlier bars"
    assert panned["count"] == before_pan["count"], "Panning must preserve time zoom"
    assert not math.isclose(panned["middle"], before_pan["middle"], rel_tol=1e-8), \
        "Vertical plot dragging must pan a manual price scale"
    assert math.isclose(panned["upper"] - panned["lower"],
                        before_pan["upper"] - before_pan["lower"], rel_tol=1e-8), \
        "Linear vertical panning must preserve the price span"
    expect(page.locator("#chartAutoScale")).to_have_attribute("aria-pressed", "false")

    # Time zoom must not silently autoscale or reset a manually positioned axis.
    point = chart_point(page, .5, .45)
    page.mouse.move(point["x"], point["y"])
    page.mouse.wheel(0, -250)
    page.wait_for_function("count => viewCount < count", arg=panned["count"])
    manual_zoom = chart_state(page)
    for key in ("upper", "lower"):
        assert math.isclose(manual_zoom[key], panned[key], rel_tol=1e-8), \
            "Time zoom must preserve a manual price range"

    before_axis = chart_state(page)
    drag(page, chart_point(page, y=.5, axis=True), 0, 45)
    after_axis = chart_state(page)
    assert_geometry(after_axis)
    assert not math.isclose(after_axis["upper"] - after_axis["lower"],
                            before_axis["upper"] - before_axis["lower"], rel_tol=1e-6), \
        "Dragging the price axis must change its scale"
    assert (after_axis["start"], after_axis["count"]) == \
        (before_axis["start"], before_axis["count"]), "Price scaling must preserve the time view"

    point = chart_point(page, axis=True)
    page.mouse.dblclick(point["x"], point["y"])
    expect(page.locator("#chartAutoScale")).to_have_attribute("aria-pressed", "true")
    assert_geometry(chart_state(page))
    assert page.evaluate("""() => {
        const g = geometry();
        return g.data.every(bar => g.y(bar.high) >= g.top - 1
            && g.y(bar.low) <= g.top + g.ph + 1);
    }"""), "Double-click autoscale must fit all visible candle highs and lows"

    # Log mode has the same pan/scale gestures and must never cross zero.
    set_mode(page, "log")
    set_auto(page, False)
    before_log_pan = chart_state(page)
    drag(page, chart_point(page), 0, 30)
    after_log_pan = chart_state(page)
    assert_geometry(after_log_pan)
    assert after_log_pan["lower"] > 0
    assert not math.isclose(after_log_pan["middle"], before_log_pan["middle"], rel_tol=1e-8)
    assert math.isclose(after_log_pan["upper"] / after_log_pan["lower"],
                        before_log_pan["upper"] / before_log_pan["lower"], rel_tol=1e-8), \
        "Logarithmic panning must preserve the price ratio"

    page.locator("#chartResetView").click()
    expect(page.locator("#chartAutoScale")).to_have_attribute("aria-pressed", "true")
    reset = chart_state(page)
    assert_geometry(reset)
    assert reset["last"] == reset["latest"], "Reset view must return to the latest bar"


def exercise_resize(page: Page) -> None:
    for width, height in ((1200, 900), (800, 900), (390, 844)):
        page.set_viewport_size({"width": width, "height": height})
        page.locator("#chart").scroll_into_view_if_needed()
        page.wait_for_function("document.documentElement.scrollWidth <= innerWidth + 1")
        assert_geometry(chart_state(page))
        canvas = page.locator("#chart").bounding_box()
        assert canvas and canvas["width"] <= width, f"Chart overflows at {width}px: {canvas}"
        # The controls must remain reachable after the chart reflows.
        expect(page.locator("#chartAutoScale")).to_be_visible()
        expect(page.locator("#chartLogScale")).to_be_visible()
        expect(page.locator("#chartScaleMenuButton")).to_be_visible()


def exercise_log_drawings_and_inversion(page: Page) -> None:
    set_mode(page, "log")
    page.locator("#chartResetView").click()
    if "active" in (page.locator("#magnet").get_attribute("class") or "").split():
        page.locator("#magnet").click()
    page.locator('[data-tool="trend"]').click()
    for x, y in ((.45, .7), (.78, .35)):
        point = chart_point(page, x, y)
        page.mouse.click(point["x"], point["y"])
    page.wait_for_function("drawings.length === 1")
    original = page.evaluate("JSON.parse(JSON.stringify(drawings[0]))")

    def line_state() -> dict:
        return page.evaluate("""() => {
            const g = geometry(), drawing = drawings[0];
            const a = anchorPoint(g, drawing.a), b = anchorPoint(g, drawing.b);
            return {drawing, a, b, middle: {x: g.box.x + (a.x+b.x)/2,
                y: g.box.y + (a.y+b.y)/2}};
        }""")

    before = line_state()
    drag(page, before["middle"], 24, 22)
    after = line_state()
    assert after["drawing"]["a"]["date"] != original["a"]["date"]
    for endpoint in ("a", "b"):
        assert math.isclose(after[endpoint]["y"] - before[endpoint]["y"], 22, abs_tol=.02), \
            f"Log drawing anchor {endpoint} must follow the actual pixel drag"
    assert math.isclose(after["drawing"]["a"]["price"] / original["a"]["price"],
                        after["drawing"]["b"]["price"] / original["b"]["price"], rel_tol=1e-8), \
        "Moving a log-scale drawing must preserve its price ratio"

    page.locator("#chartScaleMenuButton").click()
    page.locator("#chartInvertScale").click()
    expect(page.locator("#chartInvertScale")).to_have_attribute("aria-checked", "true")
    inverted = chart_state(page)
    assert inverted["finite"] and inverted["upper"] < inverted["lower"]
    assert math.isclose(inverted["roundtrip"], inverted["midpointY"], abs_tol=1e-6)
    before_inverted = line_state()
    drag(page, before_inverted["middle"], 0, 20)
    after_inverted = line_state()
    for endpoint in ("a", "b"):
        assert math.isclose(after_inverted[endpoint]["y"] - before_inverted[endpoint]["y"],
                            20, abs_tol=.02), "Inverted log drawing must follow the pointer"
        assert after_inverted["drawing"][endpoint]["price"] > \
            before_inverted["drawing"][endpoint]["price"], \
            "Moving downward on an inverted scale must increase the anchor price"

    set_auto(page, False)
    before_price_y = page.evaluate("geometry().y(rows.at(-1).close)")
    drag(page, chart_point(page, .12, .18), 0, 24)
    after_price_y = page.evaluate("geometry().y(rows.at(-1).close)")
    assert math.isclose(after_price_y - before_price_y, 24, abs_tol=.02), \
        "Manual inverted-scale panning must move candles with the pointer"
    page.locator("#chartScaleMenuButton").click()
    page.locator("#chartInvertScale").click()
    expect(page.locator("#chartInvertScale")).to_have_attribute("aria-checked", "false")
    page.locator("#chartResetView").click()


def exercise_time_axis_positioning(page: Page) -> None:
    page.locator('#chartResetView').click()
    page.evaluate('''() => {
        moveChartTime(rows.length * .3, Math.floor(rows.length * .4));
        const g = geometry(); chartScale.auto = false; chartScale.bounds = [g.low, g.high];
        draw();
    }''')
    before = chart_state(page)
    point = page.evaluate('''() => {
        const g = geometry();
        return {x:g.box.x + g.left + g.pw*.5, y:g.box.y + g.top + g.ph + 15, width:g.pw};
    }''')
    drag(page, point, 65, -12)
    after = chart_state(page)
    assert after['count'] == before['count'], 'Time-axis positioning must preserve zoom'
    assert math.isclose(after['origin'], before['origin'] - 65 / point['width'] * before['count'], abs_tol=.01)
    for key in ['lower', 'upper']:
        assert math.isclose(after[key], before[key], abs_tol=1e-8), 'Time-axis dragging must not move the price scale'
    drag(page, point, -65, 0)
    assert math.isclose(chart_state(page)['origin'], before['origin'], abs_tol=.01)
    assert page.evaluate('chartNavigationGesture === null')
    # Touch uses the same axis hit area, including while a drawing tool is active.
    page.evaluate("tool = 'trend'")
    session = page.context.new_cdp_session(page)
    session.send('Input.dispatchTouchEvent', {'type':'touchStart', 'touchPoints':[{'x':point['x'],'y':point['y']}]})
    session.send('Input.dispatchTouchEvent', {'type':'touchMove', 'touchPoints':[{'x':point['x']+40,'y':point['y']}]})
    session.send('Input.dispatchTouchEvent', {'type':'touchEnd', 'touchPoints':[]})
    session.detach()
    page.wait_for_function('origin => viewStart - chartScale.offset < origin', arg=before['origin'])
    assert chart_state(page)['count'] == before['count']
    assert page.evaluate('chartNavigationGesture === null && pending === null')
    page.evaluate("tool = 'cursor'")
    page.locator('#chartResetView').click()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1100})
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        expect(page.locator("#symbol")).to_have_text("AAPL")
        page.wait_for_function("rows.length > 30 && geometry().data.length > 30")
        expect(page.locator("#error")).to_be_hidden()
        exercise_modes(page)
        exercise_auto_panning(page)
        exercise_extended_zoom(page)
        exercise_time_axis_positioning(page)
        exercise_navigation(page)
        exercise_log_drawings_and_inversion(page)
        exercise_resize(page)
        assert not errors, errors
        context.close()
        browser.close()
    print("PASS: scale modes, price/pixel mapping, Auto preserved during diagonal/vertical drag, time zoom, manual X/Y pan, axis drag, "
          "log pan/drawings, inverted scale, double-click autoscale, reset view, responsive layout; "
          "no JavaScript errors.")


if __name__ == "__main__":
    main()
