"""Exercise chart editing in a fresh headless Edge profile against the running app.

Run with .venv/Scripts/python smoke_chart_editing.py [--url http://127.0.0.1:8765].
The test uses actual chart data and mouse interactions; it never writes app state
directly or touches the user's browser profile.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright


ARTIFACTS = Path(__file__).resolve().parent / "data"


def wait_for_chart(page: Page) -> None:
    expect(page.locator("#symbol")).to_have_text("AAPL")
    page.wait_for_function("rows.length > 30 && geometry().data.length > 30")
    expect(page.locator("#error")).to_be_hidden()
    page.locator("#chart").scroll_into_view_if_needed()


def drawings_state(page: Page) -> list[dict]:
    return page.evaluate("JSON.parse(localStorage.getItem('atlas.drawings.AAPL.1d') || '[]')")


def plot_point(page: Page, x_fraction: float, y_fraction: float) -> dict:
    """Get a real bar's x coordinate and a price coordinate inside the plot."""
    return page.evaluate(
        """([x, y]) => {
            const g = geometry();
            const i = Math.round((g.data.length - 1) * x);
            return {x: g.box.x + g.x(i), y: g.box.y + g.top + g.ph * y};
        }""",
        [x_fraction, y_fraction],
    )


def click_point(page: Page, point: dict) -> None:
    page.mouse.click(point["x"], point["y"])


def drag_point(page: Page, point: dict, dx: float, dy: float) -> None:
    page.mouse.move(point["x"], point["y"])
    page.mouse.down()
    page.mouse.move(point["x"] + dx, point["y"] + dy, steps=12)
    page.mouse.up()


def set_control(page: Page, selector: str, value: str) -> None:
    """Exercise native input controls, including color and transparency inputs."""
    control = page.locator(selector)
    if control.get_attribute("type") == "range":
        minimum = float(control.get_attribute("min") or 0)
        maximum = float(control.get_attribute("max") or 100)
        step = float(control.get_attribute("step") or 1)
        target = float(value)
        from_minimum = target - minimum <= maximum - target
        control.focus()
        control.press("Home" if from_minimum else "End")
        count = round(((target - minimum) if from_minimum else (maximum - target)) / step)
        assert 0 <= count <= 200, "Unexpected transparency control range"
        for _ in range(count):
            control.press("ArrowRight" if from_minimum else "ArrowLeft")
        control.press("Tab")
    else:
        control.fill(value)
        control.dispatch_event("change")


def shape_point(page: Page, drawing_id: str, part: str = "middle") -> dict:
    page.locator("#chart").scroll_into_view_if_needed()
    return page.evaluate(
        """([id, part]) => {
            const g = geometry(), d = drawings.find(item => item.id === id);
            const a = anchorPoint(g, d.a), b = anchorPoint(g, d.b);
            const p = part === 'a' ? a : part === 'b' ? b : part === 'ab'
                ? {x: a.x, y: b.y} : {x: (a.x + b.x) / 2, y: (a.y + b.y) / 2};
            return {x: g.box.x + p.x, y: g.box.y + p.y};
        }""",
        [drawing_id, part],
    )


def stored_shape(page: Page, drawing_id: str) -> dict:
    return next(item for item in drawings_state(page) if item["id"] == drawing_id)


def select_object(page: Page, drawing_id: str) -> None:
    if not page.locator('#workspaceSideobjects').is_visible():
        page.locator('[data-side-panel="objects"]').click()
    if page.locator("#drawingObjects").get_attribute("open") is None:
        page.locator("#drawingObjects summary").click()
    page.locator(f'[data-select-drawing="{drawing_id}"]').click()
    page.wait_for_function("id => selectedDrawing === id", arg=drawing_id)
    page.locator("#drawingObjects summary").click()
    page.locator('[data-side-panel="watchlist"]').click()
    if page.locator('#workspaceDrawingSettings').get_attribute('open') is None:
        page.locator('#workspaceDrawingSettings > summary').click()


def assert_translation(before: dict, after: dict) -> None:
    assert before["a"]["date"] != after["a"]["date"], "Dragging must move the first anchor in time"
    assert before["b"]["date"] != after["b"]["date"], "Dragging must move the second anchor in time"
    delta_a = after["a"]["price"] - before["a"]["price"]
    delta_b = after["b"]["price"] - before["b"]["price"]
    assert abs(delta_a) > 0.01, "Dragging must move the drawing in price"
    assert math.isclose(delta_a, delta_b, abs_tol=1e-8), "Whole-object drag must preserve its price span"


def exercise_drawings(page: Page) -> None:
    assert not drawings_state(page), "A fresh browser context must start without drawings"
    page.locator("#magnet").click()
    expect(page.locator("#magnet")).not_to_have_class("active")

    page.locator('[data-tool="trend"]').click()
    page.locator("#chart").scroll_into_view_if_needed()
    click_point(page, plot_point(page, 0.18, 0.68))
    click_point(page, plot_point(page, 0.45, 0.38))
    assert len(drawings_state(page)) == 1
    trend = drawings_state(page)[0]
    trend_id = trend["id"]
    expect(page.locator('[data-tool="cursor"]')).to_have_class("active")

    # A selected line can be moved as a whole; undo/redo restores exact anchors.
    click_point(page, plot_point(page, 0.10, 0.15))
    assert page.evaluate("selectedDrawing") is None
    drag_point(page, shape_point(page, trend_id), 38, -24)
    moved_trend = stored_shape(page, trend_id)
    assert page.evaluate("selectedDrawing") == trend_id
    assert_translation(trend, moved_trend)
    page.locator("#undoDrawing").click()
    assert stored_shape(page, trend_id) == trend
    page.locator("#redoDrawing").click()
    assert stored_shape(page, trend_id) == moved_trend

    # Dragging one endpoint leaves the opposite endpoint unchanged.
    drag_point(page, shape_point(page, trend_id, "b"), 24, -18)
    resized_trend = stored_shape(page, trend_id)
    assert resized_trend["a"] == moved_trend["a"]
    assert resized_trend["b"]["date"] != moved_trend["b"]["date"]
    assert resized_trend["b"]["price"] != moved_trend["b"]["price"]

    page.locator('[data-tool="rectangle"]').click()
    page.locator("#chart").scroll_into_view_if_needed()
    click_point(page, plot_point(page, 0.60, 0.67))
    click_point(page, plot_point(page, 0.84, 0.36))
    assert len(drawings_state(page)) == 2
    rectangle = drawings_state(page)[1]
    rectangle_id = rectangle["id"]
    drag_point(page, shape_point(page, rectangle_id), -24, 16)
    moved_rectangle = stored_shape(page, rectangle_id)
    assert_translation(rectangle, moved_rectangle)
    drag_point(page, shape_point(page, rectangle_id, "ab"), -20, -18)
    resized_rectangle = stored_shape(page, rectangle_id)
    assert resized_rectangle["a"]["date"] != moved_rectangle["a"]["date"]
    assert resized_rectangle["a"]["price"] == moved_rectangle["a"]["price"]
    assert resized_rectangle["b"]["date"] == moved_rectangle["b"]["date"]
    assert resized_rectangle["b"]["price"] != moved_rectangle["b"]["price"]

    set_control(page, "#drawingColor", "#ff3366")
    page.locator("#drawingWidth").select_option("4")
    page.locator("#drawingDash").select_option("dashed")
    set_control(page, "#drawingTransparency", "35")
    set_control(page, "#drawingFillColor", "#22cc88")
    set_control(page, "#drawingFillTransparency", "65")
    style = stored_shape(page, rectangle_id)["style"]
    assert style == {
        "color": "#ff3366", "width": 4, "dash": "dashed", "transparency": 35,
        "fillColor": "#22cc88", "fillTransparency": 65,
    }, style
    expect(page.locator("#drawingTransparencyValue")).to_have_text("35%")
    expect(page.locator("#drawingFillTransparencyValue")).to_have_text("65%")
    assert stored_shape(page, trend_id)["style"]["color"] != style["color"], "Style edits must affect only the selection"

    # Price entry is another edit path, with actual number-input events.
    price_a = round(stored_shape(page, rectangle_id)["a"]["price"] + 0.5, 4)
    set_control(page, "#drawingPriceA", str(price_a))
    assert stored_shape(page, rectangle_id)["a"]["price"] == price_a

    page.locator("#lockDrawing").click()
    locked = copy.deepcopy(stored_shape(page, rectangle_id))
    assert locked["locked"] is True
    expect(page.locator("#drawingColor")).to_be_disabled()
    view_start = page.evaluate("viewStart")
    drag_point(page, shape_point(page, rectangle_id), 34, 20)
    assert stored_shape(page, rectangle_id) == locked, "Locked drawings must resist dragging"
    assert page.evaluate("viewStart") == view_start, "Dragging a locked drawing must not pan the chart"
    if page.locator('#workspaceDrawingSettings').get_attribute('open') is None:
        page.locator('#workspaceDrawingSettings > summary').click()
    page.locator("#lockDrawing").click()
    page.locator("#hideDrawing").click()
    assert stored_shape(page, rectangle_id)["hidden"] is True
    click_point(page, shape_point(page, rectangle_id))
    assert page.evaluate("selectedDrawing") is None, "Hidden drawings must not be selectable on the canvas"
    select_object(page, rectangle_id)
    expect(page.locator("#hideDrawing")).to_have_text("Show")
    page.locator("#hideDrawing").click()
    assert stored_shape(page, rectangle_id)["hidden"] is False

    expected_drawings = copy.deepcopy(drawings_state(page))
    page.reload()
    wait_for_chart(page)
    assert drawings_state(page) == expected_drawings, "Anchors, styles, and visibility must survive reload"
    select_object(page, rectangle_id)
    expect(page.locator("#drawingColor")).to_have_value("#ff3366")
    expect(page.locator("#drawingFillTransparency")).to_have_value("65")

    # Delete edits text while an input is focused, then removes a selected shape
    # when focus returns to a regular chart control.
    page.locator("#search").fill("AAPL")
    page.locator("#search").press("Control+A")
    page.locator("#search").press("Delete")
    expect(page.locator("#search")).to_have_value("")
    assert drawings_state(page) == expected_drawings
    page.locator("#lockDrawing").focus()
    page.keyboard.press("Delete")
    assert len(drawings_state(page)) == 1
    assert drawings_state(page)[0]["id"] == trend_id
    page.keyboard.press("Control+z")
    assert drawings_state(page) == expected_drawings
    page.keyboard.press("Control+Shift+z")
    assert len(drawings_state(page)) == 1
    page.locator("#undoDrawing").click()
    assert drawings_state(page) == expected_drawings

    # A drawing that starts before the loaded range must retain that timestamp
    # when its visible section is moved vertically in the shorter range.
    if "active" in (page.locator("#magnet").get_attribute("class") or "").split():
        page.locator("#magnet").click()
    page.locator('[data-tool="trend"]').click()
    page.locator("#chart").scroll_into_view_if_needed()
    click_point(page, plot_point(page, 0.05, 0.17))
    click_point(page, plot_point(page, 0.90, 0.28))
    offrange_id = drawings_state(page)[-1]["id"]
    first_date = page.evaluate("rows[0].date")
    page.locator('[data-period="6M"]').click()
    page.wait_for_function("first => rows.length > 30 && rows[0].date > first", arg=first_date)
    assert page.evaluate("id => barIndex(drawings.find(d => d.id === id).a.date) < 0", offrange_id)
    select_object(page, offrange_id)
    # Keep this regression independent of whether the shorter range's price
    # scale happens to include the prices chosen on the original year's chart.
    prices = page.evaluate("() => {const g = geometry(); return [g.price(g.top + g.ph * .45), g.price(g.top + g.ph * .5)];}")
    set_control(page, "#drawingPriceA", str(prices[0]))
    set_control(page, "#drawingPriceB", str(prices[1]))
    offrange_before = stored_shape(page, offrange_id)
    page.locator("#chart").scroll_into_view_if_needed()
    visible_midpoint = page.evaluate(
        """id => {
            const g = geometry(), d = drawings.find(item => item.id === id);
            const a = anchorPoint(g, d.a), b = anchorPoint(g, d.b);
            const x = (Math.max(g.left, a.x) + Math.min(g.left + g.pw, b.x)) / 2;
            const y = a.y + (b.y - a.y) * (x - a.x) / (b.x - a.x);
            return {x: g.box.x + x, y: g.box.y + y};
        }""",
        offrange_id,
    )
    drag_point(page, visible_midpoint, 34, 14)
    offrange_after = stored_shape(page, offrange_id)
    for anchor in ("a", "b"):
        assert offrange_after[anchor]["date"] == offrange_before[anchor]["date"], "Dragging must preserve out-of-range anchor dates"
        assert offrange_after[anchor]["price"] != offrange_before[anchor]["price"]
    page.locator('[data-period="1Y"]').click()
    page.wait_for_function("first => rows.length > 30 && rows[0].date === first", arg=first_date)


def add_library_study(page: Page, study_id: str) -> None:
    page.locator("#indicatorLibrary").click()
    expect(page.locator("#indicatorDialog")).to_be_visible()
    page.locator("#indicatorSearch").fill(study_id)
    page.locator(f'[data-advanced="{study_id}"]').click()
    page.wait_for_function("id => advanced.has(id) && !pendingStudies.has(id)", arg=study_id)
    page.locator("#closeIndicators").click()
    expect(page.locator("#error")).to_be_hidden()


def exercise_indicators(page: Page) -> None:
    initial_quick = set(page.evaluate('[...indicators]'))
    add_library_study(page, "RSI")
    expect(page.locator('[data-panel="RSI"]')).to_be_visible()
    original_rsi = page.evaluate("advanced.get('RSI')")
    assert original_rsi["parameters"]["timeperiod"] == 14
    assert original_rsi["source"] == "close"

    page.locator('#studyChips [data-study-settings="RSI"]').click()
    expect(page.locator("#studySettingsDialog")).to_be_visible()
    expect(page.locator("#studyParam-timeperiod")).to_have_value("14")
    page.locator("#studyParam-timeperiod").fill("7")
    page.locator("#studySource").select_option("hlc3")
    set_control(page, "#studyColor-0", "#ab44ee")
    page.locator("#studyLineWidth").fill("2.5")
    set_control(page, "#studyTransparency", "40")
    page.locator("#studyApply").click()
    expect(page.locator("#studySettingsDialog")).to_be_hidden()
    page.wait_for_function("advanced.get('RSI')?.parameters.timeperiod === 7")
    updated_rsi = page.evaluate("advanced.get('RSI')")
    assert updated_rsi["source"] == "hlc3"
    assert updated_rsi["outputs"]["real"][:7] == [None] * 7
    assert isinstance(updated_rsi["outputs"]["real"][7], (int, float))
    assert updated_rsi["outputs"]["real"] != original_rsi["outputs"]["real"], "Applying inputs must recalculate RSI"
    rsi_config = page.evaluate("studyConfig('RSI')")
    assert rsi_config == {
        "params": {"timeperiod": 7}, "source": "hlc3", "colors": {"real": "#ab44ee"},
        "width": 2.5, "transparency": 40,
    }, rsi_config
    expect(page.locator('[data-panel="RSI"] .study-panel-settings')).to_contain_text("RSI (7)")

    # The added cloud study must produce real data for every displayed output.
    add_library_study(page, "ICHIMOKU")
    ichimoku = page.evaluate("advanced.get('ICHIMOKU')")
    assert ichimoku["overlay"] is True
    assert len(ichimoku["outputs"]) == 5
    for name, series in ichimoku["outputs"].items():
        assert len(series) == len(ichimoku["dates"]), name
        assert any(value is not None and math.isfinite(value) for value in series), name

    page.locator('.workspace-quick-studies > summary').click()
    page.locator('[data-indicator="sma"]').click()
    page.locator('#studyChips [data-study-settings="sma"]').click()
    expect(page.locator("#studyParam-timeperiod")).to_have_value("20")
    page.locator("#studyParam-timeperiod").fill("10")
    page.locator("#studyApply").click()
    expect(page.locator("#studySettingsDialog")).to_be_hidden()
    expect(page.locator('[data-indicator="sma"]')).to_have_text("SMA 10")
    expect(page.locator("#legend")).to_contain_text("SMA 10")
    quick_values = page.evaluate("quickStudySeries('sma', studyConfig('sma')).sma")
    expected_mean = page.evaluate("rows.slice(0, 10).reduce((sum, bar) => sum + bar.close, 0) / 10")
    assert quick_values[:9] == [None] * 9
    assert math.isclose(quick_values[9], expected_mean, rel_tol=1e-12)
    settings = page.evaluate("JSON.parse(localStorage.getItem('atlas.studies.v1'))")
    assert set(settings["activeAdvanced"]) == {"RSI", "ICHIMOKU"}
    assert set(settings["activeQuick"]) == initial_quick | {"sma"}
    assert settings["configurations"]["RSI"] == rsi_config

    page.reload()
    wait_for_chart(page)
    page.wait_for_function("advanced.has('RSI') && advanced.has('ICHIMOKU') && pendingStudies.size === 0")
    assert page.evaluate("studyConfig('RSI')") == rsi_config
    assert page.evaluate("advanced.get('RSI').outputs.real") == updated_rsi["outputs"]["real"]
    expect(page.locator('[data-indicator="sma"]')).to_have_text("SMA 10")
    expect(page.locator('[data-indicator="sma"]')).to_have_attribute("aria-pressed", "true")
    page.locator('#studyChips [data-study-settings="RSI"]').click()
    expect(page.locator("#studyParam-timeperiod")).to_have_value("7")
    expect(page.locator("#studySource")).to_have_value("hlc3")
    expect(page.locator("#studyColor-0")).to_have_value("#ab44ee")
    expect(page.locator("#studyTransparency")).to_have_value("40")
    expect(page.locator("#studyLineWidth")).to_have_value("2.5")
    page.locator("#cancelStudySettings").click()
    expect(page.locator("#error")).to_be_hidden()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1680, "height": 1200}, device_scale_factor=1)
        page = context.new_page()
        page.set_default_timeout(20_000)
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(args.url)
            wait_for_chart(page)
            exercise_drawings(page)
            exercise_indicators(page)
            assert not errors, f"Browser JavaScript errors: {errors}"
            page.screenshot(path=str(ARTIFACTS / "dashboard-chart-editing.png"), full_page=True)
        except Exception:
            page.screenshot(path=str(ARTIFACTS / "dashboard-chart-editing-failure.png"), full_page=True)
            raise
        finally:
            context.close()
            browser.close()
    print("PASS: movable/resizable drawings, styling, history, persistence, locking/hiding, and configurable indicators; no JavaScript errors.")


if __name__ == "__main__":
    main()
