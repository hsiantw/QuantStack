"""Exercise docked workspace panels in an isolated headless browser profile."""
from playwright.sync_api import expect, sync_playwright
import argparse


def run(url='http://127.0.0.1:8765'):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function("typeof rows !== 'undefined' && rows.length > 0")
        settings_button = page.locator("#workspaceSettingsDock #terminalSettings")
        expect(settings_button).to_be_visible()
        settings_button.click()
        expect(page.locator("#terminalSettingsDialog")).to_be_visible()
        page.locator("#terminalCancelSettings").click()
        expect(page.locator("#workspaceExpandTools")).to_be_visible()
        initial_width = page.locator("#chart").bounding_box()["width"]
        page.locator("#workspaceExpandTools").click()
        expect(page.locator("#workspaceExpandTools")).to_have_attribute("aria-expanded", "true")
        assert page.locator("#chart").bounding_box()["width"] < initial_width - 100
        page.locator('[data-tool="trend"]').click()
        page.wait_for_function("tool === 'trend'")
        page.locator('[data-tool="cursor"]').click()
        page.locator("#workspaceExpandTools").click()

        watchlist = page.locator('[data-side-panel="watchlist"]')
        watchlist.click()
        expect(page.locator("#workspaceSidebar")).not_to_be_visible()
        expect(watchlist).to_be_visible()
        assert page.locator("#chart").bounding_box()["width"] > initial_width + 200
        watchlist.click()
        page.locator('[data-side-panel="details"]').click()
        expect(page.locator("#workspaceSidedetails #price")).to_be_visible()
        page.locator('[data-side-panel="notes"]').click()
        page.locator("#workspaceNotes").fill("Watch the previous high")
        page.reload()
        page.wait_for_function("typeof rows !== 'undefined' && rows.length > 0")
        page.locator('[data-side-panel="notes"]').click()
        expect(page.locator("#workspaceNotes")).to_have_value("Watch the previous high")
        page.locator('[data-side-panel="objects"]').click()
        expect(page.locator("#drawingObjectList")).to_be_visible()
        watchlist.click()

        page.locator("#workspaceOpenData").click()
        expect(page.locator("#workspaceDataPanel")).to_be_visible()
        expect(page.locator("#workspaceOpenData")).to_have_attribute("aria-selected", "true")
        assert page.locator("dialog[open]").count() == 0
        dock = page.locator("#workspaceDock")
        chart = page.locator("#chart")
        assert chart.bounding_box()["y"] + chart.bounding_box()["height"] < dock.bounding_box()["y"]
        separator = page.locator("#workspaceDockResize")
        height = dock.bounding_box()["height"]
        separator.focus()
        separator.press("ArrowUp")
        assert dock.bounding_box()["height"] > height
        box = separator.bounding_box()
        page.mouse.move(box["x"] + 80, box["y"] + 3)
        page.mouse.down()
        page.mouse.move(box["x"] + 80, box["y"] - 37, steps=5)
        page.mouse.up()
        assert dock.bounding_box()["height"] > height + 30
        page.locator("#workspaceMaximizeDock").click()
        expect(page.locator("#workspaceMaximizeDock")).to_have_attribute("aria-pressed", "true")
        assert dock.bounding_box()["height"] > 600
        page.locator("#workspaceMaximizeDock").click()
        page.locator("#workspaceOpenScreener").click()
        expect(page.locator("#screenerSearch")).to_be_visible()
        expect(page.locator("#workspaceDataPanel")).not_to_be_visible()
        page.locator('.workspace-screen-toolbar button').click()
        expect(page.locator('#screenerCountry')).to_be_visible()
        page.locator('.workspace-screen-toolbar button').click()
        expect(page.locator('#screenerCountry')).not_to_be_visible()
        page.screenshot(path="data/workspace-screener.png")
        page.locator("#screenerSearch").fill("AAPL")
        page.locator("#screenerSearch").press("Escape")
        expect(dock).not_to_be_visible()
        page.locator("#toggleScreener").click()
        expect(dock).to_be_visible()
        page.locator("#collapseScreener").click()
        expect(dock).not_to_be_visible()
        page.screenshot(path="data/workspace-desktop.png")
        page.locator("#workspaceOpenData").click()
        page.screenshot(path="data/workspace-dock.png")
        page.locator("#workspaceOpenRisk").click()
        expect(page.locator("#workspaceRiskPanel")).to_be_visible()
        expect(page.locator("#workspaceOpenRisk")).to_have_attribute("aria-selected", "true")
        page.locator("#riskRun").click()
        expect(page.locator("#riskResults")).to_be_visible()
        expect(page.locator("#riskMetrics")).to_contain_text("Annualized volatility")
        with page.expect_download() as download:
            page.locator("#riskCSV").click()
        assert download.value.suggested_filename.endswith("-returns-risk.csv")
        page.locator("#workspaceCloseDock").click()

        for width, height in [(1024, 768), (768, 900), (390, 844)]:
            page.set_viewport_size({"width": width, "height": height})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), width
            assert chart.bounding_box()["width"] > 180, (width, chart.bounding_box())
            page.locator('[data-side-panel="notes"]').click()
            expect(page.locator("#workspaceNotes")).to_be_visible()
            page.locator("#workspaceSidenotes .workspace-panel-close").click()
            expect(page.locator("#workspaceSidebar")).not_to_be_visible()
            page.locator("#workspaceOpenData").click()
            expect(page.locator("#workspaceDataPanel")).to_be_visible()
            assert chart.bounding_box()["height"] >= 100
            page.screenshot(path=f"data/workspace-{width}.png")
            page.locator("#workspaceCloseDock").click()
        assert not errors, errors
        browser.close()
        print("Workspace checks passed: tools, panels, notes, dock resizing, tabs, mobile layout; no JavaScript errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    run(parser.parse_args().url)
