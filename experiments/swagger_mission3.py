"""Submit three glitches through real Swagger controls using installed Chrome.

Optional review dependency: pip install playwright (uses system Chrome).
Run: python experiments/swagger_mission3.py --url PUBLIC_URL
"""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    url = args.url.rstrip("/")
    folder = Path("artifacts/team_25/mission3/swagger")
    folder.mkdir(parents=True, exist_ok=True)
    example = dict(battery_v=3.86, temp_c=28.0, rings_per_day=5,
                   wifi_rssi_dbm=-66, age_months=22)
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 1000})
        page.goto(url + "/docs", wait_until="domcontentloaded", timeout=60000)
        if page.title() == "Codespaces Access Port":
            # GitHub shows this warning once per browser session even for Public
            # ports. This is our verified Codespace, so continue to its Swagger UI.
            page.get_by_text("Continue", exact=True).click()
        operation = page.locator("#operations-default-predict_predict_post")
        operation.locator(".opblock-summary").click(timeout=60000)
        operation.get_by_role("button", name="Try it out").click()
        for field, value in (("battery_v", 0), ("temp_c", -127), ("wifi_rssi_dbm", 0)):
            payload = dict(example, **{field: value})
            operation.locator("textarea.body-param__text").fill(json.dumps(payload, indent=2))
            with page.expect_response(lambda r: r.url == url + "/predict" and r.request.method == "POST", timeout=30000) as pending:
                operation.get_by_role("button", name="Execute", exact=True).click()
            response = pending.value
            body = response.json()
            assert response.status == 422, (field, response.status, body)
            assert any(field in error["loc"] for error in body["detail"])
            operation.locator(".live-responses-table").wait_for()
            page.wait_for_function(
                "field => document.querySelector('#operations-default-predict_predict_post .live-responses-table').textContent.includes(field)",
                arg=field,
            )
            page.screenshot(path=str(folder / f"{field}_422.png"), full_page=True)
            result = dict(field=field, value=value, request=payload, status=response.status,
                          response=body, interaction="Swagger Try it out → edit request JSON → Execute")
            results.append(result)
            print(f"PASS Swagger: {field}={value} → HTTP 422", flush=True)
        browser.close()
    (folder / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
