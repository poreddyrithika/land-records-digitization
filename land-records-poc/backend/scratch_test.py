import urllib.request
import json

BASE = "http://127.0.0.1:8000"

def get(path):
    req = urllib.request.Request(f"{BASE}{path}")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode())

def post(path, data=None):
    body = json.dumps(data).encode("utf-8") if data else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(f"{BASE}{path}", data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode())

def run_tests():
    passed = 0
    total = 0

    def test(name, fn):
        nonlocal passed, total
        total += 1
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    print("Running Complete Test Suite...")

    test("1. Root Endpoint", lambda: (
        lambda s, d: (_ for _ in ()).throw(AssertionError(f"Root failed: {d}")) if s != 200 or d.get("status") != "ok" else None
    )(*get("/")))

    test("2. Load 300 Demo Records", lambda: (
        lambda s, d: (_ for _ in ()).throw(AssertionError(f"Load demo failed: {d}")) if s != 200 or d.get("records_loaded") != 300 else None
    )(*post("/demo/load")))

    test("3. Dashboard Statistics Aggregation", lambda: (
        lambda s, d: (_ for _ in ()).throw(AssertionError(f"Stats check failed: {d}")) if (
            d.get("total") != 300 or
            d.get("submitted") + d.get("pending") + d.get("not_submitted") != 300 or
            "telugu" not in d.get("language_distribution", {}) or
            "english" not in d.get("language_distribution", {})
        ) else None
    )(*get("/dashboard/statistics")))

    test("4. Map Records with Coordinates", lambda: (
        lambda s, d: (_ for _ in ()).throw(AssertionError(f"Map records failed: len={len(d)}")) if (
            len(d) != 300 or not (15.0 < d[0]["latitude"] < 17.0 and 79.0 < d[0]["longitude"] < 82.0)
        ) else None
    )(*get("/map/records")))

    test("5. Filter Distinct Values", lambda: (
        lambda s, d: (_ for _ in ()).throw(AssertionError(f"Filters failed: {d}")) if (
            "Rampur" not in d.get("villages", []) or "telugu" not in d.get("languages", [])
        ) else None
    )(*get("/records/filters")))

    test("6. Record Detail & Audit Trail", lambda: (
        lambda r_s, r_d, a_s, a_d: (_ for _ in ()).throw(AssertionError(f"Record/Audit failed")) if (
            r_d.get("record_id_str") != "REC001" or len(a_d) < 1
        ) else None
    )(*get("/records/1"), *get("/audit/1")))

    test("7. Cross-Validation Engine", lambda: (
        lambda s, d: (_ for _ in ()).throw(AssertionError(f"Validation failed: {d}")) if (
            "summary" not in d or "results" not in d
        ) else None
    )(*post("/records/1/validate")))

    test("8. Duplicate Detection (True positive)", lambda: (
        lambda rec: (
            lambda s, d: (_ for _ in ()).throw(AssertionError(f"Duplicate detection missed: {d}")) if (
                not d.get("has_duplicate") or len(d.get("matches", [])) == 0
            ) else None
        )(*post("/records/check-duplicate", {
            "survey_number": rec["survey_number"],
            "village": rec["village"],
            "sub_division": rec["sub_division"]
        }))
    )(get("/records/1")[1]))

    test("9. Duplicate Detection (True negative)", lambda: (
        lambda s, d: (_ for _ in ()).throw(AssertionError(f"False duplicate flagged: {d}")) if (
            d.get("has_duplicate")
        ) else None
    )(*post("/records/check-duplicate", {
        "survey_number": "UNIQUE9999",
        "village": "NonExistentVillageXYZ",
        "sub_division": "Z"
    })))

    test("10. Officer Approve & Publish Workflow", lambda: (
        lambda s1, d1, s2, d2: (_ for _ in ()).throw(AssertionError("Approval/Publishing failed")) if (
            d1.get("verification_status") != "verified" or d2.get("verification_status") != "published"
        ) else None
    )(*post("/records/3/approve"), *post("/records/3/publish")))

    test("11. Static Frontend UI Serving", lambda: (
        lambda req: (_ for _ in ()).throw(AssertionError("UI static serving failed")) if (
            urllib.request.urlopen(req).status != 200
        ) else None
    )(urllib.request.Request(f"{BASE}/ui/")))

    test("12. Static CSS Stylesheet Serving", lambda: (
        lambda req: (_ for _ in ()).throw(AssertionError("CSS static serving failed")) if (
            urllib.request.urlopen(req).status != 200
        ) else None
    )(urllib.request.Request(f"{BASE}/ui/styles.css")))

    print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

if __name__ == "__main__":
    run_tests()
