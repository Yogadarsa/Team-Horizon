
import ast
import json
import sys
import time

def evaluate(record):
    result = {
        "model": record["model"],
        "task_id": record["task_id"],
        "difficulty": record.get("difficulty"),
        "tests_total": len(record["test_list"]),
        "tests_passed": 0,
        "test_results": [],
        "evaluation_status": "completed"
    }

    code = record["code"]

    try:
        ast.parse(code)
    except SyntaxError as exc:
        result["evaluation_status"] = "syntax_error"
        result["error"] = str(exc)
        return result

    namespace = {"__name__": "__main__"}

    try:
        for statement in record.get("test_imports", []):
            exec(statement, namespace)

        exec(code, namespace)

        for test in record["test_list"]:
            start = time.perf_counter()

            try:
                exec(test, namespace)
                passed = True
                error = None
            except Exception as exc:
                passed = False
                error = f"{type(exc).__name__}: {exc}"

            result["test_results"].append({
                "passed": passed,
                "error": error,
                "execution_time_seconds": round(
                    time.perf_counter() - start, 6
                )
            })

            if passed:
                result["tests_passed"] += 1

    except Exception as exc:
        result["evaluation_status"] = "execution_error"
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def main():
    record = json.loads(sys.stdin.read())
    result = evaluate(record)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
