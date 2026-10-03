from credit_scoring_observability.logger import build_processors, redact_identifiers


def test_identifiers_are_redacted():
    event = redact_identifiers(None, None, {"event": "x", "customer_id": "abc123"})
    assert event["customer_id"] == "***"
    assert event["event"] == "x"


def test_redaction_runs_before_file_and_renderer(tmp_path):
    processors = build_processors(json_logs=True, log_file=tmp_path / "log.jsonl")
    names = [getattr(p, "__name__", type(p).__name__) for p in processors]
    assert names.index("redact_identifiers") < names.index("JsonFileTee")
