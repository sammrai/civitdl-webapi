"""Reporting of Civitai's refusal reason on a failed download."""

from unittest.mock import MagicMock, patch

import pytest

from app.utils import _CivitaiSession

DOWNLOAD_URL = "https://civitai.com/api/download/models/3163627"
DISABLED = "The creator of this asset has disabled downloads on this file"


def response(status, payload=None):
    res = MagicMock()
    res.status_code = status
    res.json.return_value = payload if payload is not None else {}
    return res


def get_returning(res):
    """Patch the underlying requests.Session.get so only our wrapper runs."""
    return patch("requests.Session.get", return_value=res)


def test_refusal_message_is_kept():
    session = _CivitaiSession()
    with get_returning(response(401, {"error": "Unauthorized", "message": DISABLED})):
        session.get(DOWNLOAD_URL)

    assert session.refusal == DISABLED


def test_successful_download_records_nothing():
    session = _CivitaiSession()
    with get_returning(response(200)):
        session.get(DOWNLOAD_URL)

    assert session.refusal is None


def test_refusal_elsewhere_is_ignored():
    session = _CivitaiSession()
    with get_returning(response(401, {"message": "nope"})):
        session.get("https://civitai.com/api/v1/models/2805786")

    assert session.refusal is None


def test_non_json_refusal_is_survivable():
    res = response(401)
    res.json.side_effect = ValueError("not json")
    session = _CivitaiSession()
    with get_returning(res):
        session.get(DOWNLOAD_URL)

    assert session.refusal is None


def test_rate_limiting_is_not_reported_as_not_found():
    from fastapi import HTTPException
    from helpers.core.utils import APIException

    from app.utils import RATE_LIMITED, _civitdl

    with patch("app.utils.find_model_files", return_value=[]), \
            patch("app.utils.get_safe_metadata", side_effect=APIException(429, "rate limited")):
        with pytest.raises(HTTPException) as excinfo:
            _civitdl(28205, 47670)

    assert excinfo.value.status_code == 429
    assert excinfo.value.detail == RATE_LIMITED


def test_a_missing_model_is_still_a_404():
    from fastapi import HTTPException
    from helpers.core.utils import APIException

    from app.utils import _civitdl

    with patch("app.utils.find_model_files", return_value=[]), \
            patch("app.utils.get_safe_metadata", side_effect=APIException(404, "nope")):
        with pytest.raises(HTTPException) as excinfo:
            _civitdl(999999999)

    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "Model not found on Civitai."
