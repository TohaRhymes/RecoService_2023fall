from http import HTTPStatus

import pytest
from starlette.testclient import TestClient

from service.settings import ServiceConfig

GET_RECO_PATH = "/reco/{model_name}/{user_id}"
GET_HEALTH_PATH = "/health"
AUTH_HEADER = {"Authorization": "Bearer let_admin_in_lmao"}
AUTH_INVALID_HEADER = {"Authorization": "Bearer let_admin_in_lol"}


# Health check
def test_health(
    client: TestClient,
) -> None:
    with client:
        response = client.get(GET_HEALTH_PATH)
    assert response.status_code == HTTPStatus.OK


# Invalid auth token
@pytest.mark.parametrize("user_id", [5])
def test_get_reco_for_invalid_token(
    client: TestClient,
    user_id
) -> None:
    path = GET_RECO_PATH.format(model_name="random", user_id=user_id)
    with client:
        response = client.get(path, headers=AUTH_INVALID_HEADER)
    assert response.status_code == HTTPStatus.UNAUTHORIZED


# Without auth token
@pytest.mark.parametrize("user_id", [5])
def test_get_reco_without_token(
    client: TestClient,
    user_id
) -> None:
    path = GET_RECO_PATH.format(model_name="random", user_id=user_id)
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.UNAUTHORIZED


# Invalid user ID
@pytest.mark.parametrize("user_id", [10 ** 10])
def test_get_reco_for_unknown_user(
    client: TestClient,
    user_id
) -> None:
    path = GET_RECO_PATH.format(model_name="random", user_id=user_id)
    with client:
        response = client.get(path, headers=AUTH_HEADER)
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "user_not_found"


# Valid user ID with different models
@pytest.mark.parametrize("user_id,model_name,expected_status",
                         [(123, "random", HTTPStatus.OK),
                          (123, "range", HTTPStatus.OK)])
def test_get_reco_valid_user(client: TestClient, service_config: ServiceConfig,
                             user_id, model_name, expected_status) -> None:
    path = GET_RECO_PATH.format(model_name=model_name, user_id=user_id)
    with client:
        response = client.get(path, headers=AUTH_HEADER)
    assert response.status_code == expected_status
    response_json = response.json()
    assert response_json["user_id"] == user_id
    if model_name == "range":
        assert response_json["items"] == list(
            range(1, service_config.k_recs + 1))
    else:
        assert len(response_json["items"]) == service_config.k_recs


# Test invalid model name
def test_get_reco_invalid_model(client: TestClient) -> None:
    user_id = 123
    path = GET_RECO_PATH.format(model_name="shitty_model", user_id=user_id)
    with client:
        response = client.get(path, headers=AUTH_HEADER)
    assert response.status_code == HTTPStatus.NOT_FOUND
