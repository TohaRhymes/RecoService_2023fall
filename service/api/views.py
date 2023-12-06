import random
from typing import List

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from gunicorn.config import User
from pydantic import BaseModel

from models.LFM import LFM
from models.UserKnnCos70 import UserKnnCos70
from service.api.exceptions import UserNotFoundError
from service.api.security import verify_token
from service.log import app_logger


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


userknn_cos_70 = UserKnnCos70()
lfm_best = LFM()

router = APIRouter()


@router.get(
    path="/health",
    tags=["Health"],
    responses={
        200: {
            "description": "Successful Health Check",
            "content": {"text/plain": {"example": "36.6"}},
        },
        500: {
            "description": "Internal Server Error",
            "content": {"application/json": {"example": {"detail": "Internal server error"}}},
        },
    },
)
async def health() -> str:
    return "36.6"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        200: {
            "description": "Successful Response",
            "content": {"application/json": {"example": {"user_id": 1, "items": [8, 3, 6, 23, 5, 61, 78, 83, 21, 54]}}},
        },
        401: {
            "description": "Model or user not found",
            "content": {
                "application/json": {
                    "example": {
                        "errors": [
                            {
                                "error_key": "http_exception",
                                "error_message": "Invalid " "authentication " "credentials",
                                "error_loc": "null",
                            }
                        ],
                        "status_code": 401,
                    }
                }
            },
        },
        403: {
            "description": "Bearer token is not provided",
            "content": {
                "application/json": {
                    "example": {
                        "errors": [
                            {
                                "error_key": "http_exception",
                                "error_message": "Not " "authenticated",
                                "error_loc": "null",
                            }
                        ],
                        "status_code": 404,
                    }
                }
            },
        },
        404: {
            "description": "Model or user not found",
            "content": {
                "application/json": {
                    "example": {
                        "errors": [
                            {
                                "error_key": "user_not_found",
                                "error_message": "User " "2392109321 not" " found",
                                "error_loc": "null",
                            }
                        ],
                        "status_code": 404,
                    }
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {"application/json": {"example": {"detail": "Internal server error"}}},
        },
    },
)
async def get_reco(
    request: Request, model_name: str, user_id: int, current_user: User = Depends(verify_token)
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs
    if model_name == "random":
        reco = random.sample(range(1, k_recs * 5), k_recs)
    elif model_name == "range":
        reco = list(range(1, k_recs + 1))
    elif model_name == "userknn_cos_70":
        reco = userknn_cos_70.predict(user_id, k=k_recs)
    elif model_name == "lfm_best":
        reco = lfm_best.predict(user_id, k=k_recs)
    else:
        raise HTTPException(status_code=404, detail="Model doesn't exist")

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
