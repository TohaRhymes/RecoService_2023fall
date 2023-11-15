import random
from typing import List

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from gunicorn.config import User
from pydantic import BaseModel

from service.api.exceptions import UserNotFoundError
from service.api.security import get_current_user
from service.log import app_logger


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "36.6"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    current_user: User = Depends(get_current_user)
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10 ** 9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs
    if model_name == "random":
        reco = random.sample(range(1, k_recs * 5), k_recs)
    elif model_name == "range":
        reco = list(range(1, k_recs + 1))
    else:
        raise HTTPException(status_code=404, detail="Model doesn't exist")
    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
