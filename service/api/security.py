import os

from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

bearer_scheme = HTTPBearer()
load_dotenv()
TOKEN = os.getenv("MY_SECRET_TOKEN")


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)) -> None:
    if not credentials:
        raise HTTPException(status_code=403, detail="Not authenticated")
    token = credentials.credentials
    if token != TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
