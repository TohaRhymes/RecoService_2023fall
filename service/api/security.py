from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_token(token) -> Optional[str]:
    # here should be some logic (go to db/check secured dicts/etc
    # but we just use stub.
    kek_security = {'let_admin_in_lmao': 'admin'}
    if token in kek_security:
        return kek_security[token]
    return None


async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = verify_token(token)
    if not user:
        raise HTTPException(status_code=401,
                            detail="Invalid authentication credentials")
    return user
