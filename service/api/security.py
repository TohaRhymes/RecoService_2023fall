from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

bearer_scheme = HTTPBearer()


def verify_token(token) -> Optional[str]:
    # here should be some logic (go to db/check secured dicts/etc
    # but we just use stub.
    # We also can just delete this, and check provided token with reference
    kek_security = {'let_admin_in_lmao': 'admin'}
    if token in kek_security:
        return kek_security[token]
    return None


async def get_current_user(credentials: HTTPAuthorizationCredentials =
                           Security(bearer_scheme)):
    if not credentials:
        raise HTTPException(status_code=403, detail="Not authenticated")
    token = credentials.credentials
    user = verify_token(token)
    if not user:
        raise HTTPException(status_code=401,
                            detail="Invalid authentication credentials")
    return user
