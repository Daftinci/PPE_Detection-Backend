"""JWT authentication helpers."""

from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, Query, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

SECRET_KEY = "change-this-secret-key-in-production"
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 8

# Simple hardcoded user store — extend with a DB table if needed
USERS: dict[str, str] = {
    "admin": "admin123",
}

_bearer = HTTPBearer(auto_error=False)


def authenticate(username: str, password: str) -> bool:
    return USERS.get(username) == password


def create_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def _decode(token: str) -> str:
    """Decode and return the username, or raise 401."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if not username:
            raise ValueError
        return username
    except (JWTError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> str:
    """Dependency for standard Authorization: Bearer <token> requests."""
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return _decode(credentials.credentials)


def verify_token_query(
    token: str | None = Query(None, alias="token"),
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> str:
    """Dependency that also accepts ?token= query param (needed for <img src> / MJPEG streams)."""
    raw = token or (credentials.credentials if credentials else None)
    if not raw:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return _decode(raw)
