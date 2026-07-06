from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
import jwt

app = FastAPI()
security = HTTPBearer()

# Configuration
SECRET_KEY = "your-super-flexible-secret-key-change-this"
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 15  # Tokens expire quickly for security

USER_DATA = {"admin": "123"}

class LoginRequest(BaseModel):
    username: str
    password: str

# Helper: Create Token
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# 1. Login & Issue JWT
@app.post("/login")
def login(data: LoginRequest):
    if data.username in USER_DATA and USER_DATA[data.username] == data.password:
        token = create_access_token(data={"sub": data.username})
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=400, detail="Incorrect username or password")

# 2. Decode & Validate JWT
def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        # Decodes and automatically checks 'exp' claim expiration
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid security token",
        )

# 3. Secure Endpoint
@app.get("/protected-data")
def get_protected_data(token_payload: dict = Depends(verify_jwt_token)):
    return {
        "status": "authorized",
        "user": token_payload.get("sub"),
        "expires_at_epoch": token_payload.get("exp"),
        "payload": "Here is your highly classified data."
    }
