from fastapi import FastAPI

from .routers.auth import auth_router
from .routers.token import token_router

app = FastAPI()

app.include_router(auth_router, prefix='/auth')
app.include_router(token_router, prefix='/token')
app.include_router(operations_router, prefix='/action')
