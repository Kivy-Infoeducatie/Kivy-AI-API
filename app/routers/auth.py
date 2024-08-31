from fastapi import APIRouter

auth_router = APIRouter()


@auth_router.post('/login')
def login():
    pass


@auth_router.post('/register')
def register():
    pass


@auth_router.post('/logout')
def logout():
    pass
