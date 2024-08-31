from fastapi import APIRouter

token_router = APIRouter()


@token_router.post('/')
def create():
    pass


@token_router.get('/')
def find_all():
    pass


@token_router.delete('/{token_id}')
def delete(token_id: int):
    pass
