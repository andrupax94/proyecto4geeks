from fastapi import APIRouter

router = APIRouter()

@router.get('/pages')
def pages():
    return [
        {'slug': 'inicio', 'title': 'Inicio', 'body': 'Resumen del proyecto'},
        {'slug': 'arquitectura', 'title': 'Arquitectura', 'body': 'Frontend + API + modelo'},
    ]

@router.get('/pages/{slug}')
def page(slug: str):
    return {'slug': slug, 'title': slug.title(), 'body': 'Contenido base de la wiki'}
