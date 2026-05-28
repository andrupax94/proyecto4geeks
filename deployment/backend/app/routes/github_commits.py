from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from github import Github
from datetime import datetime
import os
from dotenv import load_dotenv


if(load_dotenv()==False):
    load_dotenv(dotenv_path="./deployment/backend/.env")

router = APIRouter(prefix="/github", tags=["github"])

# Configuración
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "triopeligro"
REPO_NAME = "proyecto4geeks"
BRANCH = "main"

@router.get("/commits")
async def get_commits(limit: int = 15):
    """
    Obtiene los últimos commits del repositorio
    
    Args:
        limit: Número máximo de commits a retornar (default: 15)
    
    Returns:
        Lista de commits con información formateada
    """
    
    if not GITHUB_TOKEN:
        raise HTTPException(
            status_code=500, 
            detail="GITHUB_TOKEN no configurado en variables de entorno"
        )
    
    try:
        # Conectar a GitHub
        g = Github(GITHUB_TOKEN)
        repo = g.get_user(REPO_OWNER).get_repo(REPO_NAME)
        
        # Obtener commits
        commits = repo.get_commits(sha=BRANCH)
        
        # Procesar commits
        commits_data = []
        for i, commit in enumerate(commits):
            if i >= limit:
                break
            
            commit_info = {
                "hash": commit.sha[:7],  # Hash corto (7 caracteres)
                "hash_full": commit.sha,
                "message": commit.commit.message.split('\n')[0],  # Primera línea del mensaje
                "author": commit.commit.author.name,
                "email": commit.commit.author.email,
                "date": commit.commit.author.date.isoformat(),
                "url": commit.html_url,
                "files_changed": len(commit.files) if hasattr(commit, 'files') else 0,
                "additions": commit.stats.additions if hasattr(commit, 'stats') else 0,
                "deletions": commit.stats.deletions if hasattr(commit, 'stats') else 0,
            }
            commits_data.append(commit_info)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "total": len(commits_data),
                "repository": f"{REPO_OWNER}/{REPO_NAME}",
                "branch": BRANCH,
                "commits": commits_data
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener commits: {str(e)}"
        )


@router.get("/commits/{commit_hash}")
async def get_commit_details(commit_hash: str):
    """
    Obtiene detalles completos de un commit específico
    """
    
    if not GITHUB_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="GITHUB_TOKEN no configurado"
        )
    
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_user(REPO_OWNER).get_repo(REPO_NAME)
        commit = repo.get_commit(commit_hash)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "hash": commit.sha,
                "message": commit.commit.message,
                "author": commit.commit.author.name,
                "date": commit.commit.author.date.isoformat(),
                "files_changed": len(commit.files),
                "additions": commit.stats.additions,
                "deletions": commit.stats.deletions,
                "url": commit.html_url,
                "files": [
                    {
                        "name": file.filename,
                        "status": file.status,
                        "additions": file.additions,
                        "deletions": file.deletions
                    }
                    for file in commit.files
                ]
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )