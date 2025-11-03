import os
from OCR import text_fetch
from Whisper import audio_fetch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

"""
    O serviço faz a chamada dos módulos OCR (Extração de Texto em Imagem) e Whisper (Extração de Texto em Áudio) para retornar
    informações textuais dos arquivos. Para isso, ele recebe como entrada um arquivo do tipo especificado, faz chamada dos mé-
    todos de pré-processamento e, finalmente, retorna o conteúdo.
"""

app = FastAPI(title="Extração de Conteúdo", version=1.0)

app.get("/")
def root():
    return {"message": "Rodando..."}

@app.root("/process")
async def process_file(file: UploadFile = File()):

    content_type = file.content_type
    filename = file.filename

    temp_path = os.path.join("temp", filename)
    os.makedirs("temp",exist_ok=True)

    with open(temp_path,"wb") as f:
        f.write(await file.read())

    try:
        if "image" in content_type:
            text = text_fetch(temp_path)
        elif "audio" in content_type or "video" in content_type:
            text = audio_fetch(temp_path)
        else:
            raise HTTPException(status_code=400, detail=f"Tipo de arquivo não suportado: {content_type}")

    finally:
        os.remove(temp_path)