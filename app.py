import os
import oci
import faiss
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from schemas import InputRequest, OutputResponse
from sentence_transformers import SentenceTransformer
from utils import read_docx, chunk_text, load_embeddings_index, get_top_chunks


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --------- Setup OCI Client ---------
    print("Setting up OCI Generative AI client...")
    endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    config = oci.config.from_file("config.txt")

    app.state.generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=config,
        service_endpoint=endpoint,
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240)
    )
    print("OCI client setup complete.")
    
    # --------- Load .docx and Prepare Chunks ---------
    print("Loading document and preparing chunks...")
    doc_dir = "documents"
    all_chunks = []

    for filename in os.listdir(doc_dir):
        if filename.lower().endswith(".docx"):
            file_path = os.path.join(doc_dir, filename)
            print(f"Reading: {file_path}")
            try:
                text = read_docx(file_path)
                file_chunks = chunk_text(text)
                all_chunks.extend(file_chunks)
            except Exception as e:
                print(f"Failed to read {filename}: {e}")

    chunks = all_chunks
    print("Finished loading document and preparing chunks...")
    
    # --------- Load Embedding model ---------
    print("Generating embeddings and setting up FAISS index...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings, index = load_embeddings_index(model, chunks)
    print("Finished generating embeddings and setting up FAISS index...")
    
    # --------- Load model, chunks and index to fastapi state ---------
    app.state.model = model
    app.state.chunks = chunks
    app.state.index = index

    print("Startup completed.")
    yield
    
    # Clean up and release resources steps on server shut down after yield
    print("Shutting down server...")

app = FastAPI(title="Cerner helper bot", lifespan=lifespan)

@app.get("/")
async def read_root():
    return {"message": "'Cerner helper bot' by Oracle - Operational"}

@app.post("/inference")
async def inference(request: InputRequest):
    print("Got request: ", request)
    
    try:
        user_question = request.question.strip()
        if not user_question:
            raise HTTPException(status_code=400, detail="Empty question provided.")

        top_chunks = get_top_chunks(user_question, app.state.model, app.state.chunks, app.state.index)
        context = "\n".join(top_chunks)

        full_prompt = f"""
        Context:
        {context}

        Question:
        {user_question}
        """

        print("Calling OCI LLM with prompt...")
        chat_detail = oci.generative_ai_inference.models.ChatDetails()

        chat_request = oci.generative_ai_inference.models.CohereChatRequest(
            message=full_prompt,
            max_tokens=1600,
            temperature=0.1,
            frequency_penalty=0,
            top_p=0.75,
            top_k=0
        )

        chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
            model_id="ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyanrlpnq5ybfu5hnzarg7jomak3q6kyhkzjsl4qj24fyoq"
        )
        chat_detail.chat_request = chat_request
        chat_detail.compartment_id = "ocid1.compartment.oc1..aaaaaaaakqloyfwdhn3buelfhd27z2irt4vnlxvkt6b5weefcyj5qaj57pfa"

        result = app.state.generative_ai_inference_client.chat(chat_detail)
        answer = result.data.chat_response.text
        print("Got response from LLM: ", answer)

        response = OutputResponse(response=answer)
        print("Sending response: ", response)
        return response

    except Exception as e:
        print("Failed to process inference.")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
