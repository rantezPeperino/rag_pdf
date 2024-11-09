import os
import openai
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configura tu API Key de OpenAI
API_KEY  = ''

# 1. Cargar el modelo de embeddings usando HuggingFaceEmbeddings
def cargar_modelo_embeddings():
    model_name = "all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

# 2. Cargar el documento PDF
def cargar_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# 3. Dividir el documento en fragmentos
def dividir_documento(data, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(data)

# 4. Crear la base de datos vectorial usando Chroma
def crear_base_datos_vectorial(docs, embedding_function):
    persist_directory = os.path.join(os.getcwd(), "chroma_db_local")
    
    # Si el directorio existe, lo eliminamos para evitar conflictos
    if os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    return vectorstore

# 5. Configurar el recuperador para buscar los fragmentos relevantes
def configurar_recuperador(vectorstore, k=3):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

# 6. Realizar una búsqueda
def buscar_contenido(retriever, prompt):
    # Usar el nuevo método invoke en lugar de get_relevant_documents
    resultados = retriever.invoke(prompt)
    # Extraer el contenido de los resultados
    contenido_resultados = "\n\n".join([resultado.page_content for resultado in resultados])
    return contenido_resultados

# 7. Generar respuesta usando OpenAI
def generar_respuesta_openai(prompt, contexto):
    # Formatear el mensaje para OpenAI
    mensaje = f"Pregunta: {prompt}\n\nContexto:\n{contexto}\n\nRespuesta:"
    
    # Crear una instancia del cliente con la API key
    client = openai.OpenAI(api_key=API_KEY)
    
    # Llamar a OpenAI para obtener la respuesta usando la nueva sintaxis
    respuesta = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un asistente experto en proporcionar respuestas informativas."},
            {"role": "user", "content": mensaje}
        ],
        max_tokens=150
    )
    return respuesta.choices[0].message.content.strip()

# ... (todo el código anterior se mantiene igual hasta el main)

if __name__ == "__main__":
    while True:
        print("\n=== MENÚ PRINCIPAL ===")
        print("1. Hacer una pregunta")
        print("2. Cargar nuevo PDF a la base vectorial") # Opcional, por si quieren agregar más documentos
        print("3. Salir")
        
        opcion = input("\nSeleccione una opción (1-3): ")
        
        if opcion == "1":
            try:
                # Cargar el modelo de embeddings
                print("\nCargando el modelo de embeddings...")
                embedding_function = cargar_modelo_embeddings()
                
                # Cargar la base de datos vectorial existente
                persist_directory = os.path.join(os.getcwd(), "chroma_db_local")
                if not os.path.exists(persist_directory):
                    print("\nError: No existe una base de datos vectorial. Debe cargar al menos un PDF primero.")
                    continue
                
                # Cargar Chroma desde el directorio persistente
                vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embedding_function
                )
                
                # Configurar el recuperador
                retriever = configurar_recuperador(vectorstore)
                
                # Obtener la pregunta del usuario
                prompt = input("\nIngrese su pregunta: ")
                
                # Realizar la búsqueda
                print(f"\nBuscando información para el prompt: '{prompt}'")
                contexto = buscar_contenido(retriever, prompt)

                # Generar respuesta con OpenAI
                print("Generando respuesta con OpenAI...")
                respuesta_final = generar_respuesta_openai(prompt, contexto)
                print(f"\nRespuesta generada:\n{respuesta_final}")
                
            except Exception as e:
                print(f"\nError: {str(e)}")
                
            input("\nPresione Enter para continuar...")
            
        elif opcion == "2":
            pdf_path = input("\nIngrese el path del archivo PDF: ")
            try:
                # Paso 1: Cargar el modelo de embeddings
                print("\nCargando el modelo de embeddings...")
                embedding_function = cargar_modelo_embeddings()
                
                # Paso 2: Cargar el documento
                print("Cargando el documento PDF...")
                data = cargar_pdf(pdf_path)

                # Paso 3: Dividir el documento en fragmentos
                print("Dividiendo el documento en fragmentos...")
                docs = dividir_documento(data)
                
                # Paso 4: Crear la base de datos vectorial con Chroma
                print("Creando la base de datos vectorial...")
                vectorstore = crear_base_datos_vectorial(docs, embedding_function)
                
                print("\nPDF procesado y agregado a la base de datos vectorial correctamente.")
                
            except FileNotFoundError:
                print(f"\nError: No se encontró el archivo '{pdf_path}'")
            except Exception as e:
                print(f"\nError: {str(e)}")
                
            input("\nPresione Enter para continuar...")
            
        elif opcion == "3":
            print("\n¡Hasta luego!")
            break
            
        else:
            print("\nOpción no válida. Por favor, seleccione una opción válida.")