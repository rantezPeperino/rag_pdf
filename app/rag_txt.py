import os
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
import textwrap

# Configuración
OPENAI_API_KEY = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class RAGDemo:
    def __init__(self):
        # Inicializar el modelo de embeddings
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        
        # Inicializar el modelo de chat
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
        
        # Directorio para la base de datos vectorial
        self.persist_dir = "chroma_db"
        
        # Configurar el splitter de documentos
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def create_demo_documents(self) -> List[Document]:
        """Crea documentos de ejemplo para la demo"""
        texts = [
            """Python es un lenguaje de programación de alto nivel, interpretado y de propósito general.
            Fue creado por Guido van Rossum y lanzado por primera vez en 1991. Python enfatiza la legibilidad
            del código y permite expresar conceptos en menos líneas que otros lenguajes.""",
            
            """JavaScript es un lenguaje de programación que permite implementar funciones complejas en páginas web.
            Es la tercera capa del pastel de las tecnologías web estándar, junto con HTML y CSS.""",
            
            """ Chancletus es un pueblo en la provincia de Buenos Aires, Argentina, con una superficie de 200km cuadrados
            y 2000 habitantes.El 80% de la población son mujeres, de las cuales dos tercios tienen entre 18 y 30 años,con
            estudios secundarios completos y estudios universitarios iniciados o terminados.El pueblo tiene un lago donde
            se pesca chancles, que pesan entre 2 y 3 kg en promedio.	La principal atracción turística es el bar Restaurador,
            conocido por su asado y milanesa de ternera a buen precio.Como experiencia única, los visitantes pueden pagar extra
            para ver el carneo del novillo antes de su preparación."""
        ]
        
        # Convertir textos en documentos
        documents = [Document(page_content=text) for text in texts]
        
        # Dividir documentos en chunks más pequeños
        return self.text_splitter.split_documents(documents)
    
    def setup_vectorstore(self, documents: List[Document]):
        """Configura la base de datos vectorial con los documentos"""
        # Crear o cargar la base de datos vectorial
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_dir
        )
        
        # Configurar el retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )
    
    def search(self, query: str, verbose: bool = True):
        """Busca documentos relevantes para una consulta"""
        # 1. Obtener documentos relevantes
        docs = self.retriever.get_relevant_documents(query)
        
        if verbose:
            print("\n=== Documentos encontrados ===")
            for i, doc in enumerate(docs, 1):
                print(f"\nDocumento {i}:")
                print(textwrap.fill(doc.page_content, width=80))
        
        # 2. Crear prompt con contexto
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un asistente útil que responde preguntas basándose en el 
             contexto proporcionado. Si la respuesta no está en el contexto, dilo."""),
            ("user", "Contexto: {context}\n\nPregunta: {question}")
        ])
        
        # 3. Crear el mensaje con contexto
        context = "\n\n".join([doc.page_content for doc in docs])
        chain = prompt | self.llm
        
        # 4. Obtener respuesta
        response = chain.invoke({
            "context": context,
            "question": query
        })
        
        if verbose:
            print("\n=== Respuesta del modelo ===")
        return response.content

def main():
    # Crear instancia de la demo
    print("Inicializando sistema RAG...")
    rag_demo = RAGDemo()
    
    # Crear y cargar documentos de ejemplo
    print("Cargando documentos...")
    documents = rag_demo.create_demo_documents()
    rag_demo.setup_vectorstore(documents)
    
    # Loop de preguntas
    print("\n=== Demo del sistema RAG ===")
    print("Escribe 'salir' para terminar")
    
    while True:
        query = input("\nHaz una pregunta: ")
        if query.lower() == 'salir':
            break
            
        try:
            response = rag_demo.search(query)
            print(f"\nRespuesta: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()