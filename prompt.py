from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
from dotenv import load_dotenv
import langchain

langchain.debug = True

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()

db = Chroma(
  persist_directory="emb",
  embedding_function=embeddings
)

# retriever = RedundantFilterRetriever(
#   embeddings=embeddings,
#   chroma=db
# )

retriever = db.as_retriever()


chain = RetrievalQA.from_chain_type( 
  # the internal prompts used by lamgchain can be found in the implement ation of these methods
  llm=chat,
  retriever=retriever,
  chain_type="stuff" #stuff means we're just stuffing the user's query and the result from the retriever into the llm
)

result = chain.run("What is an interesting fact about englist language?")

print(result)