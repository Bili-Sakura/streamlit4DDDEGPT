from utils import LLM

if __name__ == "__main__":
    llm = LLM()
    llm.init_vectorstore()
    print(llm.get_answer("什么是DDE"))
