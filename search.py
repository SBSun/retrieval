from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

if __name__ == "__main__":
    load_dotenv()

    search = TavilySearchResults(max_results=1)

    search_query = "2025년 대한민국 최저시급"
    search_result = search.invoke(search_query)

    template = """
    다음은 2025년 대한민국의 최저시급에 대한 검색 결과입니다:
    {search_result}
    
    위의 정보를 바탕으로 2025년 대한민국의 최저시급을 알려주세요.
    """
    prompt_template = PromptTemplate(
        input_variables=["search_result"], template=template
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    chain = prompt_template | llm | StrOutputParser()

    res = chain.invoke(input={"search_result": search_result})

    print(res)