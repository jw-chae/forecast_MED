import os
import warnings
from dotenv import load_dotenv
import gradio as gr
from typing import List, Dict, Tuple

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # Corrected import path
from langchain_openai import ChatOpenAI # Use ChatOpenAI for OpenAI-compatible APIs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document # Import Document if needed for format_docs
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

reasoning_model_id = os.getenv("REASONING_MODEL_ID", "deepseek-chat") # Default to chat model
tool_model_id = os.getenv("TOOL_MODEL_ID", "deepseek-chat") # Keep this for potential future use, but primary LLM is reasoning_model
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
use_deepseek_api = os.getenv("USE_DEEPSEEK_API", "no").lower() == "yes"
# Note: HuggingFace/Ollama logic removed as we are focusing on DeepSeek API via ChatOpenAI

if not use_deepseek_api:
    warnings.warn("USE_DEEPSEEK_API is not set to 'yes'. This script is configured for DeepSeek API.")
    # Optionally raise an error or exit if DeepSeek API is required
    # raise ValueError("This script requires USE_DEEPSEEK_API to be set to 'yes'")

if not deepseek_api_key:
    raise ValueError("DEEPSEEK_API_KEY environment variable is not set.")

# --- LangChain RAG Setup ---

# 1. Initialize Embeddings and Vector Store Retriever
print("Initializing embeddings and vector store...")
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'} # Keep on CPU for wider compatibility
    )
    db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    if not os.path.exists(db_dir):
        raise FileNotFoundError(f"Chroma database directory not found: {db_dir}. Run ingest_pdfs.py first.")

    vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    # 메타데이터 매개변수 제거
    retriever = vectordb.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 3}
    )
    print("Vector store loaded successfully.")
except Exception as e:
    print(f"Error initializing vector store: {e}")
    raise

# 2. Initialize LLM (using ChatOpenAI for DeepSeek)
print(f"Initializing LLM with DeepSeek API: {reasoning_model_id}")
llm = ChatOpenAI(
    model=reasoning_model_id,
    openai_api_key=deepseek_api_key,
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0.7,  # 온도를 낮춰서 더 일관된 응답 생성
    request_timeout=120,  # 요청 타임아웃 시간을 늘림 (초 단위)
    max_tokens=1500,  # 최대 토큰 수 제한
)

# 3. Define the RAG Prompt Template (Doctor Persona - Including Chat History)
template = """### 角色与目标
您是一名医学诊断与治疗顾问。您的任务是在**不依赖实验室确诊结果**的情况下，仅基于临床症状、病程、流行病学特征提供全面、专业、明确的医疗分析与诊断方案。

### 输入数据
**患者的病史 (Chat History):**
{chat_history}

**患者的当前陈述 (User Query):**
{question}

**流行病学特征:**
请从患者描述中提取旅行史、接触史、地区疫情等流行病学信息，这对COVID-19、结核等疾病诊断至关重要。

**发病时间和进展:**
请估计症状持续时间，将疾病分类为急性(<1周)、亚急性(1-4周)或慢性(>4周)。

**全身消耗症状:**
特别注意发热模式、盗汗、体重减轻等全身症状，这些对区分感染性、炎症性和恶性疾病至关重要。

**参考资料 (for your reference):**
{context}

### 分析与诊断任务
1. **全面分析患者信息:**
   - 详细分析患者的症状描述、病史、发病时间和流行病学特征
   - 列出关键阳性发现和重要阴性发现
   - 评估病程阶段(急性/亚急性/慢性)对诊断的影响
   - **必须单独分析流行病学风险因素**并在诊断中权衡其重要性

2. **多重诊断可能性 (≥3个):**
   - 按可能性从高到低列出至少3个可能的诊断
   - 对每个诊断必须提供:
     a) 支持证据：列出支持该诊断的症状和流行病学因素
     b) 反证据：列出与该诊断不一致的症状和发现
     c) 信任度评分：为每个诊断提供0-1之间的信任度评分
   - 所有诊断必须有明确的医学依据，直接引用参考资料
   - **假设没有实验室检查结果可用**，仅基于临床表现进行初步诊断

3. **鉴别诊断策略:**
   - 明确列出需要排除的危急重症
   - 列出至少2个被排除但临床上重要的疾病，简述为何排除("Why not X")
   - 解释为何某些常见病因在此不太可能

4. **推荐额外诊断检查:**
   - 按优先级排序(紧急/必要/可选)
   - 对于每项检查，说明具体目的(确认/排除哪种诊断)
   - 说明期望结果及其对诊断的影响

5. **详细治疗方案 (针对最可能诊断):**
   - 提供明确具体的治疗建议，包括药物名称、剂量、用法
   - 给出治疗周期和预期效果
   - 提供必要的生活方式建议和随访计划

6. **传染病专项处理 (如适用):**
   - 明确疾病传染性及传播途径
   - 详细的隔离措施建议
   - 院内感染控制方案
   - 社区传播防控措施

7. **学术引用规范:**
   - 对每个重要观点，使用"根据《文档名》[页码]，'直接引用原文'"的格式
   - 确保引用的页码准确，并使用引号标记直接引用的内容

8. **如果患者信息不足:**
   - 提出最关键的1-2个问题，说明这些问题如何帮助确定诊断

### 输出格式
**临床分析:**
- 主要症状: [列出关键症状]
- 次要症状: [列出次要症状]
- 重要阴性表现: [列出对诊断有意义的阴性发现]
- 病程分类: [急性/亚急性/慢性]
- 流行病学特征: [旅行史、接触史、地区疫情等]

**鉴别诊断:**
1. [疾病A] - 信任度:[0-1]
   - 支持证据: [列出支持该诊断的症状和发现]
   - 反证据: [列出与该诊断不一致的症状和发现]
   - 医学依据: 根据《文档名》[页码]:"直接引用"

2. [疾病B] - 信任度:[0-1]
   - 支持证据: ...
   - 反证据: ...
   - 医学依据: ...

3. [疾病C] - 信任度:[0-1]
   ...

**被排除的重要诊断:**
1. [疾病X]: 排除原因 - [简要说明]
2. [疾病Y]: 排除原因 - [简要说明]

**推荐检查:**
- [紧急] [检查名称]: 目的 - [确认/排除疾病], 预期结果 - [...]
- [必要] [检查名称]: ...
- [可选] [检查名称]: ...

**最可能诊断:** [最高信任度的疾病]

**治疗方案:**
- 药物治疗: [药物名称、剂量、用法]
- 治疗周期: [具体时间]
- 生活建议: [具体建议]
- 随访计划: [时间和重点]

**传染控制 (如适用):**
- 传染性: [高/中/低], 传播途径: [...]
- 隔离措施: [具体措施]
- 院内感染控制: [具体措施]
- 社区防控: [具体措施]

**医学免责声明:**
此分析基于有限的临床信息，不构成最终诊断。任何治疗决定都应在完整的临床评估和适当的实验室检查后由合格的医疗专业人员做出。本分析不能替代专业医疗咨询、诊断或治疗。

**开始您的回复:**"""

prompt = ChatPromptTemplate.from_template(template)

# 4. Define Helper Function to Format Docs with Source Information
def format_docs(docs: list[Document]) -> str:
    """원본 문서 텍스트 그대로를 페이지 정보와 함께 제공합니다."""
    formatted_text = ""
    for i, doc in enumerate(docs):
        # 메타데이터에서 출처 정보 추출
        source = "알 수 없는 출처"
        page_num = "?"
        
        if hasattr(doc, 'metadata') and doc.metadata:
            source = doc.metadata.get('source', '알 수 없는 출처')
            # 파일 이름만 추출
            if isinstance(source, str) and '/' in source:
                source = source.split('/')[-1]
            
            page_num = doc.metadata.get('page', '?')
        
        # 텍스트에 출처 정보를 포함하여 포맷팅 (원본 텍스트 유지)
        formatted_text += f"====== 문서 {i+1} ======\n"
        formatted_text += f"출처: {source}\n"
        formatted_text += f"페이지: {page_num}\n"
        formatted_text += f"원문:\n{doc.page_content}\n\n"
    
    return formatted_text

# Helper function to format chat history
def format_chat_history(chat_history: List[Tuple[str, str]]) -> str:
    """Formats chat history into a readable string."""
    if not chat_history:
        return "无病史记录。这是首次咨询。"
    
    formatted_history = ""
    for i, (user_msg, ai_msg) in enumerate(chat_history):
        formatted_history += f"对话 {i+1}:\n患者: {user_msg}\n医生: {ai_msg}\n\n"
    
    return formatted_history

# 5. Define the RAG Chain with chat history 
print("Defining RAG chain with chat history support...")

def create_chain(chat_history):
    """
    채팅 기록을 기반으로 RAG 체인을 생성합니다.
    
    Args:
        chat_history (list): 채팅 기록 리스트
        
    Returns:
        chain: 실행 가능한 RAG 체인
    """
    return (
        RunnableParallel(
            context=(retriever | format_docs),
            question=RunnablePassthrough(),
            chat_history=lambda _: format_chat_history(chat_history)
        )
        | prompt
        | llm
        | StrOutputParser()
    )

# Gradio 인터페이스에 참고 자료 표시 기능 추가
def chat_with_history(message, history):
    """Chat function that maintains history for the Gradio ChatInterface."""
    if not message:
        return "질문이나 증상을 입력해주세요."
    
    # Convert Gradio chat history to the format our chain expects
    chat_history = []
    print(f"현재 대화 기록 길이: {len(history)}")
    for i, (user_msg, ai_msg) in enumerate(history):
        print(f"대화 기록 {i+1}: 사용자: {user_msg[:30]}... AI: {ai_msg[:30]}...")
        if user_msg and ai_msg:  # Ensure both messages exist
            chat_history.append((user_msg, ai_msg))
    
    # Create a chain with the current chat history
    current_chain = create_chain(chat_history)
    
    print(f"Invoking RAG chain for input: {message[:100]}...")
    try:
        # 원시 검색 결과 확인
        raw_docs = retriever.invoke(message)
        print(f"검색된 문서 수: {len(raw_docs)}")
        
        # 문서 메타데이터 로깅 (디버깅 용)
        for i, doc in enumerate(raw_docs):
            source = "알 수 없음"
            page = "?"
            if hasattr(doc, 'metadata') and doc.metadata:
                source = doc.metadata.get('source', '알 수 없음')
                page = doc.metadata.get('page', '?')
            print(f"문서 {i+1} 출처: {source}, 페이지: {page}")
            print(f"문서 {i+1} 내용 앞부분: {doc.page_content[:200]}...")
        
        # Get response using the chain
        response = current_chain.invoke(message)
        print("RAG chain invocation successful.")
        print(f"응답: {response[:100]}...")
        
        return response
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error during RAG chain invocation: {e}")
        print(f"상세 오류: {error_trace}")
        return f"요청 처리 중 오류가 발생했습니다. 다시 시도해 주세요. 오류: {str(e)}"

def main():
    """Launches the Gradio chat interface with history support."""
    print("Launching Gradio chat interface...")
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("## 메디컬 진단 보조 시스템 (DeepSeek + LangChain RAG)")
        gr.Markdown("증상을 설명하거나 질문을 입력하세요. AI가 의료 보조자 역할을 하여 대화 기록과 문맥을 고려한 응답을 제공합니다.")
        
        chatbot = gr.Chatbot(
            height=500,
            bubble_full_width=False,
            show_copy_button=True,
        )
        msg = gr.Textbox(
            placeholder="여기에 증상이나 질문을 입력하세요...",
            container=False,
            scale=7,
        )
        with gr.Row():
            submit = gr.Button("전송", variant="primary", scale=1)
            clear = gr.Button("대화 초기화", scale=1)
        
        with gr.Accordion("예시 질문", open=False):
            examples = gr.Examples(
                examples=[
                    "저는 요즘 두통과 어지러움이 있습니다.",
                    "가슴이 답답하고 숨쉬기가 힘듭니다.",
                    "피부에 발진이 생겼어요.",
                    "복통과 설사가 계속됩니다.",
                    "목이 아프고 열이 납니다."
                ],
                inputs=msg
            )
        
        def respond(message, chat_history):
            if not message:
                return chat_history
            
            chat_history.append((message, "응답 생성 중..."))
            yield chat_history
            
            try:
                response = chat_with_history(message, chat_history[:-1])
                chat_history[-1] = (message, response)
            except Exception as e:
                chat_history[-1] = (message, f"오류가 발생했습니다: {str(e)}")
            
            yield chat_history
        
        def clear_chat():
            return []
        
        submit.click(
            respond, 
            [msg, chatbot], 
            [chatbot],
            queue=True
        ).then(
            lambda: "", 
            None, 
            [msg],
            queue=False
        )
        
        msg.submit(
            respond,
            [msg, chatbot],
            [chatbot],
            queue=True
        ).then(
            lambda: "",
            None,
            [msg],
            queue=False
        )
        
        clear.click(clear_chat, None, [chatbot], queue=False)
    
    # 부드러운 종료를 위한 예외 처리
    try:
        demo.launch(share=False, debug=True)
    except KeyboardInterrupt:
        print("Gradio 서버가 사용자에 의해 종료되었습니다.")
    except Exception as e:
        print(f"Gradio 서버 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()