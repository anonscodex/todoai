from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_groq import ChatGroq  # Groq integration
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace "*" with specific origins like ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Replace with your desired Groq model
    api_key=os.getenv("GROQ_API_KEY")
)

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["to_do_list"],  # Ensure this matches the request body
    template="""You are an expert cryptocurrency investment advisor with deep knowledge of market trends, technical analysis, and blockchain technology. 
    
User Question: {to_do_list}

Analyze and provide insights covering:
1. Market Analysis: Current market conditions and relevant trends
2. Technical Perspective: Key support/resistance levels and pattern analysis
3. Sentiment Analysis: Social media trends and market sentiment
4. Risk Assessment: Potential risks and mitigation strategies
5. Actionable Recommendations: Clear, practical steps based on the analysis

Important Considerations:
- Focus on data-driven insights
- Include both bull and bear scenarios
- Highlight potential regulatory impacts
- Consider market volatility and liquidity
- Emphasize risk management

Provide your analysis:"""
)

# Create an LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# SimpleSequentialChain expects an explicit input and output mapping
chain = SimpleSequentialChain(
    chains=[llm_chain],
    input_key="to_do_list",  # Define the input key for the chain
    output_key="plan"  # Define the output key for the chain
)

# Define the input model
class ToDoList(BaseModel):
    tasks: str

@app.post("/plan-day")
async def plan_day(to_do_list: ToDoList):
    try:
        # Run the chain with input data
        response = chain.run({"to_do_list": to_do_list.tasks})  # Match the input key here
        return {"plan": response}
    except Exception as e:
        return {"error": str(e)}
