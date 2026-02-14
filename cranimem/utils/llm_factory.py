# File: cranimem/utils/llm_factory.py
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFacePipeline
from langchain_community.chat_models import ChatOllama
from ..config import settings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch


_LLM_CACHE = {}

def get_llm(temperature: float = 0.7, use_4bit: bool | None = None):
    if temperature is None or temperature <= 0:
        temperature = 0.01
    provider = settings.LLM_PROVIDER.lower()
    if provider == "local_hf":
        cache_key = (provider, use_4bit)
        if cache_key in _LLM_CACHE:
            return _LLM_CACHE[cache_key]
    else:
        cache_key = (provider, temperature)
        if cache_key in _LLM_CACHE:
            return _LLM_CACHE[cache_key]

    if provider == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError("Missing GROQ_API_KEY")
        llm = ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY.get_secret_value(),
            temperature=temperature
        )
        _LLM_CACHE[(provider, temperature)] = llm
        return llm

    elif provider == "gemini":
        if not settings.GOOGLE_API_KEY:
            raise ValueError("Missing GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY.get_secret_value(),
            temperature=temperature,
            convert_system_message_to_human=True
        )
        _LLM_CACHE[(provider, temperature)] = llm
        return llm

    elif provider == "huggingface":
        if not settings.HUGGINGFACEHUB_API_TOKEN:
            raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN")
        print(f"Using Hugging Face Model: {settings.HF_REPO_ID}")
        
        llm_endpoint = HuggingFaceEndpoint(
            repo_id=settings.HF_REPO_ID,
            task="text-generation",
            max_new_tokens=512,
            do_sample=True,
            temperature=temperature,
            huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN.get_secret_value()
        )
        llm = ChatHuggingFace(llm=llm_endpoint)
        _LLM_CACHE[(provider, temperature)] = llm
        return llm
    
    elif provider == "ollama":
        print(f" Loading Local Model: {settings.OLLAMA_MODEL}")
        llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            temperature=temperature,
            base_url=settings.OLLAMA_BASE_URL,
            keep_alive="0", 
            timeout=45.0,
            max_retries=1
        )
        _LLM_CACHE[(provider, temperature)] = llm
        return llm

    elif provider == "local_hf":
        print(f"Loading Local HF Model: {settings.HF_REPO_ID}")
        tokenizer = AutoTokenizer.from_pretrained(settings.HF_REPO_ID)
        enable_4bit = bool(use_4bit) if use_4bit is not None else True
        if torch.cuda.is_available() and enable_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                settings.HF_REPO_ID,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True
            )
        elif torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                settings.HF_REPO_ID,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                settings.HF_REPO_ID,
                device_map="auto",
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
        gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=temperature
            ,
            return_full_text=False
        )
        llm = HuggingFacePipeline(pipeline=gen)
        _LLM_CACHE[(provider, use_4bit)] = llm
        return llm

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
