"""Singleton service instances — single source of truth for app components."""
from app.rl_router import RLRouter
from app.redis_cache import RedisCache
from app.vector_memory import VectorMemory
from app.profiles import UserProfileManager
from app.prompt_filter import PromptFilter
from app.parallel_executor import ParallelExecutor
from app.synthesis import ModelSynthesizer
from app.implicit_feedback import ImplicitFeedbackTracker
from app.rate_limiter import rate_limiter
from app.complexity import PromptComplexityScorer

rl_router = RLRouter()
cache = RedisCache()
memory = VectorMemory()
profile_manager = UserProfileManager()
prompt_filter = PromptFilter()
parallel_executor = ParallelExecutor(rl_router=rl_router)
synthesizer = ModelSynthesizer(parallel_executor=parallel_executor)
implicit_tracker = ImplicitFeedbackTracker()
complexity_scorer = PromptComplexityScorer()

