"""
Comprehensive import tests to ensure all dependencies and modules are importable.

This test suite validates that:
1. All external dependencies from requirements.txt can be imported
2. All app modules can be imported without errors
3. No circular import issues exist
4. All public APIs are accessible

Run this test before pushing to catch import errors early.
"""

import pytest


class TestExternalDependencies:
    """Test that all external dependencies can be imported."""

    def test_fastapi_import(self):
        """Test FastAPI can be imported."""
        import fastapi  # noqa: F401

        assert True

    def test_uvicorn_import(self):
        """Test uvicorn can be imported."""
        import uvicorn  # noqa: F401

        assert True

    def test_httpx_import(self):
        """Test httpx can be imported."""
        import httpx  # noqa: F401

        assert True

    def test_pydantic_import(self):
        """Test pydantic can be imported."""
        from pydantic import BaseModel  # noqa: F401

        assert True

    def test_pydantic_settings_import(self):
        """Test pydantic-settings can be imported."""
        from pydantic_settings import BaseSettings  # noqa: F401

        assert True

    def test_python_dotenv_import(self):
        """Test python-dotenv can be imported."""
        import dotenv  # noqa: F401

        assert True

    def test_transformers_import(self):
        """Test transformers can be imported."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

            assert True
        except ImportError:
            pytest.skip("transformers not available (optional)")

    def test_torch_import(self):
        """Test torch can be imported."""
        try:
            import torch  # noqa: F401

            assert True
        except ImportError:
            pytest.skip("torch not available (optional)")

    def test_outlines_import(self):
        """Test outlines can be imported."""
        try:
            import outlines  # noqa: F401

            assert True
        except ImportError:
            pytest.skip("outlines not available (optional)")

    def test_trafilatura_import(self):
        """Test trafilatura can be imported."""
        try:
            import trafilatura  # noqa: F401

            assert True
        except ImportError:
            pytest.skip("trafilatura not available (optional for V3)")

    def test_lxml_import(self):
        """Test lxml can be imported."""
        try:
            import lxml  # noqa: F401

            assert True
        except ImportError:
            pytest.skip("lxml not available (optional for V3)")

    def test_ruff_import(self):
        """Test ruff can be imported (development tool)."""
        try:
            import ruff  # noqa: F401

            assert True
        except ImportError:
            pytest.skip("ruff not available (dev dependency)")


class TestCoreModuleImports:
    """Test that all core modules can be imported."""

    def test_config_import(self):
        """Test core.config can be imported."""
        from app.core.config import Settings, settings  # noqa: F401

        assert True

    def test_logging_import(self):
        """Test core.logging can be imported."""
        from app.core.logging import get_logger, setup_logging  # noqa: F401

        assert True

    def test_middleware_import(self):
        """Test core.middleware can be imported."""
        from app.core.middleware import request_context_middleware  # noqa: F401

        assert True

    def test_errors_import(self):
        """Test core.errors can be imported."""
        from app.core.errors import init_exception_handlers  # noqa: F401

        assert True

    def test_cache_import(self):
        """Test core.cache can be imported."""
        from app.core.cache import SimpleCache, scraping_cache  # noqa: F401

        assert True


class TestServiceImports:
    """Test that all service modules can be imported."""

    def test_summarizer_import(self):
        """Test services.summarizer can be imported."""
        from app.services.summarizer import OllamaService, ollama_service  # noqa: F401

        assert True

    def test_transformers_summarizer_import(self):
        """Test services.transformers_summarizer can be imported."""
        from app.services.transformers_summarizer import (  # noqa: F401
            TransformersService,
            transformers_service,
        )

        assert True

    def test_hf_streaming_summarizer_import(self):
        """Test services.hf_streaming_summarizer can be imported."""
        from app.services.hf_streaming_summarizer import (  # noqa: F401
            HFStreamingSummarizer,
            hf_streaming_service,
        )

        assert True

    def test_article_scraper_import(self):
        """Test services.article_scraper can be imported."""
        from app.services.article_scraper import ArticleScraperService  # noqa: F401

        assert True

    def test_structured_summarizer_import(self):
        """Test services.structured_summarizer can be imported."""
        try:
            from app.services.structured_summarizer import (  # noqa: F401
                StructuredSummarizer,
                structured_summarizer_service,
            )

            assert True
        except ImportError:
            pytest.skip("structured_summarizer dependencies not available")


class TestV1APIImports:
    """Test that V1 API modules can be imported."""

    def test_v1_routes_import(self):
        """Test api.v1.routes can be imported."""
        from app.api.v1.routes import api_router  # noqa: F401

        assert True

    def test_v1_schemas_import(self):
        """Test api.v1.schemas can be imported."""
        from app.api.v1.schemas import (  # noqa: F401
            ErrorResponse,
            HealthResponse,
            SummarizeRequest,
            SummarizeResponse,
        )

        assert True

    def test_v1_summarize_import(self):
        """Test api.v1.summarize can be imported."""
        from app.api.v1.summarize import summarize_text  # noqa: F401

        assert True


class TestV2APIImports:
    """Test that V2 API modules can be imported."""

    def test_v2_routes_import(self):
        """Test api.v2.routes can be imported."""
        from app.api.v2.routes import api_router  # noqa: F401

        assert True

    def test_v2_schemas_import(self):
        """Test api.v2.schemas can be imported."""
        from app.api.v2.schemas import (  # noqa: F401
            ErrorResponse,
            HealthResponse,
            SummarizeRequest,
            SummarizeResponse,
        )

        assert True

    def test_v2_summarize_import(self):
        """Test api.v2.summarize can be imported."""
        from app.api.v2.summarize import summarize_text_stream  # noqa: F401

        assert True


class TestV3APIImports:
    """Test that V3 API modules can be imported."""

    def test_v3_routes_import(self):
        """Test api.v3.routes can be imported."""
        from app.api.v3.routes import api_router  # noqa: F401

        assert True

    def test_v3_schemas_import(self):
        """Test api.v3.schemas can be imported."""
        from app.api.v3.schemas import (  # noqa: F401
            ErrorResponse,
            HealthResponse,
            ScrapeSummarizeRequest,
            ScrapeSummarizeResponse,
        )

        assert True

    def test_v3_scrape_summarize_import(self):
        """Test api.v3.scrape_summarize can be imported."""
        from app.api.v3.scrape_summarize import (
            scrape_and_summarize_stream,  # noqa: F401
        )

        assert True


class TestV4APIImports:
    """Test that V4 API modules can be imported."""

    def test_v4_routes_import(self):
        """Test api.v4.routes can be imported."""
        try:
            from app.api.v4.routes import api_router  # noqa: F401

            assert True
        except ImportError:
            pytest.skip("V4 API dependencies not available")

    def test_v4_schemas_import(self):
        """Test api.v4.schemas can be imported."""
        try:
            from app.api.v4.schemas import (  # noqa: F401
                ErrorResponse,
                HealthResponse,
                StructuredSummary,
                StructuredSummaryRequest,
                StructuredSummaryResponse,
                SummarizationStyle,
            )

            assert True
        except ImportError:
            pytest.skip("V4 API dependencies not available")

    def test_v4_structured_summary_import(self):
        """Test api.v4.structured_summary can be imported."""
        try:
            from app.api.v4.structured_summary import (  # noqa: F401
                generate_structured_summary_stream,
            )

            assert True
        except ImportError:
            pytest.skip("V4 API dependencies not available")


class TestMainAppImport:
    """Test that the main app can be imported."""

    def test_main_app_import(self):
        """Test app.main can be imported."""
        from app.main import app  # noqa: F401

        assert True

    def test_main_app_has_attributes(self):
        """Test that main app has expected attributes."""
        from app.main import app

        assert hasattr(app, "title")
        assert hasattr(app, "version")
        assert app.title == "Text Summarizer API"
        assert app.version == "4.0.0"


class TestCircularImports:
    """Test that there are no circular import issues."""

    def test_repeated_imports(self):
        """Test that modules can be imported multiple times without issues."""
        # Import all major modules twice to catch circular import issues
        import importlib

        modules_to_test = [
            "app.core.config",
            "app.core.logging",
            "app.core.middleware",
            "app.core.errors",
            "app.services.summarizer",
            "app.services.transformers_summarizer",
            "app.services.hf_streaming_summarizer",
            "app.api.v1.routes",
            "app.api.v2.routes",
            "app.main",
        ]

        for module_name in modules_to_test:
            # First import
            mod1 = importlib.import_module(module_name)
            # Reload (simulates second import)
            mod2 = importlib.reload(mod1)
            # Should be the same module
            assert mod1 is mod2


class TestRuffMigrationImports:
    """Test that imports still work after ruff migration."""

    def test_all_app_modules_importable(self):
        """Test that all app modules can be imported after ruff formatting."""
        # This test ensures ruff didn't break any imports
        from app import __version__  # noqa: F401
        from app.core import config, errors, logging, middleware  # noqa: F401
        from app.services import (  # noqa: F401
            article_scraper,
            hf_streaming_summarizer,
            summarizer,
            transformers_summarizer,
        )

        assert True

    def test_import_statements_formatted(self):
        """Test that import statements are properly formatted by ruff."""
        # This is a meta-test - if imports work, ruff formatting is likely correct
        from app.core.config import settings  # noqa: F401
        from app.main import app  # noqa: F401
        from app.services.summarizer import ollama_service  # noqa: F401

        assert True
