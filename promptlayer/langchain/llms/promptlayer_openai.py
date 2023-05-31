"""PromptLayer wrapper."""
import datetime

from langchain.llms import OpenAI
from langchain.schema import LLMResult


class PromptLayerOpenAI(OpenAI):
    """Wrapper around OpenAI large language models.

    To use, you should have the ``openai`` and ``promptlayer`` python
    package installed, and the environment variable ``OPENAI_API_KEY``
    and ``PROMPTLAYER_API_KEY`` set with your openAI API key and
    promptlayer key respectively.

    All parameters that can be passed to the OpenAI LLM can also
    be passed here. The PromptLayerOpenAI LLM adds two optional
    parameters:
        ``pl_tags``: list of strings to tag the request with.
        ``return_pl_id``: If True, the PromptLayer request ID will be
            returned in the ``generation_info`` field of the
            ``Generation`` object.

    Example:
        .. code-block:: python

            from langchain.llms import PromptLayerOpenAI
            openai = PromptLayerOpenAI(model_name="text-davinci-003")
    """

    pl_tags: list[str] | None
    pl_project: str | None
    return_pl_id: bool | None = False

    def _generate(
        self, prompts: list[str], stop: list[str] | None = None
    ) -> LLMResult:
        """Call OpenAI generate and then call PromptLayer API to log the request."""
        from promptlayer.utils import promptlayer_api_request

        request_start_time = datetime.datetime.now().timestamp()
        generated_responses = super()._generate(prompts, stop)
        request_end_time = datetime.datetime.now().timestamp()
        for i in range(len(prompts)):
            prompt = prompts[i]
            generation = generated_responses.generations[i][0]
            resp = { "text": generation.text }
            request_usage = generated_responses.llm_output["token_usage"]
            pl_request_id = promptlayer_api_request(
                "langchain.PromptLayerOpenAI",
                "langchain",
                [prompt],
                self._identifying_params,
                self.pl_tags,
                resp,
                request_start_time,
                request_end_time,
                request_usage,
                metadata={ "project": self.pl_project if self.pl_project else None },
                return_pl_id=self.return_pl_id,
            )
            if self.return_pl_id:
                if generation.generation_info is None or not isinstance(
                    generation.generation_info, dict
                ):
                    generation.generation_info = {}
                generation.generation_info["pl_request_id"] = pl_request_id
        return generated_responses

    async def _agenerate(
        self, prompts: list[str], stop: list[str] | None = None
    ) -> LLMResult:
        from promptlayer.utils import promptlayer_api_request_async

        request_start_time = datetime.datetime.now().timestamp()
        generated_responses = await super()._agenerate(prompts, stop)
        request_end_time = datetime.datetime.now().timestamp()
        for i in range(len(prompts)):
            prompt = prompts[i]
            generation = generated_responses.generations[i][0]
            resp = { "text": generation.text }
            request_usage = generated_responses.llm_output["token_usage"]
            pl_request_id = await promptlayer_api_request_async(
                "langchain.PromptLayerOpenAI.async",
                "langchain",
                [prompt],
                self._identifying_params,
                self.pl_tags,
                resp,
                request_start_time,
                request_end_time,
                request_usage,
                metadata={ "project": self.pl_project if self.pl_project else None },
                return_pl_id=self.return_pl_id,
            )
            if self.return_pl_id:
                if generation.generation_info is None or not isinstance(
                    generation.generation_info, dict
                ):
                    generation.generation_info = {}
                generation.generation_info["pl_request_id"] = pl_request_id
        return generated_responses
