"""PromptLayer wrapper."""
import datetime

from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, ChatResult


class PromptLayerChatOpenAI(ChatOpenAI):
    """Wrapper around OpenAI Chat large language models and PromptLayer.
    To use, you should have the ``openai`` and ``promptlayer`` python
    package installed, and the environment variable ``OPENAI_API_KEY``
    and ``PROMPTLAYER_API_KEY`` set with your openAI API key and
    promptlayer key respectively.
    All parameters that can be passed to the OpenAI LLM can also
    be passed here. The PromptLayerChatOpenAI adds to optional
    parameters:
        ``pl_tags``: list of strings to tag the request with.
        ``return_pl_id``: If True, the PromptLayer request ID will be
            returned in the ``generation_info`` field of the
            ``Generation`` object.
    Example:
        .. code-block:: python
            from langchain.chat_models import PromptLayerChatOpenAI
            openai = PromptLayerChatOpenAI(model_name="gpt-3.5-turbo")
    """

    pl_tags: list[str] | None
    pl_project: str | None
    return_pl_id: bool | None = False

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None
    ) -> ChatResult:
        """Call ChatOpenAI generate and then call PromptLayer API to log the request."""
        from promptlayer.utils import promptlayer_api_request

        request_start_time = datetime.datetime.now().timestamp()
        generated_responses = super()._generate(messages, stop)
        request_end_time = datetime.datetime.now().timestamp()
        message_dicts, params = super()._create_message_dicts(messages, stop)
        for i, generation in enumerate(generated_responses.generations):
            response_dict, params = super()._create_message_dicts(
                [generation.message], stop
            )
            request_usage = generated_responses.llm_output["token_usage"]
            pl_request_id = promptlayer_api_request(
                "langchain.PromptLayerChatOpenAI",
                "langchain",
                message_dicts,
                params,
                self.pl_tags,
                response_dict,
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
        self, messages: list[BaseMessage], stop: list[str] | None = None
    ) -> ChatResult:
        """Call ChatOpenAI agenerate and then call PromptLayer to log."""
        from promptlayer.utils import promptlayer_api_request_async

        request_start_time = datetime.datetime.now().timestamp()
        generated_responses = await super()._agenerate(messages, stop)
        request_end_time = datetime.datetime.now().timestamp()
        message_dicts, params = super()._create_message_dicts(messages, stop)
        for i, generation in enumerate(generated_responses.generations):
            response_dict, params = super()._create_message_dicts(
                [generation.message], stop
            )
            request_usage = generated_responses.llm_output["token_usage"]
            pl_request_id = await promptlayer_api_request_async(
                "langchain.PromptLayerChatOpenAI.async",
                "langchain",
                message_dicts,
                params,
                self.pl_tags,
                response_dict,
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