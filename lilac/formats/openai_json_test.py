"""Tests for the openai_json format."""


from ..data.dataset_test_utils import TestDataMaker
from .openai_json import OpenAIConversationJSON, OpenAIJSON


def test_infer_openai_json(make_test_data: TestDataMaker) -> None:
  dataset = make_test_data(
    [
      {
        'messages': [
          {'role': 'user', 'content': 'Hello'},
          {'role': 'assistant', 'content': 'Hi'},
          {'role': 'user', 'content': 'How are you today?'},
          {'role': 'assistant', 'content': "I'm fine."},
        ],
      },
      {
        'messages': [
          {'role': 'user', 'content': 'Who are you?'},
          {'role': 'assistant', 'content': "I'm OpenChat."},
        ],
      },
    ]
  )

  assert dataset.manifest().dataset_format == OpenAIJSON()


def test_infer_openai_conversation_json(make_test_data: TestDataMaker) -> None:
  dataset = make_test_data(
    [
      {
        'conversation': [
          {'role': 'user', 'content': 'Hello'},
          {'role': 'assistant', 'content': 'Hi'},
          {'role': 'user', 'content': 'How are you today?'},
          {'role': 'assistant', 'content': "I'm fine."},
        ],
      },
      {
        'conversation': [
          {'role': 'user', 'content': 'Who are you?'},
          {'role': 'assistant', 'content': "I'm OpenChat."},
        ],
      },
    ]
  )

  assert dataset.manifest().dataset_format == OpenAIConversationJSON()
