from __future__ import annotations

from swe_data_process.utils import check_reasoning_content, check_roles


class TestCheckRoles:
    def test_valid_user_assistant(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert check_roles(messages) is True

    def test_valid_with_system_prefix(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert check_roles(messages) is True

    def test_valid_with_tool_calls(self):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "", "tool_calls": [{}]},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "done"},
        ]
        assert check_roles(messages) is True

    def test_valid_multiple_tools(self):
        messages = [
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "content": "r1"},
            {"role": "tool", "content": "r2"},
            {"role": "assistant", "content": "done"},
        ]
        assert check_roles(messages) is True

    def test_valid_user_after_assistant(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "more"},
            {"role": "assistant", "content": "ok"},
        ]
        assert check_roles(messages) is True

    def test_empty_messages(self):
        assert check_roles([]) is False

    def test_first_non_system_not_user(self):
        messages = [
            {"role": "system", "content": ""},
            {"role": "assistant", "content": "hi"},
        ]
        assert check_roles(messages) is False

    def test_last_not_assistant(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "content": "result"},
        ]
        assert check_roles(messages) is False

    def test_user_followed_by_tool(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "bad"},
            {"role": "assistant", "content": "done"},
        ]
        assert check_roles(messages) is False

    def test_assistant_followed_by_system(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": "bad"},
            {"role": "assistant", "content": "done"},
        ]
        assert check_roles(messages) is False

    def test_tool_followed_by_user(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "content": "r"},
            {"role": "user", "content": "bad"},
            {"role": "assistant", "content": "done"},
        ]
        assert check_roles(messages) is False

    def test_only_system(self):
        messages = [{"role": "system", "content": "sys"}]
        assert check_roles(messages) is False


class TestCheckReasoningContent:
    def test_fast_mode_always_true(self):
        messages = [
            {"role": "assistant", "content": "hi"},
        ]
        assert check_reasoning_content(messages, "fast", None) is True

    def test_slow_mode_all_have_reasoning(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning_content": "thinking"},
        ]
        assert check_reasoning_content(messages, "slow", None) is True

    def test_slow_mode_missing_reasoning(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert (
            check_reasoning_content(
                messages, "slow", None, reasoning_check_mode="strict"
            )
            is False
        )

    def test_slow_mode_empty_reasoning(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning_content": ""},
        ]
        assert (
            check_reasoning_content(
                messages, "slow", None, reasoning_check_mode="strict"
            )
            is False
        )

    def test_pseudo_turns_skips_initial(self):
        messages = [
            {"role": "assistant", "content": "no reasoning here"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning_content": "thinking"},
        ]
        assert (
            check_reasoning_content(
                messages, "slow", 2, reasoning_check_mode="strict"
            )
            is True
        )

    def test_pseudo_turns_zero(self):
        messages = [
            {"role": "assistant", "content": "no reasoning"},
        ]
        assert (
            check_reasoning_content(
                messages, "slow", 0, reasoning_check_mode="strict"
            )
            is False
        )

    def test_non_assistant_roles_ignored(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "reasoning_content": "yes"},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "done", "reasoning_content": "ok"},
        ]
        assert (
            check_reasoning_content(
                messages, "slow", None, reasoning_check_mode="strict"
            )
            is True
        )

    def test_adaptive_default_threshold_passes_one_fifth_reasoning(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "one", "reasoning_content": "thinking"},
            {"role": "user", "content": "more"},
            {"role": "assistant", "content": "two"},
            {"role": "assistant", "content": "three"},
            {"role": "assistant", "content": "four"},
            {"role": "assistant", "content": "five"},
        ]
        assert (
            check_reasoning_content(
                messages,
                "slow",
                None,
                reasoning_check_mode="adaptive",
            )
            is True
        )

    def test_adaptive_default_threshold_fails_below_one_fifth_reasoning(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "one", "reasoning_content": "thinking"},
            {"role": "user", "content": "more"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "three"},
            {"role": "assistant", "content": "four"},
            {"role": "assistant", "content": "five"},
            {"role": "assistant", "content": "six"},
        ]
        assert (
            check_reasoning_content(
                messages,
                "slow",
                None,
                reasoning_check_mode="adaptive",
            )
            is False
        )

    def test_adaptive_custom_threshold(self):
        messages = [
            {"role": "assistant", "content": "one", "reasoning_content": "thinking"},
            {"role": "assistant", "content": "two", "reasoning_content": "thinking"},
            {"role": "assistant", "content": "three"},
        ]
        assert (
            check_reasoning_content(
                messages,
                "slow",
                None,
                reasoning_check_mode="adaptive",
                reasoning_content_ratio_threshold=0.75,
            )
            is False
        )

        messages[2]["reasoning_content"] = "thinking"
        assert (
            check_reasoning_content(
                messages,
                "slow",
                None,
                reasoning_check_mode="adaptive",
                reasoning_content_ratio_threshold=0.75,
            )
            is True
        )

    def test_adaptive_respects_pseudo_turns(self):
        messages = [
            {"role": "assistant", "content": "pseudo"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "one", "reasoning_content": "thinking"},
            {"role": "user", "content": "more"},
            {"role": "assistant", "content": "two"},
        ]
        assert (
            check_reasoning_content(
                messages,
                "slow",
                2,
                reasoning_check_mode="adaptive",
            )
            is True
        )

    def test_adaptive_fast_mode_always_true(self):
        messages = [
            {"role": "assistant", "content": "one"},
            {"role": "assistant", "content": "two"},
        ]
        assert (
            check_reasoning_content(
                messages,
                "fast",
                None,
                reasoning_check_mode="adaptive",
                reasoning_content_ratio_threshold=1.0,
            )
            is True
        )
