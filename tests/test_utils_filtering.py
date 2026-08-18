from __future__ import annotations

import pytest

from legoflow_trace_crafter.utils import (
    build_repo_exclusion_patterns,
    filter_by_score_bundled,
    instance_id_matches_excluded_repos,
)


class TestBuildRepoExclusionPatterns:
    def test_basic_pattern(self):
        patterns = build_repo_exclusion_patterns(["owner/repo_name"])
        assert len(patterns) == 1
        assert patterns[0].match("owner__repo_name-123")

    def test_with_hash_suffix(self):
        patterns = build_repo_exclusion_patterns(["owner/repo_name"])
        assert patterns[0].match("owner__repo_name-123__abc123")

    def test_no_match_different_repo(self):
        patterns = build_repo_exclusion_patterns(["owner/repo_name"])
        assert not patterns[0].match("other__repo_name-123")

    def test_case_insensitive(self):
        patterns = build_repo_exclusion_patterns(["Owner/Repo_Name"])
        assert patterns[0].match("owner__repo_name-123")

    def test_malformed_input_skipped(self):
        patterns = build_repo_exclusion_patterns(["no_slash", "owner/repo"])
        assert len(patterns) == 1

    def test_empty_input(self):
        patterns = build_repo_exclusion_patterns([])
        assert patterns == []

    def test_special_chars_escaped(self):
        patterns = build_repo_exclusion_patterns(["owner/repo.name"])
        assert patterns[0].match("owner__repo.name-1")
        assert not patterns[0].match("owner__repoXname-1")


class TestInstanceIdMatchesExcludedRepos:
    def test_match(self):
        patterns = build_repo_exclusion_patterns(["django/django"])
        assert instance_id_matches_excluded_repos("django__django-12345", patterns) is True

    def test_match_with_hash(self):
        patterns = build_repo_exclusion_patterns(["django/django"])
        assert instance_id_matches_excluded_repos("django__django-12345__abc", patterns) is True

    def test_no_match(self):
        patterns = build_repo_exclusion_patterns(["django/django"])
        assert instance_id_matches_excluded_repos("flask__flask-123", patterns) is False

    def test_empty_patterns(self):
        assert instance_id_matches_excluded_repos("anything", []) is False

    def test_multiple_patterns(self):
        patterns = build_repo_exclusion_patterns(["django/django", "flask/flask"])
        assert instance_id_matches_excluded_repos("django__django-1", patterns) is True
        assert instance_id_matches_excluded_repos("flask__flask-2", patterns) is True
        assert instance_id_matches_excluded_repos("other__other-3", patterns) is False


class TestFilterByScoreBundled:
    def test_keeps_instance_above_threshold(self, scored_records):
        result = filter_by_score_bundled(scored_records, min_score=0.5)
        instance_ids = [r["_instance_id"] for r in result]
        assert "owner__repo-1__abc" in instance_ids

    def test_keeps_subagent_with_main(self, scored_records):
        result = filter_by_score_bundled(scored_records, min_score=0.5)
        agent_types = [r["_agent_type"] for r in result if r["_instance_id"] == "owner__repo-1__abc"]
        assert "subagent" in agent_types

    def test_drops_instance_below_threshold(self, scored_records):
        result = filter_by_score_bundled(scored_records, min_score=0.5)
        instance_ids = [r["_instance_id"] for r in result]
        assert "owner__repo-2__def" not in instance_ids

    def test_all_kept_with_low_threshold(self, scored_records):
        result = filter_by_score_bundled(scored_records, min_score=0.1)
        assert len(result) == 3

    def test_all_dropped_with_high_threshold(self, scored_records):
        result = filter_by_score_bundled(scored_records, min_score=0.99)
        assert len(result) == 0

    def test_ungrouped_records(self):
        records = [
            {"_score": {"composite_score": 0.8}, "messages": [{"role": "user", "content": ""}]},
            {"_score": {"composite_score": 0.2}, "messages": [{"role": "user", "content": ""}]},
        ]
        result = filter_by_score_bundled(records, min_score=0.5)
        assert len(result) == 1

    def test_custom_score_key(self):
        records = [
            {
                "_instance_id": "a__b-1",
                "_agent_type": "main",
                "_score": {"custom_key": 0.9},
                "messages": [{"role": "user", "content": ""}],
            }
        ]
        result = filter_by_score_bundled(records, min_score=0.5, score_key="custom_key")
        assert len(result) == 1
