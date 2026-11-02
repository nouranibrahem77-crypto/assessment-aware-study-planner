"""Fixed split and OULAD feature schema definitions."""

from __future__ import annotations

SPLIT_PRESENTATIONS: dict[str, tuple[str, ...]] = {
    "train": ("2013B", "2013J"),
    "validation": ("2014B",),
    "test": ("2014J",),
}

IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "code_module",
    "code_presentation",
    "id_student",
    "id_assessment",
)
TARGET_COLUMN = "target_score"

# Grouping is based on the actual FFF activity_type values in vle.csv.
# Content/navigation: learning materials and their navigation containers.
# Assessment: quizzes and questionnaires.
# Collaboration: discussion, wiki, and synchronous collaboration tools.
ACTIVITY_GROUPS: dict[str, tuple[str, ...]] = {
    "content_navigation": (
        "dataplus",
        "dualpane",
        "folder",
        "glossary",
        "homepage",
        "htmlactivity",
        "oucontent",
        "page",
        "repeatactivity",
        "resource",
        "subpage",
        "url",
    ),
    "assessment_activity": ("questionnaire", "quiz"),
    "collaboration": ("forumng", "oucollaborate", "ouelluminate", "ouwiki"),
}

ACTIVITY_TYPE_TO_GROUP: dict[str, str] = {
    activity_type: group
    for group, activity_types in ACTIVITY_GROUPS.items()
    for activity_type in activity_types
}

# For deadline D, these are half-open day-offset windows, from oldest to newest.
CLICK_WINDOW_OFFSETS: tuple[tuple[int, int], ...] = (
    (-28, -21),
    (-21, -14),
    (-14, -7),
    (-7, 0),
)


def click_feature_name(group: str, week: int) -> str:
    """Return the stable name for a group's 7-day click-count feature."""
    return f"click_{group}_week_{week}"
