# =============================================================================
# tests/test_predict.py
# PURPOSE: Test that core functions work correctly
# Run with: python -m pytest tests/ -v
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import clean_text


def test_clean_text_lowercase():
    """Cleaning should convert everything to lowercase."""
    result = clean_text("HELLO WORLD")
    assert result == result.lower(), "Text should be lowercase"


def test_clean_text_removes_special_chars():
    """Cleaning should remove $, !, numbers, etc."""
    result = clean_text("Win $100 now!!!")
    assert '$' not in result
    assert '!' not in result
    assert '1' not in result


def test_clean_text_removes_urls():
    """Cleaning should remove URLs."""
    result = clean_text("Click here http://spam.com to win")
    assert 'http' not in result
    assert 'spam.com' not in result


def test_clean_text_removes_stopwords():
    """Common stopwords should be removed."""
    result = clean_text("the cat sat on the mat")
    words = result.split()
    assert 'the' not in words
    assert 'on' not in words


def test_clean_text_empty_input():
    """Empty or invalid input should return empty string."""
    assert clean_text("") == ""
    assert clean_text("   ") == ""


def test_clean_text_returns_string():
    """Output should always be a string."""
    result = clean_text("test message here")
    assert isinstance(result, str)


def test_clean_text_short_words_removed():
    """Words with 2 or fewer characters should be removed."""
    result = clean_text("go to a big house")
    words = result.split()
    assert all(len(w) > 2 for w in words), "Short words should be removed"


if __name__ == "__main__":
    # Run tests manually without pytest
    tests = [
        test_clean_text_lowercase,
        test_clean_text_removes_special_chars,
        test_clean_text_removes_urls,
        test_clean_text_removes_stopwords,
        test_clean_text_empty_input,
        test_clean_text_returns_string,
        test_clean_text_short_words_removed,
    ]

    print("Running tests...\n")
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  ✅ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {test.__name__}: {e}")
        except Exception as e:
            print(f"  💥 {test.__name__} ERROR: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
