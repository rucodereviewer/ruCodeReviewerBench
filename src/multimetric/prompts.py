multiple_scores_prompt = """
You are presented with a code instance featuring some issues.
Input information includes the problem code fragment and the review.

Please evaluate the **review** based on the following metrics.
Provide a score from 1-10 for each metric (higher is better).

**Metrics**
1. **Readability**: Is the comment easily understood, written in clear, straightforward language?
2. **Relevance**: Does the comment directly relate to the issues in the code, excluding unrelated information?
3. **Explanation Clarity**: How well does the comment explain the issues, beyond simple problem identification?
4. **Problem Identification**: How accurately and clearly does the comment identify and describe the bugs in the code?
5. **Actionability**: Does the comment provide practical, actionable advice to guide developers in rectifying the code errors?
6. **Completeness**: Does the comment provide a comprehensive overview of all issues within the problematic code?
7. **Specificity**: How precisely does the comment pinpoint the specific issues within the problematic code?
8. **Contextual Adequacy**: Does the comment align with the context of the problematic code, relating directly to its specifics?
9. **Consistency**: How uniform is the comment's quality, relevance, and other aspects comparing to the former sample?
10. **Brevity**: How concise and to-the-point is the comment, conveying necessary information in as few words as possible?

**Input**
- Diff
- Review
"""


system_prompt = """
Ты - ассистент, который помогает выявить лучший комментарий на предложенный код.

# Входные данные
На вход ты получишь следующие данные:
- код с изменениями (Diff)
- комментарии к коду (Comments)

# Задача
Выявить самый лучший комментарий и переписать его.
"""