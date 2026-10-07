# Preliminary Huginn Benchmarking Plan

**Prepared by:** Ray  
**Date:** October 7, 2026  
**Status:** Draft for team discussion

This plan supports the mentor's action item for Nate and Ray: produce and pickle results at recurrent depths N = 16, 8, and 4, and periodically push work to GitHub. It covers the preliminary benchmarking stage of the broader research project on reasoning in recurrent transformer architectures.

## Research Question

How does recurrent depth (N = 4, 8, 16, 32) affect Huginn 3.5B's answer accuracy and response time on a fixed set of GSM8K problems?

The broader project investigates reasoning in the model's internal states. This preliminary comparison can assess performance changes with depth, but cannot by itself establish that the model uses internal chain-of-thought reasoning. Activation patching and early stopping are later stages of the project.

## Hypothesis

As recurrent depth increases from N = 4 to N = 32, I expect answer accuracy and response time to increase. Additional recurrent steps may help the model refine its internal representation of the problem, improving accuracy while requiring more computation.

This prediction was stated after learning about the initial N = 32 run but before reviewing the repository's additional results. It is an expectation to test, not an established finding.

## Data

The existing `Huggin.ipynb` notebook loads the `main` configuration of `openai/gsm8k`, selects the test split, and uses its first 20 problems (indices 0–19). Reference answers come from the dataset's answer field, using the final answer after the `####` delimiter.

The proposed comparison will use the same 20 problems at every recurrent depth.

**To confirm with the team:**

- Whether the additional saved runs used the same problem selection and experimental settings.
- Which existing results belong to which runs and which runs Ray still needs to complete.
- Whether those results have already been saved as pickle files and where they are stored.

## Metrics

At each recurrent depth, we will measure answer accuracy as a percentage and record response time in seconds for each problem. We will summarize response times using the mean and median.

- **Accuracy:** Number of correctly answered problems divided by the total number evaluated, multiplied by 100.
- **Mean response time:** Sum of individual response times divided by the number of timed responses.
- **Median response time:** Middle response time after sorting the times; for 20 responses, the average of the two middle times.

**To review with the team:** The answer-checking procedure and the exact start and end points for timing.

The current notebook extracts the last number in a response and compares it numerically with the reference answer. One saved response correctly states “7 dozens of eggs in 4 weeks,” but the checker extracts 4 and marks it wrong. This means the reported accuracy requires scoring review.

The repository's results file includes timings for N = 16, 8, and 4. The reviewed N = 32 notebook does not time individual model responses. We need to confirm how the saved timings were measured before making direct comparisons.

## Decisions

These are proposed choices for this preliminary experiment, pending team discussion.

- **October 7, 2026:** Use 20 GSM8K problems to keep the preliminary experiment manageable.
- **October 7, 2026:** Use the same 20 problems at every recurrent depth to help assess whether additional loops improve accuracy and how they affect response time, while keeping the problem selection fixed.

## Mistakes to Avoid

Proposed checks to discuss with my mentor and team:

- **Trusting automatic scores without checking them:** Manually inspect some model responses and compare them with the reference answers to verify that the scoring code marks them correctly.
- **Assuming results were saved successfully:** Reopen the saved pickle file and confirm that it contains the expected results before ending the notebook session.

---

**Reference materials:** Mentor's action item and research summary; the October 4 Dataset Design & AI Research Workflows lecture-chat notes; the team's [notebook](https://github.com/aaronzr/aaron-nnrr/blob/main/Huggin.ipynb) and [saved results](https://github.com/aaronzr/aaron-nnrr/blob/main/data.txt), reviewed October 7, 2026.
