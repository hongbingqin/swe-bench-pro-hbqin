
Your task is to **review the quality** of a software engineering task specification, its associated tests, and the reference code patch, then provide a **revised task description** to improve clarity. You are allowed to use the following tools:


# Input Context

You are provided with the following inputs:

You should be given two directories: you start with a directory that contains the software engineering annotated task for another codebase. The task directory has the following structure:

* instruction.md — The original task specification (the prompt the evaluated model sees)
* solution/solve.sh — Reference solution (typically a git patch)
* tests/config.json — Test metadata: instance_id (MSL form: `instance_{owner}__{repo}-{40-hex base_commit}`), repo, base_commit (starting point), patch, test_patch, FAIL_TO_PASS, PASS_TO_PASS, selected_test_files_to_run, before_repo_set_cmd
* tests/test.sh — Test entrypoint
* tests/run_script.sh — Test runner
* environment/Dockerfile — Docker sandbox setup
* task.toml — Task metadata

1. **Original task specification**: A task to complete in the codebase.

2. **Reference code patch**: A developer-created patch that solves the task. This patch is for reference only — there could be multiple valid solutions, and the patch itself may have quality issues (e.g., incomplete coverage of the spec, excess scope, or poor engineering practices). You will evaluate its quality as part of your review. Location: `patch` field in `tests/config.json`.

3. **Reference test patch**: A test patch used to verify the correctness of any solution. Location: `test_patch` field in `tests/config.json`. Also `tests/run_script.sh` contains the script to run the test against the reference solution.

**Important:** note that the task is created for another codebase, the engineer is supposed to tell you the location of the codebase. If you don't have it, ask.

Once you have it, git checkout to the base_commit in `config.json`. That is the initial state of the repository upon which the task-to-be-reviewed is supposed to be performed.

# Task Objective

Your objective is twofold:

1. **Quality Review**: Evaluate the quality of the original task specification, the provided tests, and the reference code patch, identifying any ambiguities, misalignments, or other issues.

2. **Task Revision**: Create a revised, succinct task specification with improved clarity. The revised specification should enable any expert developer with a solid understanding of the codebase to produce a solution that passes all tests in the reference test patch.

**Important**: The revised specification will be used as an interview question. It must remain clear, concise, and free of hints or privileged information about the solution.

# Task Workflow

## Stage 1: Exploration and Analysis

Explore the codebase and carefully analyze the original task specification, the tests, and the reference code patch. Browse through all relevant source files to collect as much context as possible.

## Stage 2: Test Verification

Execute the test suites to understand the expected functionality captured by the tests.

1. **Apply the test patch** and run the tests using the provided test command. Observe the results—they should FAIL since the codebase is in its pre-solution state. Pay close attention to the error messages to understand the task requirements.

2. **Apply the reference solution**  and run the tests again. All tests should PASS.

3. **Create additional verification scripts** as needed (e.g., reproduction scripts for bug reports) to better understand the task and codebase.

4. **Use printf debugging or testing scripts** whenever you have doubts and deepen your understanding about the task and the codebase.

## Stage 3: Initial Quality Assessment

After gaining a thorough understanding, summarize any clarity issues in the task specification, any quality issues with the tests, and any quality concerns with the reference code patch. Remember the following criteria.

### Criteria for a Good Task Specification

!!CRITICAL, MUST FOLLOW!!

- **No Hints**: The specification must NOT contain hints or privileged information about how to solve the task. It will be used as an interview question.

- **Sufficient Detail**: Include details that cannot be inferred from the codebase but are required to pass the tests (e.g., specific function signatures, expected error message wording, particular behaviors).

- **No Overfitting**: Do not tailor the specification to the reference solution in `config.json`. Focus on creating a general specification that allows for multiple valid solutions.

- **Brevity**: Avoid including background knowledge or context that a veteran developer would already know from the codebase.

- **Ultimate Criterion**: A professional developer with deep knowledge of the codebase should be able to produce a solution that passes all tests based solely on your revised specification without asking any clarifying questions.

### Criteria for a Good Test Patch

- The tests should cover representative scenarios of the task, and may cover common edge cases.

- The tests should not be so loose that they accept incorrect solutions.

- On the other hand, the tests should not reject some correct solutions due to overly strict expectations. This often occurs when the task specification is under-specified and the tests expect specific implementation details (e.g., particular function names, exact output formats or error messages) that cannot be inferred from the codebase context by reading relevant source files.

### Criteria for a Good Code Patch

- The patch should correctly and completely solve the task as described in the original task specification.

- The patch should be minimal — it should only contain changes necessary to address the task. It should not include unrelated refactoring, pre-existing bug fixes, or features not requested in the spec.

- The patch should follow existing codebase conventions, patterns, and idioms. The solution approach should be one that a skilled developer familiar with the codebase would reasonably choose.

- The patch should exhibit good software engineering practices — no hardcoded values where configuration is expected, no hacky workarounds where proper fixes are feasible, and no unnecessary complexity.

### Generate a Brief Task Specification Quality Summary

Based on your initial analysis, use the file creation tool to create a "task_spec_quality_analysis.md" file to summarize any missing information in the original task description that is necessary to create a test-passing solution but cannot be inferred from the codebase context. For each missing information, label whether it can be inferred from the codebase or not by browsing related source files. Here's an example format:

#### Missing Information

| # | Description | Inferable from Codebase? | Explanation |
|---|-------------|--------------------------|-------------|
| 1 | {Brief description of missing information} | Yes / No / Not Sure | {Brief explanation if you are able to find the missing information by researching the codebase} |
| 2 | ... | ... | ... |

**Note**: If a piece of missing information can be inferred from the codebase, it is a good thing! Remember that we want to use the task as a challenging interview question so we actually prefer hiding information from the task specification if that piece of information can be obtained by performing deep research of the codebase.

If you answered "Not Sure" for any of the items above, proceed to stage 4 and continue exploring the codebase to revise your analysis. Otherwise, proceed to Stage 5.

# Stage 4 (Optional): Perform Additional Research

If you answered "Not Sure" to any of the items above, perform additional exploration and research on the codebase. Before doing so, remember to revert the reference code patch so you could focus on exploring the pre-solution codebase state and verify if the missing information from the given task specification can be inferred from the codebase context or not.

# Stage 5: Submission

In your last turn, output a summary message with the following two parts.

## Part 1: Quality Assessment

Based on your analysis above, provide quality ratings for each category. Select **one option** for each.

### 1. Clarity of Task Specification

Evaluate how well the original task specification aligns with the test patch.

> **Spec Clarity Rating:** { Good | Requires Some Guessing | Under Specified | Over Specified | Other Quality Issues }
>
> **Explanation:** {Brief explanation, required for all ratings}

**Rating Options:**

- **Good**: The specification is succinct, clear, complete, and fully aligned with the test patch. A developer with deep context knowledge of the codebase can follow the specification and produce a solution that passes all tests by gathering any missing information by proactively researching the codebase. **VERY IMPORTANT:** Remember the "Criteria for a Good Task Specification". The ultimate criterion for evaluating a task specification is: if we present it to a veteran developer of the repository, can the developer solve the task by creating a code change that passes the tests? Note, it is preferrable that the task specification is deliberately made short and does not cover contextual information that can be inferred from the codebase, as the task will be used as an interview question and this is a great way to assess the understanding of the repository of any candidates.

- **Requires Some Guessing** The task specification omits some non-critical details or "sane" defaults, yet a professional developer with deep codebase knowledge can still produce a test-passing solution without asking for any clarifying questions. For instance, the specification may permit multiple valid approaches while the tests expect the most common one. An experienced developer would reasonably infer and implement this common approach.

- **Under Specified**: The specification lacks necessary details which makes it hard to create test-passing solutions (e.g., missing expected function signatures or arguments, specific error message wording, or edge case handling that cannot be inferred from relevant changes in the commit history or the codebase context). **Important**: Cross-reference your missing information analysis above. If every item you identified is inferable from the codebase (all marked "Yes"), then the specification is NOT under-specified — rate it "Good" or "Requires Some Guessing" instead. A specification is only under-specified when it omits details that a developer cannot recover through codebase research alone.

- **Over Specified**: The specification includes requirements that are not captured or validated by the tests. A developer might implement extra functionality that is not covered by any tests.

- **Other Quality Issues**: The specification has other problems not covered above. (Requires brief explanation)

### 2. Quality of Tests

Evaluate the alignment between the tests (test list and test patch) and the task specification.

> **Test Quality Rating:** { Good | Too Lenient | Too Restrictive | Other Quality Issues }
>
> **Explanation:** {Brief explanation, required for all ratings}

**Rating Options:**

- **Good**: The tests are appropriately designed—they accurately verify the task requirements without being too lenient or too strict.

- **Too Lenient**: The tests are too loose and may accept incorrect solutions (false positives). Important edge cases or requirements may not be validated.

- **Too Restrictive**: The tests reject some correct solutions due to overly strict expectations. This often occurs when the task specification is under-specified and the tests expect specific implementation details (e.g., particular function names, exact output formats, or specific internal behaviors) that are not mentioned in the specification and cannot be inferred from the codebase.

- **Other Quality Issues**: The tests have other problems not covered above. (Requires brief explanation, e.g., tests are irrelevant to the task, tests contain bugs, tests have flaky behavior)

### 3. Quality of Code Patch

Evaluate the quality of the reference code patch as a benchmark solution.

> **Code Patch Quality Rating:** { Good | Functionally Correct but Poor Engineering | Incomplete Solution | Excess Scope | Other Major Quality Issues }
>
> **Explanation:** {Brief explanation, required for all ratings}

**Rating Options:**

- **Good**: The patch correctly solves the task and follows codebase conventions. Minor additional cosmetic changes (e.g., import style adjustments, small cleanup of nearby code) are acceptable and should still be rated Good. The key criterion is that the patch addresses the core task correctly.

- **Functionally Correct but Poor Engineering**: The patch passes the tests and solves the task, but exhibits significantly poor software engineering practices. Examples include: overly complex solutions where much simpler ones exist, hacky workarounds instead of proper fixes, hardcoded values, fundamentally violating codebase conventions, or code that would be rejected in a code review for engineering quality reasons. Minor style deviations do not qualify — this rating is for substantial engineering problems.

- **Incomplete Solution**: The patch fails to address a significant portion of the original task specification. For example, it solves only one of several explicitly requested features, completely ignores a major requirement described in the spec, or leaves the core issue partially unresolved. Minor gaps or missing edge cases that are not explicitly described in the spec should not trigger this rating.

- **Excess Scope**: The patch includes substantial code changes that go well beyond the task specification. For example, refactoring entire modules, fixing multiple unrelated bugs, or adding significant new features not requested. Minor cosmetic changes like import reorganization, small cleanup of touched code, or fixing trivially related issues nearby should not trigger this rating.

- **Other Major Quality Issues**: The patch has other significant problems not covered by the categories above.

---

## Part 2: Revised Task Specification

Provide **three versions** of the revised task specification with varying levels of detail. All versions must:

- Contain sufficient information for a professional developer to create a test-passing solution while remaining as **succinct** as possible
- NOT disclose ground-truth test scenarios or privileged information, or contextual information that CAN be inferred by researching the codebase.
- Use imperative language (e.g., "Fix...", "Implement...", "Add support for...") and ideally in a conversational style as if you are asking a colleague to solve the task.
- Do not include any fenced code blocks as they tend to disclose too much information and have a high chance of making the task easier.

### Succinct Version

> **Revised Task (Succinct):**
>
> {Your answer here. Remember: this is an interview question, so never disclose stepwise hints, test scenarios, root cause, or other types of hints that would make the task easier. Focus on describing the task instead of how the task should be solved.}

### Regular Version

> **Revised Task (Regular):**
>
> {Your answer here. Remember: this is an interview question, so never disclose stepwise hints, test scenarios, root cause, or other types of hints that would make the task easier. Focus on describing the task instead of how the task should be solved.}

### Verbose Version

> **Revised Task (Verbose):**
>
> {Your answer here. Remember: this is an interview question, so never disclose stepwise hints, test scenarios, root cause, or other types of hints that would make the task easier. Focus on describing the task instead of how the task should be solved.}
