#!/bin/bash
set -euo pipefail

cd /app 2>/dev/null || cd /testbed 2>/dev/null

cat > solution_patch.diff << '__SOLUTION__'
diff --git a/src/partial_json_parser/core/myelin.py b/src/partial_json_parser/core/myelin.py
index 963867c..69bcd82 100644
--- a/src/partial_json_parser/core/myelin.py
+++ b/src/partial_json_parser/core/myelin.py
@@ -120,7 +120,7 @@ def fix_fast(json_string: str, allow_partial: Union[Allow, int] = ALL):
             # { "k
             return json_string[: stack[-1][0] + 1], join_closing_tokens(stack)

-        last_comma = json_string.rfind(",", max(stack[-1][0], last_string_end) + 1, last_string_start - 1)
+        last_comma = json_string.rfind(",", max(stack[-1][0], last_string_end) + 1, last_string_start)
         if last_comma != -1:
             # { "key": "v", "k
             # { "key": 123, "k
__SOLUTION__

git apply --verbose solution_patch.diff || patch --fuzz=5 -p1 -i solution_patch.diff
