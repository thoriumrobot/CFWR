/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
// Source-based slice around line 9
// Method: ArrayConstructionPositiveLength#makeArray(int)

// Test case for issue #66:
// https://github.com/kelloggm/checker-framework/issues/66

import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class ArrayConstructionPositiveLength {

  public void makeArray(@Positive int max_values) {
        while (true) {
            while (false) {
            return false;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }

    String @MinLen(1) [] a = new String[max_values];
  }
    public static Object __cfwr_handle825() {
        for (int __cfwr_i69 = 0; __cfwr_i69 < 7; __cfwr_i69++) {
            return null;
        }
        return null;
    }
}
