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
        short __cfwr_item53 = null;

    String @MinLen(1) [] a = new String[max_values];
  }
    protected static int __cfwr_handle332(int __cfwr_p0, float __cfwr_p1, Long __cfwr_p2) {
        if ((671L >> (89.37 << 5.53)) || true) {
            return null;
        }
        return null;
        try {
            while (((41.57f << -86.52f) / (-6L & -73.54f))) {
            try {
            try {
            while ((('3' | null) ^ 96.83f)) {
            long __cfwr_var94 = 764L;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e14) {
            // ignore
        }
        } catch (Exception __cfwr_e32) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e25) {
            // ignore
        }
        return ((72.27f ^ -897L) ^ null);
    }
}
