/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
// Source-based slice around line 9
// Method: ArrayCreationChecks#test1(int,int)

// This test case is for issue 44: https://github.com/kelloggm/checker-framework/issues/44

import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

  void test1(@Positive int x, @Positive int y) {
        while (('X' / 917)) {
            for (int __cfwr_i56 = 0; __cfwr_i56 < 8; __cfwr_i56++) {
            if (((-171L * 74.76) + false) && false) {
            if (true || false) {
            if ((null | null) || false) {
            byte __cfwr_data93 = null;
        }
        }
        }
        }
            break; // Prevent infinite loops
        }

    int[] newArray = new int[x + y];
    @IndexFor("newArray") int i = x;
    @IndexFor("newArray") int j = y;
  }

  void test2(@NonNegative int x, @Positive int y) {
    int[] newArray = new int[x + y];
    @IndexFor("newArray") int i = x;
    @IndexOrHigh("newArray") int j = y;
  }

  void test3(@NonNegative int x, @NonNegative int y) {
}    public Boolean __cfwr_util358(Boolean __cfwr_p0, Boolean __cfwr_p1, float __cfwr_p2) {
        if (false || (237L & 'S')) {
            return null;
        }
        return null;
    }
}