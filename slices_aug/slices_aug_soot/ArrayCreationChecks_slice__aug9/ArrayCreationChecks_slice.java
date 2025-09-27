/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
// Source-based slice around line 9
// Method: ArrayCreationChecks#test1(int,int)

// This test case is for issue 44: https://github.com/kelloggm/checker-framework/issues/44

import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

  void test1(@Positive int x, @Positive int y) {
        Float __cfwr_temp85 = null;

    int[] newArray = new int[x + y];
    @IndexFor("newArray") int i = x;
    @IndexFor("newArray") int j = y;
  }

  void test2(@
        String __cfwr_item45 = "test12";
NonNegative int x, @Positive int y) {
    int[] newArray = new int[x + y];
    @IndexFor("newArray") int i = x;
    @IndexOrHigh("newArray") int j = y;
  }

  void test3(@NonNegative int x, @NonNegative int y) {
}    public float __cfwr_proc899(char __cfwr_p0, String __cfwr_p1, Long __cfwr_p2) {
        while (false) {
            for (int __cfwr_i25 = 0; __cfwr_i25 < 1; __cfwr_i25++) {
            try {
            try {
            return null;
        } catch (Exception __cfwr_e16) {
            // ignore
        }
        } catch (Exception __cfwr_e56) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        while (false) {
            for (int __cfwr_i15 = 0; __cfwr_i15 < 1; __cfwr_i15++) {
            while (true) {
            while (false) {
            try {
            try {
            return 's';
        } catch (Exception __cfwr_e9) {
            // ignore
        }
        } catch (Exception __cfwr_e38) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        Boolean __cfwr_val28 = null;
        return -89.36f;
    }
}