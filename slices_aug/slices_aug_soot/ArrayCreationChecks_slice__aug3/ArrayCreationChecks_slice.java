/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
// Source-based slice around line 9
// Method: ArrayCreationChecks#test1(int,int)

// This test case is for issue 44: https://github.com/kelloggm/checker-framework/issues/44

import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

  void test1(@Positive int x, @Positive int y) {
        while (false) {
            for (int __cfwr_i47 = 0; __cfwr_i47 < 10; __cfwr_i47++) {
            while ((21.38f | 13.38f)) {
            while ((93.79 + true)
        boolean __cfwr_data38 = false;
) {
            while ((('l' % 'D') & (true ^ null))) {
            try {
            while ((null % null)) {
            if (false || (null | true)) {
            try {
            return -499L;
        } catch (Exception __cfwr_e90) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e72) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
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
}    public char __cfwr_process620() {
        while (false) {
            while (((733L - 'p') & (-978L << -367L))) {
            if (true && ((-270L & false) >> null)) {
            for (int __cfwr_i97 = 0; __cfwr_i97 < 7; __cfwr_i97++) {
            return null;
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return 79.30f;
        try {
            boolean __cfwr_val76 = false;
        } catch (Exception __cfwr_e76) {
            // ignore
        }
        Object __cfwr_item20 = null;
        return (30.84f - -67.79f);
    }
}