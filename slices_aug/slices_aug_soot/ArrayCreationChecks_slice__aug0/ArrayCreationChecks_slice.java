/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
// Source-based slice around line 9
// Method: ArrayCreationChecks#test1(int,int)

// This test case is for issue 44: https://github.com/kelloggm/checker-framework/issues/44

import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

  void test1(@Positive int x, @Positive int y) {
        if (true || (-186 >> 6.49f)) {
            try {
            char __cfwr_elem97 = 'J';
        } catch (Exception __cfwr_e50) {
            // ignore
        }
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
}    protected Long __cfwr_handle542(short __cfwr_p0, String __cfwr_p1) {
        try {
            while (((-85.02f / null) - 485L)) {
            while (true) {
            try {
            return (null / false);
        } catch (Exception __cfwr_e47) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e58) {
            // ignore
        }
        return -86.46f;
        return null;
    }
}