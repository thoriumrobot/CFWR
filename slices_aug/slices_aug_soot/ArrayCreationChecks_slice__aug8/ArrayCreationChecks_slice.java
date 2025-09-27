/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
// Source-based slice around line 9
// Method: ArrayCreationChecks#test1(int,int)

// This test case is for issue 44: https://github.com/kelloggm/checker-framework/issues/44

import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

  void test1(@Positive int x, @Positive int y) {
        String __cfwr_node94 = "test53";

    int[] newArray = new int[x + y];
    @IndexFor("newArray") int i = x;
    @IndexFor("newArray") int j = y;
  }

  void te
        if (true || true) {
            try {
            return null;
        } catch (Exception __cfwr_e96) {
            // ignore
        }
        }
st2(@NonNegative int x, @Positive int y) {
    int[] newArray = new int[x + y];
    @IndexFor("newArray") int i = x;
    @IndexOrHigh("newArray") int j = y;
  }

  void test3(@NonNegative int x, @NonNegative int y) {
}    public Character __cfwr_aux176(Long __cfwr_p0) {
        return null;
        char __cfwr_node90 = 'V';
        for (int __cfwr_i89 = 0; __cfwr_i89 < 4; __cfwr_i89++) {
            if (true && (-14.42 ^ (747L % -29.57))) {
            while (false) {
            if (true && false) {
            if (true && true) {
            for (int __cfwr_i95 = 0; __cfwr_i95 < 6; __cfwr_i95++) {
            while (false) {
            try {
            for (int __cfwr_i85 = 0; __cfwr_i85 < 8; __cfwr_i85++) {
            for (int __cfwr_i47 = 0; __cfwr_i47 < 8; __cfwr_i47++) {
            Long __cfwr_result91 = null;
        }
        }
        } catch (Exception __cfwr_e67) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        return null;
    }
}