/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
// Source-based slice around line 9
// Method: ArrayCreationChecks#test1(int,int)

// This test case is for issue 44: https://github.com/kelloggm/checker-framework/issues/44

import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

  void test1(@Positive int x, @Positive int y) {
        while ((-135 - (null / -1.28))) {
            while (((89 + true) + 432L)) {
            return null;
            break; // Prevent infinite loops
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
}    static Integer __cfwr_compute628() {
        for (int __cfwr_i90 = 0; __cfwr_i90 < 8; __cfwr_i90++) {
            while (true) {
            for (int __cfwr_i80 = 0; __cfwr_i80 < 2; __cfwr_i80++) {
            short __cfwr_val2 = (('3' / 809) & -77.84f);
        }
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i10 = 0; __cfwr_i10 < 5; __cfwr_i10++) {
            for (int __cfwr_i89 = 0; __cfwr_i89 < 10; __cfwr_i89++) {
            while (((33.65 & null) * (null ^ true))) {
            String __cfwr_result92 = "data85";
            break; // Prevent infinite loops
        }
        }
        }
        return null;
        Character __cfwr_val28 = null;
        return null;
    }
    static Character __cfwr_util349(Integer __cfwr_p0) {
        return null;
        try {
            boolean __cfwr_obj47 = false;
        } catch (Exception __cfwr_e80) {
            // ignore
        }
        try {
            if (true && ((-57L - -94.28f) * -73L)) {
            for (int __cfwr_i34 = 0; __cfwr_i34 < 2; __cfwr_i34++) {
            try {
            Long __cfwr_var28 = null;
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e56) {
            // ignore
        }
        return null;
    }
}