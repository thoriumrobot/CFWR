/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
// Source-based slice around line 9
// Method: ArrayCreationChecks#test1(int,int)

// This test case is for issue 44: https://github.com/kelloggm/checker-framework/issues/44

import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

  void test1(@Positive int x, @Positive int y) {
        for (int __cfwr_i26 = 0; __cfwr_i26 < 4; __cfwr_i26++) {
            Character __cfwr_elem6 = null;
        }

    int[] newArray = new int[x + y];
    @IndexF
        while ((-52.72 ^ (723 + null))) {
            long __cfwr_temp83 = (-438L ^ null);
            break; // Prevent infinite loops
        }
or("newArray") int i = x;
    @IndexFor("newArray") int j = y;
  }

  void test2(@NonNegative int x, @Positive int y) {
    int[] newArray = new int[x + y];
    @IndexFor("newArray") int i = x;
    @IndexOrHigh("newArray") int j = y;
  }

  void test3(@NonNegative int x, @NonNegative int y) {
}    private static Float __cfwr_calc912(Boolean __cfwr_p0, char __cfwr_p1) {
        for (int __cfwr_i2 = 0; __cfwr_i2 < 3; __cfwr_i2++) {
            return ('M' & null);
        }
        if (false && ((-717L & 45.55f) + -46.48)) {
            while (('Z' - false)) {
            if (((false >> -10.77) / null) || true) {
            try {
            while (false) {
            for (int __cfwr_i81 = 0; __cfwr_i81 < 2; __cfwr_i81++) {
            while ((('l' << null) + null)) {
            if (true && false) {
            if (true && true) {
            return null;
        }
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e73) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        return null;
    }
    public static Double __cfwr_proc617(byte __cfwr_p0) {
        try {
            for (int __cfwr_i92 = 0; __cfwr_i92 < 10; __cfwr_i92++) {
            return 29.25;
        }
        } catch (Exception __cfwr_e88) {
            // ignore
        }
        return null;
    }
}