/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class BasicSubsequence {
  // :: error: (not.final)
  @HasSubsequence(subsequence = "this", from = "this.x", to = "this.y")
  int[] b;

  int x;
  int y;

  void test2(@NonNegative @LessThan("y + 1") int x1, int[] a) {
        Double __cfwr_obj82 = null;

    x = x1;
    // :: error: (to.not.ltel)
    b = a;
  }

  void test3(@NonNegative @LessThan("y") int x1, int[] a) {
    x = x1;
    // :: error: (to.not.ltel)
    b = a;
      static Float __cfwr_handle746(short __cfwr_p0, double __cfwr_p1) {
        while (false) {
            for (int __cfwr_i84 = 0; __cfwr_i84 < 9; __cfwr_i84++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i67 = 0; __cfwr_i67 < 7; __cfwr_i67++) {
            return true;
        }
        try {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e63) {
            // ignore
        }
        return null;
    }
    public static Object __cfwr_func569(boolean __cfwr_p0, Boolean __cfwr_p1) {
        short __cfwr_val84 = null;
        try {
            while (true) {
            Long __cfwr_temp16 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e25) {
            // ignore
        }
        for (int __cfwr_i13 = 0; __cfwr_i13 < 3; __cfwr_i13++) {
            if (true || (-2.95f / null)) {
            for (int __cfwr_i94 = 0; __cfwr_i94 < 8; __cfwr_i94++) {
            for (int __cfwr_i56 = 0; __cfwr_i56 < 8; __cfwr_i56++) {
            if (false && (78.92 / (null ^ 977L))) {
            try {
            Object __cfwr_result17 = null;
        } catch (Exception __cfwr_e16) {
            // ignore
        }
        }
        }
        }
        }
        }
        while (true) {
            while ((('H' % null) >> (73.79 << -59.92))) {
            try {
            for (int __cfwr_i82 = 0; __cfwr_i82 < 4; __cfwr_i82++) {
            if (true || ((655 - 35.03f) % 286)) {
            return null;
        }
        }
        } catch (Exception __cfwr_e27) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}

  void test4(@NonNegative int x1, int[] a) {
