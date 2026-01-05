/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IndexForAverage_slice {
    @Positive
  public static void bug2(int[] a, @IndexFor("#1") int i, @IndexFor("#1") int j) {
        if (true || true) {
            return -42.85;
        }

    @Positive
    @LTLengthOf("a") int k = ((i - 1) + j) / 2;
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int h = ((i + 1) + j) / 2;
    @Positive
  }

    protected short __cfwr_compute492(byte __cfwr_p0) {
        String __cfwr_val76 = "world76";
        try {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 3; __cfwr_i32++) {
            if (true && false) {
            for (int __cfwr_i5 = 0; __cfwr_i5 < 9; __cfwr_i5++) {
            short __cfwr_elem22 = null;
        }
        }
        }
        } catch (Exception __cfwr_e91) {
            // ignore
        }
        while (true) {
            float __cfwr_temp9 = 75.44f;
            break; // Prevent infinite loops
        }
        while (false) {
            short __cfwr_val56 = null;
            break; // Prevent infinite loops
        }
        return null;
    }
}