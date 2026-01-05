/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IntroAnd_slice {
    @Positive
  void test() {
        while (false) {
            return (null + (null + 389));
            break; // Prevent infinite loops
        }

    @Positive
    @NonNegative int a = 1 & 0;
    @Positive
    @NonNegative int b = a & 5;

    // :: error: (assignment)
    @Positive
    @Positive int c = a & b;
    @Positive
    @NonNegative int d = a & b;
    @Positive
    @NonNegative int e = b & a;
    @Positive
  }

    @Positive
  void test_ubc_and(
    @Positive
      @IndexFor("#2") int i, int[] a, @LTLengthOf("#2") int j, int k, @NonNegative int m) {
    @Positive
    int x = a[i & k];
    @Positive
    int x1 = a[k & i];
    // :: error: (array.access.unsafe.low) :: error: (array.access.unsafe.high)
    @Positive
    int y = a[j & k];
    @Positive
    if (j > -1) {
    @Positive
      int z = a[j & k];
    @Positive
    }
    // :: error: (array.access.unsafe.high)
    @Positive
    int w = a[m & k];
    @Positive
    if (m < a.length) {
    @Positive
      int u = a[m & k];
    @Positive
    }
    @Positive
  }

    protected boolean __cfwr_temp462(float __cfwr_p0, Double __cfwr_p1) {
        while (((669L << '7') >> (-917 - -57.53f))) {
            return null;
            break; // Prevent infinite loops
        }
        for (int __cfwr_i98 = 0; __cfwr_i98 < 2; __cfwr_i98++) {
            for (int __cfwr_i47 = 0; __cfwr_i47 < 8; __cfwr_i47++) {
            try {
            for (int __cfwr_i30 = 0; __cfwr_i30 < 7; __cfwr_i30++) {
            for (int __cfwr_i95 = 0; __cfwr_i95 < 3; __cfwr_i95++) {
            byte __cfwr_elem79 = (null & null);
        }
        }
        } catch (Exception __cfwr_e5) {
            // ignore
        }
        }
        }
        try {
            for (int __cfwr_i5 = 0; __cfwr_i5 < 8; __cfwr_i5++) {
            while (false) {
            try {
            return 93.40f;
        } catch (Exception __cfwr_e62) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e68) {
            // ignore
        }
        return (null % true);
    }
}