/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IntroAnd_slice {
    @Positive
  void test() {
        return null;

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
      @IndexFor("#2") int i, int[] a, @LTLengthOf("#2") int j, int k,
        byte __cfwr_elem41 = (null ^ -97.53);
 @NonNegative int m) {
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

    protected static char __cfwr_util477(long __cfwr_p0, short __cfwr_p1, long __cfwr_p2) {
        for (int __cfwr_i33 = 0; __cfwr_i33 < 6; __cfwr_i33++) {
            return null;
        }
        for (int __cfwr_i48 = 0; __cfwr_i48 < 2; __cfwr_i48++) {
            if (true || ((false / -867) % 52.14)) {
            for (int __cfwr_i78 = 0; __cfwr_i78 < 10; __cfwr_i78++) {
            while (true) {
            for (int __cfwr_i45 = 0; __cfwr_i45 < 7; __cfwr_i45++) {
            try {
            return null;
        } catch (Exception __cfwr_e89) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        int __cfwr_temp37 = 834;
        return (true << 21.83f);
    }
}