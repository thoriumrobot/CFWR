/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IntroAnd_slice {
    @Positive
  void test() {
        Character __cfwr_obj98 = null;

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

    protected static int __cfwr_temp919(Long __cfwr_p0) {
        for (int __cfwr_i9 = 0; __cfwr_i9 < 5; __cfwr_i9++) {
            try {
            float __cfwr_result59 = 13.72f;
        } catch (Exception __cfwr_e87) {
            // ignore
        }
        }
        if ((-893 + (null / null)) || (-516L % 50.29)) {
            if (false || false) {
            try {
            while (('0' ^ (false + -55.46f))) {
            return ((null * -80.61f) >> 89.74f);
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e17) {
            // ignore
        }
        }
        }
        for (int __cfwr_i94 = 0; __cfwr_i94 < 9; __cfwr_i94++) {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 4; __cfwr_i74++) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        }
        }
        return ((68.87f + null) + ('Q' ^ null));
    }
}