/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IntroAnd_slice {
    @Positive
  void test() {
        double __cfwr_obj66 = 39.93;

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
      @IndexFor("#2") int i, int[] a, @LTLengthOf("#2
        try {
            Object __cfwr_val53 = null;
        } catch (Exception __cfwr_e63) {
            // ignore
        }
") int j, int k, @NonNegative int m) {
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

    private static float __cfwr_util822(Double __cfwr_p0, boolean __cfwr_p1, Double __cfwr_p2) {
        return null;
        try {
            if (true && true) {
            try {
            for (int __cfwr_i3 = 0; __cfwr_i3 < 10; __cfwr_i3++) {
            if (false || ((-17.05 >> true) % null)) {
            try {
            for (int __cfwr_i37 = 0; __cfwr_i37 < 10; __cfwr_i37++) {
            return null;
        }
        } catch (Exception __cfwr_e5) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e37) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e70) {
            // ignore
        }
        return 69.99f;
    }
    private Float __cfwr_func305(short __cfwr_p0, Float __cfwr_p1, String __cfwr_p2) {
        try {
            try {
            return null;
        } catch (Exception __cfwr_e38) {
            // ignore
        }
        } catch (Exception __cfwr_e48) {
            // ignore
        }
        return null;
    }
    protected static Float __cfwr_compute113() {
        while (((-39.69 - -33.32f) ^ -633)) {
            if (false && true) {
            while (false) {
            while (false) {
            boolean __cfwr_obj15 = false;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        while (((false << null) >> -45.65f)) {
            for (int __cfwr_i91 = 0; __cfwr_i91 < 7; __cfwr_i91++) {
            for (int __cfwr_i98 = 0; __cfwr_i98 < 2; __cfwr_i98++) {
            try {
            try {
            return -856;
        } catch (Exception __cfwr_e47) {
            // ignore
        }
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i31 = 0; __cfwr_i31 < 10; __cfwr_i31++) {
            return ((933L + -229) >> 932L);
        }
        byte __cfwr_entry1 = null;
        return null;
    }
}