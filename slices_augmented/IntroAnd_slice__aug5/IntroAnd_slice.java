/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IntroAnd_slice {
    @Positive
  void test() {
        return (null * -767);

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

    protected static double __cfwr_calc425(Boolean __cfwr_p0, char __cfwr_p1) {
        float __cfwr_val49 = -53.85f;
        for (int __cfwr_i79 = 0; __cfwr_i79 < 6; __cfwr_i79++) {
            boolean __cfwr_obj53 = true;
        }
        if (true || (false | (null & null))) {
            return 195L;
        }
        if (false || true) {
            while (true) {
            char __cfwr_item31 = 'K';
            break; // Prevent infinite loops
        }
        }
        return 58.30;
    }
    public byte __cfwr_util572(Integer __cfwr_p0, Character __cfwr_p1) {
        for (int __cfwr_i10 = 0; __cfwr_i10 < 6; __cfwr_i10++) {
            for (int __cfwr_i58 = 0; __cfwr_i58 < 4; __cfwr_i58++) {
            try {
            return (true << 95.22f);
        } catch (Exception __cfwr_e98) {
            // ignore
        }
        }
        }
        Character __cfwr_temp2 = null;
        float __cfwr_item67 = (null % -48.54f);
        if (false && false) {
            try {
            for (int __cfwr_i23 = 0; __cfwr_i23 < 7; __cfwr_i23++) {
            if (true && (-61.18f & 79.15f)) {
            char __cfwr_entry65 = (-162 & (22.39f / null));
        }
        }
        } catch (Exception __cfwr_e56) {
            // ignore
        }
        }
        return (-890 % (null & null));
    }
}