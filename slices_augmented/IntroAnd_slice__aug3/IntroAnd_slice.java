/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IntroAnd_slice {
    @Positive
  void test() {
        for (int __cfwr_i38 = 0; __cfwr_i38 < 9; __cfwr_i38++) {
            for (int __cfwr_i7 = 0; __cfwr_i7 < 1; __cfwr_i7++) {
            if ((-50.98 << null) && false) {
            for (int __cfwr_i48 = 0; __cfwr_i48 < 7; __cfwr_i48++) {
            if (((false + true) + (null ^ 'B')) && (993L - null)) {
            while (('U' >> (-705L & -39.12f))) {
            while (false) {
            try {
            try {
            try {
            try {
            for (int __cfwr_i79 = 0; __cfwr_i79 < 6; __cfwr_i79++) {
            while (((66.41 % null) ^ 543L)) {
            return ((85.11f >> 'R') * (59.01f & null));
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e91) {
            // ignore
        }
        } catch (Exception __cfwr_e66) {
            // ignore
        }
        } catch (Exception __cfwr_e45) {
            // ignore
        }
        } catch (Exception __cfwr_e64) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
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

    static Float __cfwr_util541(Boolean __cfwr_p0, Double __cfwr_p1) {
        for (int __cfwr_i64 = 0; __cfwr_i64 < 3; __cfwr_i64++) {
            return null;
        }
        return 21.23;
        return null;
    }
}