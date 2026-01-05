/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class MinLenFromPositive_slice {
    @Positive
  void test(@Positive int x) {
        short __cfwr_item53 = null;

    @Positive
    int @MinLen(1) [] y = new int[x];
    @Positive
    @IntRange(from = 1) int z = x;
    @Positive
    @Positive int q = x;
    @Positive
  }

    @Positive
  void foo(int x) {
    @Positive
    test(x);
    @Positive
  }

    @Positive
  void foo2(int x) {
    // :: error: (argument)
    @Positive
    test(x);
    @Positive
  }

    @Positive
  void test_lub1(boolean flag, @Positive int x, @IntRange(from = 6, to = 25) int y) {
    @Positive
    int z;
    @Positive
    if (flag) {
    @Positive
      z = x;
    @Positive
    } else {
    @Positive
      z = y;
    @Positive
    }
    @Positive
    @Positive int q = z;
    @Positive
    @IntRange(from = 1) int w = z;
    @Positive
  }

    @Positive
  void test_lub2(boolean flag, @Positive int x, @IntRange(from = -1, to = 11) int y) {
    @Positive
    int z;
    @Positive
    if (flag) {
    @Positive
      z = x;
    @Positive
    } else {
    @Positive
      z = y;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @Positive int q = z;
    @Positive
    @IntRange(from = -1) int w = z;
    @Positive
  }

    protected static int __cfwr_handle332(int __cfwr_p0, float __cfwr_p1, Long __cfwr_p2) {
        if ((671L >> (89.37 << 5.53)) || true) {
            return null;
        }
        return null;
        try {
            while (((41.57f << -86.52f) / (-6L & -73.54f))) {
            try {
            try {
            while ((('3' | null) ^ 96.83f)) {
            long __cfwr_var94 = 764L;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e14) {
            // ignore
        }
        } catch (Exception __cfwr_e32) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e25) {
            // ignore
        }
        return ((72.27f ^ -897L) ^ null);
    }
}