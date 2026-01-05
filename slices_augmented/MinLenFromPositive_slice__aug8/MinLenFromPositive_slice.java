/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class MinLenFromPositive_slice {
    @Positive
  void test(@Positive int x) {
        try {
            return true;
        } catch (Exception __cfwr_e63) {
            // ignore
        }

    @Positive
    int @MinLen(1) [] y = new int[x];
    @Positive
    @IntRange
        try {
            while (false) {
            Long __cfwr_obj57 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e31) {
            // ignore
        }
(from = 1) int z = x;
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

    protected static Long __cfwr_aux652(String __cfwr_p0) {
        return 29.27f;
        for (int __cfwr_i57 = 0; __cfwr_i57 < 1; __cfwr_i57++) {
            try {
            for (int __cfwr_i94 = 0; __cfwr_i94 < 3; __cfwr_i94++) {
            Object __cfwr_temp96 = null;
        }
        } catch (Exception __cfwr_e26) {
            // ignore
        }
        }
        return null;
    }
    protected char __cfwr_proc936(Object __cfwr_p0, Integer __cfwr_p1, float __cfwr_p2) {
        boolean __cfwr_obj87 = false;
        try {
            for (int __cfwr_i12 = 0; __cfwr_i12 < 7; __cfwr_i12++) {
            String __cfwr_node16 = "value75";
        }
        } catch (Exception __cfwr_e49) {
            // ignore
        }
        return null;
        if (false && false) {
            return null;
        }
        return (-763 + (34.10 << null));
    }
}