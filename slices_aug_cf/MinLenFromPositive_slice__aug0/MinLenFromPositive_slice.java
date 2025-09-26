/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class MinLenFromPositive_slice {
  void test(@Positive int x) {
        for (int __cfwr_i53 = 0; __cfwr_i53 < 5; __cfwr_i53++) {
            double __cfwr_item56 = 56.79;
        }

    int @MinLen(1) [] y = new int[x];
    @IntRange(from = 1) int z = x;
    @Positive int q = x;
  }

  @SuppressWarnings("index")
  void foo(int x) {
    test(x);
  }

  void foo2(int x) {
    // :: error: (argument)
    test(x);
  }

  void test_lub1(boolean flag, @Positive int x, @IntRange(from = 6, to = 25) int y) {
    int z;
    if (flag) {
      z = x;
    } else {
      z = y;
    }
    @Positive int q = z;
    @IntRange(from = 1) int w = z;
  }

  void test_lub2(boolean flag, @Positive int x, @IntRange(from = -1, to = 11) int y) {
    int z;
    if (flag) {
      z = x;
    } else {
      z = y;
    }
    // :: error: (assignment)
    @Positive int q = z;
    @IntRange(from = -1) int w = z;
  }

  @Positive int id(@Positive int x) {
    return x;
  }

  void test_id(int param) {
    @Positive int x = id(5);
    @IntRange(from = 1) int y = id(5);

    int @MinLen(1) [] a = new int[id(100)];
    // :: error: (assignment)
    int @MinLen(10) [] c = new int[id(100)];

    int q = id(10);

    if (param == q) {
      int @MinLen(1) [] d = new int[param];
    }
  }

    private static Object __cfwr_proc800(short __cfwr_p0) {
        try {
            byte __cfwr_item73 = (639L % (null * 'T'));
        } catch (Exception __cfwr_e59) {
            // ignore
        }
        double __cfwr_val83 = -15.13;
        return null;
        return null;
    }
}